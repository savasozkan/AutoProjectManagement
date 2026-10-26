"""Local web UI.

Flow: read a document -> find similar tasks already in the project -> the user marks the ones that
already cover part of the work and adds instructions -> generate only the missing subtasks ->
review -> create the master issue and sub-issues on GitHub.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List

from flask import Flask, jsonify, request, send_from_directory

from .client import GitHubError
from .documents import DocumentError, load_document
from . import workflow
from .planner import PlannerError, Subtask, TaskPlanner
from .projects import ProjectItem, ProjectService
from .similarity import SimilarityFinder, strength

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class Jobs:
    """In-memory store for background jobs (reading documents, generating plans)."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, work: Callable[[Callable[[str], None]], Dict[str, Any]]) -> str:
        """Run `work(log)` in a thread; its return value becomes the job result."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {"status": "running", "log": [], "result": None, "error": None}

        def log(message: str) -> None:
            with self._lock:
                self._jobs[job_id]["log"].append(message)

        def run() -> None:
            try:
                result = work(log)
                update = {"status": "done", "result": result}
            except (DocumentError, PlannerError, GitHubError) as exc:
                update = {"status": "error", "error": str(exc)}
            except Exception as exc:  # keep the UI informed instead of hanging
                update = {"status": "error", "error": f"Unexpected error: {exc}"}
            with self._lock:
                self._jobs[job_id].update(update)

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job, log=list(job["log"])) if job else None


def create_app(service: ProjectService) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads
    jobs = Jobs()
    github_lock = threading.Lock()  # one GitHub operation at a time

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/config")
    def config():
        try:
            with github_lock:
                status_field = next(
                    (f for f in service.fields() if f.name.lower() == "status"), None
                )
                project = {"title": service.title, "url": service.url}
        except GitHubError as exc:
            return jsonify(error=str(exc)), 502
        return jsonify(
            project=project,
            repo=service.config.repo,
            statuses=list(status_field.options) if status_field else [],
            model=TaskPlanner().model,
        )

    # -- Step 1+2: read the document and find similar existing tasks ---------

    @app.post("/api/read")
    def read():
        upload = request.files.get("file")
        if not upload or not upload.filename:
            return jsonify(error="Choose a .pdf, .txt or .md file."), 400
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify(error=f"Unsupported file type '{ext}'. Use .pdf, .txt or .md."), 400

        workdir = tempfile.mkdtemp(prefix="task-agent-")
        path = os.path.join(workdir, "upload" + ext)
        upload.save(path)
        filename = upload.filename
        use_vision = request.form.get("use_vision", "true") == "true"

        def work(log):
            try:
                log(f"Reading {filename}...")
                doc = load_document(path, use_vision=use_vision, log=log)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)

            log("Retrieving the tasks already in the project...")
            with github_lock:
                items = service.get_items()
            log(f"Comparing the document with {len(items)} existing task(s)...")
            finder = SimilarityFinder(log=log)
            matches = finder.find(doc.text, items, top_k=len(items))
            log(f"Ranked by {finder.method}.")
            return {
                "document_text": doc.text,
                "document_title": doc.title,
                "pages_described": doc.pages_described,
                "method": finder.method,
                "matches": [
                    dict(asdict(m.item), score=m.score, strength=strength(m.score, finder.method))
                    for m in matches
                ],
            }

        return jsonify(job_id=jobs.start(work))

    # -- Step 3+4: generate the missing subtasks ---------------------------

    @app.post("/api/plan")
    def plan():
        data = request.get_json(silent=True) or {}
        document_text = str(data.get("document_text") or "")
        if not document_text.strip():
            return jsonify(error="Read a document first."), 400
        selected_ids = set(data.get("existing_ids") or [])

        def work(log):
            log("Retrieving the tasks in the project...")
            with github_lock:
                items = service.get_items()
            proposal = workflow.propose(
                document_text,
                items,
                selected_ids,
                prompt=str(data.get("prompt") or ""),
                title=str(data.get("title") or "").strip() or None,
                document_title=data.get("document_title") or None,
                max_tasks=_clamp(data.get("max_tasks"), 1, 25, 10),
                generate_new=data.get("generate_new", True) is not False,
                log=log,
            )
            if proposal.master:
                log(f"New subtasks will go under the existing master #{proposal.master.number}.")
            return {
                "master_title": proposal.plan.master_title,
                "master_description": proposal.plan.master_description,
                "tasks": [asdict(t) for t in proposal.plan.tasks],
                # Set when a ticked master issue is reused instead of creating a new one.
                "existing_master": _master_info(proposal.master) if proposal.master else None,
                "updates": [
                    dict(_item_info(u.item), old_title=u.item.title, old_description=u.item.body or "",
                         title=u.new.title, description=u.new.description)
                    for u in proposal.updates
                ],
                "deletions": [_item_info(i) for i in proposal.deletions],
                "warnings": proposal.warnings,
            }

        return jsonify(job_id=jobs.start(work))

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="Unknown job."), 404
        return jsonify(job)

    # -- Step 5: create on GitHub -------------------------------------------

    @app.post("/api/create")
    def create():
        data = request.get_json(silent=True) or {}
        tasks = [
            Subtask(str(t.get("title", "")).strip(), str(t.get("description", "")).strip())
            for t in data.get("tasks") or []
        ]
        tasks = [t for t in tasks if t.title]
        updates_in = [u for u in data.get("updates") or [] if u.get("item_id")]
        deletions_in = [d for d in data.get("deletions") or [] if d.get("item_id")]
        if not (tasks or updates_in or deletions_in):
            return jsonify(error="Nothing selected to add, update or delete."), 400

        master_issue = data.get("existing_master") or None
        master_title = str(data.get("master_title", "")).strip()
        if tasks and not master_title and not master_issue:
            return jsonify(error="The master task needs a title."), 400
        repo = str(data.get("repo") or service.config.repo or "").strip()
        if tasks and not repo:
            return jsonify(error="Set a repository (owner/name) to create the issues in."), 400
        for d in deletions_in:
            if d.get("mode", "delete") not in workflow.REMOVE_MODES:
                return jsonify(error=f"Unknown removal mode '{d.get('mode')}'."), 400

        fields = {"Status": data["status"]} if data.get("status") else {}
        try:
            with github_lock:
                # Look tasks up again so we act on GitHub's current data, not the browser's copy.
                by_id = {i.id: i for i in service.get_items()}
                missing = [x["item_id"] for x in updates_in + deletions_in if x["item_id"] not in by_id]
                if missing:
                    return jsonify(error="Some tasks are no longer in the project. Generate again."), 409
                master = by_id.get(master_issue.get("item_id")) if master_issue else None
                result = workflow.apply(
                    service,
                    repo,
                    new_tasks=tasks,
                    master_title=master_title,
                    master_description=str(data.get("master_description", "")).strip(),
                    master=master,
                    updates=[
                        workflow.Update(by_id[u["item_id"]], Subtask(
                            str(u.get("title") or by_id[u["item_id"]].title).strip(),
                            str(u.get("description") or "").strip()))
                        for u in updates_in
                    ],
                    deletions=[(by_id[d["item_id"]], d.get("mode", "delete")) for d in deletions_in],
                    fields=fields,
                )
        except GitHubError as exc:
            return jsonify(error=str(exc)), 502
        return jsonify(
            master=result.master,
            existing_master=result.reused_master,
            subtasks=[asdict(s) for s in result.created],
            updated=[_item_info(i) for i in result.updated],
            removed=[dict(_item_info(r["item"]), mode=r["mode"]) for r in result.removed],
            errors=result.errors,
        )

    return app


def _item_info(item: ProjectItem) -> Dict[str, Any]:
    return {
        "item_id": item.id,
        "number": item.number,
        "title": item.title,
        "url": item.url,
        "type": item.type,
        "state": item.state,
        "sub_issue_count": item.sub_issue_count,
    }


def _master_info(item: ProjectItem) -> Dict[str, Any]:
    return {
        "item_id": item.id,
        "number": item.number,
        "title": item.title,
        "url": item.url,
        "content_id": item.content_id,
        "sub_issue_count": item.sub_issue_count,
    }


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def run(service: ProjectService, host: str = "127.0.0.1", port: int = 8000) -> None:
    app = create_app(service)
    print(f"task-agent UI running at http://{host}:{port}  (Ctrl-C to stop)")
    app.run(host=host, port=port, threaded=True)
