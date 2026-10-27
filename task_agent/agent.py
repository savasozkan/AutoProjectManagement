"""A conversational project agent.

The user loads a document and then steers the agent with messages. Special tokens trigger
operations; the agent remembers what happened so far in a Session:

    [SEARCH] <optional query>   find tasks already in the project (about the document, or the query)
    [TASK]  <optional guidance> generate tasks for the document that don't overlap the selected tasks
    [MERGE]  <numbers or text>   merge two or more generated tasks into one ("2, 3: guidance" or "the API tasks")
    [SPLIT]  <numbers or text>   split generated task(s) into smaller ones that don't overlap other tasks
    [ADD]                       add the generated tasks to GitHub
    [HELP]                      list the commands

Plain text without a token is remembered as guidance for the next [TASK]. Several tokens can be
used in one message ("[SEARCH] [TASK] focus on the API"); they run in order.
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import workflow
from .client import GitHubError
from .documents import Document
from .planner import Plan, PlannerError, Subtask, TaskPlanner
from .operations import parse_instructions
from .projects import ProjectItem, ProjectService, split_selection
from .similarity import SimilarityFinder, find_overlaps, strength

Log = Callable[[str], None]
Reply = Dict[str, Any]

COMMANDS = {
    "SEARCH": "find tasks already in the project that relate to the document (or to the text after the token)",
    "TASK": "generate tasks for the document that don't overlap the tasks you selected (text after the token is guidance); "
             "asked again, it keeps the current tasks and only removes the ones that overlap the selected tasks "
             "(use [TASK] regenerate to start over)",
    "MERGE": "merge two or more generated tasks into one: [MERGE] 2, 3 (text after the numbers guides the "
             "merged task) or describe them: [MERGE] the two API tasks",
    "SPLIT": "split a generated task into smaller tasks that don't overlap the others: [SPLIT] 3 (text after the "
             "number says how: [SPLIT] 3: by platform) or describe it: [SPLIT] the mobile app task",
    "ADD": "add the generated tasks to GitHub",
    "HELP": "show this list",
}
ALIASES = {"TASKS": "TASK"}  # old names that still work
_TOKEN = re.compile(r"\[(" + "|".join(list(COMMANDS) + list(ALIASES)) + r")\]", re.IGNORECASE)
_MERGE_NUMBERS = re.compile(r"^\s*#?\d+(?:\s*(?:,|&|\+|\band\b)?\s*#?\d+)*")
_REGENERATE = re.compile(r"^\s*(regenerate|start over|from scratch)\b[\s:,.-]*", re.IGNORECASE)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Every document loaded in this conversation, oldest first: the first is usually the overall
    # scope, later ones are updates. Each is {"name", "text", "title"}.
    documents: List[Dict[str, Any]] = field(default_factory=list)
    selected_ids: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)  # plain-text guidance for the next [TASK]
    proposal: Optional[workflow.Proposal] = None
    proposal_id: int = 0
    proposal_docs: int = 0  # how many documents the current tasks are based on
    added_ids: Set[str] = field(default_factory=set)  # project items created by [ADD] in this conversation
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def document_text(self) -> Optional[str]:
        """All documents as one text; later documents are marked as updates."""
        return combine_documents(self.documents) if self.documents else None

    @property
    def document_title(self) -> Optional[str]:
        return self.documents[0]["title"] if self.documents else None

    @property
    def document_name(self) -> Optional[str]:
        return self.documents[-1]["name"] if self.documents else None


def combine_documents(documents: Sequence[Dict[str, Any]]) -> str:
    if len(documents) == 1:
        return documents[0]["text"]
    parts = []
    for n, doc in enumerate(documents, 1):
        label = "original scope" if n == 1 else "update, received later"
        parts.append(f"=== Document {n} ({label}): {doc['name']} ===\n{doc['text'].strip()}")
    return "\n\n".join(parts)


def parse_message(text: str) -> List[Tuple[Optional[str], str]]:
    """Split a message into (command, argument) pairs. Text before any token has command None."""
    parts = _TOKEN.split(text or "")
    steps: List[Tuple[Optional[str], str]] = []
    if parts[0].strip():
        steps.append((None, parts[0].strip()))
    for command, argument in zip(parts[1::2], parts[2::2]):
        command = command.upper()
        steps.append((ALIASES.get(command, command), argument.strip()))
    return steps


def parse_merge(argument: str) -> Tuple[List[int], str]:
    """'2, 3 and 5: one API task' -> ([2, 3, 5], 'one API task'). Without numbers: ([], text)."""
    match = _MERGE_NUMBERS.match(argument or "")
    if not match:
        return [], (argument or "").strip()
    numbers = [int(n) for n in re.findall(r"\d+", match.group(0))]
    rest = argument[match.end():].strip().lstrip(":-–—,").strip()
    return numbers, rest


class Agent:
    def __init__(
        self,
        service: ProjectService,
        planner_factory: Callable[[], TaskPlanner] = TaskPlanner,
        finder_factory: Callable[..., SimilarityFinder] = SimilarityFinder,
        repo: Optional[str] = None,
        github_lock: Optional[threading.Lock] = None,
    ):
        self.service = service
        self.planner_factory = planner_factory
        self.finder_factory = finder_factory
        self.repo = repo or service.config.repo
        self.github_lock = github_lock or threading.Lock()

    # -- entry points -----------------------------------------------------

    def document_loaded(self, session: Session, name: str, doc: Document) -> List[Reply]:
        """Add a document to the conversation. Earlier documents and generated tasks are kept."""
        session.documents.append({"name": name, "text": doc.text, "title": doc.title})
        visuals = f", including drawings on {doc.pages_described} page(s)" if doc.pages_described else ""
        text = f"I've read **{name}** ({len(doc.text):,} characters{visuals})."
        if len(session.documents) > 1:
            earlier = ", ".join(d["name"] for d in session.documents[:-1])
            text += f" I'll treat it as an update to {earlier}, which I still remember."
        if session.proposal is not None and session.proposal.plan.tasks:
            text += (f" I still have the {len(session.proposal.plan.tasks)} task(s) generated so far: "
                     "use [TASK] to update them with this document (kept, changed, removed or added).")
        else:
            text += (" Next you can [SEARCH] for related tasks already in the project, select the ones "
                     "that already cover part of the work, then use [TASK] to generate the rest.")
        return [_text(text)]

    def handle(
        self,
        session: Session,
        message: str,
        selected_ids: Optional[Sequence[str]] = None,
        edits: Optional[Dict[str, Any]] = None,
        log: Log = lambda msg: None,
    ) -> List[Reply]:
        """Run the commands in one user message. `selected_ids`: tasks currently ticked in the UI.
        `edits`: the user's edits to the latest generated tasks (used by [ADD])."""
        if selected_ids is not None:
            session.selected_ids = set(selected_ids)
        steps = parse_message(message)
        if not steps:
            return [_text("Send a message or a command. " + _help_line())]

        replies: List[Reply] = []
        for command, argument in steps:
            try:
                if command is None:
                    replies.append(self._note(session, argument))
                elif command == "HELP":
                    replies.append(self._help())
                elif command == "SEARCH":
                    replies.append(self._search(session, argument, log))
                elif command == "TASK":
                    replies.append(self._tasks(session, argument, edits, log))
                    edits = None  # the edits were for the tasks that this step replaced
                elif command == "MERGE":
                    replies.append(self._merge(session, argument, edits, log))
                    edits = None
                elif command == "SPLIT":
                    replies.append(self._split(session, argument, edits, log))
                    edits = None
                elif command == "ADD":
                    replies.append(self._add(session, edits, log))
            except (GitHubError, PlannerError) as exc:
                replies.append(_error(f"[{command}] failed: {exc}"))
                break
        return replies

    # -- commands -----------------------------------------------------------

    def _note(self, session: Session, text: str) -> Reply:
        session.notes.append(text)
        return _text(
            "Noted. I'll use this as guidance the next time you ask for [TASK]."
            + ("" if session.document_text else " (Load a document first.)")
        )

    def _help(self) -> Reply:
        lines = [f"- **[{name}]** {what}" for name, what in COMMANDS.items()]
        return _text(
            "Commands:\n" + "\n".join(lines)
            + "\n\nAnything without a token is remembered as guidance for [TASK]. "
            "Inside [TASK] you can also write `[UPDATE] #12` or `[DELETE] #12` for selected tasks."
        )

    def _search(self, session: Session, query: str, log: Log) -> Reply:
        text = query or session.document_text
        if not text:
            return _error("Load a document first, or tell me what to search for: [SEARCH] payments")
        log("Retrieving the tasks in the project...")
        with self.github_lock:
            items = self.service.get_items()
        if not items:
            return _text("The project has no tasks yet, so there's nothing to search.")

        log(f"Comparing {'your query' if query else 'the document'} with {len(items)} task(s)...")
        finder = self.finder_factory(log=log)
        # Every task, most similar first (min_score below 0 keeps tasks with no similarity at all).
        shown = finder.find(text, items, top_k=len(items), min_score=-1.0)
        relevant = [m for m in shown if strength(m.score, finder.method) != "weak"]
        subject = f"“{query}”" if query else "the document"
        if relevant:
            summary = f"{len(relevant)} of the {len(shown)} task(s) in the project relate to {subject}."
        else:
            summary = f"None of the {len(shown)} task(s) in the project closely relate to {subject}."
        summary += (" All tasks are listed below, most similar first. Select the ones that already "
                    "cover part of the work, then use [TASK].")
        return {
            "type": "search",
            "text": summary,
            "query": query,
            "method": finder.method,
            "results": [
                dict(_item_info(m.item), score=m.score, strength=strength(m.score, finder.method),
                     selected=m.item.id in session.selected_ids)
                for m in shown
            ],
        }

    def _tasks(self, session: Session, guidance: str, edits: Optional[Dict[str, Any]], log: Log) -> Reply:
        if not session.document_text:
            return _error("Load a document first, then ask for [TASK].")
        regenerate = _REGENERATE.match(guidance)
        if regenerate:
            guidance = guidance[regenerate.end():].strip()
        elif session.proposal is not None and len(session.documents) > session.proposal_docs:
            # A new document arrived since the tasks were made: update them, don't start over.
            return self._revise(session, guidance, edits, log)
        elif session.proposal is not None:
            # Tasks were already generated: keep them and only drop what the selection covers.
            return self._refine(session, guidance, edits, log)
        log("Retrieving the tasks in the project...")
        with self.github_lock:
            items = self.service.get_items()
        prompt = "\n".join(session.notes + ([guidance] if guidance else []))
        # Tasks added earlier in this conversation count as covered, like selected ones.
        known = {i.id for i in items}
        added = (session.added_ids & known) - session.selected_ids
        if added:
            log(f"Skipping the {len(added)} task(s) added earlier in this conversation...")
        proposal = workflow.propose(
            session.document_text,
            items,
            set(session.selected_ids) | added,
            prompt=prompt,
            document_title=session.document_title,
            max_tasks=10,
            planner=self.planner_factory(),
            log=log,
        )
        session.proposal = proposal
        session.proposal_id += 1
        session.proposal_docs = len(session.documents)
        return self._proposal_reply(session, proposal, intro=None, removed=[], followed_guidance=bool(prompt))

    def _revise(self, session: Session, guidance: str, edits: Optional[Dict[str, Any]], log: Log) -> Reply:
        """Update the generated tasks for the documents loaded since they were made."""
        tasks, master_title, master_description, master, updates, deletions = self._current(session, edits)
        earlier = session.documents[:session.proposal_docs]
        newer = session.documents[session.proposal_docs:]
        prompt = parse_instructions("\n".join(session.notes + ([guidance] if guidance else []))).instructions
        planner = self.planner_factory()
        names = ", ".join(d["name"] for d in newer)
        log(f"Updating {len(tasks)} task(s) for {names} with {planner.model}...")
        revision = planner.revise_tasks(tasks, combine_documents(earlier), combine_documents(newer), prompt)

        # Tasks covered by selected project tasks still don't belong in the list.
        log("Retrieving the tasks in the project...")
        with self.github_lock:
            items = self.service.get_items()
        sel_master, covered = split_selection([i for i in items if i.id in session.selected_ids], items)
        new_tasks, changes = list(revision.tasks), list(revision.changes)
        overlap_removed: List[Dict[str, Any]] = []
        if covered and new_tasks:
            matches = find_overlaps(self.finder_factory(log=log), [f"{t.title}\n{t.description}" for t in new_tasks], covered)
            overlap_removed = [{"title": t.title, "overlaps": _item_info(m.item), "score": m.score}
                               for t, m in zip(new_tasks, matches) if m is not None]
            keep = [m is None for m in matches]
            new_tasks = [t for t, k in zip(new_tasks, keep) if k]
            changes = [c for c, k in zip(changes, keep) if k]

        master = sel_master or master
        description = master_description + "".join(
            f"\n\n### Update: {d['name']}\n{d['text'].strip()}" for d in newer)
        proposal = workflow.Proposal(Plan(master.title if master else master_title, description, new_tasks),
                                     master, updates, deletions, [])
        session.proposal = proposal
        session.proposal_id += 1
        session.proposal_docs = len(session.documents)

        counts = {c: changes.count(c) for c in ("kept", "changed", "new")}
        intro = (f"I updated the tasks for {names}: {counts['kept']} kept, {counts['changed']} changed, "
                 f"{counts['new']} new, {len(revision.removed)} removed.")
        if overlap_removed:
            intro += f" I also removed {len(overlap_removed)} that overlap the selected project task(s)."
        reply = self._proposal_reply(session, proposal, intro=intro, removed=overlap_removed,
                                     followed_guidance=bool(prompt))
        reply["revision"] = {
            "documents": [d["name"] for d in newer],
            "changes": changes,  # aligned with reply["tasks"]
            "removed": [{"title": t.title, "description": t.description, "reason": why}
                        for t, why in revision.removed],
        }
        return reply

    def _refine(self, session: Session, guidance: str, edits: Optional[Dict[str, Any]], log: Log) -> Reply:
        """Keep the generated tasks (with the user's edits) and remove those that overlap the
        selected project tasks. Nothing new is generated."""
        tasks, master_title, master_description, _, updates, deletions = self._current(session, edits)

        log("Retrieving the tasks in the project...")
        with self.github_lock:
            items = self.service.get_items()
        master, covered = split_selection([i for i in items if i.id in session.selected_ids], items)

        removed: List[Dict[str, Any]] = []
        kept = list(tasks)
        if covered and tasks:
            log(f"Checking {len(tasks)} task(s) against the {len(covered)} selected task(s)...")
            matches = find_overlaps(self.finder_factory(log=log), [f"{t.title}\n{t.description}" for t in tasks], covered)
            kept = [t for t, m in zip(tasks, matches) if m is None]
            removed = [{"title": t.title, "overlaps": _item_info(m.item), "score": m.score}
                       for t, m in zip(tasks, matches) if m is not None]

        proposal = workflow.Proposal(
            Plan(master.title if master else master_title, master_description, kept),
            master, updates, deletions, [],
        )
        session.proposal = proposal
        session.proposal_id += 1
        if guidance:
            session.notes.append(guidance)
        if removed:
            intro = (f"I kept your tasks and removed {len(removed)} that overlap the selected task(s); "
                     "no new tasks were generated.")
        elif covered:
            intro = "None of your tasks overlap the selected task(s), so I kept them all; no new tasks were generated."
        else:
            intro = "No tasks are selected, so I kept the tasks as they are."
        if guidance:
            intro += " I saved your guidance for when you use [TASK] regenerate."
        return self._proposal_reply(session, proposal, intro=intro, removed=removed, followed_guidance=False)

    def _merge(self, session: Session, argument: str, edits: Optional[Dict[str, Any]], log: Log) -> Reply:
        if session.proposal is None:
            return _error("There are no generated tasks to merge yet. Use [TASK] first.")
        tasks, master_title, master_description, master, updates, deletions = self._current(session, edits)
        if len(tasks) < 2:
            return _error("There are fewer than two tasks, so there's nothing to merge.")

        numbers, guidance = parse_merge(argument)
        planner = self.planner_factory()
        if numbers:
            bad = [n for n in numbers if not 1 <= n <= len(tasks)]
            if bad:
                return _error(f"There's no task {', '.join(map(str, bad))}; the tasks are numbered 1 to {len(tasks)}.")
            positions = sorted({n - 1 for n in numbers})
        elif guidance:
            log("Working out which tasks you mean...")
            positions = planner.pick_tasks(tasks, guidance)
        else:
            positions = []
        if len(positions) < 2:
            return _error("Tell me which tasks to merge (at least two), e.g. [MERGE] 2, 3 "
                          "or [MERGE] the two API tasks.")

        chosen = [tasks[p] for p in positions]
        log(f"Merging {len(chosen)} tasks with {planner.model}...")
        # When the tasks were described in words, that text is also the best guidance we have.
        merged = planner.merge_tasks(chosen, guidance)
        new_tasks = [merged if p == positions[0] else t for p, t in enumerate(tasks) if p not in positions[1:]]

        proposal = workflow.Proposal(Plan(master_title, master_description, new_tasks), master, updates, deletions, [])
        session.proposal = proposal
        session.proposal_id += 1
        intro = (f"I merged tasks {', '.join(str(p + 1) for p in positions)} into “{merged.title}”. "
                 "Nothing else changed.")
        reply = self._proposal_reply(session, proposal, intro=intro, removed=[], followed_guidance=False)
        reply["merged"] = {"title": merged.title, "position": positions[0] + 1,
                           "from": [t.title for t in chosen]}
        return reply

    def _split(self, session: Session, argument: str, edits: Optional[Dict[str, Any]], log: Log) -> Reply:
        if session.proposal is None:
            return _error("There are no generated tasks to split yet. Use [TASK] first.")
        tasks, master_title, master_description, master, updates, deletions = self._current(session, edits)
        if not tasks:
            return _error("There are no tasks to split.")

        numbers, guidance = parse_merge(argument)
        planner = self.planner_factory()
        if numbers:
            bad = [n for n in numbers if not 1 <= n <= len(tasks)]
            if bad:
                return _error(f"There's no task {', '.join(map(str, bad))}; the tasks are numbered 1 to {len(tasks)}.")
            positions = sorted({n - 1 for n in numbers})
        elif guidance:
            log("Working out which task you mean...")
            positions = planner.pick_tasks(tasks, guidance, action="split")
        else:
            positions = []
        if not positions:
            return _error("Tell me which task to split, e.g. [SPLIT] 3, [SPLIT] 3: by platform, "
                          "or [SPLIT] the mobile app task.")

        # Ticked project tasks also count: split parts must not redo that work either.
        covered: List[ProjectItem] = []
        if session.selected_ids:
            with self.github_lock:
                items = self.service.get_items()
            covered = split_selection([i for i in items if i.id in session.selected_ids], items)[1]

        finder = self.finder_factory(log=log)
        results = []  # one entry per task we tried to split
        new_tasks = list(tasks)
        for pos in reversed(positions):  # back to front, so earlier positions stay valid
            task = tasks[pos]
            others = [t for p, t in enumerate(tasks) if p != pos]
            log(f"Splitting task {pos + 1} “{task.title}” with {planner.model}...")
            parts = planner.split_task(task, guidance, [f"{t.title}: {t.description}" for t in others],
                                       session.document_text)
            # Drop parts that repeat another task in the list or a ticked project task.
            compare = [_as_item(p + 1, t) for p, t in enumerate(tasks) if p != pos] + covered
            overlaps = find_overlaps(finder, [f"{x.title}\n{x.description}" for x in parts], compare) if parts else []
            kept = [x for x, m in zip(parts, overlaps) if m is None]
            dropped = [{"title": x.title, "overlaps": _ref(m.item)} for x, m in zip(parts, overlaps) if m is not None]
            if len(kept) >= 2:
                new_tasks[pos:pos + 1] = kept
            results.append({"from": task.title, "position": pos + 1, "parts": [x.title for x in kept],
                            "dropped": dropped, "split": len(kept) >= 2})
        results.reverse()

        # Positions of the new parts in the final list (for highlighting).
        offset = 0
        for r in results:
            r["position"] += offset
            if r["split"]:
                offset += len(r["parts"]) - 1

        done = [r for r in results if r["split"]]
        failed = [r for r in results if not r["split"]]
        if not done:
            return _error("I couldn't split " + ", ".join(f"task {r['position']}" for r in failed)
                          + " into at least two parts that don't overlap other tasks. "
                          "Try saying how to split it, e.g. [SPLIT] 3: frontend and backend.")

        proposal = workflow.Proposal(Plan(master_title, master_description, new_tasks), master, updates, deletions, [])
        session.proposal = proposal
        session.proposal_id += 1
        intro = " ".join(f"I split “{r['from']}” into {len(r['parts'])} tasks." for r in done)
        if any(r["dropped"] for r in done):
            intro += " I left out parts that would overlap other tasks."
        if failed:
            intro += " I kept " + ", ".join(f"“{r['from']}”" for r in failed) + " as it was: it couldn't be split without overlapping other tasks."
        reply = self._proposal_reply(session, proposal, intro=intro + " Nothing else changed.", removed=[],
                                     followed_guidance=False)
        reply["split"] = results
        return reply

    def _current(self, session: Session, edits: Optional[Dict[str, Any]]):
        """The latest generated tasks as the user sees them: with their edits and ticks, if the
        edits belong to the latest proposal. Returns (tasks, master_title, master_description,
        master, updates, deletions)."""
        old = session.proposal
        if edits and edits.get("proposal_id") == session.proposal_id:
            tasks, master_title, master_description, master, updates, _ = self._apply_edits(old, edits)
            ticked_ids = {d["item_id"] for d in edits.get("deletions") or []}
            deletions = [i for i in old.deletions if i.id in ticked_ids]
            return tasks, master_title, master_description, master, updates, deletions
        return (list(old.plan.tasks), old.plan.master_title, old.plan.master_description,
                old.master, list(old.updates), list(old.deletions))

    def _proposal_reply(self, session: Session, proposal: workflow.Proposal, intro: Optional[str],
                        removed: List[Dict[str, Any]], followed_guidance: bool) -> Reply:
        parts = [intro] if intro else []
        where = (f"under the existing master #{proposal.master.number}" if proposal.master
                 else "under a new master task")
        if not proposal.plan.tasks:
            parts.append("No new tasks are needed: the selected tasks already cover the document.")
        elif intro:
            parts.append(f"{len(proposal.plan.tasks)} task(s) remain, {where}.")
        else:
            parts.append(f"I generated {len(proposal.plan.tasks)} new task(s) {where}.")
            if session.selected_ids:
                parts.append(f"I skipped work covered by the {len(session.selected_ids)} task(s) you selected.")
        if proposal.updates or proposal.deletions:
            parts.append(f"Also proposed: {len(proposal.updates)} update(s), {len(proposal.deletions)} deletion(s).")
        if followed_guidance:
            parts.append("I followed your guidance.")
        parts.append("Review and edit them below, then send [ADD] to add them to GitHub.")
        return {
            "type": "proposal",
            "text": " ".join(parts),
            "proposal_id": session.proposal_id,
            "master": _item_info(proposal.master) if proposal.master else None,
            "master_title": proposal.plan.master_title,
            "master_description": proposal.plan.master_description,
            "tasks": [asdict(t) for t in proposal.plan.tasks],
            "updates": [
                dict(_item_info(u.item), old_title=u.item.title, old_description=u.item.body or "",
                     title=u.new.title, description=u.new.description)
                for u in proposal.updates
            ],
            "deletions": [_item_info(i) for i in proposal.deletions],
            "warnings": proposal.warnings,
            "removed": removed,  # tasks dropped because they overlap a selected project task
        }

    def _add(self, session: Session, edits: Optional[Dict[str, Any]], log: Log) -> Reply:
        proposal = session.proposal
        if proposal is None:
            return _error("There's nothing to add yet. Use [TASK] to generate tasks first.")
        if edits and edits.get("proposal_id") not in (None, session.proposal_id):
            return _error("Those edits belong to an older set of tasks. Use the latest one, or [TASK] again.")

        tasks, master_title, master_description, master, updates, deletions = self._apply_edits(proposal, edits)
        if not (tasks or updates or deletions):
            return _error("Nothing is selected to add.")
        if tasks and not self.repo:
            return _error("No repository is set (GITHUB_REPO in .env), so I can't create issues.")

        log("Adding to GitHub...")
        with self.github_lock:
            by_id = {i.id: i for i in self.service.get_items()}  # act on current data
            result = workflow.apply(
                self.service, self.repo, new_tasks=tasks,
                master_title=master_title, master_description=master_description,
                master=by_id.get(master.id) if master else None,
                updates=[workflow.Update(by_id[u.item.id], u.new) for u in updates if u.item.id in by_id],
                deletions=[(by_id[i.id], mode) for i, mode in deletions if i.id in by_id],
            )
        session.proposal = None  # applied: can't be added twice
        # Remember what we added, so later [TASK] (e.g. after an update document) don't redo it.
        session.added_ids.update(c.item_id for c in result.created)
        if result.master and result.master.get("item_id"):
            session.added_ids.add(result.master["item_id"])

        bits = []
        if result.created:
            where = "the existing master" if result.reused_master else "a new master"
            bits.append(f"Added {len(result.created)} task(s) under {where} #{result.master['number']}.")
        if result.updated:
            bits.append(f"Updated {len(result.updated)} task(s).")
        if result.removed:
            bits.append(f"Removed {len(result.removed)} task(s).")
        if result.errors:
            bits.append(f"{len(result.errors)} change(s) failed.")
        return {
            "type": "added",
            "text": " ".join(bits) or "Nothing was changed.",
            "master": result.master,
            "subtasks": [dict(asdict(c), title=t.title) for c, t in zip(result.created, tasks)],
            "updated": [_item_info(i) for i in result.updated],
            "removed": [dict(_item_info(r["item"]), mode=r["mode"]) for r in result.removed],
            "errors": result.errors,
        }

    @staticmethod
    def _apply_edits(proposal: workflow.Proposal, edits: Optional[Dict[str, Any]]):
        """Combine the stored proposal with the user's edits from the UI (if any)."""
        if not edits:
            return (list(proposal.plan.tasks), proposal.plan.master_title,
                    proposal.plan.master_description, proposal.master, list(proposal.updates),
                    [(i, "delete" if i.type == "ISSUE" else "remove") for i in proposal.deletions])

        tasks = [Subtask(str(t.get("title", "")).strip(), str(t.get("description", "")).strip())
                 for t in edits.get("tasks") or []]
        tasks = [t for t in tasks if t.title]
        master = proposal.master if edits.get("use_existing_master", True) else None
        updates_by_id = {u.item.id: u for u in proposal.updates}
        updates = [
            workflow.Update(updates_by_id[u["item_id"]].item,
                            Subtask(str(u.get("title") or "").strip() or updates_by_id[u["item_id"]].item.title,
                                    str(u.get("description") or "").strip()))
            for u in edits.get("updates") or [] if u.get("item_id") in updates_by_id
        ]
        deletions_by_id = {i.id: i for i in proposal.deletions}
        deletions = [
            (deletions_by_id[d["item_id"]], d.get("mode", "delete"))
            for d in edits.get("deletions") or []
            if d.get("item_id") in deletions_by_id and d.get("mode", "delete") in workflow.REMOVE_MODES
        ]
        return (tasks, str(edits.get("master_title") or proposal.plan.master_title).strip(),
                str(edits.get("master_description") or proposal.plan.master_description).strip(),
                master, updates, deletions)


# -- helpers ----------------------------------------------------------------

def _text(text: str) -> Reply:
    return {"type": "text", "text": text}


def _error(text: str) -> Reply:
    return {"type": "error", "text": text}


def _help_line() -> str:
    return "Commands: " + ", ".join(f"[{c}]" for c in COMMANDS) + "."


def _as_item(position: int, task: Subtask) -> ProjectItem:
    """A generated task wrapped as a project item, so it can be compared like one."""
    return ProjectItem(id=f"task-{position}", type="DRAFT_ISSUE", title=task.title, url=None,
                       values={}, body=task.description)


def _ref(item: ProjectItem) -> Dict[str, Any]:
    """How to refer to an item that a split part overlaps: a project task or another generated task."""
    if item.id.startswith("task-"):
        return {"label": f"task {item.id[5:]}", "title": item.title, "url": None}
    return {"label": f"#{item.number}", "title": item.title, "url": item.url}


def _item_info(item: ProjectItem) -> Dict[str, Any]:
    return {
        "item_id": item.id,
        "number": item.number,
        "title": item.title,
        "url": item.url,
        "type": item.type,
        "state": item.state,
        "status": item.status,
        "body": item.body,
        "parent": item.parent,
        "sub_issue_count": item.sub_issue_count,
        "is_done": item.is_done,
    }
