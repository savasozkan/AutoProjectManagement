from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from .client import GitHubError
from .documents import Document, DocumentError, load_document, split_title
from .planner import ExistingTask, Plan, PlannerError, TaskPlanner
from .projects import ProjectItem, ProjectService, split_selection
from .similarity import SimilarityFinder, strength
from . import workflow


def _parse_field_args(pairs: Optional[List[str]]) -> Dict[str, str]:
    fields = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"Invalid --field '{pair}'. Use NAME=VALUE, e.g. Status=Todo")
        name, value = pair.split("=", 1)
        fields[name.strip()] = value.strip()
    return fields


def cmd_info(service: ProjectService, args: argparse.Namespace) -> None:
    print(f"{service.title}  ({service.url})\n")
    print("Fields:")
    for f in service.fields():
        extra = f"  [{', '.join(f.options)}]" if f.options else ""
        print(f"  - {f.name} ({f.data_type}){extra}")


def cmd_list(service: ProjectService, args: argparse.Namespace) -> None:
    items = service.get_items(status=args.status)
    if args.json:
        print(json.dumps([asdict(i) for i in items], indent=2))
        return
    if not items:
        print("No items." if args.status is None else f"No items with Status '{args.status}'.")
        return
    for item in items:
        ref = f"#{item.number}" if item.number else "draft"
        state = f" ({item.state.lower()})" if item.state and item.state != "OPEN" else ""
        parent = f"  ↳ sub-issue of #{item.parent['number']}" if item.parent else ""
        print(f"[{item.status or '-'}]  {ref}  {item.title}{state}{parent}")
    print(f"\n{len(items)} item(s)")


def cmd_add(service: ProjectService, args: argparse.Namespace) -> None:
    fields = _parse_field_args(args.field)
    if args.url:
        item_id = service.add_existing(args.url, fields)
    else:
        item_id = service.add_draft(args.title, args.body, fields)
    print(f"Added item {item_id}")


def _load_file(path: str, args: argparse.Namespace) -> Document:
    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    doc = load_document(
        path, args.vision_model, not args.no_vision, log=log, instructions=args.prompt
    )
    if doc.pages_described:
        log(f"Described visuals on {doc.pages_described} page(s).")
    return doc


def _read_description(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    """Return the description text and, if the document has one, its title."""
    path = args.file
    if not path and args.description:
        # A path to an existing file is read; anything else is the description itself.
        text = args.description
        looks_like_path = " " not in text.strip() and os.path.splitext(text)[1] != ""
        if os.path.isfile(text) or looks_like_path:
            path = text
        else:
            return text, None
    if path:
        doc = _load_file(path, args)
        return doc.text, doc.title
    if not sys.stdin.isatty():
        return sys.stdin.read(), None
    print("Describe the master task, then press Ctrl-D on an empty line:")
    return sys.stdin.read(), None


def _ask(prompt: str, lower: bool = True) -> str:
    # Read the answer from the terminal even if the description came from a pipe.
    try:
        with open("/dev/tty") as tty:
            print(prompt, end="", flush=True)
            answer = tty.readline().strip()
            return answer.lower() if lower else answer
    except OSError:
        return ""


def _choose_items(items: List[ProjectItem], document: str, args: argparse.Namespace) -> List[ProjectItem]:
    if args.covered:
        wanted = {int(n.strip().lstrip("#")) for n in args.covered.split(",") if n.strip()}
        chosen = [i for i in items if i.number in wanted]
        missing = wanted - {i.number for i in chosen}
        if missing:
            raise RuntimeError(f"Not in the project: {', '.join(f'#{n}' for n in sorted(missing))}")
        return chosen
    if args.no_similar or args.yes:
        return []

    print(f"Comparing with {len(items)} task(s) already in the project...", file=sys.stderr)
    finder = SimilarityFinder(log=lambda m: print(m, file=sys.stderr))
    matches = [m for m in finder.find(document, items, top_k=15)
               if strength(m.score, finder.method) != "weak"]
    if not matches:
        print("No similar tasks found in the project.\n")
        return []

    print("\nSimilar tasks already in the project:")
    for n, m in enumerate(matches, 1):
        ref = f"#{m.item.number}" if m.item.number else "draft"
        level = strength(m.score, finder.method)
        master = f", master with {m.item.sub_issue_count} sub-issues" if m.item.is_master else ""
        print(f"  {n}. [{m.item.status or '-'}] {ref} {m.item.title}  ({level} match{master})")
    print("Tip: tick a master to add the new subtasks under it instead of creating a new master.")
    answer = _ask("Which already cover part of this work? Numbers like 1,3 (Enter for none): ")
    picked = []
    for part in answer.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(matches):
            picked.append(matches[int(part) - 1].item)
    return picked


def _to_existing(item) -> ExistingTask:
    return ExistingTask(item.title or "", item.body or "", item.status, item.number, item.url)


def cmd_plan(service: ProjectService, args: argparse.Namespace) -> None:
    repo = args.repo or service.config.repo
    if not repo and not args.drafts and not args.dry_run:
        raise RuntimeError(
            "No repository set for the issues. Add GITHUB_REPO=owner/name to .env, "
            "pass --repo owner/name, or use --drafts."
        )
    text, doc_title = _read_description(args)
    if args.show_text:
        print(f"--- Text sent to the planner ---\n{split_title(text)[1]}\n--------------------------------")
    fields = _parse_field_args(args.field)

    items = service.get_items()
    picked = _choose_items(items, text, args)
    master, covered = split_selection(picked, items)
    if master:
        if args.drafts:
            raise RuntimeError("Draft items can't be added under a master issue. Drop --drafts.")
        print(f"New subtasks will be added under the existing master #{master.number} "
              f"{master.title} ({master.sub_issue_count} sub-issues).")
    if covered:
        print(f"Won't generate subtasks for: {', '.join(_to_existing(i).label() for i in covered)}")

    instructions = args.prompt
    if instructions is None and not args.yes:
        print("Instructions may include lines like '[UPDATE] #33' or '[DELETE] Task name'.")
        instructions = _ask("Extra instructions for the planner (Enter to skip): ", lower=False)
    planner = TaskPlanner(model=args.model)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    while True:
        proposal = workflow.propose(
            text, items, {i.id for i in picked}, prompt=instructions, title=args.title,
            document_title=doc_title, max_tasks=args.max_tasks, min_tasks=args.min_tasks,
            generate_new=not args.no_new, planner=planner, log=log,
        )
        deletions = [(i, args.delete_mode if i.type == "ISSUE" else "remove") for i in proposal.deletions]
        _print_proposal(proposal, deletions)
        changes = len(proposal.plan.tasks) + len(proposal.updates) + len(deletions)
        if not changes:
            print("Nothing to change: the existing tasks already cover this document.")
            return
        if args.dry_run:
            return
        if args.yes:
            break
        answer = _ask(f"Apply these {changes} change(s)? [y]es / [r]egenerate / [n]o: ")
        if answer in ("y", "yes"):
            break
        if answer not in ("r", "regenerate"):
            print("Nothing changed.")
            return

    new_tasks = proposal.plan.tasks
    if args.drafts and new_tasks:
        ids = service.add_plan_as_drafts(proposal.plan, fields)
        print(f"Added master task and {len(ids) - 1} subtasks as drafts to '{service.title}'.")
        new_tasks = []

    result = workflow.apply(
        service, repo, new_tasks=new_tasks,
        master_title=proposal.plan.master_title, master_description=proposal.plan.master_description,
        master=proposal.master, updates=proposal.updates, deletions=deletions, fields=fields,
    )
    if result.created:
        verb = "Added to the existing master" if result.reused_master else "Created master"
        print(f"{verb} #{result.master['number']}  {result.master['url']}")
        for issue in result.created:
            print(f"  sub-issue: #{issue.number}  {issue.url}")
    for item in result.updated:
        print(f"Updated #{item.number}  {item.url or ''}".rstrip())
    for removed in result.removed:
        print(f"Removed #{removed['item'].number} {removed['item'].title} ({_MODE_TEXT[removed['mode']]})")
    for error in result.errors:
        print(f"Error: {error}", file=sys.stderr)


_MODE_TEXT = {"delete": "deleted permanently", "close": "closed", "remove": "removed from the project"}


def _print_proposal(proposal: "workflow.Proposal", deletions: List[Tuple[ProjectItem, str]]) -> None:
    for warning in proposal.warnings:
        print(f"Warning: {warning}")
    if proposal.updates:
        print("\nTasks to update:")
        for u in proposal.updates:
            title = u.new.title if u.new.title == u.item.title else f"{u.item.title}  ->  {u.new.title}"
            print(f"  #{u.item.number} {title}\n     old: {u.item.body or '(empty)'}\n     new: {u.new.description}")
    if deletions:
        print("\nTasks to delete:")
        for item, mode in deletions:
            print(f"  #{item.number} {item.title}  ({_MODE_TEXT[mode]})")
    if proposal.plan.tasks:
        where = f"under existing master #{proposal.master.number}" if proposal.master else "new master"
        print(f"\nNew subtasks ({where}: {proposal.plan.master_title}):")
        for i, task in enumerate(proposal.plan.tasks, 1):
            print(f"  {i}. {task.title}")
            if task.description:
                print(f"     {task.description}")
    print()


def cmd_ui(service: ProjectService, args: argparse.Namespace) -> None:
    from .web import run  # imported here so Flask is only needed for the UI

    run(service, args.host, args.port)


def cmd_set(service: ProjectService, args: argparse.Namespace) -> None:
    service.set_fields(args.item_id, _parse_field_args(args.field))
    print("Updated.")


def cmd_delete(service: ProjectService, args: argparse.Namespace) -> None:
    service.delete_item(args.item_id)
    print("Deleted.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="task-agent", description="Manage a GitHub Project (v2).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show project title and fields").set_defaults(func=cmd_info)
    list_ = sub.add_parser("list", help="List items in the project")
    list_.add_argument("--status", help="Only items with this Status, e.g. Todo")
    list_.add_argument("--json", action="store_true", help="Print the items as JSON")
    list_.set_defaults(func=cmd_list)

    add = sub.add_parser("add", help="Add a draft item, or an existing issue/PR by URL")
    target = add.add_mutually_exclusive_group(required=True)
    target.add_argument("--title", help="Title for a new draft item")
    target.add_argument("--url", help="URL of an existing issue or pull request")
    add.add_argument("--body", help="Body text for a draft item")
    add.add_argument("--field", action="append", metavar="NAME=VALUE", help="Set a field (repeatable)")
    add.set_defaults(func=cmd_add)

    plan = sub.add_parser("plan", help="Break a master-task description into subtasks with an AI model")
    plan.add_argument(
        "description", nargs="?",
        help="Path to a .txt, .md or .pdf file with the description, or the description itself",
    )
    plan.add_argument("--file", help="Read the description from a text file")
    plan.add_argument(
        "--title",
        help="Master-task title (default: a '# Title' first line in the file, else the model's suggestion)",
    )
    plan.add_argument("--model", help="Ollama model (default: $OLLAMA_MODEL or qwen3:4b)")
    plan.add_argument("--min-tasks", type=int, default=3)
    plan.add_argument("--max-tasks", type=int, default=10)
    plan.add_argument("--field", action="append", metavar="NAME=VALUE", help="Set on every created item")
    plan.add_argument("--repo", help="owner/name to create the issues in (default: $GITHUB_REPO)")
    plan.add_argument("--drafts", action="store_true", help="Add draft items instead of linked issues")
    plan.add_argument("--vision-model", help="Ollama model for PDF drawings (default: $OLLAMA_VISION_MODEL or qwen3-vl:4b)")
    plan.add_argument("--no-vision", action="store_true", help="Read only the text of PDFs; skip drawings")
    plan.add_argument(
        "-p", "--prompt",
        help="Extra instructions for the planner, e.g. 'Focus on the database part'",
    )
    plan.add_argument(
        "--covered", metavar="NUMBERS",
        help="Issue numbers already covering part of the work, e.g. 12,15 (skips the question)",
    )
    plan.add_argument("--no-similar", action="store_true", help="Don't look for similar existing tasks")
    plan.add_argument(
        "--delete-mode", choices=["delete", "close", "remove"], default="delete",
        help="What [DELETE] does to an issue: delete it permanently (default), close it, "
             "or only remove it from the project",
    )
    plan.add_argument("--no-new", action="store_true", help="Only apply [UPDATE]/[DELETE]; don't generate new subtasks")
    plan.add_argument("--show-text", action="store_true", help="Print the text extracted from the document")
    plan.add_argument("--dry-run", action="store_true", help="Show the subtasks without adding them")
    plan.add_argument("-y", "--yes", action="store_true", help="Add without asking")
    plan.set_defaults(func=cmd_plan)

    ui = sub.add_parser("ui", help="Open the web UI for loading documents and reviewing subtasks")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: this computer only)")
    ui.set_defaults(func=cmd_ui)

    set_ = sub.add_parser("set", help="Set fields on an existing item")
    set_.add_argument("item_id")
    set_.add_argument("--field", action="append", metavar="NAME=VALUE", required=True)
    set_.set_defaults(func=cmd_set)

    delete = sub.add_parser("delete", help="Remove an item from the project")
    delete.add_argument("item_id")
    delete.set_defaults(func=cmd_delete)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        service = ProjectService.from_env()
        args.func(service, args)
    except (GitHubError, PlannerError, DocumentError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
