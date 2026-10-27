"""Turn a document + the user's choices into a set of proposed changes, then apply them.

propose(): new subtasks, [UPDATE] rewrites and [DELETE] removals, for the user to review.
apply():   carry out the changes the user kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

from .client import GitHubError
from .documents import split_title
from .operations import parse_instructions, resolve_commands
from .planner import ExistingTask, Plan, Subtask, TaskPlanner
from .projects import CreatedIssue, FieldValue, ProjectItem, ProjectService, split_selection

Log = Callable[[str], None]
REMOVE_MODES = ("delete", "close", "remove")


@dataclass
class Update:
    item: ProjectItem
    new: Subtask  # proposed title + description


@dataclass
class Proposal:
    plan: Plan  # new subtasks (may be empty)
    master: Optional[ProjectItem] = None  # existing master the new subtasks go under
    updates: List[Update] = field(default_factory=list)
    deletions: List[ProjectItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def propose(
    document_text: str,
    items: Sequence[ProjectItem],
    selected_ids: Set[str] = frozenset(),
    prompt: Optional[str] = None,
    title: Optional[str] = None,
    document_title: Optional[str] = None,
    max_tasks: int = 10,
    min_tasks: int = 3,
    generate_new: bool = True,
    planner: Optional[TaskPlanner] = None,
    log: Log = lambda msg: None,
) -> Proposal:
    planner = planner or TaskPlanner()
    parsed = parse_instructions(prompt)
    instructions = parsed.instructions or None
    warnings: List[str] = []

    # 1. What the user ticked: a master to reuse, and work that's already covered.
    master, covered = split_selection([i for i in items if i.id in selected_ids], list(items))

    # 2. [UPDATE] / [DELETE] commands.
    to_update: List[ProjectItem] = []
    deletions: List[ProjectItem] = []
    for r in resolve_commands(parsed.commands, items):
        label = f"[{r.command.action}] {r.command.target}"
        if r.item is None:
            warnings.append(f"{label}: {r.note}")
            continue
        ref = f"#{r.item.number} '{r.item.title}'" if r.item.number else f"'{r.item.title}'"
        # Only tasks the user ticked in step 2, and only unfinished ones, may be changed.
        if r.item.id not in selected_ids:
            warnings.append(f"{label}: skipped, {ref} isn't ticked. Tick it in step 2 to change it.")
            continue
        if r.item.is_done:
            warnings.append(f"{label}: skipped, {ref} is already done.")
            continue
        if r.note:
            warnings.append(f"{label}: matched {ref} (closest match).")
        bucket = to_update if r.command.action == "UPDATE" else deletions
        if r.item not in bucket:
            bucket.append(r.item)

    for item in [i for i in to_update if i in deletions]:
        warnings.append(f"#{item.number} '{item.title}' is marked for both update and delete; it will only be deleted.")
        to_update.remove(item)
    if master and master in deletions:
        warnings.append(f"The ticked master #{master.number} is being deleted, so a new master will be created.")
        master = None
    for item in deletions:
        if item.is_master:
            warnings.append(f"#{item.number} is a master with {item.sub_issue_count} sub-issue(s); "
                            "they'll stay but lose their master.")

    heading_title, description = split_title(document_text)

    # 3. Rewrite the tasks to update.
    updates: List[Update] = []
    for n, item in enumerate(to_update, 1):
        log(f"Updating #{item.number or 'draft'} '{item.title}' ({n}/{len(to_update)})...")
        updates.append(Update(item, planner.rewrite_task(description, item.title, item.body or "", instructions)))

    # 4. New subtasks, skipping covered, updated and deleted work.
    # Updated tasks are described by their new version.
    rewritten = {u.item.id: u.new for u in updates}
    existing = [_existing(i, rewritten.get(i.id)) for i in covered if i not in deletions]
    known = {i.id for i in covered}
    existing += [_existing(u.item, u.new) for u in updates if u.item.id not in known]
    existing += [
        ExistingTask(f"{i.title} (being removed: not needed)", "", None, i.number, i.url)
        for i in deletions
    ]
    plan_title = master.title if master else (title or heading_title or document_title)
    if generate_new:
        notes = []
        if existing:
            notes.append(f"skipping work covered by {len(existing)} existing task(s)")
        if instructions:
            notes.append("using your instructions")
        log(f"Generating new subtasks with {planner.model}"
            + (f" ({', '.join(notes)})" if notes else "") + "...")
        # Only tasks the user ticked as covered can make "nothing new" a valid answer;
        # tasks named in [UPDATE]/[DELETE] don't cover the rest of the document.
        low = 0 if covered else min(min_tasks, max_tasks)
        plan = planner.plan(description, plan_title, low, max_tasks,
                            instructions=instructions, existing=existing)
    else:
        plan = Plan(plan_title or "Master task", description, [])

    log(f"Proposed {len(plan.tasks)} new subtask(s), {len(updates)} update(s), {len(deletions)} deletion(s).")
    return Proposal(plan, master, updates, deletions, warnings)


@dataclass
class ApplyResult:
    master: Optional[Dict] = None  # {"number", "url", "title"}
    reused_master: bool = False
    created: List[CreatedIssue] = field(default_factory=list)
    updated: List[ProjectItem] = field(default_factory=list)
    removed: List[Dict] = field(default_factory=list)  # {"item": ProjectItem, "mode": str}
    errors: List[str] = field(default_factory=list)


def apply(
    service: ProjectService,
    repo: Optional[str],
    new_tasks: Sequence[Subtask] = (),
    master_title: str = "",
    master_description: str = "",
    master: Optional[ProjectItem] = None,
    updates: Sequence[Update] = (),
    deletions: Sequence[tuple] = (),  # (ProjectItem, mode)
    fields: Optional[Dict[str, FieldValue]] = None,
) -> ApplyResult:
    """Create, update and remove tasks. Keeps going after a failure and reports each error."""
    result = ApplyResult()

    if new_tasks:
        if not repo:
            result.errors.append("No repository set, so the new subtasks weren't created.")
        else:
            try:
                if master:
                    result.created = service.add_to_master(master.content_id, list(new_tasks), repo, fields)
                    result.master = {"number": master.number, "url": master.url, "title": master.title,
                                     "item_id": master.id}
                    result.reused_master = True
                else:
                    created, subs = service.add_plan(
                        Plan(master_title or "Master task", master_description, list(new_tasks)), repo, fields
                    )
                    result.master = {"number": created.number, "url": created.url, "title": master_title,
                                     "item_id": created.item_id}
                    result.created = subs
            except GitHubError as exc:
                result.errors.append(f"Creating the new subtasks failed: {exc}")

    for update in updates:
        item, new = update.item, update.new
        if item.is_done:  # may have been finished since the changes were prepared
            result.errors.append(f"Skipped updating #{item.number} '{item.title}': it's already done.")
            continue
        try:
            new_title = new.title if new.title.strip() and new.title != item.title else None
            service.update_task(item, new_title, new.description)
            result.updated.append(item)
        except GitHubError as exc:
            result.errors.append(f"Updating #{item.number} '{item.title}' failed: {exc}")

    for item, mode in deletions:
        if item.is_done:
            result.errors.append(f"Skipped removing #{item.number} '{item.title}': it's already done.")
            continue
        try:
            service.remove_task(item, mode)
            result.removed.append({"item": item, "mode": mode})
        except GitHubError as exc:
            result.errors.append(f"Removing #{item.number} '{item.title}' failed: {exc}")

    return result


def _existing(item: ProjectItem, new: Optional[Subtask] = None) -> ExistingTask:
    if new:
        return ExistingTask(new.title, new.description, item.status, item.number, item.url)
    return ExistingTask(item.title or "", item.body or "", item.status, item.number, item.url)
