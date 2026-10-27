from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .ollama import OllamaClient, OllamaError

DEFAULT_MODEL = "qwen3:4b"

SYSTEM_PROMPT = """You are a project planner. The user describes a master task.
Break it into concrete subtasks that together fully cover the master task.

Rules:
- Each subtask is a single, actionable piece of work someone could pick up and finish.
- Order subtasks in the sequence they should be done.
- Do not repeat the master task as a subtask, and do not add work the description doesn't imply.
- Titles are short (under 80 characters) and start with a verb.
- Each description is 1-3 sentences saying what to do and when it counts as done.
- Return between {min_tasks} and {max_tasks} subtasks.
"""

# Added to the system prompt when the user gives extra instructions.
INSTRUCTIONS_RULE = """
The user also gives planning instructions after the description. Follow them exactly; they take
priority over the rules above.
- If they emphasize a part of the task, split that part into several detailed subtasks.
- Still include at least one subtask for every other part of the task, unless the instructions
  explicitly say to skip it.
- If they say how many subtasks a part should get, use exactly that number.
"""

# Added when the user marks existing project tasks as already covering part of the work.
EXISTING_RULE = """
IMPORTANT: some of the work is already done or planned. Those tasks are listed after the
description under "Already existing tasks". Return ONLY NEW subtasks for work that none of the
existing tasks covers. Never copy, rename or reword an existing task. If nothing is missing,
return an empty task list.
"""

REWRITE_PROMPT = """You update one existing task in a project plan.
The user gives a document describing the work, the task as it is now, and maybe instructions.
Write an UPDATED description for this task:
- Include every requirement from the document and the instructions that concerns this task,
  especially anything new that the current description is missing.
- The instructions have the highest priority; the new description must reflect them.
- First list new_details: each requirement the current description is missing.
- Then write the description so it includes all of new_details. Do not copy the current
  description unchanged.
- Say what to do and when it counts as done, in 2-5 sentences. Describe only this task.
- Keep the current title unless it no longer fits the work; then give a short new title
  (under 80 characters, starting with a verb).
"""

# Fields are generated in this order: listing what's new first stops the model from just
# copying the current description back.
TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "new_details": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Requirements from the instructions and document that the current description is missing",
        },
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["new_details", "title", "description"],
}

MERGE_PROMPT = """You merge several tasks from a project plan into ONE task.
Write a single task that covers all the work of the given tasks, without dropping anything and
without adding new work. If the user gives guidance, follow it.
- Title: short (under 80 characters), starts with a verb, describes the combined work.
- Description: what to do and when it counts as done, in 2-5 sentences.
"""

MERGE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
    "required": ["title", "description"],
}

PICK_PROMPT = """The user wants to {action} some tasks from a numbered list. Read the request and
return the numbers of ALL the tasks the request refers to ({how_many}), in list order.
"""

SPLIT_PROMPT = """You split ONE task from a project plan into smaller tasks.
- Together the parts must cover all the work of the original task, and nothing more.
- Each part is a separate, meaningful piece of work that someone can pick up on its own.
- Parts must not overlap each other, and must not repeat or overlap any of the OTHER tasks listed.
- If the user says how many parts or how to split, follow that exactly; otherwise use 2 to 5 parts.
- Titles are short (under 80 characters) and start with a verb.
- Each description says what to do and when it counts as done, in 1-3 sentences.
"""

SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
                "required": ["title", "description"],
            },
        }
    },
    "required": ["parts"],
}

PICK_SCHEMA = {
    "type": "object",
    "properties": {"numbers": {"type": "array", "items": {"type": "integer"}}},
    "required": ["numbers"],
}

REVISE_PROMPT = """You maintain the task list of a project.
The current tasks were made from the EARLIER documents. A NEWER document has arrived (for example
an update or a change of plan). Update the task list so that it matches the newer document:
- A task the newer document does not affect stays exactly as it is: change = "kept".
- A task whose work the newer document changes is rewritten: change = "changed".
- First, in "update_changes", list every change the newer document makes: new needs, changed
  requirements, and anything it says is not wanted, not needed (for now), handled another way, or
  replaced. Read between the lines: if people say they don't want something, that is a change.
- A task whose work is no longer wanted, not needed, handled another way, or replaced is NOT
  returned; list it under "removed" with a short reason.
- New work that the newer document asks for becomes a new task: change = "new", number = 0.
- Do NOT drop or change tasks just because the newer document doesn't mention them.
- Refer to existing tasks by their number. Keep the order of the current list; put new tasks
  where they fit best. Titles start with a verb; descriptions are 1-3 sentences.
"""

# Generated in this order: summarising the update first makes the model notice implicit changes.
REVISE_SCHEMA = {
    "type": "object",
    "properties": {
        "update_changes": {"type": "array", "items": {"type": "string"}},
        "removed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"number": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["number", "reason"],
            },
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "change": {"type": "string", "enum": ["kept", "changed", "new"]},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["number", "change", "title", "description"],
            },
        },
    },
    "required": ["update_changes", "removed", "tasks"],
}


CHECK_PROMPT = """A project received an update. You get the changes the update makes and ONE task.
Decide what the update means for this task:
- "not_needed": a change says the work this WHOLE task is about is removed, not wanted, not
  needed (for now), handled another way, or replaced by something else.
- "partly": only PART of this task is no longer needed; the rest of the task is still needed.
- "needed": otherwise. Changes about other parts of the project do not affect this task.
If unsure, answer "needed". First give the reason in one short sentence, then the decision.
"""

# "reason" comes first so the model thinks before deciding.
CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "decision": {"type": "string", "enum": ["needed", "partly", "not_needed"]},
    },
    "required": ["reason", "decision"],
}


@dataclass
class Revision:
    tasks: List["Subtask"]
    changes: List[str]  # per task: "kept", "changed" or "new"
    removed: List[tuple]  # (Subtask, reason)

# JSON shape the model must follow. Ollama enforces this at generation time.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "master_title": {"type": "string"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title", "description"],
            },
        },
    },
    "required": ["master_title", "tasks"],
}


class PlannerError(Exception):
    """Raised when the model can't be reached or returns an unusable plan."""


@dataclass
class Subtask:
    title: str
    description: str


@dataclass
class ExistingTask:
    """A task already in the project that the user says covers part of the work."""

    title: str
    description: str = ""
    status: Optional[str] = None
    number: Optional[int] = None
    url: Optional[str] = None

    def label(self) -> str:
        ref = f"#{self.number} " if self.number else ""
        return f"{ref}{self.title}"


@dataclass
class Plan:
    master_title: str
    master_description: str
    tasks: List[Subtask]


class TaskPlanner:
    """Turns a master-task description into subtasks using a local Ollama model."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = 600.0,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self._ollama = OllamaClient(host, timeout)

    def plan(
        self,
        description: str,
        title: Optional[str] = None,
        min_tasks: int = 3,
        max_tasks: int = 10,
        instructions: Optional[str] = None,
        existing: Optional[List[ExistingTask]] = None,
    ) -> Plan:
        """Generate subtasks.

        `instructions`: optional extra guidance, e.g. what to focus on.
        `existing`: tasks already in the project; no new subtasks are made for that work.
        """
        description = description.strip()
        instructions = (instructions or "").strip()
        if not description:
            raise PlannerError("The master-task description is empty.")

        # min_tasks=0 lets the model answer "nothing new" (when ticked tasks cover everything).
        system = SYSTEM_PROMPT.format(min_tasks=min_tasks, max_tasks=max_tasks)
        user_message = f"Master task description:\n\n{description}"
        if title:
            user_message = f"Master task title: {title}\n\n" + user_message
        if existing:
            system += EXISTING_RULE
            lines = []
            for task in existing:
                status = f" [{task.status}]" if task.status else ""
                summary = " ".join(task.description.split())[:200]
                lines.append(f"- {task.label()}{status}" + (f": {summary}" if summary else ""))
            user_message += (
                "\n\nAlready existing tasks (already covered; do NOT include these or variants of them):\n"
                + "\n".join(lines)
            )
        if instructions:
            system += INSTRUCTIONS_RULE
            # Placed last so a small model gives it the most weight.
            user_message += f"\n\nPlanning instructions (follow these):\n{instructions}"

        raw = self._chat(system, user_message)
        plan = self._parse(raw, description, title, max_tasks, allow_empty=bool(existing) or min_tasks == 0)
        if existing:
            # Small models sometimes copy existing tasks anyway; drop those.
            plan.tasks = [
                t for t in plan.tasks
                if not any(_same_task(t.title, e.title) for e in existing)
            ]
        if instructions:
            plan.master_description += f"\n\n### Planning focus\n{instructions}"
        if existing:
            links = [
                f"- [{t.label()}]({t.url})" if t.url else f"- {t.label()}" for t in existing
            ]
            plan.master_description += "\n\n### Already covered by existing tasks\n" + "\n".join(links)
        return plan

    def rewrite_task(
        self,
        document: str,
        title: str,
        description: str = "",
        instructions: Optional[str] = None,
    ) -> Subtask:
        """New title + description for an existing task, based on the document and instructions."""
        user = (
            f"Document:\n{document.strip()}\n\n"
            f"Task to update\nTitle: {title}\n"
            f"Current description: {description.strip() or '(empty)'}"
        )
        if instructions and instructions.strip():
            user += (
                f"\n\nInstructions (the updated description must reflect these):\n"
                f"{instructions.strip()}"
            )
        raw = self._chat(REWRITE_PROMPT, user, schema=TASK_SCHEMA)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise PlannerError(f"Model returned invalid JSON:\n{raw}") from None
        new_title = str(data.get("title") or "").strip() or title
        new_description = str(data.get("description") or "").strip()
        if not new_description:
            raise PlannerError(f"Model returned an empty description for '{title}'.")
        return Subtask(new_title, new_description)

    def revise_tasks(
        self,
        tasks: List[Subtask],
        earlier: str,
        newer: str,
        guidance: Optional[str] = None,
    ) -> Revision:
        """Update `tasks` (made from the `earlier` documents) for the `newer` document(s).

        Tasks the model neither returns nor removes are kept: nothing is forgotten silently.
        """
        listing = "\n".join(f"{n}. {t.title}: {t.description}" for n, t in enumerate(tasks, 1))
        user = (f"EARLIER documents:\n{earlier.strip()}\n\nNEWER document:\n{newer.strip()}\n\n"
                f"Current tasks:\n{listing}")
        if guidance and guidance.strip():
            user += f"\n\nInstructions (follow these):\n{guidance.strip()}"
        data = self._json(REVISE_PROMPT, user, REVISE_SCHEMA)

        # Second, focused pass: with the update's changes listed, is each task still needed?
        # One question per task: a small model mixes up task numbers when judging a whole list.
        not_needed = {}  # number -> reason: drop the task
        trimmed = {}  # number -> rewritten task without the part that's no longer needed
        changes_listed = [str(c).strip() for c in data.get("update_changes") or [] if str(c).strip()]
        if changes_listed:
            changes_text = "\n".join(f"- {c}" for c in changes_listed)
            for number, task in enumerate(tasks, 1):
                check = self._json(
                    CHECK_PROMPT,
                    f"Changes made by the update:\n{changes_text}\n\n"
                    f"Task: {task.title}\nDescription: {task.description or '(none)'}",
                    CHECK_SCHEMA,
                )
                reason = str(check.get("reason") or "").strip()
                if check.get("decision") == "not_needed":
                    not_needed[number] = reason or "Not needed after the update."
                elif check.get("decision") == "partly":
                    trimmed[number] = self.rewrite_task(
                        newer, task.title, task.description,
                        f"Remove the part of this task that is no longer needed ({reason}). Keep the rest.",
                    )

        result: List[Subtask] = []
        changes: List[str] = []
        origins: List[int] = []  # original number of each result task (0 = new)
        used = set()
        for item in data.get("tasks") or []:
            number = item.get("number") if isinstance(item.get("number"), int) else 0
            change = item.get("change")
            title = str(item.get("title") or "").strip()
            description = str(item.get("description") or "").strip()
            if 1 <= number <= len(tasks) and number in not_needed:
                continue  # removed below
            existing = 1 <= number <= len(tasks) and number not in used
            if existing and number in trimmed:
                result.append(trimmed[number])
                changes.append("changed")
                origins.append(number)
                used.add(number)
                continue
            if existing and ":" in title and title.split(":", 1)[0].strip().lower() == tasks[number - 1].title.lower():
                title = tasks[number - 1].title  # the model glued the description onto the title
            if existing and change == "kept":
                result.append(tasks[number - 1])
                changes.append("kept")
                origins.append(number)
                used.add(number)
            elif existing and change == "changed":
                old = tasks[number - 1]
                new = Subtask(title or old.title, description or old.description)
                same = (new.title, new.description) == (old.title, old.description)
                result.append(old if same else new)
                changes.append("kept" if same else "changed")
                origins.append(number)
                used.add(number)
            elif title and change == "new" and not any(_same_task(title, t.title) for t in result + tasks):
                head, _, tail = title.partition(":")
                if tail.strip() and len(tail.strip()) > 20:  # "Title: description" glued together
                    title, description = head.strip(), description or tail.strip()
                result.append(Subtask(title, description))
                changes.append("new")
                origins.append(0)

        removed = []
        for number, reason in sorted(not_needed.items()):
            if number not in used:
                removed.append((tasks[number - 1], reason))
                used.add(number)
        for item in data.get("removed") or []:
            number = item.get("number")
            if isinstance(number, int) and 1 <= number <= len(tasks) and number not in used:
                removed.append((tasks[number - 1], str(item.get("reason") or "").strip()))
                used.add(number)

        # Never lose a task silently: anything the model skipped stays, right after the
        # nearest earlier original task (or at the top).
        for number, task in enumerate(tasks, 1):
            if number in used:
                continue
            task, change = (trimmed[number], "changed") if number in trimmed else (task, "kept")
            earlier_positions = [k for k, o in enumerate(origins) if 0 < o < number]
            at = earlier_positions[-1] + 1 if earlier_positions else 0
            result.insert(at, task)
            changes.insert(at, change)
            origins.insert(at, number)
        return Revision(result, changes, removed)

    def merge_tasks(self, tasks: List[Subtask], guidance: Optional[str] = None) -> Subtask:
        """One task that covers all of `tasks`."""
        listing = "\n".join(f"{n}. {t.title}: {t.description}" for n, t in enumerate(tasks, 1))
        user = f"Tasks to merge:\n{listing}"
        if guidance and guidance.strip():
            user += f"\n\nGuidance for the merged task:\n{guidance.strip()}"
        data = self._json(MERGE_PROMPT, user, MERGE_SCHEMA)
        title = str(data.get("title") or "").strip()
        description = str(data.get("description") or "").strip()
        if not title:
            raise PlannerError("The model returned a merged task without a title.")
        return Subtask(title, description)

    def split_task(
        self,
        task: Subtask,
        guidance: Optional[str] = None,
        others: Sequence[str] = (),
        document: Optional[str] = None,
    ) -> List[Subtask]:
        """Smaller tasks that together cover `task`, avoiding the work in `others`."""
        user = ""
        if document and document.strip():
            user += f"Project document (for context):\n{document.strip()}\n\n"
        if others:
            user += "OTHER tasks (do not repeat or overlap these):\n" + "\n".join(f"- {o}" for o in others) + "\n\n"
        user += f"Task to split:\nTitle: {task.title}\nDescription: {task.description or '(none)'}"
        if guidance and guidance.strip():
            user += f"\n\nHow to split (follow this):\n{guidance.strip()}"
        data = self._json(SPLIT_PROMPT, user, SPLIT_SCHEMA)
        parts, seen = [], set()
        for item in data.get("parts") or []:
            title = str(item.get("title") or "").strip()
            if title and title.lower() not in seen:
                seen.add(title.lower())
                parts.append(Subtask(title, str(item.get("description") or "").strip()))
        return parts

    def pick_tasks(self, tasks: List[Subtask], request: str, action: str = "merge") -> List[int]:
        """0-based positions of the tasks a free-text merge/split request refers to."""
        listing = "\n".join(f"{n}. {t.title}" for n, t in enumerate(tasks, 1))
        system = PICK_PROMPT.format(action=action, how_many="at least two" if action == "merge" else "usually one")
        data = self._json(system, f"Tasks:\n{listing}\n\nRequest: {request.strip()}", PICK_SCHEMA)
        picked: List[int] = []
        for n in data.get("numbers") or []:
            if isinstance(n, int) and 1 <= n <= len(tasks) and n - 1 not in picked:
                picked.append(n - 1)
        return sorted(picked)

    def _json(self, system: str, user: str, schema: dict) -> dict:
        raw = self._chat(system, user, schema=schema)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise PlannerError(f"Model returned invalid JSON:\n{raw}") from None
        if not isinstance(data, dict):
            raise PlannerError(f"Model returned unexpected JSON:\n{raw}")
        return data

    def _chat(self, system: str, user: str, schema: Optional[dict] = None) -> str:
        """Ask the model for JSON. A broken or empty reply is retried once with memory freed."""
        for attempt in (1, 2):
            if attempt == 2:
                # Low GPU memory (several models loaded) can produce cut-off replies
                # without an error. Unload everything and try again from fresh.
                self._ollama.unload_all()
            try:
                raw = self._ollama.chat(
                    self.model,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    format=schema or PLAN_SCHEMA,
                    # Larger context so long documents (e.g. PDFs) aren't cut off.
                    options={"temperature": 0.3, "num_ctx": 16384},
                    think=False,  # skip Qwen3's reasoning output; the schema does the work
                )
            except OllamaError as exc:
                raise PlannerError(str(exc)) from None
            if _is_json_object(raw):
                return raw
        return raw  # still broken: the caller reports it

    @staticmethod
    def _parse(
        raw: str, description: str, title: Optional[str], max_tasks: int, allow_empty: bool = False
    ) -> Plan:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise PlannerError(f"Model returned invalid JSON:\n{raw}") from None

        tasks = []
        seen = set()
        for item in data.get("tasks") or []:
            task_title = str(item.get("title", "")).strip()
            if not task_title or task_title.lower() in seen:
                continue
            seen.add(task_title.lower())
            tasks.append(Subtask(task_title, str(item.get("description", "")).strip()))

        if not tasks and not allow_empty:
            raise PlannerError(f"Model returned no subtasks:\n{raw}")

        master_title = title or str(data.get("master_title", "")).strip() or "Master task"
        return Plan(master_title, description, tasks[:max_tasks])


def _words(title: str) -> set:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words if len(w) > 2}


def _same_task(a: str, b: str) -> bool:
    """True if two task titles describe the same work (equal or nearly the same words)."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return a.strip().lower() == b.strip().lower()
    overlap = len(wa & wb) / len(wa | wb)
    smaller, larger = sorted((wa, wb), key=len)
    return overlap >= 0.7 or (len(smaller) >= 3 and smaller <= larger)


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (TypeError, ValueError):
        return False
