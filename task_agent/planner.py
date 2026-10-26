from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

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
