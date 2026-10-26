"""[UPDATE] / [DELETE] commands written in the instructions box.

    [UPDATE] Implement kitchen dashboard    -> rewrite that task from the document + instructions
    [DELETE] #33                            -> remove that task

A command names a task by its number (#33), its title, or both ("#33 Implement kitchen
dashboard"). Every other line stays a normal planning instruction.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .projects import ProjectItem

ACTIONS = ("UPDATE", "DELETE")
_COMMAND = re.compile(r"\[(UPDATE|DELETE)\]", re.IGNORECASE)
_NUMBER = re.compile(r"^#(\d+)\b")
FUZZY_MATCH = 0.6  # how close a title must be (0..1) when it isn't typed exactly


@dataclass
class Command:
    action: str  # "UPDATE" or "DELETE"
    target: str  # what the user typed after the token


@dataclass
class ResolvedCommand:
    command: Command
    item: Optional[ProjectItem] = None
    note: str = ""  # e.g. "closest match" or why nothing was found


@dataclass
class ParsedInstructions:
    instructions: str  # the text left after removing the commands
    commands: List[Command] = field(default_factory=list)


def parse_instructions(text: Optional[str]) -> ParsedInstructions:
    """Split the instructions box into plain instructions and [UPDATE]/[DELETE] commands."""
    commands: List[Command] = []
    kept: List[str] = []
    for line in (text or "").splitlines():
        parts = _COMMAND.split(line)
        # parts = [before, ACTION, target, ACTION, target, ...]
        before = parts[0]
        if before.strip(" -*\t"):
            kept.append(before.rstrip())
        for action, target in zip(parts[1::2], parts[2::2]):
            target = target.strip().strip("\"'`").strip()
            if target:
                commands.append(Command(action.upper(), target))
        if len(parts) == 1 and not before.strip():
            kept.append("")
    return ParsedInstructions("\n".join(kept).strip(), commands)


def resolve_commands(
    commands: Sequence[Command], items: Sequence[ProjectItem]
) -> List[ResolvedCommand]:
    """Match each command to a project item. Unmatched commands get an explanatory note."""
    candidates = [i for i in items if i.type != "REDACTED" and i.title]
    resolved = []
    for command in commands:
        item, note = _find(command.target, candidates)
        resolved.append(ResolvedCommand(command, item, note))
    return resolved


def _find(target: str, items: Sequence[ProjectItem]) -> Tuple[Optional[ProjectItem], str]:
    number = _NUMBER.match(target)
    if number:
        wanted = int(number.group(1))
        for item in items:
            if item.number == wanted:
                return item, ""
        return None, f"No task #{wanted} in the project."

    wanted = _normalize(target)
    exact = [i for i in items if _normalize(i.title) == wanted]
    if len(exact) == 1:
        return exact[0], ""
    if len(exact) > 1:
        refs = ", ".join(f"#{i.number}" if i.number else "a draft" for i in exact)
        return None, f"Several tasks are called '{target}' ({refs}). Use the number, e.g. [UPDATE] #{exact[0].number}."

    scored = sorted(
        ((difflib.SequenceMatcher(None, wanted, _normalize(i.title)).ratio(), i) for i in items),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if scored and scored[0][0] >= FUZZY_MATCH:
        best_score, best = scored[0]
        if len(scored) > 1 and scored[1][0] >= FUZZY_MATCH and best_score - scored[1][0] < 0.05:
            other = scored[1][1]
            return None, (
                f"'{target}' could be '{best.title}' or '{other.title}'. Use the task number instead."
            )
        return best, f"Closest match for '{target}'."
    return None, f"No task matching '{target}' in the project."


def _normalize(title: Optional[str]) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (title or "").lower()))
