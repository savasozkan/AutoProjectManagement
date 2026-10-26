"""Turn a task-description file (text, Markdown or PDF) into plain text for the planner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .ollama import OllamaClient, OllamaError

DEFAULT_VISION_MODEL = "qwen3-vl:4b"

# A page with more vector paths than this is treated as containing a drawing.
# Simple tables and underlines use a handful; diagrams and CAD exports use many more.
DRAWING_PATH_THRESHOLD = 30
# Pages with less text than this are probably scanned, so we look at them as images.
SCANNED_TEXT_THRESHOLD = 40

VISION_PROMPT = """This is page {page} of a document that describes a piece of work to plan.
Describe every drawing, diagram, sketch, chart, table, screenshot or photo on this page.
Focus on what matters for planning the work: components, layout, labels, measurements,
connections, flows, steps and any handwritten notes. Be specific and factual.
If the page contains only plain text and no visuals, transcribe the text instead.
Do not add commentary or advice."""

VISION_FOCUS = """
The reader is especially interested in the following, so describe related details thoroughly
(but still mention everything else on the page): {instructions}"""


class DocumentError(Exception):
    """Raised when a document can't be read."""


@dataclass
class Document:
    text: str
    title: Optional[str] = None
    pages_described: int = 0  # how many pages were sent to the vision model


Log = Callable[[str], None]


def load_document(
    path: str,
    vision_model: Optional[str] = None,
    use_vision: bool = True,
    ollama: Optional[OllamaClient] = None,
    log: Log = lambda msg: None,
    instructions: Optional[str] = None,
) -> Document:
    """Read a .txt/.md/.pdf file. For PDFs, drawings are described by a vision model.

    `instructions` (what the user wants to emphasize) steers the drawing descriptions.
    """
    if not os.path.isfile(path):
        raise DocumentError(f"File not found: {path}")

    if path.lower().endswith(".pdf"):
        return _load_pdf(
            path,
            vision_model or os.getenv("OLLAMA_VISION_MODEL", DEFAULT_VISION_MODEL),
            use_vision,
            ollama or OllamaClient(),
            log,
            instructions,
        )

    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise DocumentError(f"Can't read '{path}': {exc}") from None
    return Document(text)


def _load_pdf(
    path: str,
    vision_model: str,
    use_vision: bool,
    ollama: OllamaClient,
    log: Log,
    instructions: Optional[str] = None,
) -> Document:
    focus = VISION_FOCUS.format(instructions=instructions.strip()) if (instructions or "").strip() else ""

    try:
        import pymupdf
    except ImportError:
        raise DocumentError("PDF support needs PyMuPDF: pip install pymupdf") from None

    try:
        pdf = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises several types for bad files
        raise DocumentError(f"Can't open PDF '{path}': {exc}") from None

    sections: List[str] = []
    described = 0
    vision_ok = use_vision

    with pdf:
        title = (pdf.metadata or {}).get("title") or None
        for number, page in enumerate(pdf, 1):
            text = page.get_text("text").strip()
            parts = [text] if text else []

            if vision_ok and _has_visuals(page, text):
                log(f"Describing drawings on page {number} with {vision_model}...")
                try:
                    png = page.get_pixmap(dpi=_dpi_for(page)).tobytes("png")
                    visual = ollama.describe_image(
                        vision_model, png, VISION_PROMPT.format(page=number) + focus
                    )
                    if visual:
                        parts.append(f"[Visual content on page {number}]\n{visual}")
                        described += 1
                except OllamaError as exc:
                    log(f"Warning: skipping drawings ({exc}). Continuing with text only.")
                    vision_ok = False

            if parts:
                sections.append(f"--- Page {number} ---\n" + "\n\n".join(parts))

    if described:
        ollama.unload(vision_model)  # free memory for the embedding and planning models

    if not sections:
        raise DocumentError(
            f"No text or describable content found in '{path}'."
            + ("" if vision_ok else " It may be scanned; a vision model is needed to read it.")
        )
    return Document("\n\n".join(sections), title, described)


def _has_visuals(page, text: str) -> bool:
    if len(text) < SCANNED_TEXT_THRESHOLD:
        return True
    if page.get_images(full=False):
        return True
    return len(page.get_drawings()) > DRAWING_PATH_THRESHOLD


def _dpi_for(page, max_side_px: int = 1600) -> int:
    """Render big enough to read labels, small enough to keep the vision model fast."""
    longest_side_pt = max(page.rect.width, page.rect.height) or 1
    return max(72, min(200, int(max_side_px / longest_side_pt * 72)))


def split_title(text: str) -> Tuple[Optional[str], str]:
    """Use a leading '# Title' line as the master-task title. Returns (title, rest)."""
    stripped = text.strip()
    first, _, rest = stripped.partition("\n")
    if first.startswith("# "):
        return first[2:].strip() or None, rest.strip()
    return None, stripped
