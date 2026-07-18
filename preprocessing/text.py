"""Loss-minimizing normalization for student writing."""

from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"[^\S\n]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: object) -> str:
    """Remove invalid Unicode and normalize whitespace while preserving prose."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(char for char in value if char in "\n\t" or unicodedata.category(char) != "Cc")
    value = _SPACE_RE.sub(" ", value)
    value = _NEWLINES_RE.sub("\n\n", value)
    return value.strip()


def compose_context(prompt_name: object, assignment: object, sources: list[object]) -> str:
    """Create labeled context from the prompt, assignment, and available sources."""
    sections = []
    if normalize_text(prompt_name):
        sections.append(f"Prompt title: {normalize_text(prompt_name)}")
    if normalize_text(assignment):
        sections.append(f"Assignment: {normalize_text(assignment)}")
    for index, source in enumerate(sources, start=1):
        cleaned = normalize_text(source)
        if cleaned and cleaned.lower() != "nan":
            sections.append(f"Source {index}: {cleaned}")
    return "\n\n".join(sections)


def compose_baseline_input(
    essay: object,
    prompt_name: object,
    assignment: object,
    sources: list[object],
    source_character_limit: int = 1_500,
) -> str:
    """Compose a bounded text input for the fast TF-IDF baseline."""
    bounded_sources = [normalize_text(source)[:source_character_limit] for source in sources]
    context = compose_context(prompt_name, assignment, bounded_sources)
    return f"{context}\n\nStudent essay: {normalize_text(essay)}".strip()
