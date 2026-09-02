"""
Documents Tool Handler
-------------------------
Gives the 'documents' tool (registered since Phase I, never wired) a
real implementation: CIEL can actually create text/markdown files on
disk. Deliberately restricted to one dedicated folder — a filename like
'../../../important_file.txt' must NOT be able to write outside that
folder, which is a real safety property, not a formality (per the
constitution's Preservation law: CIEL having more reach than intended is
exactly the kind of risk least-privilege design exists to prevent).
"""

from __future__ import annotations

import pathlib

DOCUMENTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "ciel_documents"


def create_document(filename: str, content: str) -> str:
    """Creates a text file inside DOCUMENTS_DIR only. Rejects any
    filename that would escape that folder (path traversal via '..',
    absolute paths, etc.) rather than silently sanitizing it — an
    explicit rejection is safer than a silent "best guess" fix, since a
    silent fix could mask a bug (or a malicious input) that deserves to
    be seen, not quietly worked around."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve what the final path WOULD be, and verify it's still inside
    # DOCUMENTS_DIR — this is the actual security check, not just a
    # string search for ".." (which is easy to bypass with tricks like
    # encoded separators; resolving the real path is not).
    candidate_path = (DOCUMENTS_DIR / filename).resolve()
    try:
        candidate_path.relative_to(DOCUMENTS_DIR.resolve())
    except ValueError:
        return (f"Refused to create '{filename}': this path would escape the "
                f"documents folder ({DOCUMENTS_DIR}). This is a safety restriction, not a bug.")

    if candidate_path.exists():
        return f"Refused to create '{filename}': a file already exists there. Choose a different name."

    candidate_path.write_text(content, encoding="utf-8")
    return f"Created '{filename}' in {DOCUMENTS_DIR} ({len(content)} characters)."