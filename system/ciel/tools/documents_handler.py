"""
Documents Tool Handler
-------------------------
Gives the 'documents' tool a real implementation: CIEL can create and
read text/markdown files on disk. Deliberately restricted to one
dedicated folder — a filename like '../../../important_file.txt' must
NOT be able to escape that folder, for both writes AND reads.
"""

from __future__ import annotations

import pathlib

DOCUMENTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "ciel_documents"


def create_document(filename: str, content: str) -> str:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
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


def read_file(filename: str) -> str:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = (DOCUMENTS_DIR / filename).resolve()
    try:
        candidate_path.relative_to(DOCUMENTS_DIR.resolve())
    except ValueError:
        return (f"Refused to read '{filename}': this path would escape the "
                f"documents folder ({DOCUMENTS_DIR}). This is a safety restriction, not a bug.")
    if not candidate_path.exists():
        return f"No file named '{filename}' found in {DOCUMENTS_DIR}."
    if not candidate_path.is_file():
        return f"'{filename}' is not a file (it may be a directory)."
    try:
        return candidate_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"'{filename}' isn't a text file — can't read it as text."


def list_files() -> str:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in DOCUMENTS_DIR.iterdir() if p.is_file())
    if not files:
        return f"No files in {DOCUMENTS_DIR} yet."
    return f"Files in {DOCUMENTS_DIR}: " + ", ".join(files)