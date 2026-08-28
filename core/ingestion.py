"""
Document loading and processing for resumes.
Converts PDF files to LangChain Document objects with rich metadata.
"""

import os
from pathlib import Path
from typing import Callable, Iterable
from langchain.schema import Document
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import get_logger, load_pdf

logger = get_logger(__name__)


def _get_parser():
    """Build one ResumeParser, or return None if Gemini is unavailable.

    Callers reuse a single instance across a whole ingestion run — building a
    parser per resume also builds a ChatGoogleGenerativeAI client per resume.
    """
    try:
        from core.parsing import ResumeParser
        return ResumeParser()
    except Exception as e:
        logger.warning(f"ResumeParser unavailable, falling back to basic metadata: {e}")
        return None


def _try_parse_resume(text: str, candidate_id: str, parser=None) -> dict:
    """Attempt to parse resume text with ResumeParser (Gemini).

    Returns a dict with keys like 'skills', 'location', 'experience', 'name'.
    If the parser is unavailable or fails, returns an empty dict so ingestion
    can still proceed with basic metadata.

    Each call is one LLM round-trip, so this is the expensive step of ingestion —
    it should run once per resume, not once per app start.
    """
    if parser is None:
        parser = _get_parser()
    if parser is None:
        return {}
    try:
        return parser.parse_resume(text, candidate_id)
    except Exception as e:
        logger.warning(f"Resume parsing failed for candidate {candidate_id}: {e}")
        return {}


def build_resume_document(file_path: str, filename: str = None, parser=None) -> Document | None:
    """Read one resume PDF and return a fully-parsed Document, or None.

    This is the unit of work shared by bulk indexing and single-file uploads.
    """
    filename = filename or Path(file_path).name
    text = load_pdf(file_path)
    if not text:
        return None

    candidate_id = f"c_{Path(filename).stem}"
    fallback_name = Path(filename).stem.replace('_', ' ').title()

    parsed = _try_parse_resume(text, candidate_id, parser=parser)

    return Document(
        page_content=text,
        metadata={
            "source": file_path,
            "filename": filename,
            "candidate_id": candidate_id,
            "name": parsed.get('name') or fallback_name,
            "skills": parsed.get('skills', []),
            "location": parsed.get('location', 'Unknown'),
            "experience": parsed.get('experience', 0),
        },
    )


def list_resume_files(directory: str) -> list[str]:
    """Return the PDF filenames in `directory`, sorted. Empty if it doesn't exist."""
    if not directory or not os.path.isdir(directory):
        logger.warning(f"Resume directory not found: {directory}")
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith('.pdf'))


def load_resumes(directory: str, only_files: Iterable[str] = None,
                 progress: Callable[[int, int, str], None] = None) -> list[Document]:
    """Load resume PDFs from `directory`, parse them with Gemini, and return
    Document objects with rich metadata (skills, location, experience).

    Args:
        directory: folder holding the resume PDFs.
        only_files: if given, parse just these filenames — this is what makes
            incremental indexing cheap, since already-indexed resumes are skipped
            instead of being sent to Gemini again.
        progress: optional callback ``(done, total, filename)`` for CLI/UI output.

    A missing or empty directory yields an empty list rather than raising.
    """
    available = list_resume_files(directory)
    if not available:
        return []

    targets = [f for f in available if f in set(only_files)] if only_files is not None else available
    if not targets:
        return []

    # One parser (and one LLM client) for the whole run.
    parser = _get_parser()

    resumes = []
    total = len(targets)
    for i, file in enumerate(targets, start=1):
        file_path = os.path.join(directory, file)
        try:
            if progress:
                progress(i, total, file)
            doc = build_resume_document(file_path, filename=file, parser=parser)
            if doc is not None:
                resumes.append(doc)
        except Exception as e:
            logger.error(f"Failed to load resume {file_path}: {e}")
            continue
    return resumes

def _default_data_dirs():
    """Return sensible default directories relative to project root."""
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    resumes_dir = project_root / "data" / "resumes"
    return str(resumes_dir)


def _print_sample_documents(docs, label: str, max_display: int = 3):
    print(f"\n{label}: {len(docs)} found")
    for i, doc in enumerate(docs[:max_display]):
        meta = getattr(doc, 'metadata', {}) or {}
        print(f"[{i+1}] id: {meta.get('candidate_id', meta.get('jd_id', 'n/a'))} | filename: {meta.get('filename', meta.get('title', 'n/a'))}")
        snippet = (getattr(doc, 'page_content', '') or '')[:200]
        # Flatten outside the f-string: backslashes in f-string expressions are
        # a SyntaxError before Python 3.12, and this project pins 3.11.
        flat_snippet = snippet.replace('\n', ' ')[:180]
        print(f"    snippet: {flat_snippet}...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ingestion demo: load resumes')
    default_resumes = _default_data_dirs()
    parser.add_argument('--resumes-dir', type=str, default=default_resumes, help='Path to resumes directory (PDFs)')
    parser.add_argument('--show', action='store_true', help='Print a short sample of loaded documents')

    args = parser.parse_args()

    print(f"Using resumes dir: {args.resumes_dir}")

    try:
        resumes = load_resumes(args.resumes_dir)

        if args.show:
            _print_sample_documents(resumes, 'Resumes')
        else:
            print(f"Loaded {len(resumes)} resumes.")

    except Exception as e:
        print(f"Ingestion demo failed: {e}")
