"""Unit tests for core/ingestion.py — no external API calls required."""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingestion import load_resumes


@pytest.fixture(autouse=True)
def no_llm_parsing():
    """Keep ingestion tests offline.

    Without this, load_resumes builds a real ResumeParser and sends every
    fixture resume to Gemini — slow, billable, and non-deterministic (the
    returned name overrides the filename-derived fallback these tests assert
    on). Returning None makes ingestion use its no-parser fallback path.
    """
    with patch("core.ingestion._get_parser", return_value=None):
        yield


# ---------------------------------------------------------------------------
# load_resumes
# ---------------------------------------------------------------------------

class TestLoadResumes:
    def test_returns_empty_for_nonexistent_directory(self):
        result = load_resumes("/nonexistent/path/that/does/not/exist")
        assert result == []

    def test_returns_empty_for_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_resumes(tmp)
        assert result == []

    def test_returns_empty_when_no_pdfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create a non-PDF file
            (Path(tmp) / "resume.txt").write_text("some text")
            result = load_resumes(tmp)
        assert result == []

    def test_loads_pdfs_from_directory(self):
        """load_resumes should call load_pdf for each .pdf file it finds."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create dummy PDF files (content doesn't matter — we mock load_pdf)
            for name in ["alice.pdf", "bob.pdf"]:
                (Path(tmp) / name).write_bytes(b"%PDF-1.4 fake")

            with patch("core.ingestion.load_pdf", return_value="Candidate resume text") as mock_load:
                result = load_resumes(tmp)

        assert len(result) == 2
        assert mock_load.call_count == 2

    def test_document_metadata_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "alice_smith.pdf").write_bytes(b"%PDF-1.4 fake")
            with patch("core.ingestion.load_pdf", return_value="Resume text"):
                result = load_resumes(tmp)

        doc = result[0]
        assert doc.metadata["filename"] == "alice_smith.pdf"
        assert doc.metadata["candidate_id"] == "c_alice_smith"
        assert doc.metadata["name"] == "Alice Smith"

    def test_skips_pdfs_with_no_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "empty.pdf").write_bytes(b"%PDF-1.4 fake")
            with patch("core.ingestion.load_pdf", return_value=""):
                result = load_resumes(tmp)
        assert result == []

    def test_ignores_non_pdf_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "resume.pdf").write_bytes(b"%PDF-1.4 fake")
            (Path(tmp) / "notes.txt").write_text("ignore me")
            (Path(tmp) / "data.csv").write_text("ignore me too")
            with patch("core.ingestion.load_pdf", return_value="Resume text") as mock_load:
                result = load_resumes(tmp)
        assert len(result) == 1
