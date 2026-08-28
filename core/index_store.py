"""
On-disk persistence for the BM25 corpus and the resume index manifest.

The expensive part of indexing is not BM25 — it is reading every PDF and making
one Gemini call per resume to extract name/skills/location/experience. That work
must happen once, not on every app start.

This module stores two things under ``data/hybrid_index/``:

``corpus.pkl``
    The lowercased resume texts and their parallel metadata — everything
    ``HybridIndexer`` needs to rebuild BM25Okapi in memory with no API calls.

``manifest.json``
    One entry per indexed PDF, keyed by filename, holding the file's SHA-256.
    This is what makes incremental indexing possible: a file whose hash matches
    the manifest has already been parsed and can be skipped.

BM25Okapi itself is deliberately *not* pickled. Rebuilding it from the stored
texts takes milliseconds and avoids breaking whenever rank_bm25 changes its
internal layout.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.utils import get_logger

logger = get_logger(__name__)

# Bump when the on-disk layout changes in a way older files can't satisfy.
MANIFEST_VERSION = 1

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "data" / "hybrid_index"

_CORPUS_FILE = "corpus.pkl"
_MANIFEST_FILE = "manifest.json"


def file_fingerprint(path: str | Path) -> str:
    """Return a SHA-256 of the file's bytes.

    Used instead of mtime so that copying or re-downloading an unchanged resume
    does not trigger a re-parse (and another Gemini call).
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class IndexStore:
    """Reads and writes the persisted resume index."""

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.corpus_path = self.index_dir / _CORPUS_FILE
        self.manifest_path = self.index_dir / _MANIFEST_FILE

    # -- reading ------------------------------------------------------------

    def exists(self) -> bool:
        """True if a usable persisted index is present."""
        return self.corpus_path.is_file() and self.manifest_path.is_file()

    def load_corpus(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Return ``(texts, metadata)``; empty lists if nothing is stored."""
        if not self.corpus_path.is_file():
            return [], []
        try:
            with open(self.corpus_path, "rb") as fh:
                payload = pickle.load(fh)
            texts = payload.get("texts", [])
            metadata = payload.get("metadata", [])
            if len(texts) != len(metadata):
                logger.warning(
                    "Corpus is inconsistent (%d texts vs %d metadata entries); "
                    "treating the index as empty. Re-run indexing with --force.",
                    len(texts), len(metadata),
                )
                return [], []
            return texts, metadata
        except Exception as e:
            logger.error(f"Failed to read persisted corpus: {e}")
            return [], []

    def load_manifest(self) -> Dict[str, Any]:
        """Return the manifest, or an empty skeleton if absent/unreadable."""
        if not self.manifest_path.is_file():
            return self._empty_manifest()
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if manifest.get("version") != MANIFEST_VERSION:
                logger.warning(
                    "Manifest version %s does not match %s; treating as empty. "
                    "Re-run indexing with --force.",
                    manifest.get("version"), MANIFEST_VERSION,
                )
                return self._empty_manifest()
            return manifest
        except Exception as e:
            logger.error(f"Failed to read manifest: {e}")
            return self._empty_manifest()

    def indexed_files(self) -> Dict[str, Dict[str, Any]]:
        """Map of ``filename -> {sha256, candidate_id, indexed_at}``."""
        return self.load_manifest().get("files", {})

    # -- writing ------------------------------------------------------------

    def save(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]],
        files: Dict[str, Dict[str, Any]],
        embedding_model: Optional[str] = None,
    ) -> bool:
        """Persist the corpus and manifest atomically enough for local use."""
        if len(texts) != len(metadata):
            logger.error("Refusing to save: texts and metadata lengths differ.")
            return False
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)

            tmp_corpus = self.corpus_path.with_suffix(".pkl.tmp")
            with open(tmp_corpus, "wb") as fh:
                pickle.dump({"texts": texts, "metadata": metadata}, fh)
            tmp_corpus.replace(self.corpus_path)

            manifest = {
                "version": MANIFEST_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "resume_count": len(texts),
                "embedding_model": embedding_model,
                "files": files,
            }
            tmp_manifest = self.manifest_path.with_suffix(".json.tmp")
            with open(tmp_manifest, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            tmp_manifest.replace(self.manifest_path)

            logger.info(f"Persisted index for {len(texts)} resumes to {self.index_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to persist index: {e}")
            return False

    def clear(self) -> None:
        """Delete the persisted index files (used by a forced rebuild)."""
        for path in (self.corpus_path, self.manifest_path):
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Could not remove {path}: {e}")

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _empty_manifest() -> Dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "updated_at": None,
            "resume_count": 0,
            "embedding_model": None,
            "files": {},
        }
