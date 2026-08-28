"""
The single place where resume indexing happens.

Indexing is expensive: one PDF read plus one Gemini call per resume, then an
embedding and a Pinecone upsert. It is therefore a deliberate, explicit step —
run once from the CLI (or the API, or the sidebar button) — never something the
Streamlit app does implicitly on every start.

Three operations, shared by every entry point:

``build_index``   parse resumes and write the index. Incremental by default:
                  resumes whose file hash already appears in the manifest are
                  skipped, so adding 2 PDFs to a folder of 50 costs 2 LLM calls.
                  ``force=True`` re-parses and re-upserts everything.
``load_index``    hydrate a HybridIndexer from disk. No PDF reads, no LLM calls,
                  no Pinecone writes — this is what app startup uses.
``add_resumes``   parse and append specific new files (the upload path), then
                  persist so the work survives a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from core.index_store import IndexStore, file_fingerprint
from core.ingestion import build_resume_document, list_resume_files, load_resumes, _get_parser
from utils.utils import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESUMES_DIR = _PROJECT_ROOT / "data" / "resumes"

ProgressFn = Callable[[str], None]


@dataclass
class IndexReport:
    """What a build_index run actually did."""
    parsed: int = 0            # resumes sent to Gemini this run
    skipped: int = 0           # already in the manifest, untouched
    failed: List[str] = field(default_factory=list)
    total_indexed: int = 0     # size of the index after the run
    vectors_upserted: int = 0
    forced: bool = False
    persisted: bool = False

    def summary(self) -> str:
        bits = [f"{self.parsed} parsed", f"{self.skipped} skipped",
                f"{self.total_indexed} total in index"]
        if self.failed:
            bits.append(f"{len(self.failed)} failed")
        return ", ".join(bits)


def _rebuild_bm25(indexer) -> None:
    """Rebuild the in-memory BM25 index from the indexer's current corpus."""
    if indexer.resume_texts:
        indexer.bm25_resumes = BM25Okapi([t.split() for t in indexer.resume_texts])
    else:
        indexer.bm25_resumes = None


def _embedding_model_name(indexer) -> Optional[str]:
    embeddings = getattr(indexer.vector_store, "embeddings", None)
    return getattr(embeddings, "model_name", None)


def load_index(indexer, store: IndexStore = None, quiet: bool = False) -> bool:
    """Hydrate `indexer` from the persisted corpus. Returns True if it loaded.

    This is the cheap path used on every app start: it rebuilds BM25 in memory
    from stored text and makes no external calls of any kind.
    """
    store = store or IndexStore()
    texts, metadata = store.load_corpus()
    if not texts:
        if not quiet:
            logger.info("No persisted index found — run 'python index_resumes.py' first.")
        return False

    indexer.resume_texts = texts
    indexer.resume_metadata = metadata
    _rebuild_bm25(indexer)
    logger.info(f"Loaded persisted index: {len(texts)} resumes (no API calls)")
    return True


def build_index(
    indexer,
    resumes_dir: str | Path = DEFAULT_RESUMES_DIR,
    force: bool = False,
    store: IndexStore = None,
    progress: ProgressFn = None,
) -> IndexReport:
    """Parse resumes, index them into BM25 + Pinecone, and persist the result.

    Incremental unless `force` is set: only PDFs whose content hash is absent
    from the manifest are parsed. Everything already indexed is reused from
    the stored corpus.
    """
    store = store or IndexStore()
    resumes_dir = str(resumes_dir)
    report = IndexReport(forced=force)

    def say(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    available = list_resume_files(resumes_dir)
    if not available:
        say(f"No PDF resumes found in {resumes_dir}")
        return report

    if force:
        say(f"Force rebuild: re-parsing all {len(available)} resumes")
        store.clear()
        kept_texts: List[str] = []
        kept_metadata: List[Dict[str, Any]] = []
        known_files: Dict[str, Dict[str, Any]] = {}
        to_parse = available
    else:
        kept_texts, kept_metadata = store.load_corpus()
        known_files = dict(store.indexed_files())

        # A file is new if its name is unknown, or its bytes changed since it
        # was indexed (edited or replaced PDF).
        to_parse = []
        for filename in available:
            recorded = known_files.get(filename)
            if not recorded:
                to_parse.append(filename)
                continue
            try:
                if recorded.get("sha256") != file_fingerprint(Path(resumes_dir) / filename):
                    to_parse.append(filename)
            except OSError as e:
                logger.warning(f"Could not fingerprint {filename}: {e}")
                to_parse.append(filename)

        # Drop anything from the corpus that we are about to re-parse, plus any
        # resume whose PDF has since been deleted from the folder.
        reparse = set(to_parse)
        still_present = set(available)
        surviving = [
            (text, meta)
            for text, meta in zip(kept_texts, kept_metadata)
            if meta.get("filename") in still_present
            and meta.get("filename") not in reparse
        ]
        kept_texts = [t for t, _ in surviving]
        kept_metadata = [m for _, m in surviving]
        known_files = {
            name: info for name, info in known_files.items()
            if name in still_present and name not in reparse
        }
        report.skipped = len(kept_texts)

    if not to_parse:
        say(f"Index already up to date — {len(kept_texts)} resumes, nothing to parse")
        _apply_corpus(indexer, kept_texts, kept_metadata)
        report.total_indexed = len(kept_texts)
        report.persisted = True
        return report

    say(f"Parsing {len(to_parse)} resume(s) with Gemini "
        f"({report.skipped} already indexed, skipped)")

    parser = _get_parser()
    new_docs: List[Document] = []
    for i, filename in enumerate(to_parse, start=1):
        path = Path(resumes_dir) / filename
        say(f"  [{i}/{len(to_parse)}] {filename}")
        try:
            doc = build_resume_document(str(path), filename=filename, parser=parser)
            if doc is None:
                logger.warning(f"No text extracted from {filename}")
                report.failed.append(filename)
                continue
            new_docs.append(doc)
            known_files[filename] = {
                "sha256": file_fingerprint(path),
                "candidate_id": doc.metadata.get("candidate_id"),
            }
        except Exception as e:
            logger.error(f"Failed to index {filename}: {e}")
            report.failed.append(filename)

    report.parsed = len(new_docs)

    # Merge the freshly parsed resumes into the retained corpus.
    texts = list(kept_texts) + [d.page_content.lower().strip() for d in new_docs]
    metadata = list(kept_metadata) + [d.metadata for d in new_docs]
    _apply_corpus(indexer, texts, metadata)
    report.total_indexed = len(texts)

    # Only the new documents need embedding + upserting; Pinecone already holds
    # the rest (upsert is idempotent by candidate_id anyway).
    if new_docs and indexer.vector_store.is_ready():
        say(f"Upserting {len(new_docs)} vector(s) into Pinecone")
        if indexer.vector_store.add_resumes(new_docs):
            report.vectors_upserted = len(new_docs)
        else:
            say("Pinecone upsert failed — BM25 index is still usable")
    elif new_docs:
        say("Vector store not ready — indexed for BM25 only")

    report.persisted = store.save(
        texts, metadata, known_files, embedding_model=_embedding_model_name(indexer)
    )
    say(f"Done: {report.summary()}")
    return report


def add_resumes(indexer, docs: List[Document], store: IndexStore = None,
                resumes_dir: str | Path = DEFAULT_RESUMES_DIR) -> bool:
    """Append already-parsed resumes to the live index and persist them.

    Used by the upload path, where the PDF has just been parsed. Replaces any
    existing entry with the same candidate_id so re-uploading a resume updates
    it instead of duplicating it.
    """
    if not docs:
        return False
    store = store or IndexStore()

    new_by_id = {d.metadata.get("candidate_id"): d for d in docs}
    texts = [t for t, m in zip(indexer.resume_texts, indexer.resume_metadata)
             if m.get("candidate_id") not in new_by_id]
    metadata = [m for m in indexer.resume_metadata
                if m.get("candidate_id") not in new_by_id]

    texts += [d.page_content.lower().strip() for d in docs]
    metadata += [d.metadata for d in docs]
    _apply_corpus(indexer, texts, metadata)

    if indexer.vector_store.is_ready():
        indexer.vector_store.add_resumes(docs)

    files = dict(store.indexed_files())
    for doc in docs:
        filename = doc.metadata.get("filename")
        if not filename:
            continue
        path = Path(doc.metadata.get("source") or (Path(resumes_dir) / filename))
        try:
            files[filename] = {
                "sha256": file_fingerprint(path),
                "candidate_id": doc.metadata.get("candidate_id"),
            }
        except OSError as e:
            logger.warning(f"Could not fingerprint uploaded {filename}: {e}")

    return store.save(texts, metadata, files,
                      embedding_model=_embedding_model_name(indexer))


def index_status(indexer=None, store: IndexStore = None) -> Dict[str, Any]:
    """Describe the persisted index without touching PDFs or the LLM."""
    store = store or IndexStore()
    manifest = store.load_manifest()
    resumes_on_disk = list_resume_files(str(DEFAULT_RESUMES_DIR))
    indexed_names = set(manifest.get("files", {}))

    status = {
        "index_exists": store.exists(),
        "indexed_resumes": manifest.get("resume_count", 0),
        "updated_at": manifest.get("updated_at"),
        "embedding_model": manifest.get("embedding_model"),
        "index_dir": str(store.index_dir),
        "pdfs_on_disk": len(resumes_on_disk),
        "unindexed_pdfs": sorted(set(resumes_on_disk) - indexed_names),
        "missing_pdfs": sorted(indexed_names - set(resumes_on_disk)),
    }

    if indexer is not None and indexer.vector_store.is_ready():
        status["pinecone_vector_count"] = indexer.vector_store.get_stats().get(
            "total_vector_count", 0
        )
    return status


def _apply_corpus(indexer, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
    indexer.resume_texts = texts
    indexer.resume_metadata = metadata
    _rebuild_bm25(indexer)
