"""One-time resume indexing for HireFlow.

Run this once after adding resumes, then start the app. The Streamlit UI and the
FastAPI backend both read the index this produces; neither of them re-parses
PDFs or calls Gemini on startup.

    python index_resumes.py              # incremental — only new/changed PDFs
    python index_resumes.py --force      # full rebuild, re-parses everything
    python index_resumes.py --status     # what's indexed, no work done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.hybrid_indexer import HybridIndexer
from core.indexing_service import DEFAULT_RESUMES_DIR, build_index, index_status


def _print_status(indexer) -> int:
    status = index_status(indexer)
    print("HireFlow index status")
    print("---------------------")
    print(f"  Index directory : {status['index_dir']}")
    print(f"  Index exists    : {status['index_exists']}")
    print(f"  Indexed resumes : {status['indexed_resumes']}")
    print(f"  PDFs on disk    : {status['pdfs_on_disk']}")
    print(f"  Last updated    : {status['updated_at'] or 'never'}")
    print(f"  Embedding model : {status['embedding_model'] or 'unknown'}")
    if "pinecone_vector_count" in status:
        print(f"  Pinecone vectors: {status['pinecone_vector_count']}")

    if status["unindexed_pdfs"]:
        print(f"\n  {len(status['unindexed_pdfs'])} PDF(s) not yet indexed:")
        for name in status["unindexed_pdfs"][:10]:
            print(f"    - {name}")
        if len(status["unindexed_pdfs"]) > 10:
            print(f"    ... and {len(status['unindexed_pdfs']) - 10} more")
        print("\n  Run 'python index_resumes.py' to index them.")

    if status["missing_pdfs"]:
        print(f"\n  {len(status['missing_pdfs'])} indexed resume(s) no longer on disk:")
        for name in status["missing_pdfs"][:10]:
            print(f"    - {name}")
        print("\n  They are dropped from the index on the next run.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index resume PDFs into BM25 + Pinecone (one-time step).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-parse and re-index every resume, ignoring what is already indexed. "
             "Costs one Gemini call per resume.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show what is currently indexed and exit without doing any work.",
    )
    parser.add_argument(
        "--resumes-dir", type=str, default=str(DEFAULT_RESUMES_DIR),
        help="Folder holding the resume PDFs (default: data/resumes/).",
    )
    args = parser.parse_args()

    indexer = HybridIndexer()

    if args.status:
        return _print_status(indexer)

    if not indexer.vector_store.is_ready():
        print("Warning: Pinecone is not available — indexing BM25 only.")
        print("         Check PINECONE_API_KEY in the module-root .env file.\n")

    report = build_index(
        indexer,
        resumes_dir=args.resumes_dir,
        force=args.force,
        progress=print,
    )

    if report.failed:
        print(f"\n{len(report.failed)} resume(s) failed:")
        for name in report.failed:
            print(f"  - {name}")

    if report.total_indexed == 0:
        print("\nNothing was indexed. Add PDF resumes to data/resumes/ and try again.")
        return 1

    if not report.persisted:
        print("\nWarning: the index could not be written to disk — "
              "the app will not see it. Check the logs above.")
        return 1

    print(f"\nIndex ready: {report.total_indexed} resumes. "
          f"Start the app with 'uv run streamlit run streamlit/app.py'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
