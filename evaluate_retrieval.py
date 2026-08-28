"""Offline evaluation of HireFlow's retrieval quality.

Runs the test cases in data/eval/test_cases.json through hybrid search and
reports MRR, MAP, and NDCG. Uses the index that already exists — it does not
re-parse resumes and makes no Gemini calls.

    python evaluate_retrieval.py                # run every case
    python evaluate_retrieval.py --top-k 5      # score the top 5 instead of 10
    python evaluate_retrieval.py --detail       # show each ranked hit and score
    python evaluate_retrieval.py --list-skills  # skills available for ground truth
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.hybrid_indexer import HybridIndexer
from core.indexing_service import load_index
from core.retrieval_eval import DEFAULT_CASES_PATH, load_cases, run_evaluation


def _print_case(result, show_detail: bool) -> None:
    rank = result.first_hit_rank
    first = f"#{rank}" if rank else "not found"
    print(f"\n  {result.query}")
    if result.note:
        print(f"    {result.note}")
    print(f"    relevant in corpus: {result.relevant_count:<3}  first hit: {first}")
    print(f"    RR {result.reciprocal_rank:.3f}   AP {result.average_precision:.3f}   "
          f"NDCG {result.ndcg:.3f}")

    if show_detail:
        print(f"    {'':2} {'candidate':<28} {'bm25':>7} {'vector':>7} {'combined':>9}")
        for position, (hit, doc) in enumerate(zip(result.hits, result.retrieved), start=1):
            mark = "hit " if hit else "miss"
            print(f"    {mark} {position:>2}. {doc['name'][:26]:<26} "
                  f"{doc['bm25_score']:>7.4f} {doc['vector_score']:>7.4f} "
                  f"{doc['combined_score']:>9.4f}")


def _list_skills(indexer) -> int:
    counts = collections.Counter()
    for meta in indexer.resume_metadata:
        for skill in (meta.get("skills") or []):
            counts[str(skill).strip().lower()] += 1

    if not counts:
        print("No skills in the index. Run 'python index_resumes.py' first.")
        return 1

    print(f"Skills across {len(indexer.resume_metadata)} indexed resumes")
    print("(use these names in data/eval/test_cases.json)\n")
    for skill, count in counts.most_common():
        print(f"  {count:>3}  {skill}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score retrieval quality with MRR, MAP, and NDCG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many results to score per query (default: 10).")
    parser.add_argument("--detail", action="store_true",
                        help="Show every retrieved candidate with its three scores.")
    parser.add_argument("--cases", type=str, default=str(DEFAULT_CASES_PATH),
                        help="Path to the test-case JSON file.")
    parser.add_argument("--list-skills", action="store_true",
                        help="List indexed skills and exit — useful when writing cases.")
    args = parser.parse_args()

    indexer = HybridIndexer()
    if not load_index(indexer):
        print("No index found. Run 'python index_resumes.py' first.")
        return 1

    if args.list_skills:
        return _list_skills(indexer)

    cases = load_cases(args.cases)
    if not cases:
        print(f"No test cases found in {args.cases}")
        return 1

    print(f"Evaluating {len(cases)} queries against {len(indexer.resume_metadata)} "
          f"resumes, scoring the top {args.top_k}")
    print("=" * 72)

    report = run_evaluation(indexer, cases, top_k=args.top_k)
    for result in report["results"]:
        _print_case(result, args.detail)

    summary = report["summary"]
    print("\n" + "=" * 72)
    print(f"Queries scored: {summary['queries']}"
          + (f"   (skipped {report['skipped']} with no relevant candidates)"
             if report["skipped"] else ""))
    print()
    print(f"  MRR   {summary['mrr']:.3f}   mean reciprocal rank — how high the first hit lands")
    print(f"  MAP   {summary['map']:.3f}   mean average precision — are all the hits near the top")
    print(f"  NDCG  {summary['ndcg']:.3f}   position-discounted gain vs. a perfect ranking")

    return 0 if summary["queries"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
