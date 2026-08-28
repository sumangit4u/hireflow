"""Unit tests for the metrics in core/retrieval_eval.py — pure maths, no I/O.

These double as worked examples: each expected value is written out by hand so
the formula can be checked by reading the test.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retrieval_eval import (
    aggregate,
    average_precision,
    dcg,
    evaluate_ranking,
    ndcg,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    @pytest.mark.parametrize("position,expected", [(1, 1.0), (2, 0.5), (4, 0.25)])
    def test_scores_by_first_hit_position(self, position, expected):
        ranked = [f"c{i}" for i in range(1, 6)]
        relevant = {f"c{position}"}
        assert reciprocal_rank(ranked, relevant) == pytest.approx(expected)

    def test_zero_when_nothing_relevant_retrieved(self):
        assert reciprocal_rank(["a", "b"], {"z"}) == 0.0

    def test_ignores_hits_after_the_first(self):
        """Only the first hit matters — extra hits do not raise the score."""
        one_hit = reciprocal_rank(["a", "x", "y"], {"a"})
        many_hits = reciprocal_rank(["a", "b", "c"], {"a", "b", "c"})
        assert one_hit == many_hits == 1.0

    def test_empty_ranking(self):
        assert reciprocal_rank([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------

class TestAveragePrecision:
    def test_perfect_ranking_scores_one(self):
        assert average_precision(["a", "b", "x"], {"a", "b"}) == pytest.approx(1.0)

    def test_worked_example(self):
        """ranked [good, bad, good, bad], 2 relevant.

        hit at 1 -> 1/1 = 1.000
        hit at 3 -> 2/3 = 0.667
        AP = (1.000 + 0.667) / 2 = 0.8333
        """
        result = average_precision(["a", "x", "b", "y"], {"a", "b"})
        assert result == pytest.approx((1 / 1 + 2 / 3) / 2)

    def test_order_matters(self):
        """Same hits, better positions, higher score."""
        early = average_precision(["a", "b", "x", "y"], {"a", "b"})
        late = average_precision(["x", "y", "a", "b"], {"a", "b"})
        assert early > late

    def test_divisor_caps_at_list_length(self):
        """30 relevant candidates but only 2 slots — a perfect top 2 scores 1.0."""
        relevant = {f"c{i}" for i in range(30)}
        assert average_precision(["c0", "c1"], relevant) == pytest.approx(1.0)

    def test_zero_without_ground_truth(self):
        assert average_precision(["a"], set()) == 0.0

    def test_zero_when_no_hits(self):
        assert average_precision(["x", "y"], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# dcg / ndcg
# ---------------------------------------------------------------------------

class TestDcg:
    def test_position_discount(self):
        """A hit at position 1 is worth 1.0; at position 2, 1/log2(3)."""
        assert dcg(["a"], {"a"}) == pytest.approx(1.0)
        assert dcg(["x", "a"], {"a"}) == pytest.approx(1 / math.log2(3))

    def test_accumulates_over_hits(self):
        assert dcg(["a", "b"], {"a", "b"}) == pytest.approx(1.0 + 1 / math.log2(3))


class TestNdcg:
    def test_perfect_ranking_scores_one(self):
        assert ndcg(["a", "b", "x"], {"a", "b"}) == pytest.approx(1.0)

    def test_worst_ranking_scores_lowest(self):
        best = ndcg(["a", "b", "x", "y"], {"a", "b"})
        worst = ndcg(["x", "y", "a", "b"], {"a", "b"})
        assert best == pytest.approx(1.0)
        assert 0 < worst < best

    def test_bounded_between_zero_and_one(self):
        for ranked in (["a", "x", "b"], ["x", "a", "y"], ["x", "y", "z"]):
            assert 0.0 <= ndcg(ranked, {"a", "b"}) <= 1.0

    def test_zero_when_no_hits(self):
        assert ndcg(["x", "y"], {"a"}) == 0.0

    def test_empty_inputs(self):
        assert ndcg([], {"a"}) == 0.0
        assert ndcg(["a"], set()) == 0.0


# ---------------------------------------------------------------------------
# evaluate_ranking / aggregate
# ---------------------------------------------------------------------------

class TestEvaluateRanking:
    def test_returns_all_three_metrics(self):
        result = evaluate_ranking(["a", "x"], {"a"})
        assert set(result) == {"reciprocal_rank", "average_precision", "ndcg"}
        assert result["reciprocal_rank"] == pytest.approx(1.0)


class TestAggregate:
    def test_averages_across_queries(self):
        summary = aggregate([
            {"reciprocal_rank": 1.0, "average_precision": 1.0, "ndcg": 1.0},
            {"reciprocal_rank": 0.5, "average_precision": 0.0, "ndcg": 0.5},
        ])
        assert summary["mrr"] == pytest.approx(0.75)
        assert summary["map"] == pytest.approx(0.5)
        assert summary["ndcg"] == pytest.approx(0.75)
        assert summary["queries"] == 2

    def test_empty_input_is_zeroed(self):
        summary = aggregate([])
        assert summary == {"mrr": 0.0, "map": 0.0, "ndcg": 0.0, "queries": 0}


# ---------------------------------------------------------------------------
# How the three metrics differ — the reason we report all of them
# ---------------------------------------------------------------------------

class TestMetricsDisagree:
    def test_mrr_blind_to_later_hits(self):
        """Both rankings hit at position 1, so MRR is identical...

        ...but the second also has hits at 2 and 3, which MAP and NDCG reward.
        This is exactly why MRR alone is not enough.
        """
        one_good = ["a", "x", "y"]
        all_good = ["a", "b", "c"]
        relevant = {"a", "b", "c"}

        assert reciprocal_rank(one_good, relevant) == reciprocal_rank(all_good, relevant)
        assert average_precision(one_good, relevant) < average_precision(all_good, relevant)
        assert ndcg(one_good, relevant) < ndcg(all_good, relevant)
