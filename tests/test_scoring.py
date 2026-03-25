"""Tests for conformal prediction scoring."""

import math
import pytest
from cfa.utils.scoring import compute_cp_score


class TestComputeCPScore:
    def test_single_response(self):
        """Single response should get a score based on frequency=1/1=1.0."""
        freq = {"response_a": 5}
        scores, entropy = compute_cp_score(freq, weight=1.0, weight_2=1.0)
        assert "response_a" in scores
        # freq/total = 1.0, so base = 10 - 10 = 0
        # entropy of single item = 0, so score = 0 + 0 = 0
        assert scores["response_a"] == pytest.approx(0.0, abs=0.01)

    def test_uniform_distribution(self):
        """Uniform distribution should have maximum entropy for its support."""
        freq = {"a": 10, "b": 10, "c": 10}
        scores, entropy = compute_cp_score(freq, weight=0.0, weight_2=0.0)
        # Each has freq/total = 1/3, score = 10 - 3.33 = 6.67
        for key in scores:
            assert scores[key] == pytest.approx(10 - 10/3, abs=0.01)
        # Normalized entropy: H / log(total_freq) = log(3) / log(30)
        expected = math.log(3) / math.log(30)
        assert entropy == pytest.approx(expected, abs=0.01)

    def test_skewed_distribution(self):
        """Higher frequency response should have lower score (more confident)."""
        freq = {"common": 8, "rare": 2}
        scores, entropy = compute_cp_score(freq, weight=0.0, weight_2=0.0)
        assert scores["common"] < scores["rare"]

    def test_weight_increases_score(self):
        """Higher weight should increase scores (via entropy term)."""
        freq = {"a": 5, "b": 3, "c": 2}
        scores_low, _ = compute_cp_score(freq, weight=0.0, weight_2=0.0)
        scores_high, _ = compute_cp_score(freq, weight=2.0, weight_2=0.0)
        # With positive entropy, higher weight should increase scores
        for key in freq:
            assert scores_high[key] >= scores_low[key]

    def test_zero_weights(self):
        """With zero weights, score depends only on frequency ratio."""
        freq = {"a": 7, "b": 3}
        scores, _ = compute_cp_score(freq, weight=0.0, weight_2=0.0)
        assert scores["a"] == pytest.approx(10 - 7.0, abs=0.01)
        assert scores["b"] == pytest.approx(10 - 3.0, abs=0.01)

    def test_entropy_computation(self):
        """Verify normalized entropy for a known distribution."""
        freq = {"a": 5, "b": 5}
        _, entropy = compute_cp_score(freq, weight=0.0, weight_2=0.0)
        # H = -2*(0.5*log(0.5)) = log(2), normalized by log(10) = log(2)/log(10)
        expected = (-2 * 0.5 * math.log(0.5)) / math.log(10)
        assert entropy == pytest.approx(expected, abs=0.01)
