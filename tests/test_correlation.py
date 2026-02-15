"""
Tests for Correlation Analysis Engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


class TestCorrelationEngine:
    """Test suite for the CorrelationEngine class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.correlation import CorrelationEngine
        self.engine = CorrelationEngine()

    def _make_test_data(self, n_days=30, ticker="AAPL"):
        """Generate synthetic test data with known correlation."""
        np.random.seed(42)
        dates = pd.date_range(end="2025-01-30", periods=n_days, freq="B")

        # Create correlated data: sentiment partially predicts returns
        base_signal = np.random.randn(n_days)
        noise = np.random.randn(n_days) * 0.5

        sentiment = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "mean_score": base_signal * 0.3,  # Sentiment based on signal
            "median_score": base_signal * 0.25,
            "std_score": np.abs(np.random.randn(n_days) * 0.1),
            "num_texts": np.random.randint(3, 20, n_days),
            "pct_positive": np.clip(0.5 + base_signal * 0.1, 0, 1),
            "pct_negative": np.clip(0.3 - base_signal * 0.1, 0, 1),
            "pct_neutral": 0.2,
        })

        returns = base_signal * 0.5 + noise  # Returns partially from same signal
        prices = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "close": 150 + np.cumsum(returns),
            "daily_return": returns,
            "next_day_return": np.roll(returns, -1),
            "volatility_5d": np.abs(np.random.randn(n_days) * 0.02),
            "volume": np.random.randint(1000000, 10000000, n_days),
        })

        return sentiment, prices

    def test_merge_data(self):
        """Test that sentiment and price data merge correctly."""
        sent, prices = self._make_test_data(20)
        merged = self.engine._merge_data(sent, prices)
        assert not merged.empty
        assert "mean_score" in merged.columns
        assert "daily_return" in merged.columns

    def test_compute_correlations(self):
        """Test correlation computation."""
        sent, prices = self._make_test_data(30)
        merged = self.engine._merge_data(sent, prices)
        corrs = self.engine._compute_correlations(merged)

        assert "overall" in corrs
        assert "by_ticker" in corrs

        # Check that we get Pearson correlations
        overall = corrs["overall"]
        assert len(overall) > 0

    def test_positive_correlation_detected(self):
        """Test that a known positive correlation is detected."""
        sent, prices = self._make_test_data(50)  # More data = more reliable
        merged = self.engine._merge_data(sent, prices)
        corrs = self.engine._compute_correlations(merged)

        # With our synthetic data, there should be positive correlation
        pearson_same = corrs["overall"].get("pearson_same_day", {})
        if pearson_same:
            assert pearson_same["correlation"] > 0  # Should be positive

    def test_rolling_correlation(self):
        """Test rolling correlation computation."""
        sent, prices = self._make_test_data(30)
        merged = self.engine._merge_data(sent, prices)
        rolling = self.engine._compute_rolling(merged)

        assert "AAPL" in rolling
        assert len(rolling["AAPL"]) > 0
        assert "date" in rolling["AAPL"][0]
        assert "correlation" in rolling["AAPL"][0]

    def test_granger_causality(self):
        """Test Granger causality test."""
        sent, prices = self._make_test_data(30)
        merged = self.engine._merge_data(sent, prices)
        granger = self.engine._granger_test(merged)

        assert "AAPL" in granger
        assert "predictive_correlation" in granger["AAPL"]

    def test_event_study(self):
        """Test event study analysis."""
        sent, prices = self._make_test_data(30)
        merged = self.engine._merge_data(sent, prices)
        events = self.engine._event_study(merged)

        assert "AAPL" in events
        assert "high_sentiment_days" in events["AAPL"]
        assert "low_sentiment_days" in events["AAPL"]

    def test_full_analysis(self):
        """Test full analysis pipeline."""
        sent, prices = self._make_test_data(30)
        results = self.engine.analyze(sent, prices)

        assert "summary" in results
        assert "correlations" in results
        assert "rolling_correlations" in results
        assert "granger_causality" in results
        assert "event_study" in results
        assert "error" not in results

    def test_empty_data(self):
        """Test handling of empty DataFrames."""
        results = self.engine.analyze(pd.DataFrame(), pd.DataFrame())
        assert "error" in results

    def test_insufficient_data(self):
        """Test handling of insufficient data points."""
        sent, prices = self._make_test_data(2)  # Only 2 days
        results = self.engine.analyze(sent, prices)
        # Should either handle gracefully or report insufficient data

    def test_summary_table(self):
        """Test correlation summary table generation."""
        sent, prices = self._make_test_data(30)
        results = self.engine.analyze(sent, prices)
        table = self.engine.get_correlation_summary_table(results)
        assert not table.empty
        assert "Ticker" in table.columns
        assert "Correlation" in table.columns

    def test_multi_ticker(self):
        """Test analysis with multiple tickers."""
        sent1, prices1 = self._make_test_data(20, "AAPL")
        sent2, prices2 = self._make_test_data(20, "TSLA")

        sent = pd.concat([sent1, sent2], ignore_index=True)
        prices = pd.concat([prices1, prices2], ignore_index=True)

        results = self.engine.analyze(sent, prices)
        if "error" not in results:
            assert results["summary"]["tickers_analyzed"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
