"""
Tests for FinBERT Sentiment Analysis Pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest


class TestSentimentAnalyzer:
    """Test suite for the SentimentAnalyzer class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.sentiment import SentimentAnalyzer
        self.analyzer = SentimentAnalyzer()

    def test_model_loads(self):
        """Test that FinBERT model loads successfully."""
        self.analyzer._load_model()
        assert self.analyzer._model is not None
        assert self.analyzer._tokenizer is not None

    def test_single_positive_text(self):
        """Test sentiment classification of a clearly positive text."""
        results = self.analyzer.analyze_texts(["Company reports record earnings beating all expectations"])
        assert len(results) == 1
        r = results[0]
        assert "positive" in r
        assert "negative" in r
        assert "neutral" in r
        assert "score" in r
        assert "label" in r
        # Probabilities should sum to ~1
        total = r["positive"] + r["negative"] + r["neutral"]
        assert abs(total - 1.0) < 0.01

    def test_single_negative_text(self):
        """Test sentiment classification of a clearly negative text."""
        results = self.analyzer.analyze_texts(["Stock crashes amid fraud investigation and bankruptcy fears"])
        assert len(results) == 1
        assert results[0]["label"] == "negative"
        assert results[0]["score"] < 0

    def test_batch_processing(self):
        """Test batch processing of multiple texts."""
        texts = [
            "Revenue growth exceeds expectations",
            "Company faces severe losses",
            "Market opens unchanged today",
        ]
        results = self.analyzer.analyze_texts(texts)
        assert len(results) == 3

    def test_empty_text(self):
        """Test handling of empty/whitespace text."""
        results = self.analyzer.analyze_texts(["", "   "])
        assert len(results) == 2
        # Should not crash

    def test_long_text_truncation(self):
        """Test that very long text is handled without errors."""
        long_text = "Financial markets are volatile. " * 500  # >2000 chars
        results = self.analyzer.analyze_texts([long_text])
        assert len(results) == 1
        assert "score" in results[0]

    def test_analyze_dataframe(self):
        """Test DataFrame analysis integration."""
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "ticker": ["AAPL", "TSLA"],
            "headline": ["Apple stock surges on earnings beat", "Tesla recalls vehicles"],
        })
        result = self.analyzer.analyze_dataframe(df, text_column="headline")
        assert "sentiment_score" in result.columns
        assert "sentiment_label" in result.columns
        assert len(result) == 2

    def test_score_range(self):
        """Test that composite scores are in [-1, 1] range."""
        texts = [
            "Stock rises sharply",
            "Company goes bankrupt",
            "Markets trade sideways",
        ]
        results = self.analyzer.analyze_texts(texts)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0

    def test_daily_aggregation(self):
        """Test daily sentiment aggregation."""
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "headline": ["Good news", "Bad news", "Neutral news"],
            "sentiment_score": [0.8, -0.6, 0.0],
            "sentiment_label": ["positive", "negative", "neutral"],
        })
        daily = self.analyzer.aggregate_daily(df)
        assert len(daily) == 2  # Two unique dates
        assert "mean_score" in daily.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
