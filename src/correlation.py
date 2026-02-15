"""
Market Sentiment Analysis - Correlation Engine
================================================
Merges sentiment and price data, then computes:
  - Pearson/Spearman correlations
  - Rolling correlations
  - Granger causality tests
  - Event study around extreme sentiment days
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class CorrelationEngine:
    """Analyzes the relationship between sentiment and stock price movements."""

    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        self.min_samples = config.CORRELATION_MIN_SAMPLES
        self.rolling_window = config.ROLLING_WINDOW

    def analyze(
        self,
        sentiment_daily: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> dict:
        """
        Run full correlation analysis between sentiment and price data.

        Args:
            sentiment_daily: Daily aggregated sentiment (from SentimentAnalyzer.aggregate_daily)
            price_data: Stock price data (from StockDataFetcher.fetch)

        Returns:
            Dictionary with all analysis results.
        """
        if sentiment_daily.empty or price_data.empty:
            print("  [Correlation] Insufficient data for analysis")
            return {"error": "Insufficient data"}

        # Merge sentiment + price data on date and ticker
        merged = self._merge_data(sentiment_daily, price_data)

        if merged.empty or len(merged) < self.min_samples:
            msg = f"Only {len(merged)} merged rows, need {self.min_samples}"
            print(f"  [Correlation] {msg}")
            error_result = {"error": msg}
            self._save_results(error_result)
            return error_result

        print(f"  [Correlation] Analyzing {len(merged)} matched data points...")

        results = {
            "summary": self._compute_summary(merged),
            "correlations": self._compute_correlations(merged),
            "rolling_correlations": self._compute_rolling(merged),
            "granger_causality": self._granger_test(merged),
            "event_study": self._event_study(merged),
            "merged_data": merged.to_dict(orient="records"),
        }

        # Save results
        self._save_results(results)
        return results

    def _merge_data(self, sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Merge sentiment and price data on date + ticker."""
        sentiment = sentiment.copy()
        prices = prices.copy()

        sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.normalize()
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

        merged = pd.merge(
            sentiment,
            prices[["date", "ticker", "close", "daily_return", "next_day_return",
                     "volatility_5d", "volume"]],
            on=["date", "ticker"],
            how="inner",
        )

        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        print(f"  [Merge] {len(sentiment)} sentiment rows × {len(prices)} price rows → {len(merged)} matched")
        return merged

    def _compute_summary(self, df: pd.DataFrame) -> dict:
        """Compute overall summary statistics."""
        return {
            "total_data_points": len(df),
            "tickers_analyzed": df["ticker"].nunique(),
            "ticker_list": sorted(df["ticker"].unique().tolist()),
            "date_range": {
                "start": str(df["date"].min().date()),
                "end": str(df["date"].max().date()),
            },
            "avg_sentiment_score": round(df["mean_score"].mean(), 4),
            "avg_daily_return": round(df["daily_return"].mean(), 4),
            "sentiment_std": round(df["mean_score"].std(), 4),
            "return_std": round(df["daily_return"].std(), 4),
        }

    def _compute_correlations(self, df: pd.DataFrame) -> dict:
        """Compute Pearson & Spearman correlations per ticker and overall."""
        results = {"overall": {}, "by_ticker": {}}

        # Overall correlations
        for method in ["pearson", "spearman"]:
            for target, label in [("daily_return", "same_day"), ("next_day_return", "next_day")]:
                valid = df.dropna(subset=["mean_score", target])
                if len(valid) >= self.min_samples:
                    if method == "pearson":
                        corr, pval = stats.pearsonr(valid["mean_score"], valid[target])
                    else:
                        corr, pval = stats.spearmanr(valid["mean_score"], valid[target])
                    results["overall"][f"{method}_{label}"] = {
                        "correlation": round(corr, 4),
                        "p_value": round(pval, 4),
                        "significant": pval < 0.05,
                        "n_samples": len(valid),
                    }

        # Per-ticker correlations
        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker]
            ticker_results = {}

            for target, label in [("daily_return", "same_day"), ("next_day_return", "next_day")]:
                valid = ticker_df.dropna(subset=["mean_score", target])
                if len(valid) >= self.min_samples:
                    corr, pval = stats.pearsonr(valid["mean_score"], valid[target])
                    ticker_results[label] = {
                        "correlation": round(corr, 4),
                        "p_value": round(pval, 4),
                        "significant": pval < 0.05,
                        "n_samples": len(valid),
                    }

            results["by_ticker"][ticker] = ticker_results

        return results

    def _compute_rolling(self, df: pd.DataFrame) -> dict:
        """Compute rolling correlation for each ticker."""
        results = {}
        window = self.rolling_window

        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker].sort_values("date")

            if len(ticker_df) < window:
                continue

            rolling_corrs = []
            for i in range(window, len(ticker_df) + 1):
                window_data = ticker_df.iloc[i - window:i]
                valid = window_data.dropna(subset=["mean_score", "daily_return"])
                if len(valid) >= 3:
                    corr, _ = stats.pearsonr(valid["mean_score"], valid["daily_return"])
                    rolling_corrs.append({
                        "date": str(window_data.iloc[-1]["date"].date()),
                        "correlation": round(corr, 4),
                    })

            results[ticker] = rolling_corrs

        return results

    def _granger_test(self, df: pd.DataFrame) -> dict:
        """
        Simple Granger-like causality test:
        Does lagged sentiment predict next-day returns better than returns alone?
        Uses linear regression comparison.
        """
        results = {}

        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker].sort_values("date")
            valid = ticker_df.dropna(subset=["mean_score", "next_day_return"])

            if len(valid) < self.min_samples + 2:
                continue

            try:
                # Simple approach: correlation between sentiment today and return tomorrow
                corr, pval = stats.pearsonr(valid["mean_score"], valid["next_day_return"])

                # Also test: is mean sentiment on positive-return days higher?
                pos_returns = valid[valid["next_day_return"] > 0]["mean_score"]
                neg_returns = valid[valid["next_day_return"] <= 0]["mean_score"]

                if len(pos_returns) >= 2 and len(neg_returns) >= 2:
                    t_stat, t_pval = stats.ttest_ind(pos_returns, neg_returns)
                else:
                    t_stat, t_pval = 0, 1.0

                results[ticker] = {
                    "predictive_correlation": round(corr, 4),
                    "predictive_pvalue": round(pval, 4),
                    "sentiment_predicts_direction": pval < 0.05,
                    "mean_sentiment_before_up": round(pos_returns.mean(), 4) if len(pos_returns) > 0 else None,
                    "mean_sentiment_before_down": round(neg_returns.mean(), 4) if len(neg_returns) > 0 else None,
                    "t_test_pvalue": round(t_pval, 4),
                    "significant_difference": t_pval < 0.05,
                }
            except Exception as e:
                results[ticker] = {"error": str(e)}

        return results

    def _event_study(self, df: pd.DataFrame) -> dict:
        """
        Event study: What are average returns around extreme sentiment days?
        Defines 'extreme' as sentiment in the top/bottom 20th percentile.
        """
        results = {}

        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

            if len(ticker_df) < self.min_samples:
                continue

            scores = ticker_df["mean_score"]
            high_threshold = scores.quantile(0.8)
            low_threshold = scores.quantile(0.2)

            high_sent = ticker_df[scores >= high_threshold]
            low_sent = ticker_df[scores <= low_threshold]

            results[ticker] = {
                "high_sentiment_days": {
                    "count": len(high_sent),
                    "avg_same_day_return": round(high_sent["daily_return"].mean(), 4) if not high_sent.empty else None,
                    "avg_next_day_return": round(high_sent["next_day_return"].mean(), 4)
                        if not high_sent.empty and "next_day_return" in high_sent else None,
                    "threshold": round(high_threshold, 4),
                },
                "low_sentiment_days": {
                    "count": len(low_sent),
                    "avg_same_day_return": round(low_sent["daily_return"].mean(), 4) if not low_sent.empty else None,
                    "avg_next_day_return": round(low_sent["next_day_return"].mean(), 4)
                        if not low_sent.empty and "next_day_return" in low_sent else None,
                    "threshold": round(low_threshold, 4),
                },
            }

        return results

    def _save_results(self, results: dict):
        """Save analysis results to JSON."""
        import math

        def clean_for_json(obj):
            """Recursively replace NaN/Inf with None for JSON compatibility."""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj

        # Create a serializable copy (exclude merged_data for the JSON file)
        save_data = {k: v for k, v in results.items() if k != "merged_data"}
        save_data = clean_for_json(save_data)

        output_file = self.results_dir / "correlation_analysis.json"
        with open(output_file, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"  [Results] Saved to {output_file}")

    def get_correlation_summary_table(self, results: dict) -> pd.DataFrame:
        """Convert correlation results to a readable DataFrame."""
        if "correlations" not in results:
            return pd.DataFrame()

        rows = []
        corrs = results["correlations"]

        # Overall
        for key, data in corrs.get("overall", {}).items():
            rows.append({
                "Ticker": "ALL",
                "Metric": key.replace("_", " ").title(),
                "Correlation": data["correlation"],
                "P-Value": data["p_value"],
                "Significant": "✓" if data["significant"] else "✗",
                "N": data["n_samples"],
            })

        # Per-ticker
        for ticker, metrics in corrs.get("by_ticker", {}).items():
            for label, data in metrics.items():
                rows.append({
                    "Ticker": ticker,
                    "Metric": f"Pearson {label.replace('_', ' ').title()}",
                    "Correlation": data["correlation"],
                    "P-Value": data["p_value"],
                    "Significant": "✓" if data["significant"] else "✗",
                    "N": data["n_samples"],
                })

        return pd.DataFrame(rows)
