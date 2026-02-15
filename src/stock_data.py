"""
Market Sentiment Analysis - Stock Data Fetcher
================================================
Fetches historical stock price data using yfinance.
Computes returns, volatility, and momentum metrics.
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class StockDataFetcher:
    """Fetches and processes stock price data from Yahoo Finance."""

    def __init__(self):
        self.cache_dir = config.PRICES_DIR

    def fetch(self, tickers: list[str] = None, days: int = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for given tickers and compute derived metrics.

        Returns DataFrame with columns:
            date, ticker, open, high, low, close, volume,
            daily_return, volatility_5d, momentum_5d
        """
        tickers = tickers or config.DEFAULT_TICKERS
        days = days or config.LOOKBACK_DAYS

        # Add buffer days for derived calculations
        fetch_days = days + 20

        all_data = []
        for ticker in tickers:
            cache_file = self._cache_path(ticker, days)

            # Check cache (valid for 1 hour for price data)
            if cache_file.exists():
                age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
                if age_hours < 1:
                    cached = pd.read_csv(cache_file, parse_dates=["date"])
                    all_data.append(cached)
                    print(f"  [Cache] {ticker}: loaded from cache")
                    continue

            try:
                print(f"  [yfinance] Fetching {ticker}...")
                stock = yf.Ticker(ticker)
                start_date = (datetime.now() - timedelta(days=fetch_days)).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")

                hist = stock.history(start=start_date, end=end_date)

                if hist.empty:
                    print(f"    [Warning] No data returned for {ticker}")
                    continue

                df = self._process_ticker(hist, ticker, days)
                df.to_csv(cache_file, index=False)
                all_data.append(df)
                print(f"  [yfinance] {ticker}: {len(df)} trading days")

            except Exception as e:
                print(f"  [Error] Failed to fetch {ticker}: {e}")

        if not all_data:
            print("  [Warning] No stock data fetched!")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        return result

    def _process_ticker(self, hist: pd.DataFrame, ticker: str, days: int) -> pd.DataFrame:
        """Process raw yfinance data into analysis-ready format."""
        df = hist.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Rename Date/Datetime column
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})

        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # ── Derived metrics ──

        # Daily returns (percentage)
        df["daily_return"] = df["close"].pct_change() * 100

        # 5-day rolling volatility (annualized)
        df["volatility_5d"] = df["daily_return"].rolling(window=5).std() * np.sqrt(252)

        # 5-day price momentum (cumulative return over 5 days)
        df["momentum_5d"] = df["close"].pct_change(periods=5) * 100

        # Next-day return (for predictive analysis — "can sentiment predict tomorrow?")
        df["next_day_return"] = df["daily_return"].shift(-1)

        # Trim to requested date range (after computing metrics that need history)
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["date"] >= cutoff]

        # Select final columns
        cols = ["date", "ticker", "open", "high", "low", "close", "volume",
                "daily_return", "volatility_5d", "momentum_5d", "next_day_return"]
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].dropna(subset=["daily_return"])

        return df

    def get_summary(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics per ticker."""
        if price_data.empty:
            return pd.DataFrame()

        summary = price_data.groupby("ticker").agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            trading_days=("date", "count"),
            avg_close=("close", "mean"),
            total_return=("daily_return", "sum"),
            avg_daily_return=("daily_return", "mean"),
            max_daily_return=("daily_return", "max"),
            min_daily_return=("daily_return", "min"),
            avg_volatility=("volatility_5d", "mean"),
        ).round(4)

        return summary

    def _cache_path(self, ticker: str, days: int) -> Path:
        """Generate cache file path for a ticker."""
        return self.cache_dir / f"prices_{ticker}_{days}d.csv"
