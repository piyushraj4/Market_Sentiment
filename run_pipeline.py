"""
Market Sentiment Analysis - Pipeline Orchestration
====================================================
Main script that runs the full analysis pipeline:
  1. Collect financial news headlines (NewsAPI + RSS)
  2. Run FinBERT sentiment classification
  3. Fetch stock price data
  4. Run correlation analysis
  5. Print summary
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.news_scraper import NewsCollector
from src.sentiment import SentimentAnalyzer
from src.stock_data import StockDataFetcher
from src.correlation import CorrelationEngine


def run_pipeline(tickers: list[str] = None, days: int = None):
    """Execute the full sentiment analysis pipeline."""

    tickers = tickers or config.DEFAULT_TICKERS
    days = days or config.LOOKBACK_DAYS

    print("=" * 70)
    print("  MARKET SENTIMENT ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Lookback: {days} days")
    print("=" * 70)
    start = time.time()

    # -- Step 1: Collect News Headlines --
    print("\n[1/4] Collecting News Headlines...")
    news_collector = NewsCollector()
    news_df = news_collector.collect(tickers, days)
    print(f"  >> {len(news_df)} headlines collected\n")

    # -- Step 2: Run Sentiment Analysis --
    print("[2/4] Running FinBERT Sentiment Analysis...")
    analyzer = SentimentAnalyzer()

    if not news_df.empty:
        news_df = analyzer.analyze_dataframe(news_df, text_column="headline")
        print(f"  >> {len(news_df)} headlines analyzed")

    # Prepare sentiment data (normalize text column)
    sent_cols = ["date", "ticker", "text", "sentiment_positive", "sentiment_negative",
                 "sentiment_neutral", "sentiment_label", "sentiment_score"]

    all_sentiment = pd.DataFrame()
    if not news_df.empty and "sentiment_score" in news_df.columns:
        ndf = news_df.copy()
        ndf["text"] = ndf.get("headline", ndf.get("text", ""))
        all_sentiment = ndf[[c for c in sent_cols if c in ndf.columns]]

    # Aggregate to daily
    daily_sentiment = analyzer.aggregate_daily(all_sentiment)
    print(f"  >> Daily sentiment: {len(daily_sentiment)} ticker-days\n")

    # Save sentiment data
    if not all_sentiment.empty:
        all_sentiment.to_csv(config.RESULTS_DIR / "all_sentiment.csv", index=False)
    if not daily_sentiment.empty:
        daily_sentiment.to_csv(config.RESULTS_DIR / "daily_sentiment.csv", index=False)

    # -- Step 3: Fetch Stock Price Data --
    print("[3/4] Fetching Stock Price Data...")
    fetcher = StockDataFetcher()
    price_data = fetcher.fetch(tickers, days)
    print(f"  >> {len(price_data)} price records\n")

    if not price_data.empty:
        price_data.to_csv(config.RESULTS_DIR / "price_data.csv", index=False)

    # -- Step 4: Correlation Analysis --
    print("[4/4] Running Correlation Analysis...")
    engine = CorrelationEngine()
    results = engine.analyze(daily_sentiment, price_data)

    elapsed = time.time() - start

    # -- Print Summary --
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Time elapsed: {elapsed:.1f}s")

    if "error" not in results:
        _print_summary(results, engine)
    else:
        print(f"  [!] {results['error']}")

    print("\n  Results saved to:", config.RESULTS_DIR)
    print("  Run the dashboard:  python -m streamlit run dashboard.py")
    print("=" * 70)

    return results


def _print_summary(results: dict, engine: CorrelationEngine):
    """Pretty-print the analysis summary."""

    summary = results.get("summary", {})
    print(f"\n  Data Points  : {summary.get('total_data_points', 'N/A')}")
    print(f"  Tickers      : {', '.join(summary.get('ticker_list', []))}")
    print(f"  Date Range   : {summary.get('date_range', {}).get('start', '?')} -> "
          f"{summary.get('date_range', {}).get('end', '?')}")
    print(f"  Avg Sentiment: {summary.get('avg_sentiment_score', 'N/A')}")
    print(f"  Avg Return   : {summary.get('avg_daily_return', 'N/A')}%")

    # Correlation table
    table = engine.get_correlation_summary_table(results)
    if not table.empty:
        print("\n  +--- Correlation Results -----------------------------------------+")
        for _, row in table.iterrows():
            sig = row['Significant']
            print(f"  | {row['Ticker']:>5} | {row['Metric']:<28} | "
                  f"r={row['Correlation']:+.3f} | p={row['P-Value']:.3f} {sig} |")
        print("  +---------------------------------------------------------------+")

    # Predictive analysis
    granger = results.get("granger_causality", {})
    if granger:
        print("\n  +--- Predictive Analysis (Sentiment -> Next-Day Return) --------+")
        for ticker, g in granger.items():
            if "error" not in g:
                pred = "YES *" if g.get("sentiment_predicts_direction") else "NO"
                print(f"  | {ticker:>5} | Predicts direction: {pred:<6} | "
                      f"r={g.get('predictive_correlation', 0):+.3f} |")
        print("  +---------------------------------------------------------------+")

    # Event study
    events = results.get("event_study", {})
    if events:
        print("\n  +--- Event Study (Extreme Sentiment Days) ----------------------+")
        for ticker, ev in events.items():
            high = ev.get("high_sentiment_days", {})
            low = ev.get("low_sentiment_days", {})
            h_ret = high.get("avg_same_day_return")
            l_ret = low.get("avg_same_day_return")
            if h_ret is not None and l_ret is not None:
                print(f"  | {ticker:>5} | High sent -> {h_ret:+.2f}% | "
                      f"Low sent -> {l_ret:+.2f}% |")
        print("  +---------------------------------------------------------------+")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Sentiment Analysis Pipeline")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Stock tickers to analyze (e.g., AAPL TSLA MSFT)")
    parser.add_argument("--days", type=int, default=None,
                        help="Lookback period in days (default: 30)")
    args = parser.parse_args()

    run_pipeline(tickers=args.tickers, days=args.days)
