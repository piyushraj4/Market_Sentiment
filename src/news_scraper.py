"""
Market Sentiment Analysis - News Scraper
=========================================
Collects financial news headlines from NewsAPI, Google News RSS,
and multiple financial RSS feeds for comprehensive coverage.
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import feedparser
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class NewsCollector:
    """Fetches financial news headlines from multiple sources."""

    # General financial RSS feeds
    GENERAL_RSS_FEEDS = {
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        "MarketWatch Top": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "MarketWatch Markets": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        "CNBC Top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "CNBC Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "CNBC Earnings": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135",
        "Investing.com": "https://www.investing.com/rss/news.rss",
        "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
        "Benzinga": "https://www.benzinga.com/feed",
    }

    def __init__(self):
        self.api_key = config.NEWSAPI_KEY
        self.cache_dir = config.HEADLINES_DIR

    def collect(self, tickers: list[str] = None, days: int = None) -> pd.DataFrame:
        """
        Collect headlines for given tickers from all available sources.
        Uses NewsAPI + Google News RSS + general RSS feeds.
        """
        tickers = tickers or config.DEFAULT_TICKERS
        days = days or config.LOOKBACK_DAYS

        # Check cache first
        cache_file = self._cache_path(tickers, days)
        if cache_file.exists():
            cached = pd.read_csv(cache_file, parse_dates=["date"])
            age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            if age_hours < 6:
                print(f"  [Cache] Loaded {len(cached)} headlines from cache")
                return cached

        all_headlines = []

        # --- Source 1: NewsAPI (everything endpoint) ---
        if self.api_key:
            print("  [NewsAPI] Fetching headlines...")
            for ticker in tickers:
                company = config.TICKER_TO_COMPANY.get(ticker, ticker)
                headlines = self._fetch_newsapi(company, ticker, days)
                all_headlines.extend(headlines)
                # Also try with just the ticker symbol for broader matches
                extra = self._fetch_newsapi(f"{ticker} stock", ticker, days)
                all_headlines.extend(extra)
            print(f"  [NewsAPI] Got {len(all_headlines)} articles")

        # --- Source 2: Google News RSS (per-ticker search) ---
        print("  [Google News] Fetching per-ticker headlines...")
        for ticker in tickers:
            company = config.TICKER_TO_COMPANY.get(ticker, ticker)
            gn_headlines = self._fetch_google_news(company, ticker)
            all_headlines.extend(gn_headlines)
        print(f"  [Google News] Running total: {len(all_headlines)} articles")

        # --- Source 3: General financial RSS feeds ---
        print("  [RSS] Fetching from financial RSS feeds...")
        rss_headlines = self._fetch_rss(tickers)
        all_headlines.extend(rss_headlines)

        if not all_headlines:
            print("  [!] No headlines found. Check your network connection.")

        df = pd.DataFrame(all_headlines)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            # Filter to lookback window
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df["date"] >= cutoff]
            df = df.drop_duplicates(subset=["headline"]).sort_values("date", ascending=False)
            df.to_csv(cache_file, index=False)
            print(f"  [News] Total: {len(df)} unique headlines across {df['ticker'].nunique()} tickers")

        return df

    def _fetch_newsapi(self, query: str, ticker: str, days: int) -> list[dict]:
        """Fetch from NewsAPI everything endpoint."""
        headlines = []
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 100,
                "apiKey": self.api_key,
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                for article in data.get("articles", []):
                    title = article.get("title", "")
                    if title and title != "[Removed]":
                        headlines.append({
                            "date": article.get("publishedAt", "")[:10],
                            "source": article.get("source", {}).get("name", "NewsAPI"),
                            "headline": title,
                            "ticker": ticker,
                        })
            elif resp.status_code == 426:
                print(f"    [NewsAPI] Free plan limits reached for query: {query[:30]}...")
        except Exception as e:
            print(f"    [NewsAPI] Error for {ticker}: {e}")
        return headlines

    def _fetch_google_news(self, company: str, ticker: str) -> list[dict]:
        """Fetch from Google News RSS search for a specific company/ticker."""
        headlines = []
        # Try multiple search queries for broader coverage
        queries = [
            f"{company} stock",
            f"{ticker} stock market",
            f"{company} earnings revenue",
        ]
        for query in queries:
            try:
                encoded = quote_plus(query)
                url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    title = entry.get("title", "")
                    if not title:
                        continue
                    # Parse date
                    try:
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            date_str = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
                        else:
                            date_str = datetime.now().strftime("%Y-%m-%d")
                    except Exception:
                        date_str = datetime.now().strftime("%Y-%m-%d")

                    # Google News titles often have " - Source" at the end
                    source = "Google News"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        title = parts[0]
                        source = parts[1] if len(parts) > 1 else "Google News"

                    headlines.append({
                        "date": date_str,
                        "source": source,
                        "headline": title,
                        "ticker": ticker,
                    })
            except Exception as e:
                pass  # Silently skip failed Google News queries
        return headlines

    def _fetch_rss(self, tickers: list[str]) -> list[dict]:
        """Fetch from general financial RSS feeds and match to tickers."""
        headlines = []
        for source_name, url in self.GENERAL_RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:50]:
                    title = entry.get("title", "")
                    # Parse date
                    try:
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            date_str = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
                        else:
                            date_str = datetime.now().strftime("%Y-%m-%d")
                    except Exception:
                        date_str = datetime.now().strftime("%Y-%m-%d")

                    # Match headlines to tickers
                    matched_tickers = self._match_tickers(title, tickers)
                    for ticker in matched_tickers:
                        headlines.append({
                            "date": date_str,
                            "source": source_name,
                            "headline": title,
                            "ticker": ticker,
                        })
            except Exception as e:
                print(f"    [RSS] Error fetching {source_name}: {e}")
        return headlines

    def _match_tickers(self, text: str, tickers: list[str]) -> list[str]:
        """Match a headline to relevant tickers by company name or symbol."""
        text_upper = text.upper()
        matched = []
        for ticker in tickers:
            company = config.TICKER_TO_COMPANY.get(ticker, "").upper()
            # Match on ticker symbol or company name
            if ticker in text_upper or (company and company in text_upper):
                matched.append(ticker)
        return matched

    def _cache_path(self, tickers: list[str], days: int) -> Path:
        """Generate a cache file path based on query parameters."""
        key = hashlib.md5(f"{'_'.join(sorted(tickers))}_{days}".encode()).hexdigest()[:8]
        return self.cache_dir / f"headlines_{key}.csv"
