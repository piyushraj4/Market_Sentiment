"""
Market Sentiment Analysis - Configuration
==========================================
Central configuration for API keys, model settings, and analysis parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (contains API keys — not committed to git)
load_dotenv(Path(__file__).parent / ".env")

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
HEADLINES_DIR = DATA_DIR / "headlines"
PRICES_DIR = DATA_DIR / "prices"
RESULTS_DIR = DATA_DIR / "results"

# Ensure all data directories exist
for d in [HEADLINES_DIR, PRICES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# API Keys (loaded from .env file)
# ─────────────────────────────────────────────
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

# ─────────────────────────────────────────────
# FinBERT Model Settings
# ─────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"
SENTIMENT_BATCH_SIZE = 16
MAX_TOKEN_LENGTH = 512

# ─────────────────────────────────────────────
# Target Stocks
# ─────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "JPM": "JPMorgan",
    "NFLX": "Netflix",
    "AMD": "AMD",
}

# ─────────────────────────────────────────────
# Analysis Settings
# ─────────────────────────────────────────────
# >>> CHANGE THIS to adjust the analysis date period (in days) <<<
LOOKBACK_DAYS = 30          # Options: 7, 14, 30, 60, 90
ROLLING_WINDOW = 7          # Rolling correlation window (days)
CORRELATION_MIN_SAMPLES = 2 # Minimum data points for correlation


