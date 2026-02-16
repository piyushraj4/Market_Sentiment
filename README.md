#  Market Sentiment Analysis

A comprehensive financial sentiment analysis tool that scrapes news from multiple sources, applies FinBERT deep learning for sentiment classification, and correlates sentiment with stock price movements to identify predictive signals.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Key Features

- **Multi-Source News Aggregation**: Collects 1000+ articles from NewsAPI, Google News RSS, and 9 major financial feeds
- **FinBERT Sentiment Analysis**: State-of-the-art transformer model (ProsusAI/finbert) for financial text classification
- **Statistical Correlation Engine**: Pearson, Spearman, rolling correlations, and Granger causality tests
- **Event Study Analysis**: Measures abnormal returns around extreme sentiment days
- **Interactive Dashboard**: Real-time Streamlit dashboard with visualizations and insights
- **Scalable Architecture**: Modular design with caching, batch processing, and error handling

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- [NewsAPI Key](https://newsapi.org/register) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/market_sentiment.git
cd market_sentiment

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your NewsAPI key
```

### Running the Analysis

```bash
# Run the full pipeline (news collection → sentiment → correlation)
python run_pipeline.py

# Launch the dashboard
python -m streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501`.

##  Project Structure

```
market_sentiment/
├── src/
│   ├── news_scraper.py      # Multi-source news collection
│   ├── sentiment.py          # FinBERT sentiment pipeline
│   ├── stock_data.py         # yfinance price fetcher
│   └── correlation.py        # Statistical analysis engine
├── tests/                    # Unit tests (pytest)
├── data/                     # Auto-generated results (gitignored)
├── config.py                 # Centralized configuration
├── run_pipeline.py           # Main orchestration script
├── dashboard.py              # Streamlit interactive dashboard
├── requirements.txt          # Python dependencies
├── .env.example              # Template for API keys
└── .gitignore                # Excludes .env, data/, cache
```

##  Configuration

Edit `config.py` to customize:

```python
# Change analysis period (7, 14, 30, 60, 90 days)
LOOKBACK_DAYS = 30

# Change target stocks
DEFAULT_TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

# Adjust correlation thresholds
CORRELATION_MIN_SAMPLES = 2
ROLLING_WINDOW = 7
```

Or pass arguments at runtime:

```bash
python run_pipeline.py --tickers AAPL MSFT --days 60
```

##  Dashboard Features

### 1. Sentiment Feed
- Recent headlines with color-coded sentiment scores
- Sentiment distribution pie chart
- Score histogram

### 2. Sentiment Trends
- Time-series visualization overlaying sentiment and stock price
- Per-ticker sentiment summaries
- Volume-weighted sentiment analysis

### 3. Correlation Analysis
- Overall and per-ticker correlation heatmaps
- Scatter plots with trendlines
- Rolling 7-day correlation charts
- Statistical significance indicators

### 4. Insights
- **Predictive Analysis**: Does sentiment predict next-day returns?
- **Event Study**: Average returns on high vs. low sentiment days
- **Top Bullish/Bearish Headlines**: Extreme sentiment examples

##  Technical Approach

### 1. Data Collection
- **NewsAPI**: Everything endpoint with dual queries per ticker (company + symbol)
- **Google News RSS**: 3 search queries per ticker for broad coverage
- **Financial RSS Feeds**: Yahoo Finance, MarketWatch, CNBC, Seeking Alpha, Benzinga, etc.
- **Result**: ~1400 articles per 30-day period with full date coverage

### 2. Sentiment Classification
- **Model**: ProsusAI/finbert (FinBERT) - BERT fine-tuned on financial text
- **Output**: Positive/Negative/Neutral probabilities + composite score [-1, +1]
- **Batch Processing**: GPU-accelerated with automatic fallback to CPU
- **Aggregation**: Daily mean/median scores per ticker

### 3. Statistical Analysis
- **Pearson & Spearman Correlations**: Same-day and next-day returns
- **Rolling Correlations**: 7-day window for temporal dynamics
- **Granger Causality**: Tests if sentiment predicts future returns
- **Event Study**: Abnormal returns on extreme sentiment days (80th/20th percentile)

### 4. Key Findings (Sample Run)
- **90 data points** across 5 tickers (Jan 20 - Feb 13, 2026)
- **Significant correlations**:
  - MSFT: r=+0.632, p=0.005 (same-day)
  - AAPL: r=+0.499, p=0.030 (same-day)
  - GOOGL: r=+0.502, p=0.047 (next-day, **predictive**)
- **Event study**: High sentiment days → +2.03% returns (AAPL), +0.81% (MSFT)

##  Advanced Usage

### Custom Tickers & Periods
```bash
python run_pipeline.py --tickers NVDA AMD INTC --days 90
```

### Programmatic API
```python
from src.news_scraper import NewsCollector
from src.sentiment import SentimentAnalyzer

collector = NewsCollector()
analyzer = SentimentAnalyzer()

# Collect news
news_df = collector.collect(["AAPL"], days=30)

# Analyze sentiment
results = analyzer.analyze_dataframe(news_df, text_column="headline")
print(results[["headline", "sentiment_label", "sentiment_score"]])
```

### Running Tests
```bash
pytest tests/ -v
```

Expected output:
```
tests/test_correlation.py::test_merge_data PASSED
tests/test_sentiment.py::test_model_loading PASSED
...
==================== 20 passed in 12.34s ====================
```

##  Performance

- **News Collection**: ~30 sec (1400+ articles via 3 sources)
- **Sentiment Analysis**: ~60 sec (FinBERT batch inference on CPU)
- **Correlation Engine**: <1 sec (90 data points)
- **Total Pipeline**: **~90 seconds** end-to-end

##  Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **FinBERT**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) - Financial sentiment classification
- **NewsAPI**: [newsapi.org](https://newsapi.org/) - News aggregation service
- **yfinance**: [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data fetcher
- **Streamlit**: [streamlit.io](https://streamlit.io/) - Dashboard framework

## 📧 Contact

For questions or suggestions, please open an issue or contact me at [piyush.35raj@gmail.com](mailto:piyush.35raj@gmail.com).

---

** If you found this project helpful, please give it a star!**
