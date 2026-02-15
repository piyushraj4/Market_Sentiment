"""
Market Sentiment Analysis - Streamlit Dashboard
=================================================
Interactive dashboard with 4 tabs:
  1. Sentiment Feed - recent headlines with sentiment color-coding
  2. Sentiment Trends - time-series overlaid with stock price
  3. Correlation Analysis - heatmaps, scatter plots, rolling correlations
  4. Insights - key stats, Granger causality, event study
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.8;
        margin: 0.5rem 0 0 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .positive { color: #00e676 !important; }
    .negative { color: #ff5252 !important; }
    .neutral  { color: #ffd740 !important; }

    .sentiment-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-positive { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }
    .badge-negative { background: rgba(255,82,82,0.15); color: #ff5252; border: 1px solid rgba(255,82,82,0.3); }
    .badge-neutral  { background: rgba(255,215,64,0.15); color: #ffd740; border: 1px solid rgba(255,215,64,0.3); }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    """Load analysis results from data directory."""
    data = {}

    # Sentiment data
    sent_file = config.RESULTS_DIR / "all_sentiment.csv"
    if sent_file.exists():
        data["sentiment"] = pd.read_csv(sent_file, parse_dates=["date"])

    # Daily aggregated sentiment
    daily_file = config.RESULTS_DIR / "daily_sentiment.csv"
    if daily_file.exists():
        data["daily_sentiment"] = pd.read_csv(daily_file, parse_dates=["date"])

    # Price data
    price_file = config.RESULTS_DIR / "price_data.csv"
    if price_file.exists():
        data["prices"] = pd.read_csv(price_file, parse_dates=["date"])

    # Correlation results
    corr_file = config.RESULTS_DIR / "correlation_analysis.json"
    if corr_file.exists():
        with open(corr_file) as f:
            data["correlations"] = json.load(f)

    return data


def check_data_available(data: dict) -> bool:
    """Check if required data files exist."""
    return bool(data.get("sentiment") is not None and len(data.get("sentiment", [])) > 0)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Market Sentiment Analysis</h1>
    <p>FinBERT-powered NLP sentiment classification correlated with stock price movements</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
data = load_data()

if not check_data_available(data):
    st.warning("⚠️ No analysis data found. Run the pipeline first:")
    st.code("python run_pipeline.py --demo", language="bash")
    st.info("After running the pipeline, refresh this page to see results.")
    st.stop()

sentiment_df = data.get("sentiment", pd.DataFrame())
daily_df = data.get("daily_sentiment", pd.DataFrame())
price_df = data.get("prices", pd.DataFrame())
corr_results = data.get("correlations", {})


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filters")

    available_tickers = sorted(sentiment_df["ticker"].unique().tolist()) if not sentiment_df.empty else []
    selected_tickers = st.multiselect(
        "Tickers",
        options=available_tickers,
        default=available_tickers,
    )

    st.markdown("---")
    st.markdown("### 📈 Quick Stats")

    if not sentiment_df.empty:
        total_texts = len(sentiment_df)
        avg_score = sentiment_df["sentiment_score"].mean()
        pct_pos = (sentiment_df["sentiment_label"] == "positive").mean() * 100
        pct_neg = (sentiment_df["sentiment_label"] == "negative").mean() * 100

        st.metric("Total Texts Analyzed", f"{total_texts:,}")
        st.metric("Avg Sentiment Score", f"{avg_score:+.3f}")
        st.metric("% Positive", f"{pct_pos:.1f}%")
        st.metric("% Negative", f"{pct_neg:.1f}%")

# Filter data
if selected_tickers:
    sentiment_df = sentiment_df[sentiment_df["ticker"].isin(selected_tickers)]
    if not daily_df.empty:
        daily_df = daily_df[daily_df["ticker"].isin(selected_tickers)]
    if not price_df.empty:
        price_df = price_df[price_df["ticker"].isin(selected_tickers)]


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📰 Sentiment Feed",
    "📈 Sentiment Trends",
    "🔗 Correlation Analysis",
    "💡 Insights",
])


# ── TAB 1: Sentiment Feed ──
with tab1:
    st.markdown("### Recent Headlines with Sentiment Scores")

    if not sentiment_df.empty:
        # Format display
        display_df = sentiment_df.copy()
        text_col = "text" if "text" in display_df.columns else ("headline" if "headline" in display_df.columns else None)

        display_cols = ["date", "ticker"] + ([text_col] if text_col else []) + ["sentiment_label", "sentiment_score"]
        available = [c for c in display_cols if c in display_df.columns]
        display_df = display_df[available].sort_values("date", ascending=False).head(100)

        # Color-code by sentiment
        def color_sentiment(row):
            score = row.get("sentiment_score", 0)
            if score > 0.1:
                return [""] * len(row)  # Will use Streamlit's built-in
            elif score < -0.1:
                return ["background-color: rgba(255,82,82,0.1)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(color_sentiment, axis=1),
            use_container_width=True,
            height=600,
        )

        # Sentiment distribution
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                sentiment_df,
                names="sentiment_label",
                title="Sentiment Distribution",
                color="sentiment_label",
                color_discrete_map={
                    "positive": "#00e676",
                    "negative": "#ff5252",
                    "neutral": "#ffd740",
                },
                hole=0.4,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                sentiment_df,
                x="sentiment_score",
                nbins=40,
                title="Sentiment Score Distribution",
                color_discrete_sequence=["#3a7bd5"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                xaxis_title="Sentiment Score",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data available.")


# ── TAB 2: Sentiment Trends ──
with tab2:
    st.markdown("### Sentiment vs Stock Price Over Time")

    if not daily_df.empty and not price_df.empty:
        for ticker in selected_tickers:
            ticker_sent = daily_df[daily_df["ticker"] == ticker].sort_values("date")
            ticker_price = price_df[price_df["ticker"] == ticker].sort_values("date")

            if ticker_sent.empty or ticker_price.empty:
                continue

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(f"{ticker} — Stock Price", f"{ticker} — Daily Sentiment Score"),
                row_heights=[0.6, 0.4],
            )

            # Stock price with candlestick-style coloring
            fig.add_trace(
                go.Scatter(
                    x=ticker_price["date"],
                    y=ticker_price["close"],
                    mode="lines",
                    name="Close Price",
                    line=dict(color="#3a7bd5", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(58,123,213,0.1)",
                ),
                row=1, col=1,
            )

            # Sentiment as bar chart with color coding
            colors = ["#00e676" if s > 0 else "#ff5252" if s < 0 else "#ffd740"
                       for s in ticker_sent["mean_score"]]

            fig.add_trace(
                go.Bar(
                    x=ticker_sent["date"],
                    y=ticker_sent["mean_score"],
                    name="Sentiment",
                    marker_color=colors,
                    opacity=0.8,
                ),
                row=2, col=1,
            )

            # Add zero line on sentiment
            fig.add_hline(y=0, line_dash="dash", line_color="white",
                          opacity=0.3, row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=500,
                showlegend=False,
                margin=dict(l=50, r=30, t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Volume-weighted sentiment
        st.markdown("### Sentiment by Ticker (Summary)")
        summary_data = daily_df.groupby("ticker").agg(
            avg_sentiment=("mean_score", "mean"),
            total_texts=("num_texts", "sum"),
            days_tracked=("date", "nunique"),
        ).reset_index()

        fig = px.bar(
            summary_data,
            x="ticker",
            y="avg_sentiment",
            color="avg_sentiment",
            color_continuous_scale=["#ff5252", "#ffd740", "#00e676"],
            color_continuous_midpoint=0,
            title="Average Sentiment by Ticker",
            text="avg_sentiment",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for trend analysis. Run the pipeline first.")


# ── TAB 3: Correlation Analysis ──
with tab3:
    st.markdown("### Sentiment ↔ Stock Price Correlation")

    if corr_results and "correlations" in corr_results:
        corrs = corr_results["correlations"]

        # Overall correlation metrics
        st.markdown("#### Overall Correlations")
        cols = st.columns(4)
        overall = corrs.get("overall", {})
        for i, (key, data_item) in enumerate(overall.items()):
            with cols[i % 4]:
                corr_val = data_item.get("correlation", 0)
                pval = data_item.get("p_value", 1)
                sig = "✓" if data_item.get("significant") else "✗"
                label = key.replace("_", " ").title()
                st.metric(
                    label=label,
                    value=f"{corr_val:+.4f}",
                    delta=f"p={pval:.3f} {sig}",
                )

        # Per-ticker correlation heatmap
        st.markdown("#### Correlation Heatmap by Ticker")
        by_ticker = corrs.get("by_ticker", {})
        if by_ticker:
            heatmap_data = []
            for ticker, metrics in by_ticker.items():
                row = {"Ticker": ticker}
                for label, vals in metrics.items():
                    row[label.replace("_", " ").title()] = vals.get("correlation", 0)
                heatmap_data.append(row)

            hm_df = pd.DataFrame(heatmap_data).set_index("Ticker")

            fig = px.imshow(
                hm_df.values,
                labels=dict(x="Metric", y="Ticker", color="Correlation"),
                x=hm_df.columns.tolist(),
                y=hm_df.index.tolist(),
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                text_auto=".3f",
                title="Pearson Correlation: Sentiment → Returns",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Scatter plot: sentiment vs returns
        st.markdown("#### Sentiment vs Returns Scatter")
        if not daily_df.empty and not price_df.empty:
            merged_viz = pd.merge(
                daily_df[["date", "ticker", "mean_score"]],
                price_df[["date", "ticker", "daily_return"]],
                on=["date", "ticker"],
                how="inner",
            )
            if not merged_viz.empty:
                fig = px.scatter(
                    merged_viz,
                    x="mean_score",
                    y="daily_return",
                    color="ticker",
                    title="Daily Sentiment Score vs Same-Day Returns",
                    labels={
                        "mean_score": "Sentiment Score",
                        "daily_return": "Daily Return (%)",
                    },
                    trendline="ols",
                    opacity=0.7,
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Rolling correlation
        st.markdown("#### Rolling Correlation (7-day window)")
        rolling = corr_results.get("rolling_correlations", {})
        if rolling:
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, (ticker, data_pts) in enumerate(rolling.items()):
                if ticker not in selected_tickers or not data_pts:
                    continue
                rc_df = pd.DataFrame(data_pts)
                fig.add_trace(go.Scatter(
                    x=rc_df["date"],
                    y=rc_df["correlation"],
                    mode="lines",
                    name=ticker,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=400,
                title="Rolling 7-Day Correlation (Sentiment vs Returns)",
                yaxis_title="Correlation",
                xaxis_title="Date",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No correlation data available. Run the pipeline first.")


# ── TAB 4: Insights ──
with tab4:
    st.markdown("### Key Insights & Predictive Analysis")

    # Granger causality
    granger = corr_results.get("granger_causality", {})
    if granger:
        st.markdown("#### 🔮 Does Sentiment Predict Next-Day Returns?")

        for ticker, g in granger.items():
            if ticker not in selected_tickers:
                continue
            if "error" in g:
                continue

            col1, col2, col3 = st.columns(3)

            predicts = g.get("sentiment_predicts_direction", False)
            with col1:
                st.metric(
                    f"{ticker} — Predictive Power",
                    "YES ✓" if predicts else "NO ✗",
                    delta=f"r = {g.get('predictive_correlation', 0):+.3f}",
                )
            with col2:
                mean_up = g.get("mean_sentiment_before_up")
                if mean_up is not None:
                    st.metric(
                        "Avg Sentiment Before Up Day",
                        f"{mean_up:+.3f}",
                    )
            with col3:
                mean_down = g.get("mean_sentiment_before_down")
                if mean_down is not None:
                    st.metric(
                        "Avg Sentiment Before Down Day",
                        f"{mean_down:+.3f}",
                    )

    # Event study
    events = corr_results.get("event_study", {})
    if events:
        st.markdown("---")
        st.markdown("#### 📅 Event Study: Extreme Sentiment Days")

        event_rows = []
        for ticker, ev in events.items():
            if ticker not in selected_tickers:
                continue
            high = ev.get("high_sentiment_days", {})
            low = ev.get("low_sentiment_days", {})
            event_rows.append({
                "Ticker": ticker,
                "High Sentiment Days": high.get("count", 0),
                "Avg Return (High)": f"{high.get('avg_same_day_return', 0):+.2f}%",
                "Low Sentiment Days": low.get("count", 0),
                "Avg Return (Low)": f"{low.get('avg_same_day_return', 0):+.2f}%",
            })

        if event_rows:
            st.dataframe(pd.DataFrame(event_rows), use_container_width=True)

    # Top bullish/bearish headlines
    if not sentiment_df.empty:
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 Most Bullish Headlines")
            text_col = "text" if "text" in sentiment_df.columns else ("headline" if "headline" in sentiment_df.columns else None)
            if not text_col:
                st.info("No text data available. Re-run the pipeline to include text.")
                st.stop()
            bullish = sentiment_df.nlargest(10, "sentiment_score")
            for _, row in bullish.iterrows():
                score = row["sentiment_score"]
                st.markdown(
                    f'<div style="padding:8px;margin:4px 0;border-radius:8px;'
                    f'background:rgba(0,230,118,0.08);border-left:3px solid #00e676;">'
                    f'<strong>{row["ticker"]}</strong> ({score:+.3f})<br>'
                    f'<span style="font-size:0.9rem;">{row[text_col]}</span></div>',
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown("#### 🔴 Most Bearish Headlines")
            bearish = sentiment_df.nsmallest(10, "sentiment_score")
            for _, row in bearish.iterrows():
                score = row["sentiment_score"]
                st.markdown(
                    f'<div style="padding:8px;margin:4px 0;border-radius:8px;'
                    f'background:rgba(255,82,82,0.08);border-left:3px solid #ff5252;">'
                    f'<strong>{row["ticker"]}</strong> ({score:+.3f})<br>'
                    f'<span style="font-size:0.9rem;">{row[text_col]}</span></div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8892b0;font-size:0.85rem;'>"
    "Market Sentiment Analysis | FinBERT + yfinance | Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True,
)
