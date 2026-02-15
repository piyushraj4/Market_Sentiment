"""
Market Sentiment Analysis - FinBERT Sentiment Pipeline
=======================================================
Uses ProsusAI/finbert to classify financial text sentiment.
Produces per-text probabilities and composite scores in [-1, 1].
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

warnings.filterwarnings("ignore", category=FutureWarning)


class SentimentAnalyzer:
    """FinBERT-based sentiment analysis for financial text."""

    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self):
        self.model_name = config.FINBERT_MODEL
        self.batch_size = config.SENTIMENT_BATCH_SIZE
        self.max_length = config.MAX_TOKEN_LENGTH
        self.device = self._get_device()
        self._model = None
        self._tokenizer = None

    def _get_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  [FinBERT] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("  [FinBERT] Using CPU")
        return device

    def _load_model(self):
        """Lazily load the FinBERT model and tokenizer."""
        if self._model is None:
            print(f"  [FinBERT] Loading model: {self.model_name}")
            print("  [FinBERT] (First run downloads ~420MB of weights)")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            print("  [FinBERT] Model loaded successfully ✓")

    def analyze_texts(self, texts: list[str]) -> list[dict]:
        """
        Classify a list of texts into sentiment categories.

        Returns list of dicts, each containing:
            - positive: probability [0, 1]
            - negative: probability [0, 1]
            - neutral:  probability [0, 1]
            - label:    most probable class
            - score:    composite score in [-1, 1]
        """
        self._load_model()

        results = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            if total_batches > 1:
                print(f"  [FinBERT] Processing batch {batch_num}/{total_batches}...")

            # Clean texts
            clean_batch = [self._clean_text(t) for t in batch]

            # Tokenize
            inputs = self._tokenizer(
                clean_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Process results
            for j, prob in enumerate(probs):
                prob_dict = {
                    "positive": round(prob[0].item(), 4),
                    "negative": round(prob[1].item(), 4),
                    "neutral": round(prob[2].item(), 4),
                }
                label_idx = torch.argmax(prob).item()
                prob_dict["label"] = self.LABEL_MAP[label_idx]
                # Composite score: positive - negative (range [-1, 1])
                prob_dict["score"] = round(prob[0].item() - prob[1].item(), 4)
                results.append(prob_dict)

        return results

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame with a text column.
        Adds sentiment columns directly to the DataFrame.
        """
        if df.empty:
            for col in ["sentiment_positive", "sentiment_negative", "sentiment_neutral",
                        "sentiment_label", "sentiment_score"]:
                df[col] = []
            return df

        # Use 'headline' or 'text' column
        if text_column not in df.columns:
            text_column = "text" if "text" in df.columns else df.columns[0]

        texts = df[text_column].fillna("").tolist()
        sentiments = self.analyze_texts(texts)

        # Merge results back
        sent_df = pd.DataFrame(sentiments)
        sent_df.columns = [f"sentiment_{c}" for c in sent_df.columns]

        result = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
        return result

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment scores to daily level per ticker.

        Returns DataFrame with columns:
            date, ticker, mean_score, median_score, num_texts,
            pct_positive, pct_negative, pct_neutral
        """
        if df.empty or "sentiment_score" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        agg = df.groupby(["date", "ticker"]).agg(
            mean_score=("sentiment_score", "mean"),
            median_score=("sentiment_score", "median"),
            std_score=("sentiment_score", "std"),
            num_texts=("sentiment_score", "count"),
            pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
            pct_negative=("sentiment_label", lambda x: (x == "negative").mean()),
            pct_neutral=("sentiment_label", lambda x: (x == "neutral").mean()),
        ).reset_index()

        agg = agg.round(4)
        return agg

    def _clean_text(self, text: str) -> str:
        """Clean text for model input."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        import re
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Truncate very long text
        if len(text) > 2000:
            text = text[:2000]
        return text.strip() if text.strip() else "neutral"
