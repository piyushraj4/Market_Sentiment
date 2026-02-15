# Contributing to Market Sentiment Analysis

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## 🐛 Reporting Bugs

If you find a bug, please open an issue with:
- A clear title and description
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, etc.)
- Relevant logs or screenshots

## 💡 Suggesting Features

Feature requests are welcome! Please open an issue with:
- A clear description of the feature
- Use cases and benefits
- Any implementation ideas (optional)

## 🔧 Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/market_sentiment.git
   cd market_sentiment
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set up your `.env` file with API keys

## 🧪 Running Tests

Before submitting a PR, ensure all tests pass:

```bash
pytest tests/ -v
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable/function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

Example docstring format:
```python
def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of financial text using FinBERT.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with sentiment_label, sentiment_score, and probabilities
    """
    pass
```

## 🚀 Pull Request Process

1. Create a new branch:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
2. Make your changes
3. Run tests to ensure nothing breaks
4. Commit with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/YourFeatureName
   ```
6. Open a Pull Request with:
   - Clear title describing the change
   - Description of what changed and why
   - Any related issues (e.g., "Fixes #123")

## 🎯 Areas for Contribution

Some ideas for contributions:

- **Additional Data Sources**: Integrate more news APIs or financial feeds
- **Advanced NLP**: Explore other sentiment models or entity extraction
- **Visualization**: Add more chart types or dashboard features
- **Performance**: Optimize batch processing or add caching strategies
- **Testing**: Increase test coverage
- **Documentation**: Improve docstrings, add tutorials, or create examples

## ❓ Questions

If you have questions, feel free to:
- Open a discussion issue
- Reach out via email at [your-email@example.com]

Thank you for contributing! 🙏
