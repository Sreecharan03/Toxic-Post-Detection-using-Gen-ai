# Twitter Toxicity Detection System 🛡️

## Project Overview
This project is an advanced, dual-layer toxicity detection system designed for a B.Tech project. It automatically fetches tweets from X (formerly Twitter), analyzes their content for harmful or toxic language, and provides AI-driven explanations and suggested improvements.

The system is designed to handle the strict constraints of the Twitter API Free Tier through intelligent caching and rate-limit management.

## Key Features
- **Twitter API v2 Integration**: Fetches real-time tweet data including metrics and author info.
- **Dual-Layer Analysis**:
  - **Layer 1 (Fast)**: Uses a pre-trained `toxic-bert` model (HuggingFace Transformers) for immediate classification.
  - **Layer 2 (Deep Context)**: Uses Google Gemini 1.5 Pro (via API) for contextual understanding, sarcasm detection, and generating polite reformulations.
- **Intelligent Caching**: Saves every fetched tweet to a local JSON cache (`cache/tweets/`) to bypass API rate limits on repeated runs.
- **Smart Rate-Limit Management**: Automatically detects `429 Too Many Requests` errors and pauses execution until the API window resets.
- **Explainable AI**: Not only detects toxicity but explains *why* a post was flagged.

## Implementation Details

### 1. Twitter Client (`simple_twitter_client.py`)
Implemented using direct HTTP requests to the Twitter API v2. It includes:
- **Normalization**: Handles "Bearer " prefixing in environment variables.
- **Caching Logic**: Checks the `cache/` directory before making network requests.
- **Error Handling**: Detailed logging for 401, 403, 404, and 429 status codes.

### 2. Toxicity Detector (`src/toxic_detector.py`)
The brain of the project. It fuses scores from two different models:
- **ML Layer**: Implements `unitary/toxic-bert` for specialized toxicity scoring across 6 categories (toxic, severe_toxic, obscene, threat, insult, identity_hate).
- **LLM Layer**: Sends a specialized JSON-output prompt to Gemini 1.5 Pro to get a human-like explanation of the toxicity.

### 3. Pipeline (`test_working_pipeline.py`)
A single-execution wrapper that ensures no parallel instances are running (using a lock file) and coordinates the flow from URL parsing to final AI analysis.

## How to Run

### Prerequisites
- Python 3.10+
- A Google Gemini API Key
- A Twitter Developer Bearer Token

### Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install tweepy requests transformers torch google-generativeai python-dotenv
   ```
3. Configure your `.env` file:
   ```env
   TWITTER_BEARER_TOKEN=your_token_here
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-1.5-pro
   ```

### Execution
Run the main test pipeline:
```bash
python test_working_pipeline.py
```

## Rate Limit Handling
The system is optimized for the **Free Tier**:
- **15-Minute Window**: The system respects the standard reset period.
- **Wait Mechanism**: If a limit is hit, the console will show: `🛑 429 Rate Limited. Sleeping for X seconds...`
- **Cache**: Once a tweet is analyzed, it never costs another API credit to analyze it again.

## Results Preview
The system outputs a detailed report:
- **Overall Toxicity Score** (0.0 to 1.0)
- **Confidence Level** based on model agreement.
- **AI Explanation**: "The text uses aggressive language directed at a specific group..."
- **Suggested Reformulation**: "I disagree with this approach because..."
