# Daily Summary Telegram

An AI-powered financial news aggregation and multilingual distribution bot that automatically searches for US financial market news, captures chart images, and delivers formatted summaries to Telegram channels in 5 languages.

## Overview

Daily Summary Telegram uses a CrewAI-powered multi-agent workflow to:
- Search for latest US financial news from multiple sources
- Extract and analyze financial images from articles
- Generate concise market summaries with AI
- Translate content to multiple languages
- Deliver formatted updates to language specific Telegram bots

## Key Features

### Automated News Aggregation
- **1-hour news search** using Tavily API
- **Multi-source coverage**: Yahoo Finance, CNBC, Reuters, Bloomberg, WSJ, and more
- **Intelligent filtering** for relevant market news

### AI-Powered Chart Extraction
- **Image extraction technology** captures image from article URLs
- **Vision AI analysis** using Gemini 2.5 Flash-exp for image description matching
- **Smart paragraph extraction** from article text using Crawl4AI
- **Automated image descriptions** by matching charts with article content

### Professional Market Summaries
- **Two-message format** for Telegram delivery:
  - Message 1: Short image caption with chart description
  - Message 2: Comprehensive market summary (≤250 words)
- **Structured sections**: Market Overview, Macro News, Notable Stocks, Commodities & FX
- **Live chart links** : for major indices
- **Dynamic titles** based on the day's main market theme

### Multilingual Distribution
- **5 languages supported**: English, Arabic, Hindi, Hebrew, German
- **Separate Telegram bots** for each language
- **Automated translation** while preserving financial data accuracy
- **Stock symbols and numbers** remain unchanged across translations


### Key Technologies

- **CrewAI**: Multi-agent orchestration framework
- **Google Gemini 2.5 Flash**: LLM for summaries and vision analysis
- **Tavily API**: Financial news search
- **Playwright**: Browser automation for chart screenshots
- **Crawl4AI**: Article text extraction
- **BeautifulSoup**: HTML parsing
- **python-telegram-bot**: Telegram API integration

## Installation


### Step 1: Clone Repository

```bash
git clone https://github.com/wahid18-maqs/financial_market_summary.git
cd financial_market_summary
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Playwright Browsers

```bash
playwright install
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API (Required)
GOOGLE_API_KEY=your_gemini_api_key_here

# Tavily API (Required)
TAVILY_API_KEY=your_tavily_api_key_here

# Telegram Bots (At least one required)
TELEGRAM_BOT_TOKEN=your_english_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Optional: Additional Language Bots
TELEGRAM_BOT_TOKEN_ARABIC=your_arabic_bot_token_here
TELEGRAM_CHAT_ID_ARABIC=your_arabic_chat_id_here

TELEGRAM_BOT_TOKEN_HINDI=your_hindi_bot_token_here
TELEGRAM_CHAT_ID_HINDI=your_hindi_chat_id_here

TELEGRAM_BOT_TOKEN_HEBREW=your_hebrew_bot_token_here
TELEGRAM_CHAT_ID_HEBREW=your_hebrew_chat_id_here

TELEGRAM_BOT_TOKEN_GERMAN=your_german_bot_token_here
TELEGRAM_CHAT_ID_GERMAN=your_german_chat_id_here
```

### Getting API Keys

#### Google Gemini API
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Click "Get API Key"
4. Copy your API key to `.env`

#### Tavily API
1. Visit [Tavily](https://tavily.com/)
2. Sign up for an account
3. Navigate to your dashboard
4. Copy your API key to `.env`

#### Telegram Bot Setup
1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` command
3. Follow prompts to create your bot
4. Copy the bot token to `.env`
5. To get chat ID:
   - Send a message to your bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find `"chat":{"id":` in the response
   - Copy the chat ID to `.env`

## Usage

### Run the Bot

```bash
python -m src.financial_market_summary.main
```

### Testing

To run the test suite:

```bash
pytest -v
```

### Expected Execution Times

- **Free Tier APIs**: 8-12 minutes (conservative rate limiting)
- **Paid Tier APIs**: 3-5 minutes (standard rate limiting)

The system automatically detects your API tier and adjusts behavior accordingly.

### Workflow Process

1. **Pre-flight Checks** (10-15 seconds)
   - Validates environment variables
   - Tests API connections (Gemini, Tavily, Telegram)
   - Detects API tier (free vs paid)

2. **News Search** (30-60 seconds)
   - Searches Tavily for financial news (24-hour window)
   - Stores search results in `output/search_results/`

3. **Image Extraction** (60-120 seconds)
   - Visits article URLs with Playwright
   - Captures image
   - Saves to `output/screenshots/`

4. **AI Analysis** (30-60 seconds per article)
   - Analyzes image with Vision AI
   - Extracts article text with Crawl4AI
   - Matches image with descriptions
   - Saves results to `output/image_results/`

5. **Summary Creation** (30-45 seconds)
   - Generates two-message format summary
   - Includes image caption and full summary
   - Adds live chart links

6. **Distribution** (10-20 seconds per language)
   - Sends to English Telegram bot
   - Translates to 4 additional languages
   - Sends to language-specific bots

7. **Completion**
   - Saves workflow results to `logs/workflow-result_YYYYMMDD.json`
   - Automatically cleans up files older than 10 days
   - Logs execution time and status

## Output

### Telegram Messages

**Message 1 (Image Caption)**:
- Short engaging caption (≤150 words)
- Description of the financial chart

**Message 2 (Full Summary)**:
- **Title**: Based on day's main market theme
- **Market Overview**: Dow, S&P 500, Nasdaq performance
- **Macro News**: 2-3 key background events
- **Notable Stocks**: 2-3 significant movers with explanations
- **Commodities & FX**: Brief descriptions if relevant
- **Live Charts**: Links to major indices
- **Disclaimer**: Investment advice notice

### Saved Files

All outputs are saved in the `output/` directory:

```
output/
├── search_results/          # Tavily search results (JSON)
├── screenshots/             # Chart images (PNG)
├── image_results/          # Image analysis results (JSON)
└── workflow_result_*.json  # Final execution summary
```

## Project Structure

```
src/financial_market_summary/
├── main.py                 # Entry point with API checks
├── crew_bot.py             # CrewAI workflow orchestration
├── agents.py               # AI agent definitions
├── LLM_config.py          # Rate limiting configuration
└── tools/
    ├── tavily_search.py    # News search tool
    ├── image_finder.py     # Chart extraction & AI analysis
    ├── translator.py       # Multi-language translation
    └── telegram_sender.py  # Telegram delivery
```

