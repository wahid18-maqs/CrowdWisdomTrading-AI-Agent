# CrowdWisdomTrading AI Agent 📈

An AI-powered financial news aggregation and distribution system built with CrewAI that automatically searches, summarizes, and distributes US financial market updates to Telegram channels in multiple languages.

##  Features

- **Real-time Financial News Search**: Aggregates latest US market news from trusted sources
- **AI-Powered Summarization**: Creates concise, actionable market summaries under 500 words
- **Multi-language Support**: Translates content to Arabic, Hindi, and Hebrew
- **Visual Content Integration**: Automatically finds and includes relevant financial charts and graphs
- **Telegram Distribution**: Sends formatted summaries to designated Telegram channels
- **Rate Limiting & Error Handling**: Built-in API quota management and retry mechanisms
- **Comprehensive Logging**: Detailed execution tracking and error reporting

##  Architecture

The system uses a multi-agent architecture powered by CrewAI:

- **Search Agent**: Gathers financial news from multiple sources (Tavily, Serper APIs)
- **Summary Agent**: Creates professional market analysis using Gemini LLM
- **Formatting Agent**: Enhances content with relevant charts and visual elements
- **Translation Agent**: Provides accurate multilingual translations
- **Distribution Agent**: Manages Telegram delivery with retry logic

##  Prerequisites

- Python 3.8+
- API keys for:
  - Google Gemini API (for LLM operations)
  - Tavily API (for news search)
  - Serper API (backup search)
  - Telegram Bot Token & Chat ID

##  Working

https://github.com/user-attachments/assets/f6e98a73-c86a-4082-bd58-1d834e2fef6b

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/wahid18-maqs/financial_market_summary.git
   cd financial-market-summary
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   
   GOOGLE_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

##  Project Structure

```
src/financial_market_summary/
├── main.py                 # Entry point and orchestration
├── crew.py                 # CrewAI workflow management
├── agents.py               # AI agent definitions
├── tasks.py                # Task definitions (alternative workflow)
├── LLM_config.py          # Rate limiting and configuration
└── tools/
    ├── tavily_search.py    # News search tools
    ├── telegram_sender.py  # Telegram distribution
    ├── translator.py       # Multi-language translation
    └── image_finder.py     # Financial chart finder
```

##  Configuration

### API Tier Detection
The system automatically detects your API tier and adjusts behavior:
- **Free Tier**: Conservative rate limiting, reduced search results
- **Paid Tier**: Standard rate limiting, full feature set

### Workflow Settings
Customize in `LLM_config.py`:
- `max_summary_words`: Summary length limit (default: 500)
- `max_search_results`: Number of news articles to process (default: 8)
- `search_hours_back`: Time window for news search (default: 2 hours)
- `languages`: Target translation languages

##  Usage

### Basic Execution
```bash
python -m src.financial_market_summary.main
```

### API Connection Testing
The system includes pre-flight checks that test all API connections before execution.

### Expected Execution Times
- **Free Tier**: 8-12 minutes (conservative rate limiting)
- **Paid Tier**: 3-5 minutes (standard rate limiting)

##  Output

The system generates:

1. **English Summary**: Professional market analysis with key sections:
   - Market Overview
   - Key Movers (top performing stocks)
   - Sector Analysis
   - Economic Highlights
   - Tomorrow's Watch

2. **Visual Content**: 1-2 relevant financial charts or graphs
3. **Translations**: Accurate translations preserving financial terminology
4. **Telegram Delivery**: Formatted messages sent to configured channels

##  Troubleshooting

### Common Issues

**Rate Limit Errors (429)**
- Wait 1-2 minutes before retrying
- Consider upgrading to paid API tiers
- Check quota limits at respective API providers

**Missing API Keys**
- Ensure all required keys are in `.env` file
- Verify API key permissions and quotas

**Translation Failures**
- Stock symbols and numbers are preserved in original language
- Financial terms may be kept in English with local language in parentheses

##  Workflow Process

 <img width="343" height="720" alt="financial_workflow_flowchart" src="https://github.com/user-attachments/assets/1c0da789-641c-4cec-804b-bdb142453013" />

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
