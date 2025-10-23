from crewai import Task
from datetime import datetime

class FinancialTasks:
    """
    Streamlined task factory for financial workflow using only free, accessible sources.
    """
    
    def __init__(self, agents_factory):
        self.agents = agents_factory
    
    def create_all_tasks(self):
        """Create optimized workflow tasks for free financial sources."""
        
        # Task 1: Search Free Financial News
        search_task = Task(
            description="""Search for BREAKING/LIVE US financial news from the past 1 HOUR ONLY using 
            sources.

            STRICT REQUIREMENTS:
            1. ONLY articles published in the last 60 minutes
            2. Focus on BREAKING NEWS, LIVE UPDATES, or TODAY's top market movers/activity which has the most impact
            3. Target sources: Yahoo Finance, MarketWatch, Investing.com, Benzinga, CNBC
            4. Look for: Live market updates, breaking earnings, Fed announcements, major stock moves

            SEARCH STRATEGY:
            - Use terms like "breaking news today", "live updates", "stock market today"
            - Prioritize articles with "live", "today", "now", "breaking" in titles
            - Only include articles that are genuinely recent (within 1 hour)

            CRITICAL: Use tavily_financial_search with hours_back=1 - be extremely selective about recency.""",
            expected_output="Only the most recent breaking financial news from the past hour with verified timestamps and accessible URLs.",
            agent=self.agents.search_agent()
        )
        
        # Task 2: Create Structured Summary
        summary_task = Task(
            #description=f"Create a financial summary from this news data. Include a title, 3 key points, and 2 market implications. Keep it under 200 words.",
            #expected_output="A structured financial summary with title, key points, and market implications.",
            description = (
                                "Write a daily financial market summary in the style of 'The Crowd Wisdom's summary'. "
                                "The summary should be concise, fluent, and professional, following this structure: "
                                "1) Title (e.g., 'The Crowd Wisdom's summary'), "
                                "2) Market overview – summarize Dow Jones, S&P 500, and Nasdaq performance, "
                                "3) Macro news – include 1–2 short items about key background events (start each with 🔍), "
                                "4) Notable stocks – highlight 2–3 stocks that moved significantly with short explanations (use 🟢🔵🟡 to distinguish them), "
                                "5) Commodities or FX if relevant, "
                                "6) Disclaimer – 'The above does not constitute investment advice…'. "
                                "Keep it factual, engaging, and under 200 words.\n\n"
                                "GUARD CLAUSE - PREVENT DUPLICATES:\n"
                                "- Do NOT repeat the same news item, stock, or event multiple times in the summary\n"
                                "- Each company/stock should be mentioned only once\n"
                                "- Each macro news event should appear only once\n"
                                "- If multiple search results discuss the same event, consolidate into ONE mention\n"
                                "- Ensure each bullet point or section discusses a DIFFERENT topic"
                            ),

            expected_output = (
                                "A structured daily market summary under 200 words, including a title, market overview, "
                                "macro news items, notable stock updates, commodities/FX section if relevant, "
                                "and a closing disclaimer. Use emojis as specified to mark sections. "
                                "Each news item, stock, and event must appear only once with no duplicates."
                            ),
            agent=self.agents.summary_agent(),
            context=[search_task]
        )
        
        # Task 3: Format with Free Images
        format_task = Task(
            description="""Format summary for Telegram and find FREE financial charts.
            
            Use financial_image_finder tool to get charts from:
            - Yahoo Finance (priority)
            - Finviz charts
            - Other free financial sources
            
            Ensure images match stocks mentioned in summary.""",
            expected_output="Clean Telegram-ready format with free, verified financial images integrated.",
            agent=self.agents.formatting_agent(),
            context=[summary_task]
        )
        
        # Task 4: Send English version with engaging hook
        send_english_task = Task(
            description="""Send English version to Telegram using telegram_sender tool with language='english'.

            CRITICAL - CREATE ENGAGING HOOK FROM IMAGE DESCRIPTION:
            Before sending, transform the image description into an attention-grabbing hook that:
            1. Creates curiosity and urgency ("Markets just shifted dramatically...", "Breaking: Major move in...")
            2. Highlights the most dramatic/interesting aspect from the description
            3. Uses power words: "surged", "plunged", "breaking", "historic", "unexpected", "dramatic"
            4. Asks a compelling question OR states a surprising fact
            5. Keeps it under 20 words - ultra punchy and direct
            6. Makes users want to read the full summary

            EXAMPLES OF GOOD HOOKS (≤20 WORDS):
            Original: "Wednesday's slide in major averages was led by the Dow Jones..."
            Hook: "⚠️ Dow plunges 234 points! What's triggering the sell-off? 👇" (9 words)

            Original: "The S&P 500 rose 1.2% to close at 5,850.23..."
            Hook: "🚀 S&P 500 surges to 5,850 - best session in weeks! What's fueling this? 👇" (14 words)

            Original: "Tesla shares tumbled 8% following weak delivery figures..."
            Hook: "⚡ Tesla crashes 8% on weak deliveries! The shocking reason 👇" (10 words)

            The telegram_sender will automatically:
            - Find latest screenshot from output/screenshots/
            - Extract AI description from image_results JSON
            - Transform description into engaging hook (as specified above)
            - Send Message 1: Image + Engaging Hook
            - Send Message 2: Full summary with charts
            - Route to English bot if configured

            REMEMBER: The hook should make users STOP scrolling and READ the full summary!""",
            expected_output="Confirmation English version sent to Telegram with engaging hook that drives readership.",
            agent=self.agents.send_agent(),
            context=[format_task]
        )

        # Task 5: Translate to Arabic and send with engaging hook
        translate_arabic_task = Task(
            description="""1. Use financial_translator tool to translate content to Arabic
            2. Use telegram_sender tool with language='arabic' to send to Arabic bot

            ENGAGING HOOK STRATEGY (same as English):
            - Transform image description into attention-grabbing Arabic hook
            - Use Arabic power words and cultural relevance
            - Create urgency and curiosity in Arabic style
            - Keep under 20 words in Arabic -  punchy!
            - Add relevant emojis (⚠️ 🚀 ⚡ 📈 📉 💰)

            CRITICAL: Preserve stock symbols, numbers, HTML tags, and two-message format.""",
            expected_output="Arabic translation sent to Arabic Telegram bot with engaging hook.",
            agent=self.agents.send_agent(),
            context=[format_task]
        )

        # Task 6: Translate to Hindi and send with engaging hook
        translate_hindi_task = Task(
            description="""1. Use financial_translator tool to translate content to Hindi
            2. Use telegram_sender tool with language='hindi' to send to Hindi bot

            ENGAGING HOOK STRATEGY (same as English):
            - Transform image description into attention-grabbing Hindi hook
            - Use Hindi power words and cultural relevance
            - Create urgency and curiosity in Hindi style
            - Keep under 20 words in Hindi -  punchy!
            - Add relevant emojis (⚠️ 🚀 ⚡ 📈 📉 💰)

            CRITICAL: Preserve stock symbols, numbers, HTML tags, and two-message format.""",
            expected_output="Hindi translation sent to Hindi Telegram bot with engaging hook.",
            agent=self.agents.send_agent(),
            context=[format_task]
        )

        # Task 7: Translate to Hebrew and send with engaging hook
        translate_hebrew_task = Task(
            description="""1. Use financial_translator tool to translate content to Hebrew
            2. Use telegram_sender tool with language='hebrew' to send to Hebrew bot

            ENGAGING HOOK STRATEGY (same as English):
            - Transform image description into attention-grabbing Hebrew hook
            - Use Hebrew power words and cultural relevance
            - Create urgency and curiosity in Hebrew style
            - Keep under 20 words in Hebrew -  punchy!
            - Add relevant emojis (⚠️ 🚀 ⚡ 📈 📉 💰)

            CRITICAL: Preserve stock symbols, numbers, HTML tags, and two-message format.""",
            expected_output="Hebrew translation sent to Hebrew Telegram bot with engaging hook.",
            agent=self.agents.send_agent(),
            context=[format_task]
        )
        
        return [
            search_task,
            summary_task,
            format_task,
            send_english_task,
            translate_arabic_task,
            translate_hindi_task,
            translate_hebrew_task
        ]