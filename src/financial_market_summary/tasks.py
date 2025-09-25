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
        
        # Task 1: Search Free Financial News (1-hour enforced)
        search_task = Task(
            description="""Search for BREAKING/LIVE US financial news from the past 1 HOUR ONLY using 
            sources.

            STRICT REQUIREMENTS:
            1. ONLY articles published in the last 60-90 minutes
            2. Focus on BREAKING NEWS, LIVE UPDATES, or TODAY's market activity
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
            description=f"Create a financial summary from this news data. Include a title, 3 key points, and 2 market implications. Keep it under 200 words.",
            expected_output="A structured financial summary with title, key points, and market implications.",
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
        
        # Task 4: Translate to Arabic
        translate_arabic_task = Task(
            description="Translate to Arabic keeping stock symbols (AAPL, MSFT) and all numbers unchanged.",
            expected_output="Accurate Arabic translation with preserved financial data.",
            agent=self.agents.translation_agent(),
            context=[format_task]
        )
        
        # Task 5: Translate to Hindi  
        translate_hindi_task = Task(
            description="Translate to Hindi keeping stock symbols and all numbers unchanged.",
            expected_output="Accurate Hindi translation with preserved financial data.",
            agent=self.agents.translation_agent(),
            context=[format_task]
        )
        
        # Task 6: Translate to Hebrew
        translate_hebrew_task = Task(
            description="Translate to Hebrew keeping stock symbols and all numbers unchanged.",
            expected_output="Accurate Hebrew translation with preserved financial data.",
            agent=self.agents.translation_agent(),
            context=[format_task]
        )
        
        # Task 7: Send to Telegram with Free Images
        send_task = Task(
            description="""Send all versions to Telegram using telegram_sender tool.
            
            Message format for each language:
            - Catchy title (bold)
            - Key points (bullets)
            - Market implications (bullets)
            
            The enhanced telegram_sender will:
            - Verify all images are from free sources
            - Send text-only if no free images available
            - Ensure all article links are accessible without paywalls""",
            expected_output="Confirmation all versions sent to Telegram with free, accessible content only.",
            agent=self.agents.send_agent(),
            context=[format_task, translate_arabic_task, translate_hindi_task, translate_hebrew_task]
        )
        
        return [
            search_task,
            summary_task,
            format_task,
            translate_arabic_task,
            translate_hindi_task,
            translate_hebrew_task,
            send_task
        ]