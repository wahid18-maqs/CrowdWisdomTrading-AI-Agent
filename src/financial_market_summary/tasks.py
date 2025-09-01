# tasks.py
from crewai import Task
from datetime import datetime

class FinancialTasks:
    """
    Factory class for creating all financial analysis and distribution tasks.
    It encapsulates the logic for how tasks are defined and chained together.
    """
    
    def __init__(self, agents_factory):
        """
        Initializes the task factory with an instance of the agent factory
        to ensure tasks are assigned to the correct, pre-initialized agents.
        """
        self.agents = agents_factory
    
    def create_all_tasks(self):
        """
        Creates and returns a list of all tasks in the correct workflow order.
        Each task is configured with a description, expected output, the agent
        responsible, and any prerequisite tasks (context).
        """
        
        # ⚙️ Task 1: Search for Financial News
        search_task = Task(
            description="""Search for recent US financial news from the last 1-2 hours.
            
            Focus on gathering comprehensive data regarding:
            - Major stock movements (significant gains or losses)
            - Corporate earnings reports and major announcements
            - Key economic data releases (e.g., inflation, employment)
            - News related to the Federal Reserve's policies
            - Overall market sector performance
            
            Use both the tavily_financial_search and serper_financial_search tools
            to ensure complete coverage of the latest market-moving information.""",
            expected_output="""A structured JSON or markdown object containing a list of 5-10
            highly relevant news items. Each item must include a title, source, timestamp,
            and a concise summary. Also, identify and include any mentioned stock symbols.""",
            agent=self.agents.search_agent()
        )
        
        # ⚙️ Task 2: Create a Financial Summary
        summary_task = Task(
            description="""Analyze the provided financial news data and create a concise,
            actionable financial market summary suitable for investors. The total length
            should be under 500 words.
            
            The summary must be structured with the following sections:
            1.  **Market Overview**: A brief (2-3 sentence) summary of the overall market sentiment.
            2.  **Key Movers**: List the top 3-5 performing and underperforming stocks.
            3.  **Sector Analysis**: Highlight the best and worst-performing market sectors.
            4.  **Economic Highlights**: Note any significant economic data that was released.
            5.  **Tomorrow's Watch**: Briefly mention what investors should look out for next.
            
            Maintain a professional tone, use CAPS for stock symbols (e.g., AAPL), and
            include specific figures like percentage changes.""",
            expected_output="""A professionally written financial summary, under 500 words,
            with clear, well-defined sections as described.""",
            agent=self.agents.summary_agent(),
            context=[search_task]  # This task depends on the output of search_task
        )
        
        # ⚙️ Task 3: Format the Summary with Images
        format_task = Task(
            description="""Take the financial summary and format it for publication.
            
            Use the financial_image_finder tool to find 1 or 2 relevant charts or graphs
            (e.g., a stock chart for a key mover, or a market index graph).
            
            Integrate these images seamlessly into the summary using markdown.
            Ensure the final output is well-formatted, clean, and visually appealing.""",
            expected_output="""A fully formatted financial summary in markdown,
            including the original text and 1-2 relevant, working image URLs with captions.""",
            agent=self.agents.formatting_agent(),
            context=[summary_task]  # This task depends on the output of summary_task
        )
        
        # ⚙️ Task 4: Translate to Arabic
        translate_arabic_task = Task(
            description="""Translate the formatted financial summary to Arabic.
            
            It is CRITICAL to ensure that:
            - All financial terminology is translated accurately.
            - All stock symbols (e.g., TSLA, GOOGL) remain in English and unchanged.
            - All numbers, percentages, and currency figures are preserved exactly.
            - The original markdown formatting (headers, bold text, image links) is maintained.""",
            expected_output="""An accurate and complete Arabic translation of the
            formatted financial summary, with all critical requirements met.""",
            agent=self.agents.translation_agent(),
            context=[format_task]  # Depends on the formatted English version
        )
        
        # ⚙️ Task 5: Translate to Hindi
        translate_hindi_task = Task(
            description="""Translate the formatted financial summary to Hindi, following
            the same critical requirements: preserve financial terms, keep stock symbols
            in English, maintain all numbers, and keep markdown formatting intact.""",
            expected_output="""An accurate and complete Hindi translation of the
            formatted financial summary.""",
            agent=self.agents.translation_agent(),
            context=[format_task]
        )
        
        # ⚙️ Task 6: Translate to Hebrew
        translate_hebrew_task = Task(
            description="""Translate the formatted financial summary to Hebrew, following
            the same critical requirements: preserve financial terms, keep stock symbols
            in English, maintain all numbers, and keep markdown formatting intact.""",
            expected_output="""An accurate and complete Hebrew translation of the
            formatted financial summary.""",
            agent=self.agents.translation_agent(),
            context=[format_task]
        )
        
        # ⚙️ Task 7: Send All Summaries to Telegram
        send_task = Task(
            description="""Distribute the financial summaries to the designated Telegram channel.
            
            Send each language version as a separate message in the following order:
            1.  English (from the formatting task)
            2.  Arabic (from the Arabic translation task)
            3.  Hindi (from the Hindi translation task)
            4.  Hebrew (from the Hebrew translation task)
            
            Use the telegram_sender tool for each message, ensuring each includes the
            appropriate language-specific header and a timestamp.""",
            expected_output="""A final confirmation message stating that all summaries
            (English, Arabic, Hindi, and Hebrew) have been successfully sent to Telegram.""",
            agent=self.agents.send_agent(),
            # This final task depends on all the content being ready
            context=[format_task, translate_arabic_task, translate_hindi_task, translate_hebrew_task]
        )
        
        # ✅ Return the full list of tasks
        return [
            search_task,
            summary_task,
            format_task,
            translate_arabic_task,
            translate_hindi_task,
            translate_hebrew_task,
            send_task
        ]