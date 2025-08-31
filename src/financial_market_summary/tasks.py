from crewai import Task
from datetime import datetime, timedelta

class FinancialTasks:
    """Factory class for creating specialized financial analysis tasks"""
    
    def __init__(self, agents):
        self.agents = agents  # Store the FinancialAgents instance
    
    def search_financial_news(self):
        """Task for searching recent financial news"""
        return Task(
            description="""Search for the most recent and relevant US financial news from the last hour.
            
            Focus on:
            1. Major stock movements (gains/losses > 5%)
            2. Earnings announcements and results
            3. Economic data releases (GDP, inflation, employment)
            4. Federal Reserve updates and monetary policy news
            5. Major corporate announcements (mergers, acquisitions, partnerships)
            6. Market-moving events (geopolitical impacts, sector rotations)
            
            Search Requirements:
            - Use both Tavily and Serper tools for comprehensive coverage
            - Focus on the last 1 hour of market activity
            - Prioritize news from reliable sources (Bloomberg, Reuters, CNBC, MarketWatch)
            - Include specific stock symbols, percentage changes, and dollar amounts
            - Gather at least 5-10 relevant news items
            
            Output should include:
            - Source and publication time for each news item
            - Stock symbols and price changes
            - Brief summary of each news item
            - Market sector affected
            - Potential trading impact
            
            Use the tavily_financial_search and serper_financial_search tools to gather comprehensive news data.""",
            expected_output="""A comprehensive list of recent US financial news containing:
            - 5-10 relevant news items from the last hour
            - Each item with: title, source, timestamp, summary, affected stocks/sectors
            - Formatted as structured data ready for analysis
            - Focus on market-moving and trading-relevant information
            - Include specific stock symbols, percentage changes, and dollar amounts""",
            agent=self.agents.search_agent()
        )
    
    def create_summary(self):
        """Task for creating financial market summary"""
        return Task(
            description="""Create a concise daily financial market summary under 500 words based on the search results.
            
            Structure the summary as follows:
            1. **Market Overview** (2-3 sentences on overall market sentiment)
            2. **Key Movers** (Top 3-5 stocks with significant movements)
            3. **Sector Analysis** (Which sectors performed best/worst)
            4. **Economic Highlights** (Important economic data or Fed news)
            5. **Tomorrow's Watch List** (What to monitor)
            
            Requirements:
            - Keep under 500 words total
            - Include specific numbers (percentages, dollar amounts)
            - Mention stock symbols in CAPS (e.g., AAPL, MSFT)
            - Use professional, analytical tone suitable for traders and investors
            - Focus on actionable insights
            - Include market close data if available
            - Highlight any unusual market activity
            
            Style Guidelines:
            - Write for financial professionals
            - Use bullet points for key information
            - Include relevant financial metrics (P/E, volume, etc.)
            - Maintain objective, factual tone
            - Provide forward-looking insights where appropriate
            
            Context: Expects structured news data from the search_financial_news task.""",
            expected_output="""A professional financial market summary under 500 words featuring:
            - Clear market overview and sentiment analysis
            - Specific stock movements with percentages and symbols
            - Sector performance breakdown
            - Key economic developments and their impact
            - Forward-looking insights for next trading session
            - Professional formatting with sections and bullet points
            - Actionable information for financial professionals""",
            agent=self.agents.summary_agent()
        )
    
    def format_with_images(self):
        """Task for formatting summary with relevant images"""
        return Task(
            description="""Format the financial summary by finding and integrating relevant charts and images.
            
            Objectives:
            1. Find 2 relevant financial charts/graphs that support the summary content
            2. Integrate images logically within the text at appropriate sections
            3. Ensure images are accessible and enhance understanding
            4. Maintain professional document presentation
            
            Image Requirements:
            - Search for stock charts of companies mentioned in the summary
            - Include market index charts (S&P 500, NASDAQ, DOW) if relevant
            - Ensure images are from reliable financial sources (Yahoo Finance, TradingView, etc.)
            - Verify image URLs are accessible and working
            - Add appropriate captions and descriptions
            
            Formatting Standards:
            - Use proper markdown image syntax: ![Alt Text](URL)
            - Place images at contextually relevant sections
            - Add descriptive alt text for accessibility
            - Maintain document flow and readability
            - Include image source credits where possible
            
            Fallback Strategy:
            - If specific company charts unavailable, use relevant market index charts
            - Use generic financial visualization if needed
            - Ensure at least 1 relevant image is successfully integrated
            - Maintain professional appearance throughout
            
            Context: Expects a text summary from the create_summary task.
            Use the financial_image_finder tool to locate appropriate charts and graphs.""",
            expected_output="""A well-formatted financial summary enhanced with visual content:
            - Original summary content with 1-2 relevant financial charts integrated
            - Proper markdown formatting for all images
            - Descriptive captions and alt text for each image
            - Images placed at contextually appropriate locations
            - Maintained readability and professional presentation
            - Working image URLs from reliable financial sources""",
            agent=self.agents.formatting_agent()
        )
    
    def translate_content(self, target_language: str):
        """Task for translating content to specified language"""
        language_mapping = {
            'arabic': 'Arabic',
            'hindi': 'Hindi', 
            'hebrew': 'Hebrew'
        }
        
        lang_display = language_mapping.get(target_language.lower(), target_language.title())
        
        return Task(
            description=f"""Translate the formatted financial summary to {lang_display} while maintaining accuracy and professional quality.
            
            Requirements:
            - Maintain the original meaning and tone
            - Ensure financial terminology is accurately translated
            - Preserve markdown formatting, including image syntax
            - Handle any errors gracefully
            
            Context: Expects a markdown-formatted summary from the format_with_images task.""",
            expected_output=f"""Translated financial summary in {lang_display} with:
            - Preserved meaning and professional tone
            - Accurate financial terminology
            - Consistent markdown formatting, including images
            - Error-free translation""",
            agent=self.agents.translation_agent()
        )
    
    def send_to_telegram(self):
        """Task for sending content to Telegram"""
        return Task(
            description="""Send the formatted summaries in all languages to the Telegram channel.
            
            Requirements:
            - Send English version first
            - Then send translations in order: Arabic, Hindi, Hebrew
            - Include appropriate headers and timestamps
            - Handle any errors gracefully
            
            Context: Expects a dictionary with:
            - 'english': markdown-formatted English summary
            - 'arabic': Arabic translation
            - 'hindi': Hindi translation
            - 'hebrew': Hebrew translation""",
            expected_output="""Confirmation of successful delivery to Telegram or error details.""",
            agent=self.agents.send_agent()
        )
    
    def comprehensive_market_analysis(self):
        """Task for comprehensive market analysis"""
        return Task(
            description="""Perform deep market analysis including sentiment and risk assessment.
            
            Requirements:
            - Analyze market sentiment based on news and price movements
            - Assess risk factors (volatility, geopolitical events, etc.)
            - Provide detailed insights for traders
            
            Context: Expects structured news data from the search_financial_news task.""",
            expected_output="""Detailed market analysis report with:
            - Sentiment analysis
            - Risk assessment
            - Actionable trading insights""",
            agent=self.agents.summary_agent()
        )
    
    def create_trading_alerts(self):
        """Task for creating trading alerts"""
        return Task(
            description="""Generate actionable trading alerts based on current market conditions.
            
            Requirements:
            - Identify trading opportunities (buy/sell signals)
            - Include specific stock symbols and price targets
            - Provide rationale for each alert
            
            Context: Expects structured news data from the search_financial_news task.""",
            expected_output="""List of trading alerts with:
            - Stock symbols and price targets
            - Buy/sell recommendations
            - Supporting rationale""",
            agent=self.agents.summary_agent()
        )
    
    def monitor_market_anomalies(self):
        """Task for monitoring market anomalies"""
        return Task(
            description="""Detect and report unusual market activity and anomalies.
            
            Requirements:
            - Identify unusual price movements or volume spikes
            - Flag potential market manipulations or errors
            - Provide detailed explanations
            
            Context: Expects structured news data from the search_financial_news task.""",
            expected_output="""Report on detected market anomalies with:
            - Details of unusual activity
            - Potential causes
            - Recommended actions""",
            agent=self.agents.search_agent()
        )
    
    def create_crew_tasks(self):
        """Create core tasks for the crew"""
        core_tasks = [
            self.search_financial_news(),
            self.create_summary(),
            self.format_with_images()
        ]
        
        translation_tasks = self.create_parallel_translation_tasks()
        
        # Delivery task
        delivery_task = self.send_to_telegram()
        
        # Set up dependencies
        # Core sequence
        for i in range(1, len(core_tasks)):
            core_tasks[i].context = [core_tasks[i-1]]
        
        # Translation tasks depend on formatted summary
        for task in translation_tasks:
            task.context = [core_tasks[2]]  # Formatted summary
        
        # Delivery depends on formatted summary and translations
        delivery_task.context = [core_tasks[2]] + translation_tasks
        
        return core_tasks + translation_tasks + [delivery_task]
    
    def create_parallel_translation_tasks(self):
        """Create translation tasks that can run in parallel"""
        return [
            self.translate_content("arabic"),
            self.translate_content("hindi"),
            self.translate_content("hebrew")
        ]
    
    def create_conditional_tasks(self, market_conditions: dict):
        """Create tasks based on current market conditions"""
        base_tasks = [
            self.search_financial_news(),
            self.create_summary(),
            self.format_with_images()
        ]
        
        # Add conditional tasks based on market conditions
        conditional_tasks = []
        
        if market_conditions.get('high_volatility', False):
            conditional_tasks.append(self.monitor_market_anomalies())
        
        if market_conditions.get('earnings_season', False):
            conditional_tasks.append(self.create_trading_alerts())
        
        if market_conditions.get('market_stress', False):
            conditional_tasks.append(self.comprehensive_market_analysis())
        
        # Always include translations and delivery
        translation_tasks = self.create_parallel_translation_tasks()
        delivery_task = self.send_to_telegram()
        
        all_tasks = base_tasks + conditional_tasks + translation_tasks + [delivery_task]
        
        # Set up dependencies
        for i, task in enumerate(all_tasks):
            if i > 0 and i < len(base_tasks):
                task.context = [all_tasks[i-1]]
            elif i >= len(base_tasks) and i < len(base_tasks) + len(conditional_tasks):
                task.context = [base_tasks[1]]  # Use summary as context
            elif i >= len(base_tasks) + len(conditional_tasks) and i < len(all_tasks) - 1:
                task.context = [base_tasks[2]]  # Translations use formatted summary
            else:  # Delivery task
                task.context = [base_tasks[2]] + translation_tasks
        
        return all_tasks
    
    def get_task_by_name(self, task_name: str):
        """Get specific task by name"""
        task_mapping = {
            'search': self.search_financial_news,
            'summary': self.create_summary,
            'format': self.format_with_images,
            'translate_arabic': lambda: self.translate_content("arabic"),
            'translate_hindi': lambda: self.translate_content("hindi"),
            'translate_hebrew': lambda: self.translate_content("hebrew"),
            'send': self.send_to_telegram,
            'analysis': self.comprehensive_market_analysis,
            'alerts': self.create_trading_alerts,
            'anomalies': self.monitor_market_anomalies
        }
        
        task_func = task_mapping.get(task_name.lower())
        if task_func:
            return task_func()
        else:
            available_tasks = list(task_mapping.keys())
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {available_tasks}")
    
    def create_custom_task(self, name: str, description: str, expected_output: str, agent=None):
        """Create a custom task with specified parameters"""
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent
        )
    
    def get_task_execution_order(self, task_names: list):
        """Get recommended execution order for named tasks"""
        execution_priority = {
            'search': 1,
            'summary': 2,
            'analysis': 3,
            'format': 4,
            'alerts': 5,
            'anomalies': 6,
            'translate_arabic': 7,
            'translate_hindi': 7,
            'translate_hebrew': 7,
            'send': 8
        }
        
        # Sort task names by priority
        sorted_task_names = sorted(
            task_names,
            key=lambda x: execution_priority.get(x.lower(), 999)
        )
        
        return sorted_task_names
    
    def validate_task_dependencies(self, tasks: list):
        """Validate that task dependencies are properly set"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        for i, task in enumerate(tasks):
            task_desc = getattr(task, 'description', 'Unknown Task')[:50] + "..."
            
            # Check if task has required context (except first task)
            if i > 0 and not hasattr(task, 'context'):
                validation_results['warnings'].append(
                    f"Task {i} ({task_desc}) has no context dependency"
                )
            
            # Check for circular dependencies
            if hasattr(task, 'context') and task.context:
                if task in task.context:
                    validation_results['errors'].append(
                        f"Task {i} ({task_desc}) has circular dependency on itself"
                    )
                    validation_results['valid'] = False
        
        return validation_results
    
    def get_task_summary(self):
        """Get summary of all available tasks"""
        return {
            'core_tasks': {
                'search_financial_news': 'Search for recent US financial news from reliable sources',
                'create_summary': 'Generate professional <500 word market summary',
                'format_with_images': 'Add relevant financial charts and formatting',
                'translate_content': 'Translate to Arabic, Hindi, or Hebrew',
                'send_to_telegram': 'Deliver content to Telegram channel'
            },
            'advanced_tasks': {
                'comprehensive_market_analysis': 'Deep market analysis with sentiment and risk assessment',
                'create_trading_alerts': 'Generate actionable trading opportunities',
                'monitor_market_anomalies': 'Detect unusual market activity and anomalies'
            },
            'task_sequences': {
                'basic': 'Core 7-task sequence for daily summaries',
                'advanced': 'Extended sequence with comprehensive analysis',
                'conditional': 'Dynamic task selection based on market conditions'
            }
        }