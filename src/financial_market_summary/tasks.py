from crewai import Task
from datetime import datetime, timedelta

class FinancialTasks:
    """Factory class for creating specialized financial analysis tasks"""
    
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
    
    def create_summary(self,):
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
            - Provide forward-looking insights where appropriate""",
            expected_output="""A professional financial market summary under 500 words featuring:
            - Clear market overview and sentiment analysis
            - Specific stock movements with percentages and symbols
            - Sector performance breakdown
            - Key economic developments and their impact
            - Forward-looking insights for next trading session
            - Professional formatting with sections and bullet points
            - Actionable information for financial professionals""",
            agent=self.agents.search_agent()
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
            
            Use the financial_image_finder tool to locate appropriate charts and graphs.""",
            expected_output="""A well-formatted financial summary enhanced with visual content:
            - Original summary content with 1-2 relevant financial charts integrated
            - Proper markdown formatting for all images
            - Descriptive captions and alt text for each image
            - Images placed at contextually appropriate locations
            - Maintained readability and professional presentation
            - Working image URLs from reliable financial sources""",
            agent=self.agents.search_agent()
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
            
            Critical Translation Requirements:
            1. Preserve Financial Accuracy
               - Keep all stock symbols unchanged (AAPL, MSFT, GOOGL, etc.)
               - Maintain exact numerical data (percentages, dollar amounts)
               - Preserve financial terminology precision
            
            2. Formatting Preservation
               - Keep markdown formatting intact (headers, bold, italics)
               - Maintain bullet points and list structures
               - Preserve image references and captions
               - Keep document hierarchy and sections
            
            3. Language Quality Standards
               - Use professional financial terminology in {lang_display}
               - Ensure natural flow appropriate for financial professionals
               - Maintain cultural appropriateness for target audience
               - Use consistent terminology throughout the document
            
            4. Technical Considerations
               - Handle right-to-left text formatting for Arabic and Hebrew
               - Use appropriate number formatting conventions
               - Maintain readability on mobile devices
               - Ensure proper character encoding
            
            Financial Terminology Guidelines:
            - Accurately translate market concepts (bull/bear market, volatility, etc.)
            - Maintain precision in financial data presentation
            - Use standard financial terms recognized in target language markets
            - When uncertain about financial terms, provide English term in parentheses
            
            Use the financial_translator tool for accurate financial translation.""",
            expected_output=f"""A professionally translated financial summary in {lang_display} containing:
            - Complete translation of all text content while preserving meaning
            - All stock symbols and numerical data kept exactly as original
            - Proper financial terminology appropriate for {lang_display}-speaking professionals
            - Maintained markdown formatting and document structure
            - Natural language flow suitable for target audience
            - Cultural appropriateness and professional tone
            - Preserved image references and captions""",
            agent=self.agents.search_agent()
        )
    
    def send_to_telegram(self):
        """Task for sending content to Telegram channel"""
        return Task(
            description="""Send the complete financial summaries to the Telegram channel in all languages with proper formatting.
            
            Distribution Strategy:
            1. **Sequential Delivery**
               - Send English summary first with clear identification
               - Follow with Arabic translation (marked with 🇸🇦 Arabic)
               - Then Hindi translation (marked with 🇮🇳 Hindi)
               - Finally Hebrew translation (marked with 🇮🇱 Hebrew)
            
            2. **Message Formatting**
               - Use Telegram's markdown formatting for enhanced readability
               - Include appropriate emojis for visual appeal and language identification
               - Add timestamps and source attribution
               - Ensure mobile-friendly formatting
               - Keep individual messages within Telegram's character limits (4096 chars)
            
            3. **Quality Assurance**
               - Verify each message sends successfully before proceeding
               - Include delivery confirmation in logs
               - Handle message truncation if content exceeds limits
               - Retry failed sends with exponential backoff
            
            4. **Professional Presentation**
               - Add consistent headers with CrowdWisdomTrading branding
               - Include generation timestamp and AI attribution
               - Maintain professional tone across all language versions
               - Use consistent formatting patterns
            
            Message Format Template:
            📊 **CrowdWisdomTrading Daily Market Summary**
            🤖 **Generated by AI Agent**
            🕐 **[Timestamp]**
            🌐 **Language: [Language Name]**
            
            [Summary Content]
            
            📈 **Automated Financial Analysis**
            🔄 **Next update: Tomorrow 01:30 IST**
            
            Error Handling:
            - Log any delivery failures with detailed error messages
            - Implement retry mechanism for failed sends
            - Provide fallback notification methods if needed
            - Maintain delivery status tracking
            
            Use the telegram_sender tool to deliver all content successfully.""",
            expected_output="""Complete delivery confirmation report containing:
            - Successful delivery confirmation for all 4 language versions (English + 3 translations)
            - Individual message delivery timestamps and status
            - Proper formatting maintained across all messages
            - Professional presentation with consistent branding
            - Error handling results and any retry attempts
            - Final delivery status summary
            - Message length optimization results
            - Channel engagement metrics if available""",
            agent=self.agents.search_agent()
        )
    
    def comprehensive_market_analysis(self):
        """Advanced task for deep market analysis"""
        return Task(
            description="""Perform comprehensive market analysis beyond basic news summary.
            
            Analysis Framework:
            1. **Market Sentiment Analysis**
               - Bullish vs bearish indicators from news sentiment
               - Fear & greed index implications
               - Volatility patterns (VIX analysis)
               - Social sentiment from financial news sources
            
            2. **Technical Analysis Insights**
               - Key support and resistance levels mentioned in news
               - Moving average trends for major indices
               - Volume analysis and unusual activity
               - Momentum indicators and trend changes
            
            3. **Fundamental Analysis**
               - P/E ratios and valuation concerns from earnings news
               - Earnings growth trends and guidance changes
               - Economic indicator impacts (GDP, inflation, employment)
               - Interest rate environment effects
            
            4. **Sector and Industry Analysis**
               - Sector rotation patterns and drivers
               - Industry-specific news impact analysis
               - Relative strength changes between sectors
               - Defensive vs. growth positioning shifts
            
            5. **Risk Assessment**
               - Geopolitical risk factors from news
               - Economic policy risks and uncertainties
               - Market structure risks (liquidity, volatility)
               - Potential black swan event indicators
            
            6. **Forward-Looking Analysis**
               - Upcoming earnings calendar impact
               - Economic data release schedule
               - Federal Reserve meeting implications
               - Seasonal and cyclical factors""",
            expected_output="""Comprehensive market analysis report featuring:
            - Executive summary of overall market health and direction
            - Detailed sentiment analysis with supporting data points
            - Technical analysis insights with key levels and patterns
            - Fundamental valuation assessment with trend analysis
            - Sector rotation analysis with investment implications
            - Risk assessment matrix with probability weightings
            - Forward-looking market outlook with scenario planning
            - Investment strategy recommendations based on analysis""",
            agent=None
        )

    def create_trading_alerts(self):
        """Task for generating actionable trading alerts"""
        return Task(
            description="""Generate specific, actionable trading alerts based on the market analysis.
            
            Alert Categories:
            1. **Technical Breakout Alerts**
               - Stocks breaking above/below key resistance/support levels
               - Volume confirmation requirements for validity
               - Price targets based on technical analysis
               - Appropriate stop-loss levels for risk management
            
            2. **Earnings-Based Opportunities**
               - Pre-earnings positioning strategies
               - Post-earnings reaction plays
               - Expected move calculations vs. actual moves
               - Historical earnings performance patterns
            
            3. **Economic Data Plays**
               - Federal Reserve announcement positioning
               - Economic data release trading strategies
               - Currency and bond market correlation plays
               - Inflation and interest rate sensitive sectors
            
            4. **Sector Rotation Opportunities**
               - Emerging sector strength identification
               - Sector relative strength momentum changes
               - ETF rotation strategies
               - Defensive vs. cyclical positioning
            
            5. **News-Driven Catalyst Plays**
               - Merger and acquisition rumors/announcements
               - Regulatory approval/disapproval impacts
               - Management changes and guidance updates
               - Product launch and partnership announcements
            
            Alert Specifications:
            - Specific entry price ranges and timing
            - Multiple price targets (conservative and aggressive)
            - Defined stop-loss levels with risk percentage
            - Position sizing recommendations
            - Time horizon for the trade
            - Key catalysts to monitor
            - Risk/reward ratio calculations""",
            expected_output="""Set of 3-5 actionable trading alerts including:
            - Specific stock/ETF symbols with current prices
            - Detailed entry criteria with price levels and timing
            - Multiple exit strategies (profit targets and stop losses)
            - Risk management parameters and position sizing
            - Catalyst timeline and key events to monitor
            - Expected risk/reward ratios with probability assessments
            - Confidence levels and trade rationale
            - Alternative plays if primary setups fail""",
            agent=self.agents.search_agent()
        )
    
    def monitor_market_anomalies(self,agent):
        """Task for detecting unusual market activity"""
        return Task(
            description="""Identify and analyze unusual market activity and potential anomalies from the news data.
            
            Anomaly Detection Framework:
            1. **Volume and Liquidity Anomalies**
               - Unusual volume spikes (>200% of average daily volume)
               - After-hours and pre-market volume irregularities
               - Dark pool activity increases
               - Bid-ask spread widening or narrowing
            
            2. **Price Movement Anomalies**
               - Gap ups/downs exceeding 3% without clear catalysts
               - Intraday reversal patterns exceeding normal ranges
               - Cross-market arbitrage opportunities
               - Unexplained price decoupling from sector peers
            
            3. **Options and Derivatives Activity**
               - Unusual options volume spikes
               - Volatility skew changes indicating sentiment shifts
               - Large block trades and institutional whale activity
               - Put/call ratio extremes
            
            4. **Sector and Market Structure Anomalies**
               - Abnormal sector rotation speed or magnitude
               - Correlation breakdowns between related assets
               - Flight-to-safety patterns without clear triggers
               - Market cap weighted vs. equal weight divergences
            
            5. **Cross-Asset Anomalies**
               - Bond-equity correlation changes
               - Currency market stress signals
               - Commodity price dislocations
               - Credit spread anomalies
            
            6. **Sentiment and Behavioral Anomalies**
               - Extreme sentiment readings from news analysis
               - Contrarian indicators reaching extremes
               - Institutional vs. retail positioning divergences
               - Social media sentiment vs. price action disconnects
            
            Analysis Requirements:
            - Identify anomaly type and severity level (1-5 scale)
            - Determine potential causes and market implications
            - Provide historical context and similar precedents
            - Assess probability of mean reversion vs. trend continuation
            - Recommend monitoring actions and key levels to watch""",
            expected_output="""Market anomaly detection report containing:
            - List of identified anomalies with severity ratings (1-5)
            - Detailed analysis of potential causes and implications
            - Historical context and precedent analysis
            - Probability assessments for various outcome scenarios
            - Recommended monitoring strategies and key levels
            - Risk assessment for portfolio impact
            - Actionable steps for capitalizing on or protecting against anomalies
            - Timeline for resolution or escalation of anomalies""",
            agent=agent
        )
    
    def create_task_sequence(self):
        """Create the complete task sequence for the financial flow"""
        tasks = [
            self.search_financial_news(),
            self.create_summary(),
            self.format_with_images(),
            self.translate_content("arabic"),
            self.translate_content("hindi"), 
            self.translate_content("hebrew"),
            self.send_to_telegram()
        ]
        
        # Set up task dependencies for sequential execution
        for i in range(1, len(tasks)):
            tasks[i].context = [tasks[i-1]]
        
        return tasks
    
    def create_advanced_task_sequence(self):
        """Create advanced task sequence with comprehensive analysis"""
        # Core tasks
        core_tasks = [
            self.search_financial_news(),
            self.create_summary(),
            self.format_with_images()
        ]
        
        # Advanced analysis tasks (can run in parallel after summary)
        analysis_tasks = [
            self.comprehensive_market_analysis(),
            self.create_trading_alerts(),
            self.monitor_market_anomalies()
        ]
        
        # Translation tasks (run in parallel)
        translation_tasks = [
            self.translate_content("arabic"),
            self.translate_content("hindi"),
            self.translate_content("hebrew")
        ]
        
        # Delivery task
        delivery_task = self.send_to_telegram()
        
        # Set up dependencies
        # Core sequence
        for i in range(1, len(core_tasks)):
            core_tasks[i].context = [core_tasks[i-1]]
        
        # Analysis tasks depend on summary
        for task in analysis_tasks:
            task.context = [core_tasks[1]]  # Summary task
        
        # Translation tasks depend on formatted summary
        for task in translation_tasks:
            task.context = [core_tasks[2]]  # Formatted summary
        
        # Delivery depends on all translations
        delivery_task.context = translation_tasks
        
        return core_tasks + analysis_tasks + translation_tasks + [delivery_task]
    
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
                task.context = translation_tasks
        
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