from crewai import Task
from textwrap import dedent
from datetime import datetime

class FinancialMarketTasks:
    def __init__(self, agents, tools):
        self.agents = agents
        self.tools = tools
        
    def search_financial_news_task(self):
        return Task(
            description=dedent(f"""\
                Search for the latest US financial market news and developments from the past 2 hours.
                
                Your search should focus on:
                1. Major US stock market movements (NYSE, NASDAQ, S&P 500, Dow Jones)
                2. Significant earnings reports and guidance updates
                3. Federal Reserve news and economic indicators
                4. Major corporate announcements (mergers, acquisitions, IPOs)
                5. Sector-specific developments affecting US markets
                6. Economic data releases and their market impact
                
                CRITICAL REQUIREMENTS:
                - Capture COMPLETE article URLs for each news item found
                - Ensure articles are from trusted financial sources (Bloomberg, Reuters, WSJ, CNBC, etc.)
                - Focus specifically on US market news (not international markets)
                - Include article metadata (title, source, publication date, relevance score)
                - Prioritize recent news (past 2 hours) over older content
                
                Search timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                Return a comprehensive list of articles with:
                - Article titles and summaries
                - Complete URLs for image extraction
                - Source information
                - Publication timestamps
                - Content relevance scores
            """),
            expected_output=dedent("""\
                A structured collection of US financial market news containing:
                
                1. ARTICLE_URLS: List of complete URLs for all articles found (needed for image extraction)
                2. ARTICLES: Detailed information for each article including:
                   - Title
                   - Source domain
                   - Content summary
                   - Publication date/time
                   - Financial relevance score
                   - Full article URL
                
                3. SEARCH_METADATA:
                   - Total articles found
                   - Time range searched
                   - Source distribution
                   - Search query effectiveness
                
                Format: JSON-structured data that can be processed by subsequent agents
            """),
            agent=self.agents.search_agent(),
            tools=[self.tools['tavily_search']],
            output_file="logs/search_results.json"
        )
    
    def extract_images_task(self):
        return Task(
            description=dedent("""\
                Extract relevant financial charts, graphs, and visual content from the news articles 
                found in the search phase.
                
                Your objectives:
                1. Access each article URL from the search results
                2. Scrape article pages to find embedded images
                3. Identify financial charts, graphs, infographics, and data visualizations
                4. Filter out non-relevant images (ads, author photos, logos, etc.)
                5. Validate image accessibility and format compatibility
                6. Select the TOP 2 most relevant images that support the news content
                
                SPECIFIC CRITERIA FOR IMAGE SELECTION:
                - Charts showing stock performance, market data, or economic indicators
                - Infographics explaining financial concepts or market movements
                - Graphs displaying earnings data, sector performance, or economic trends
                - Visual content that directly relates to the key news stories found
                
                TECHNICAL REQUIREMENTS:
                - Ensure images are accessible and properly formatted
                - Verify images meet Telegram size and format requirements
                - Include proper source attribution for each image
                - Maintain relationship between images and their source articles
                
                EXCLUSIONS:
                - Generic stock photos not related to specific news
                - Author headshots or company logos
                - Advertisement banners or promotional content
                - Images that are too small or low quality
            """),
            expected_output=dedent("""\
                A curated collection of financial visual content containing:
                
                1. SELECTED_IMAGES: Top 2 most relevant images with:
                   - Image data (downloaded and validated)
                   - Source article URL and title
                   - Image caption or description
                   - Source attribution (publication name)
                   - Relevance score and reasoning
                   - Technical specifications (size, format, dimensions)
                
                2. IMAGE_CONTEXT: For each selected image:
                   - Which news story/article it relates to
                   - What financial data or concept it illustrates
                   - How it supports the overall market narrative
                
                3. EXTRACTION_METADATA:
                   - Total images found across all articles
                   - Number of images that passed relevance filtering
                   - Technical validation results
                   - Any extraction issues encountered
                
                Format: Structured data ready for integration with written analysis
            """),
            agent=self.agents.image_extraction_agent(),
            tools=[self.tools['image_finder']],
            context=[self.search_financial_news_task()],
            output_file="logs/extracted_images.json"
        )
    
    def create_market_summary_task(self):
        return Task(
            description=dedent("""\
                Create a comprehensive, professional US financial market summary based on the 
                news articles found and considering the available visual content.
                
                Your analysis should include:
                
                1. MARKET OVERVIEW (100-150 words):
                   - Overall market sentiment and direction
                   - Major index movements and key drivers
                   - Trading volume and market participation
                
                2. KEY MOVERS (100-120 words):
                   - Top performing stocks and reasons
                   - Significant decliners and causes
                   - Sector rotation patterns observed
                
                3. EARNINGS & CORPORATE NEWS (80-100 words):
                   - Notable earnings reports and reactions
                   - Major corporate announcements
                   - Guidance updates and their impact
                
                4. ECONOMIC HIGHLIGHTS (80-100 words):
                   - Economic data releases and interpretation
                   - Federal Reserve developments
                   - Policy implications for markets
                
                5. SECTOR ANALYSIS (60-80 words):
                   - Best and worst performing sectors
                   - Sector-specific news and trends
                
                6. TOMORROW'S WATCH (40-60 words):
                   - Key events to monitor
                   - Potential market catalysts
                
                WRITING GUIDELINES:
                - Professional, analytical tone suitable for serious investors
                - Include specific data points (percentages, dollar amounts) when available
                - Reference visual content when it supports your analysis
                - Keep total length under 500 words
                - Use clear, actionable language
                - Focus on US markets specifically
                
                VISUAL CONTENT INTEGRATION:
                - Be aware of what charts/graphs are available
                - Reference visual content naturally in your analysis
                - Don't duplicate information that's clearly shown in charts
                - Explain what the visual content demonstrates
            """),
            expected_output=dedent("""\
                A professional market summary structured as follows:
                
                **📊 US MARKET SUMMARY - [Date/Time]**
                
                **Market Overview**
                [Professional analysis of overall market conditions, major index performance, 
                and primary market drivers]
                
                **Key Movers**
                [Analysis of top gainers/losers with specific reasons and implications]
                
                **Earnings & Corporate News**
                [Coverage of significant corporate developments and earnings impacts]
                
                **Economic Highlights**
                [Analysis of economic data and Federal Reserve developments]
                
                **Sector Analysis**
                [Sector performance breakdown and rotation patterns]
                
                **Tomorrow's Watch**
                [Key events and potential catalysts for next trading session]
                
                **Visual Content References**
                [Notes on how available charts/graphs support the analysis]
                
                Total length: 450-500 words
                Tone: Professional, analytical, actionable
                Format: Markdown-ready for Telegram distribution
            """),
            agent=self.agents.summary_agent(),
            tools=[],
            context=[self.search_financial_news_task(), self.extract_images_task()],
            output_file="logs/market_summary.md"
        )
    
    def format_content_with_visuals_task(self):
        return Task(
            description=dedent("""\
                Integrate the written market summary with visual content to create a cohesive,
                professional market report optimized for Telegram distribution.
                
                Your responsibilities:
                1. Combine the written market analysis with available visual content
                2. Create appropriate captions for charts and graphs
                3. Ensure visual content complements (not duplicates) written analysis
                4. Optimize formatting for Telegram's message structure
                5. Prepare content sequencing (text first, then supporting visuals)
                6. Add proper source attribution for all visual content
                
                CONTENT INTEGRATION GUIDELINES:
                - Reference visual content naturally in the text where relevant
                - Create descriptive captions that explain what each chart shows
                - Ensure charts support and enhance the written narrative
                - Maintain professional presentation throughout
                
                TELEGRAM OPTIMIZATION:
                - Format text with appropriate markdown for readability
                - Ensure message length fits Telegram limits (4096 chars per message)
                - Prepare image captions with source attribution
                - Create logical content flow (summary → supporting visuals)
                
                VISUAL CONTENT HANDLING:
                - Include only high-quality, relevant charts/graphs
                - Ensure all images have proper captions and source attribution
                - Verify images support the key themes in the written summary
                - Remove any visual content that doesn't add clear value
            """),
            expected_output=dedent("""\
                A complete, formatted market report package containing:
                
                1. FORMATTED_SUMMARY:
                   - Telegram-optimized markdown text
                   - Professional structure and flow
                   - Appropriate length for platform limits
                   - Clear section headers and formatting
                
                2. VISUAL_CONTENT_PACKAGE:
                   - Selected images with optimized captions
                   - Source attribution for each visual element
                   - Context explanations for charts/graphs
                   - Technical specifications confirmed for Telegram
                
                3. CONTENT_SEQUENCE:
                   - Recommended order for message delivery
                   - Timing suggestions between text and images
                   - Message splitting strategy if needed
                
                4. INTEGRATION_NOTES:
                   - How visual content supports written analysis
                   - Key points illustrated by charts/graphs
                   - Overall content cohesion assessment
                
                Ready for multi-language translation and distribution
            """),
            agent=self.agents.formatting_agent(),
            tools=[],
            context=[self.create_market_summary_task(), self.extract_images_task()],
            output_file="logs/formatted_content.json"
        )
    
    def translate_content_task(self):
        return Task(
            description=dedent("""\
                Translate the formatted market summary into Arabic, Hindi, and Hebrew while
                preserving financial accuracy and professional tone.
                
                TRANSLATION REQUIREMENTS:
                
                1. PRESERVE TECHNICAL ACCURACY:
                   - Keep all stock symbols unchanged (AAPL, TSLA, etc.)
                   - Maintain numerical data exactly as provided
                   - Preserve company names and exchange names
                   - Keep percentage and dollar amounts in original format
                
                2. CULTURAL ADAPTATION:
                   - Use appropriate financial terminology for each language
                   - Maintain professional investment analysis tone
                   - Adapt market concepts for regional understanding
                   - Ensure currency references are clear
                
                3. FORMAT CONSISTENCY:
                   - Maintain markdown formatting across languages
                   - Preserve section structure and headers
                   - Keep message length suitable for Telegram
                   - Ensure readability in target languages
                
                4. VISUAL CONTENT CAPTIONS:
                   - Translate image captions accurately
                   - Maintain source attribution in all languages
                   - Explain chart content clearly in target language
                   - Preserve technical chart terminology
                
                LANGUAGE-SPECIFIC GUIDELINES:
                - Arabic: Right-to-left formatting, financial terminology preservation
                - Hindi: Technical terms with English equivalents in parentheses
                - Hebrew: Professional financial language, maintain English technical terms
                
                QUALITY STANDARDS:
                - Native-level fluency in translations
                - Professional investment analysis tone
                - Cultural appropriateness for financial content
                - Technical accuracy in all financial terms
            """),
            expected_output=dedent("""\
                Complete translated content package containing:
                
                1. ARABIC_VERSION:
                   - Fully translated market summary
                   - Translated image captions with source attribution
                   - Proper RTL formatting considerations
                   - Financial terminology correctly adapted
                
                2. HINDI_VERSION:
                   - Complete Hindi translation
                   - Technical terms with English references where appropriate
                   - Image captions in Hindi with source attribution
                   - Cultural adaptation for Indian financial markets context
                
                3. HEBREW_VERSION:
                   - Professional Hebrew financial translation
                   - Maintained technical accuracy
                   - Translated image captions and attributions
                   - Appropriate formatting for Hebrew text
                
                4. TRANSLATION_QUALITY_NOTES:
                   - Key terminology choices made
                   - Cultural adaptations implemented
                   - Any technical terms kept in English with explanations
                   - Quality assurance checklist completion
                
                All versions ready for immediate Telegram distribution
            """),
            agent=self.agents.translation_agent(),
            tools=[self.tools['translator']],
            context=[self.format_content_with_visuals_task()],
            output_file="logs/translated_content.json"
        )
    
    def distribute_to_telegram_task(self):
        return Task(
            description=dedent("""\
                Distribute the complete financial market report (text + images) to Telegram channels
                in all target languages with proper sequencing and error handling.
                
                DISTRIBUTION STRATEGY:
                
                1. CONTENT DELIVERY ORDER:
                   - Send main summary text first for each language
                   - Follow with supporting visual content
                   - Include proper captions and source attribution for images
                   - Maintain logical flow between text and visuals
                
                2. MULTI-LANGUAGE COORDINATION:
                   - Distribute to English channel first
                   - Follow with Arabic, Hindi, and Hebrew versions
                   - Use same visual content across all languages
                   - Translate image captions appropriately for each language
                
                3. QUALITY ASSURANCE:
                   - Verify message delivery success for each language
                   - Confirm image uploads completed successfully
                   - Check message formatting appears correctly
                   - Validate source attributions are included
                
                4. ERROR HANDLING:
                   - Retry failed message deliveries
                   - Log any images that fail to upload
                   - Provide fallback for content that encounters issues
                   - Generate detailed delivery status report
                
                TECHNICAL REQUIREMENTS:
                - Respect Telegram rate limits between messages
                - Optimize image sizes for platform requirements
                - Handle message length limits appropriately
                - Ensure proper markdown formatting renders correctly
                
                REPORTING REQUIREMENTS:
                - Track successful deliveries per language
                - Record any failed content delivery attempts
                - Monitor image upload success rates
                - Generate comprehensive distribution report
            """),
            expected_output=dedent("""\
                Complete distribution report containing:
                
                1. DELIVERY_SUCCESS_SUMMARY:
                   - Total messages sent successfully by language
                   - Image upload success rates
                   - Overall distribution success percentage
                   - Time stamps for all deliveries
                
                2. DETAILED_DELIVERY_LOG:
                   English Channel:
                   - ✅/❌ Main summary delivered
                   - ✅/❌ Image 1 delivered with caption
                   - ✅/❌ Image 2 delivered with caption
                   
                   Arabic Channel:
                   - ✅/❌ Translated summary delivered
                   - ✅/❌ Images with Arabic captions delivered
                   
                   [Similar for Hindi and Hebrew channels]
                
                3. CONTENT_QUALITY_VERIFICATION:
                   - Message formatting verified in all languages
                   - Image captions properly translated and attributed
                   - Source attributions included for all visual content
                   - Professional presentation maintained across channels
                
                4. ERROR_RESOLUTION_LOG:
                   - Any delivery failures and resolution attempts
                   - Image upload issues and workarounds applied
                   - Message formatting problems and corrections
                   - Final status of any unresolved issues
                
                5. PERFORMANCE_METRICS:
                   - Total execution time for distribution
                   - Success rate by content type (text vs images)
                   - Success rate by language
                   - Recommendations for future improvements
            """),
            agent=self.agents.distribution_agent(),
            tools=[],  # Tools will be handled by crew execution
            context=[self.translate_content_task(), self.format_content_with_visuals_task()],
            output_file="logs/distribution_report.json"
        )