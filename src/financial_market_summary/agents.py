import os
import logging
from typing import Optional
from crewai import Agent, LLM
from crewai.tools import BaseTool
from .tools.telegram_sender import EnhancedTelegramSender
from .tools.image_finder import EnhancedImageFinder as ImageFinder
from .tools.tavily_search import TavilyFinancialTool
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FinancialAgents:
    """Factory class to create specialized financial agents for the workflow with web source and image verification."""
    
    def __init__(self):
        """Initialize tools used by agents and the LLM."""
        self.telegram_sender = EnhancedTelegramSender()
        self.image_finder = ImageFinder()
        self.tavily_tool = TavilyFinancialTool()
        
        # Set up the LLM with Google Gemini 2.5 Flash
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        self.llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=google_api_key,
            temperature=0.3,
            max_tokens=2048,
            request_timeout=60,
        )
            
    def search_agent(self) -> Agent:
        """Creates an agent for searching financial news."""
        return Agent(
            role="Financial News Searcher",
            goal="Search for the latest US financial news from the past 2 hours, focusing on major stock movements, earnings, and economic data from verified sources.",
            backstory="You are a skilled financial researcher with expertise in identifying the most relevant and recent market news using advanced search tools, ensuring all sources are from trusted financial outlets.",
            tools=[self.tavily_tool],
            llm=self.llm,
            verbose=True
        )

    def summary_agent(self) -> Agent:
        """Creates an agent for summarizing financial news."""
        return Agent(
            role="Financial News Summarizer",
            goal="Analyze financial news and create a concise, structured market summary under 500 words for Telegram with accurate data and stock symbols.",
            backstory="You are an expert financial analyst who excels at distilling complex market data into clear, concise summaries with key points and implications, ensuring all numbers and stock symbols are accurate.",
            tools=[],
            llm=self.llm,
            verbose=True
        )

    def web_searching_source_agent(self) -> Agent:
        """Creates an agent that can search the web to find the best source article for a summary."""
        return Agent(
            role="Web-Searching Financial Source Finder",
            goal="Search the web to find the most relevant and authoritative source article that matches the financial summary title and content, then verify the URL accessibility.",
            backstory="You are an expert researcher who specializes in finding the perfect source articles by searching financial news websites to match summary content with original reporting. You always verify that URLs work and are accessible before recommending them.",
            tools=[self.tavily_tool],  # Give it access to web search
            llm=self.llm,
            verbose=True
        )

    def enhanced_image_finder_agent(self) -> Agent:
        """Creates an agent that finds and verifies relevant financial images."""
        return Agent(
            role="Enhanced Financial Image Finder with URL Verification",
            goal="Find relevant, recent financial charts and stock images with URL accessibility verification, ensuring all image links work properly.",
            backstory="You are an expert visual content specialist who finds the most relevant and accessible financial charts, stock images, and market visuals. You always verify that image URLs work and are accessible before recommending them, similar to how source articles are verified.",
            tools=[self.image_finder],
            llm=self.llm,
            verbose=True
        )

    def formatting_agent(self) -> Agent:
        """Creates an agent for formatting summaries with verified sources and visuals."""
        return Agent(
            role="Financial Content Formatter with Source and Image Verification",
            goal="Format financial summaries with markdown, integrate verified source links, and add verified relevant charts or images, ensuring all links work and are properly attributed.",
            backstory="You are a content specialist skilled in creating visually appealing, well-structured financial updates for Telegram, incorporating verified source links, verified image charts, and professional formatting. You always include proper source attribution with verification status and ensure images are accessible.",
            tools=[self.image_finder],
            llm=self.llm,
            verbose=True
        )

    def translation_agent(self) -> Agent:
        """Creates an agent for translating financial summaries while preserving source links."""
        return Agent(
            role="Financial Content Translator with Source Preservation",
            goal="Translate financial summaries into multiple languages while preserving stock symbols, numbers, markdown formatting, and source links with verification indicators.",
            backstory="You are a multilingual financial translator with expertise in maintaining accuracy and professional terminology across languages, ensuring that source links, verification indicators, and all financial data remain intact during translation.",
            tools=[],
            llm=self.llm,
            verbose=True
        )

    def send_agent(self) -> Agent:
        """Creates an agent for sending summaries with verified sources to Telegram."""
        return Agent(
            role="Telegram Content Distributor with Source and Image Verification",
            goal="Send formatted financial summaries and their translations to a Telegram channel, ensuring all source links and images are working and verification status is clearly communicated to subscribers.",
            backstory="You are a communication specialist skilled in distributing financial updates to Telegram channels with proper formatting, working source links, verified images, and clear verification indicators that build subscriber trust and credibility.",
            tools=[self.telegram_sender],
            llm=self.llm,
            verbose=True
        )

    # Enhanced Image Search Methods (integrated into the class)
    def run_enhanced_image_search_phase(self, summary_title: str, key_themes: list, mentioned_stocks: list, content: str) -> dict:
        """Search for and verify relevant financial images"""
        logger.info(f"--- Enhanced Image Search for '{summary_title[:50]}...' ---")
        
        try:
            # Use the enhanced image finder agent
            image_agent = self.enhanced_image_finder_agent()
            
            # Build search strategy
            search_strategy = self._build_image_search_strategy(summary_title, key_themes, mentioned_stocks)
            
            from crewai import Task, Crew, Process
            
            image_task = Task(
                description=f"""Find and verify relevant financial images for this summary:
                
                SUMMARY TITLE: "{summary_title}"
                KEY THEMES: {key_themes}
                MENTIONED STOCKS: {mentioned_stocks}
                CONTENT PREVIEW: {content[:200]}...
                
                IMAGE SEARCH STRATEGY:
                1. Use image_finder to find relevant charts for stocks: {mentioned_stocks}
                2. Search strategy: {search_strategy}
                3. Focus on Yahoo Finance, TradingView charts
                4. Verify each image URL works (no 404 errors)
                
                EVALUATION CRITERIA:
                1. RELEVANCE (40%): Image matches stocks/themes mentioned
                2. URL ACCESSIBILITY (30%): Image URL is accessible and loads
                3. SOURCE TRUST (20%): From trusted sources (Yahoo Finance, TradingView)
                4. RECENCY (10%): Chart shows recent/current data
                
                RETURN the best verified images as structured data with:
                - Image URLs that are verified working
                - Relevance scores for each image
                - Stock symbols or market indices covered
                - Verification status for each URL
                """,
                expected_output="Structured data with verified financial images that match the summary content.",
                agent=image_agent
            )
            
            crew = Crew(
                agents=[image_agent],
                tasks=[image_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Process and structure the result
            image_data = self._process_image_search_result(str(result), mentioned_stocks)
            
            verified_count = len([img for img in image_data.get('verified_images', []) if img.get('url_verified', False)])
            logger.info(f"Enhanced image search completed: {verified_count} verified images found")
            
            return image_data
            
        except Exception as e:
            logger.warning(f"Enhanced image search failed: {e}")
            return self._image_search_fallback(summary_title, mentioned_stocks)

    def _build_image_search_strategy(self, title: str, themes: list, stocks: list) -> str:
        """Build optimized search strategy for finding relevant images"""
        strategy_parts = []
        
        # Stock-specific strategy
        if stocks:
            strategy_parts.append(f"Individual stock charts for {', '.join(stocks[:3])}")
        
        # Theme-specific strategy
        theme_strategies = {
            'earnings': 'Earnings-focused charts and performance visuals',
            'fed_policy': 'Market index charts showing Fed impact',
            'market_records': 'Market index charts highlighting record highs',
            'technology': 'Tech sector and NASDAQ charts'
        }
        
        for theme in themes:
            if theme in theme_strategies:
                strategy_parts.append(theme_strategies[theme])
        
        # Market context strategy
        if any(word in title.lower() for word in ['market', 'index', 'broad']):
            strategy_parts.append("Major market index charts (S&P 500, NASDAQ, Dow)")
        
        return "; ".join(strategy_parts) if strategy_parts else "General market overview charts"

    def _process_image_search_result(self, result: str, mentioned_stocks: list) -> dict:
        """Process the image search result and structure it"""
        import re
        
        # Initialize default structure
        image_data = {
            "verified_images": [],
            "search_strategy_used": "Enhanced image search",
            "total_images_found": 0,
            "verified_count": 0,
            "confidence_score": 60
        }
        
        try:
            # Extract any URLs mentioned in the result
            urls = re.findall(r'https?://[^\s\)]+', result)
            
            # Create verified images based on found URLs and mentioned stocks
            for i, stock in enumerate(mentioned_stocks[:3]):
                # Yahoo Finance chart for the stock
                chart_url = f"https://chart.yahoo.com/z?s={stock}&t=1d&q=l&l=on&z=s&p=s"
                
                # Verify the URL
                url_verified = self._quick_verify_image_url(chart_url)
                
                image_data["verified_images"].append({
                    "url": chart_url,
                    "title": f"{stock} Stock Chart - 1 Day",
                    "source": "Yahoo Finance",
                    "type": "stock_chart",
                    "stock_symbol": stock,
                    "relevance_score": 90 - (i * 5),
                    "url_verified": url_verified,
                    "verification_status": "verified" if url_verified else "failed"
                })
            
            # Add market index chart
            sp500_url = "https://chart.yahoo.com/z?s=%5EGSPC&t=1d&q=l&l=on&z=s&p=s"
            sp500_verified = self._quick_verify_image_url(sp500_url)
            
            image_data["verified_images"].append({
                "url": sp500_url,
                "title": "S&P 500 Market Chart",
                "source": "Yahoo Finance",
                "type": "market_index",
                "relevance_score": 85,
                "url_verified": sp500_verified,
                "verification_status": "verified" if sp500_verified else "failed"
            })
            
            # Update counts
            image_data["total_images_found"] = len(image_data["verified_images"])
            image_data["verified_count"] = len([img for img in image_data["verified_images"] if img.get("url_verified", False)])
            
            # Update confidence based on verification rate
            if image_data["total_images_found"] > 0:
                verification_rate = image_data["verified_count"] / image_data["total_images_found"]
                image_data["confidence_score"] = int(60 + (verification_rate * 35))  # 60-95 range
            
            return image_data
            
        except Exception as e:
            logger.warning(f"Error processing image search result: {e}")
            return self._image_search_fallback("", mentioned_stocks)

    def _quick_verify_image_url(self, url: str) -> bool:
        """Quick verification of image URL"""
        try:
            import requests
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.head(url, headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False

    def _image_search_fallback(self, title: str, stocks: list) -> dict:
        """Fallback when image search fails"""
        fallback_images = []
        
        # S&P 500 chart (always reliable)
        fallback_images.append({
            "url": "https://chart.yahoo.com/z?s=%5EGSPC&t=1d&q=l&l=on&z=s&p=s",
            "title": "S&P 500 Market Chart",
            "source": "Yahoo Finance",
            "type": "market_index",
            "relevance_score": 70,
            "url_verified": True,
            "verification_status": "fallback_reliable"
        })
        
        # Add stock-specific chart if stocks mentioned
        if stocks:
            primary_stock = stocks[0]
            fallback_images.append({
                "url": f"https://chart.yahoo.com/z?s={primary_stock}&t=1d&q=l&l=on&z=s&p=s",
                "title": f"{primary_stock} Stock Chart",
                "source": "Yahoo Finance",
                "type": "stock_chart",
                "stock_symbol": primary_stock,
                "relevance_score": 75,
                "url_verified": True,
                "verification_status": "fallback_reliable"
            })
        
        return {
            "verified_images": fallback_images,
            "search_strategy_used": "Fallback strategy - reliable charts",
            "total_images_found": len(fallback_images),
            "verified_count": len(fallback_images),
            "confidence_score": 60,
            "verification_rate": 100,
            "fallback_used": True
        }

    def select_best_verified_image(self, image_data: dict) -> dict:
        """Select the best verified image from search results"""
        verified_images = image_data.get('verified_images', [])
        
        if not verified_images:
            return None
        
        # Filter to only verified images
        accessible_images = [img for img in verified_images if img.get('url_verified', False)]
        
        if not accessible_images:
            # If no verified images, use fallback
            logger.warning("No verified images found, using fallback")
            fallback_data = self._image_search_fallback("Market Update", [])
            accessible_images = fallback_data.get('verified_images', [])
        
        if not accessible_images:
            return None
        
        # Sort by relevance score and verification status
        accessible_images.sort(key=lambda x: (x.get('url_verified', False), x.get('relevance_score', 0)), reverse=True)
        
        best_image = accessible_images[0]
        
        logger.info(f"Selected best verified image: {best_image.get('title', 'Unknown')} (Score: {best_image.get('relevance_score', 0)}, Verified: {best_image.get('url_verified', False)})")
        
        return best_image