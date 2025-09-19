import json
import logging
import os
import re
import time
from dotenv import load_dotenv
from typing import Any, Dict, List, Type
from urllib.parse import urljoin, urlparse
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

class ImageFinderInput(BaseModel):
    """Input schema for the free financial image finder tool."""
    search_content: str = Field(..., description="The news content to find relevant financial images for.")
    max_images: int = Field(default=3, description="The maximum number of images to find.")

class ImageFinder(BaseTool):
    """
    Finds free, accessible financial images based on news content.
    Uses only free financial sources to avoid paywall issues.
    """

    name: str = "financial_image_finder"
    description: str = (
        "Find free, accessible financial charts and images from reliable sources like Yahoo Finance, "
        "Finviz, and other free financial platforms based on news content."
    )
    args_schema: Type[BaseModel] = ImageFinderInput

    def _run(self, search_content: str, max_images: int = 3) -> str:
        """Find free financial images based on news content."""
        try:
            logger.info(f"Searching for free financial images - max: {max_images}")

            content_analysis = self._analyze_content(search_content)
            logger.info(f"Content analysis: {content_analysis}")

            if not content_analysis["has_financial_content"]:
                return json.dumps([])

            # Search for free images using multiple strategies
            free_images = self._search_free_images(content_analysis, max_images)

            if not free_images:
                logger.warning("No free images found")
                return json.dumps([])

            # Verify all images are accessible
            verified_images = self._verify_free_images(free_images)

            if verified_images:
                json_output = []
                for img in verified_images:
                    json_output.append({
                        "url": img["url"],
                        "title": img.get("title", "Financial Chart"),
                        "source": img.get("source", "free_source"),
                        "relevance_score": img.get("relevance_score", 0)
                    })
                return json.dumps(json_output)
            else:
                return json.dumps([])

        except Exception as e:
            error_msg = f"Free image search error: {e}"
            logger.error(error_msg, exc_info=True)
            return json.dumps([])

    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content to extract financial information for image search."""
        analysis = {
            "stocks": [],
            "companies": [],
            "key_events": [],
            "sectors": [],
            "has_financial_content": False,
            "key_movers": []
        }

        content_lower = content.lower()

        # Extract major stock symbols
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, content)
        
        major_stocks = {
            "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "GOOG": "Alphabet",
            "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "Nvidia", "META": "Meta",
            "NFLX": "Netflix", "AMD": "AMD", "INTC": "Intel", "CRM": "Salesforce",
            "UBER": "Uber", "SPOT": "Spotify", "JPM": "JPMorgan", "BAC": "Bank of America",
            "V": "Visa", "MA": "Mastercard", "WMT": "Walmart", "HD": "Home Depot",
            "DIS": "Disney", "KO": "Coca-Cola", "PFE": "Pfizer", "JNJ": "Johnson & Johnson"
        }

        # Validate stocks and identify key movers
        for stock in potential_stocks:
            if stock in major_stocks:
                stock_contexts = [
                    f"{stock} stock", f"{stock} shares", f"{stock} trading", 
                    f"{stock} price", f"{stock} earnings"
                ]
                
                if any(context.lower() in content_lower for context in stock_contexts):
                    analysis["stocks"].append(stock)
                    analysis["companies"].append(major_stocks[stock])
                    analysis["has_financial_content"] = True
                    
                    # Check for key movers with performance data
                    key_mover_patterns = [
                        f"{stock}.*?(?:surge|jump|gain|rise|rally|drop|fall|decline).*?\\d+(?:\\.\\d+)?%",
                        f"(?:surge|jump|gain|rise|rally|drop|fall|decline).*?{stock}.*?\\d+(?:\\.\\d+)?%"
                    ]
                    
                    for pattern in key_mover_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            analysis["key_movers"].append({
                                "symbol": stock,
                                "company": major_stocks[stock]
                            })
                            break

        # Extract key financial events
        event_patterns = {
            "earnings": [r"earnings", r"quarterly results", r"revenue"],
            "market_movement": [r"surge", r"rally", r"drop", r"decline"],
            "fed_policy": [r"federal reserve", r"fed", r"interest rate"]
        }

        for event_type, patterns in event_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    analysis["key_events"].append(event_type)
                    analysis["has_financial_content"] = True
                    break

        # Remove duplicates
        for key in analysis:
            if isinstance(analysis[key], list) and key != "key_movers":
                analysis[key] = list(set(analysis[key]))

        return analysis

    def _search_free_images(self, analysis: Dict[str, Any], max_images: int) -> List[Dict[str, Any]]:
        """Search for images from free financial sources."""
        images = []

        # Strategy 1: Direct Yahoo Finance charts for key stocks
        images.extend(self._get_yahoo_charts(analysis["stocks"][:2]))
        
        # Strategy 2: Finviz charts for technical analysis
        images.extend(self._get_finviz_charts(analysis["stocks"][:2]))
        
        # Strategy 3: General financial charts via Serper (free sources only)
        if len(images) < max_images:
            images.extend(self._search_serper_free_images(analysis, max_images - len(images)))

        return images[:max_images]

    def _get_yahoo_charts(self, stocks: List[str]) -> List[Dict[str, Any]]:
        """Get charts directly from Yahoo Finance (always free)."""
        charts = []
        
        for stock in stocks:
            chart_urls = [
                f"https://chart.yahoo.com/z?s={stock}&t=1d&q=l&l=on&z=s&p=s",
                f"https://chart.yahoo.com/z?s={stock}&t=5d&q=l&l=on&z=m&p=s"
            ]
            
            for i, url in enumerate(chart_urls):
                charts.append({
                    "url": url,
                    "title": f"{stock} {'Daily' if i == 0 else '5-Day'} Chart",
                    "source": "Yahoo Finance",
                    "relevance_score": 15 - (i * 2)  # Prefer daily charts
                })
                
        return charts

    def _get_finviz_charts(self, stocks: List[str]) -> List[Dict[str, Any]]:
        """Get technical charts from Finviz (free tier available)."""
        charts = []
        
        for stock in stocks:
            chart_url = f"https://finviz.com/chart.ashx?t={stock}&ty=c&ta=1&p=d&s=l"
            charts.append({
                "url": chart_url,
                "title": f"{stock} Technical Chart",
                "source": "Finviz",
                "relevance_score": 12
            })
                
        return charts

    def _search_serper_free_images(self, analysis: Dict[str, Any], max_images: int) -> List[Dict[str, Any]]:
        """Search for images using Serper, filtering for free sources only."""
        serper_key = os.getenv("SERPER_API_KEY")
        if not serper_key:
            return []

        images = []
        search_queries = self._build_free_queries(analysis)
        
        for query in search_queries[:2]:  # Limit to 2 queries
            try:
                url = "https://google.serper.dev/images"
                payload = {
                    "q": f"{query} site:yahoo.com OR site:finviz.com OR site:marketwatch.com OR site:investing.com",
                    "num": 8,
                    "gl": "us",
                    "safe": "active"
                }
                headers = {
                    "X-API-KEY": serper_key,
                    "Content-Type": "application/json"
                }

                response = requests.post(url, json=payload, headers=headers, timeout=20)
                response.raise_for_status()
                data = response.json()

                for img_data in data.get("images", []):
                    if len(images) >= max_images:
                        break

                    img_url = img_data.get("imageUrl") or img_data.get("link")
                    title = img_data.get("title", "")
                    source = img_data.get("source", "")

                    if self._is_free_financial_image(img_url, title, source):
                        images.append({
                            "url": img_url,
                            "title": title,
                            "source": self._extract_free_source(source),
                            "relevance_score": self._calculate_relevance(title, source, analysis)
                        })

                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Serper query '{query}' failed: {e}")
                continue

        return images

    def _build_free_queries(self, analysis: Dict[str, Any]) -> List[str]:
        """Build search queries targeting free financial sources."""
        queries = []

        # Stock-specific queries
        for stock in analysis["stocks"][:2]:
            queries.append(f"{stock} stock chart")
            
        # Event-specific queries
        if "earnings" in analysis["key_events"]:
            queries.append("earnings report chart")
        if "market_movement" in analysis["key_events"]:
            queries.append("stock market chart today")

        # General fallback
        queries.append("financial market chart")

        return queries

    def _is_free_financial_image(self, url: str, title: str, source: str) -> bool:
        """Check if image is from free financial sources."""
        if not url or not url.startswith("http"):
            return False

        text_to_check = f"{title} {source} {url}".lower()

        # Must be from free financial sources
        free_sources = ['yahoo', 'finviz', 'marketwatch', 'investing', 'cnbc']
        is_from_free_source = any(source_name in text_to_check for source_name in free_sources)

        # Must have financial content
        financial_indicators = ['chart', 'stock', 'market', 'trading', 'financial']
        has_financial_content = any(indicator in text_to_check for indicator in financial_indicators)

        # Exclude non-chart content
        exclude_indicators = ['logo', 'icon', 'profile', 'avatar', 'banner']
        is_excluded = any(indicator in text_to_check for indicator in exclude_indicators)

        return is_from_free_source and has_financial_content and not is_excluded

    def _extract_free_source(self, source: str) -> str:
        """Extract clean source name from free financial sites."""
        source_lower = source.lower()
        
        if 'yahoo' in source_lower:
            return "Yahoo Finance"
        elif 'finviz' in source_lower:
            return "Finviz"
        elif 'marketwatch' in source_lower:
            return "MarketWatch"
        elif 'investing' in source_lower:
            return "Investing.com"
        elif 'cnbc' in source_lower:
            return "CNBC"
        else:
            return "Free Financial Source"

    def _calculate_relevance(self, title: str, source: str, analysis: Dict[str, Any]) -> int:
        """Calculate relevance score for free images."""
        score = 0
        text = f"{title} {source}".lower()

        # Higher points for key movers
        for key_mover in analysis.get("key_movers", []):
            symbol = key_mover.get("symbol", "")
            if symbol and symbol.lower() in text:
                score += 15

        # Points for other stock mentions
        for stock in analysis["stocks"]:
            if stock.lower() in text:
                score += 8

        # Points for key events
        for event in analysis["key_events"]:
            if event in text:
                score += 5

        # Points for free source quality
        free_source_scores = {
            'yahoo': 10, 'finviz': 8, 'marketwatch': 6, 
            'investing': 6, 'cnbc': 7
        }
        
        for source_name, points in free_source_scores.items():
            if source_name in text:
                score += points

        return score

    def _verify_free_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify that free images are accessible."""
        verified = []
        
        for img in images:
            url = img.get("url")
            if not url:
                continue
                
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.head(url, timeout=5, headers=headers, allow_redirects=True)
                
                if response.status_code == 200:
                    # Extra verification for Yahoo and Finviz charts
                    if "yahoo.com" in url or "finviz.com" in url:
                        verified.append(img)
                        logger.info(f"Verified free image: {img.get('title', 'Unknown')} from {img.get('source', 'Unknown')}")
                    else:
                        # Additional content-type check for other sources
                        content_type = response.headers.get("content-type", "").lower()
                        if "image" in content_type:
                            verified.append(img)
                    
            except Exception as e:
                logger.debug(f"Free image verification failed for {url}: {e}")
                continue
                
        return verified