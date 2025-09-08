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
    """Input schema for the contextual image finder tool."""

    search_content: str = Field(
        ..., description="The news content or search results to find relevant financial images for."
    )
    max_images: int = Field(
        default=3, description="The maximum number of images to find."
    )

class ImageFinder(BaseTool):
    """
    Finds real, contextually relevant financial images based on actual news content.
    Uses Serper API to search for images that match the specific stories and stocks mentioned.
    """

    name: str = "financial_image_finder"
    description: str = (
        "Find contextually relevant financial charts and images based on actual news content. "
        "Uses real search results to find images that match the specific stories, stocks, and events mentioned."
    )
    args_schema: Type[BaseModel] = ImageFinderInput

    def _run(self, search_content: str, max_images: int = 3) -> str:
        """
        Find contextually relevant images based on the actual search content.
        """
        try:
            logger.info(f"Starting contextual image search for {max_images} images")

            # Analyze the content to extract key information
            content_analysis = self._analyze_content_deeply(search_content)
            logger.info(f"Content analysis: {content_analysis}")

            if not content_analysis["has_financial_content"]:
                return "No financial content found to generate relevant images for."

            # Search for contextually relevant images using Serper
            relevant_images = self._search_contextual_images(content_analysis, max_images)

            if not relevant_images:
                logger.warning("No contextual images found, trying backup search")
                relevant_images = self._backup_search(content_analysis, max_images)

            # Verify images are accessible
            verified_images = self._verify_images(relevant_images)

            return self._format_results(verified_images, content_analysis)

        except Exception as e:
            error_msg = f"Contextual image search error: {e}"
            logger.error(error_msg)
            return error_msg

    def _analyze_content_deeply(self, content: str) -> Dict[str, Any]:
        """
        Deep analysis of content to extract all relevant financial information.
        """
        analysis = {
            "stocks": [],
            "companies": [],
            "key_events": [],
            "sectors": [],
            "financial_terms": [],
            "news_topics": [],
            "has_financial_content": False
        }

        content_lower = content.lower()

        # Extract stock symbols with better context validation
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, content)
        
        # Major stocks with high confidence
        major_stocks = {
            "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "GOOG": "Alphabet",
            "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "Nvidia", "META": "Meta",
            "NFLX": "Netflix", "AMD": "AMD", "INTC": "Intel", "CRM": "Salesforce",
            "ADBE": "Adobe", "PYPL": "PayPal", "UBER": "Uber", "SPOT": "Spotify",
            "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo",
            "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup",
            "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "UNH": "UnitedHealth",
            "XOM": "ExxonMobil", "CVX": "Chevron", "WMT": "Walmart", "HD": "Home Depot",
            "DIS": "Disney", "V": "Visa", "MA": "Mastercard", "KO": "Coca-Cola"
        }

        # Validate stocks by context
        for stock in potential_stocks:
            if stock in major_stocks:
                # Check if it appears in financial context
                stock_contexts = [
                    f"{stock} stock", f"{stock} shares", f"{stock} trading", 
                    f"{stock} price", f"{stock} earnings", f"{stock} revenue",
                    f"${stock}", f"({stock})", f"{stock})"
                ]
                
                if any(context.lower() in content_lower for context in stock_contexts):
                    analysis["stocks"].append(stock)
                    analysis["companies"].append(major_stocks[stock])
                    analysis["has_financial_content"] = True

        # Extract company names directly mentioned
        company_patterns = [
            r'\b(Apple|Microsoft|Google|Amazon|Tesla|Meta|Netflix|Nvidia)\b',
            r'\b(JPMorgan|Goldman Sachs|Bank of America|Wells Fargo)\b',
            r'\b(Salesforce|Adobe|PayPal|Intel|AMD)\b'
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match.title() not in analysis["companies"]:
                    analysis["companies"].append(match.title())
                    analysis["has_financial_content"] = True

        # Extract key financial events and topics
        event_patterns = {
            "earnings": [r"earnings", r"quarterly results", r"revenue", r"profit"],
            "market_movement": [r"surge", r"rally", r"drop", r"decline", r"gain", r"loss"],
            "corporate_actions": [r"merger", r"acquisition", r"deal", r"buyout"],
            "fed_policy": [r"federal reserve", r"fed", r"interest rate", r"monetary policy"],
            "economic_data": [r"inflation", r"gdp", r"unemployment", r"jobs report"],
            "analyst_actions": [r"upgrade", r"downgrade", r"target price", r"rating"]
        }

        for event_type, patterns in event_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    analysis["key_events"].append(event_type)
                    analysis["has_financial_content"] = True
                    break

        # Extract news headlines/topics for search queries
        # Look for news article titles or key phrases
        headline_patterns = [
            r'\*\*(.*?)\*\*',  # Bold text (likely headlines)
            r'^\d+\.\s+(.+)$',  # Numbered items
            r'Source: (.+?) \|',  # Source lines
        ]

        for pattern in headline_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if len(match) > 10 and len(match) < 100:  # Reasonable headline length
                    analysis["news_topics"].append(match.strip())

        # Extract financial terms for better search context
        financial_terms = [
            "stock market", "trading", "investment", "portfolio", "dividend",
            "market cap", "valuation", "IPO", "SEC", "NYSE", "NASDAQ"
        ]

        for term in financial_terms:
            if term in content_lower:
                analysis["financial_terms"].append(term)

        # Remove duplicates
        for key in analysis:
            if isinstance(analysis[key], list):
                analysis[key] = list(set(analysis[key]))

        logger.info(f"Deep analysis found: {len(analysis['stocks'])} stocks, {len(analysis['companies'])} companies, {len(analysis['key_events'])} events")
        return analysis

    def _search_contextual_images(self, analysis: Dict[str, Any], max_images: int) -> List[Dict[str, Any]]:
        """
        Search for images using Serper API based on the actual content analysis.
        """
        serper_key = os.getenv("SERPER_API_KEY")
        if not serper_key:
            logger.error("SERPER_API_KEY not found")
            return []

        images = []
        
        # Create multiple targeted search queries based on content
        search_queries = self._build_contextual_queries(analysis)
        
        for query in search_queries[:3]:  # Try up to 3 different queries
            if len(images) >= max_images:
                break
                
            logger.info(f"Searching images for: {query}")
            
            try:
                url = "https://google.serper.dev/images"
                payload = {
                    "q": query,
                    "num": 10,  # Get more to filter better
                    "gl": "us",
                    "safe": "active",
                    "tbs": "qdr:w",  # Recent images from past week
                }
                headers = {
                    "X-API-KEY": serper_key,
                    "Content-Type": "application/json"
                }

                response = requests.post(url, json=payload, headers=headers, timeout=20)
                response.raise_for_status()
                data = response.json()

                # Process results
                for img_data in data.get("images", []):
                    if len(images) >= max_images:
                        break

                    img_url = img_data.get("imageUrl") or img_data.get("link")
                    title = img_data.get("title", "")
                    source = img_data.get("source", "")

                    if self._is_relevant_financial_image(img_url, title, source, analysis):
                        images.append({
                            "url": img_url,
                            "title": title,
                            "source": source,
                            "query": query,
                            "relevance_score": self._calculate_relevance(title, source, analysis)
                        })

                # Brief delay between searches
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Search query '{query}' failed: {e}")
                continue

        # Sort by relevance score
        images.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return images[:max_images]

    def _build_contextual_queries(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Build targeted search queries based on content analysis.
        """
        queries = []

        # Stock-specific queries
        for stock in analysis["stocks"][:3]:  # Top 3 stocks
            company = next((comp for comp in analysis["companies"] 
                           if any(stock_name in comp for stock_name in [stock])), stock)
            
            if "earnings" in analysis["key_events"]:
                queries.append(f"{stock} {company} earnings chart stock price today")
            elif "market_movement" in analysis["key_events"]:
                queries.append(f"{stock} {company} stock chart performance today")
            else:
                queries.append(f"{stock} stock chart financial graph")

        # Company-specific queries
        for company in analysis["companies"][:2]:
            if "earnings" in analysis["key_events"]:
                queries.append(f"{company} earnings report chart financial results")
            else:
                queries.append(f"{company} stock price chart market performance")

        # Event-specific queries
        if "fed_policy" in analysis["key_events"]:
            queries.append("federal reserve interest rates market impact chart")
        
        if "economic_data" in analysis["key_events"]:
            queries.append("economic data market chart inflation jobs report")

        # General market queries if specific content found
        if analysis["stocks"] or analysis["companies"]:
            queries.append("stock market chart today financial news trading")
            queries.append("market analysis chart wall street performance")

        # Ensure we have at least some queries
        if not queries:
            queries = [
                "financial market chart today",
                "stock market news chart",
                "trading chart financial analysis"
            ]

        return queries

    def _is_relevant_financial_image(self, url: str, title: str, source: str, analysis: Dict[str, Any]) -> bool:
        """
        Check if image is relevant to our content analysis.
        """
        if not url or not url.startswith("http"):
            return False

        # Combine text for analysis
        text_to_check = f"{title} {source} {url}".lower()

        # Check for financial relevance
        financial_indicators = [
            "chart", "graph", "stock", "market", "trading", "financial",
            "price", "performance", "analysis", "earnings", "revenue"
        ]

        has_financial_content = any(indicator in text_to_check for indicator in financial_indicators)

        # Check for content-specific relevance
        content_relevance = False
        
        # Check against our specific stocks/companies
        for stock in analysis["stocks"]:
            if stock.lower() in text_to_check:
                content_relevance = True
                break
                
        for company in analysis["companies"]:
            if company.lower() in text_to_check:
                content_relevance = True
                break

        # Check for quality sources
        quality_sources = [
            "yahoo", "bloomberg", "reuters", "cnbc", "marketwatch",
            "investing.com", "tradingview", "finviz", "wsj"
        ]
        
        from_quality_source = any(source_name in text_to_check for source_name in quality_sources)

        # Exclude low-quality content
        exclude_indicators = [
            "meme", "joke", "cartoon", "logo", "icon", "template",
            "wallpaper", "avatar", "profile", "social"
        ]
        
        is_low_quality = any(indicator in text_to_check for indicator in exclude_indicators)

        return (has_financial_content and not is_low_quality and 
                (content_relevance or from_quality_source))

    def _calculate_relevance(self, title: str, source: str, analysis: Dict[str, Any]) -> int:
        """
        Calculate relevance score for ranking images.
        """
        score = 0
        text = f"{title} {source}".lower()

        # Points for stock mentions
        for stock in analysis["stocks"]:
            if stock.lower() in text:
                score += 10

        # Points for company mentions
        for company in analysis["companies"]:
            if company.lower() in text:
                score += 8

        # Points for key events
        for event in analysis["key_events"]:
            if event in text:
                score += 5

        # Points for quality sources
        quality_sources = ["yahoo", "bloomberg", "reuters", "cnbc", "marketwatch"]
        for source_name in quality_sources:
            if source_name in text:
                score += 3

        # Points for chart/graph indicators
        chart_indicators = ["chart", "graph", "performance", "analysis"]
        for indicator in chart_indicators:
            if indicator in text:
                score += 2

        return score

    def _backup_search(self, analysis: Dict[str, Any], max_images: int) -> List[Dict[str, Any]]:
        """
        Backup search if primary search fails.
        """
        backup_images = []
        
        # Simple fallback queries
        backup_queries = [
            "stock market chart financial news today",
            "trading chart market analysis",
            "financial graph stock performance"
        ]

        serper_key = os.getenv("SERPER_API_KEY")
        if not serper_key:
            return []

        for query in backup_queries[:1]:  # Try just one backup
            try:
                url = "https://google.serper.dev/images"
                payload = {
                    "q": query,
                    "num": 5,
                    "gl": "us",
                    "safe": "active"
                }
                headers = {
                    "X-API-KEY": serper_key,
                    "Content-Type": "application/json"
                }

                response = requests.post(url, json=payload, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()

                for img_data in data.get("images", [])[:max_images]:
                    img_url = img_data.get("imageUrl") or img_data.get("link")
                    title = img_data.get("title", "Financial Chart")
                    
                    if img_url and "chart" in title.lower():
                        backup_images.append({
                            "url": img_url,
                            "title": title,
                            "source": "backup_search",
                            "query": query,
                            "relevance_score": 1
                        })

            except Exception as e:
                logger.warning(f"Backup search failed: {e}")

        return backup_images

    def _verify_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify that images are accessible.
        """
        verified = []
        
        for img in images:
            url = img.get("url")
            if not url:
                continue
                
            try:
                # Quick verification
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.head(url, timeout=5, headers=headers, allow_redirects=True)
                
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()
                    if "image" in content_type or any(ext in url.lower() for ext in [".jpg", ".png", ".gif", ".webp"]):
                        verified.append(img)
                        logger.info(f"Verified image: {img.get('title', 'Unknown')}")
                    
            except Exception as e:
                logger.debug(f"Image verification failed for {url}: {e}")
                continue
                
        return verified

    def _format_results(self, images: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """
        Format the results for output.
        """
        if not images:
            return "No contextually relevant financial images found for the provided content."

        result_parts = ["=== CONTEXTUAL FINANCIAL IMAGES FOUND ===\n"]
        
        # Add context information
        if analysis["stocks"]:
            result_parts.append(f"📈 Related to stocks: {', '.join(analysis['stocks'][:5])}")
        if analysis["companies"]:
            result_parts.append(f"🏢 Companies mentioned: {', '.join(analysis['companies'][:5])}")
        if analysis["key_events"]:
            result_parts.append(f"📊 Key events: {', '.join(analysis['key_events'][:3])}")
        
        result_parts.append("")

        for i, img in enumerate(images, 1):
            image_info = f"""
Image {i}:
- URL: {img['url']}
- Type: contextual_search
- Description: {img.get('title', 'Financial Chart')}
- Source: {img.get('source', 'web_search')}
- Search Query: {img.get('query', 'N/A')}
- Relevance Score: {img.get('relevance_score', 0)}
---
"""
            result_parts.append(image_info)

        result_parts.append(f"Total contextual images: {len(images)}")
        result_parts.append(f"Content analysis confidence: {'High' if analysis['has_financial_content'] else 'Low'}")
        
        return "\n".join(result_parts)