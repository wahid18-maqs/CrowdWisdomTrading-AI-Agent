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
                return json.dumps([])  # Return empty JSON array instead of string

            # Search for contextually relevant images using Serper
            relevant_images = self._search_contextual_images(content_analysis, max_images)

            if not relevant_images:
                logger.warning("No contextual images found, trying backup search")
                relevant_images = self._backup_search(content_analysis, max_images)

            # Verify images are accessible
            verified_images = self._verify_images(relevant_images)

            # Return both JSON format for programmatic use AND formatted string
            if verified_images:
                # Create simplified JSON output for the Telegram sender
                json_output = []
                for img in verified_images:
                    json_output.append({
                        "url": img["url"],
                        "title": img.get("title", "Financial Chart"),
                        "source": img.get("source", "web_search"),
                        "relevance_score": img.get("relevance_score", 0)
                    })
                return json.dumps(json_output)
            else:
                return json.dumps([])  # Empty array if no images found

        except Exception as e:
            error_msg = f"Contextual image search error: {e}"
            logger.error(error_msg, exc_info=True)
            return json.dumps([])  # Return empty JSON on error

    def _analyze_content_deeply(self, content: str) -> Dict[str, Any]:
        """
        Deep analysis of content to extract all relevant financial information.
        Enhanced to focus on key movers and specific stock mentions.
        """
        analysis = {
            "stocks": [],
            "companies": [],
            "key_events": [],
            "sectors": [],
            "financial_terms": [],
            "news_topics": [],
            "has_financial_content": False,
            "key_movers": []  # New field for key movers
        }

        content_lower = content.lower()

        # Extract stock symbols with better context validation
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, content)
        
        # Expanded major stocks dictionary for better coverage
        major_stocks = {
            "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "GOOG": "Alphabet",
            "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "Nvidia", "META": "Meta",
            "NFLX": "Netflix", "AMD": "AMD", "INTC": "Intel", "CRM": "Salesforce",
            "ADBE": "Adobe", "PYPL": "PayPal", "UBER": "Uber", "SPOT": "Spotify",
            "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo",
            "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup",
            "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "UNH": "UnitedHealth",
            "XOM": "ExxonMobil", "CVX": "Chevron", "WMT": "Walmart", "HD": "Home Depot",
            "DIS": "Disney", "V": "Visa", "MA": "Mastercard", "KO": "Coca-Cola",
            "IBM": "IBM", "ORCL": "Oracle", "CSCO": "Cisco", "QCOM": "Qualcomm",
            "BABA": "Alibaba", "SHOP": "Shopify", "SQ": "Block", "ZM": "Zoom"
        }

        # Validate stocks by context and identify key movers
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
                    
                    # Check if it's a key mover (has performance indicators)
                    key_mover_patterns = [
                        f"{stock}.*?(?:surge|jump|gain|rise|rally|drop|fall|decline|lose).*?\\d+(?:\\.\\d+)?%",
                        f"(?:surge|jump|gain|rise|rally|drop|fall|decline|lose).*?{stock}.*?\\d+(?:\\.\\d+)?%"
                    ]
                    
                    for pattern in key_mover_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            analysis["key_movers"].append({
                                "symbol": stock,
                                "company": major_stocks[stock]
                            })
                            break

        # Extract company names directly mentioned with performance
        company_performance_pattern = r'(\w+(?:\s+\w+)*?)\s+(?:surged?|jumped?|gained?|rose|rallied|dropped?|fell|declined?|lost)\s+[\d.]+%'
        company_movers = re.findall(company_performance_pattern, content, re.IGNORECASE)
        
        for company in company_movers:
            company_clean = company.strip().title()
            if len(company_clean.split()) <= 3:  # Reasonable company name length
                analysis["key_movers"].append({
                    "symbol": None,
                    "company": company_clean
                })
                if company_clean not in analysis["companies"]:
                    analysis["companies"].append(company_clean)
                    analysis["has_financial_content"] = True

        # Extract key financial events and topics (unchanged)
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

        # Remove duplicates
        for key in analysis:
            if isinstance(analysis[key], list) and key != "key_movers":
                analysis[key] = list(set(analysis[key]))

        logger.info(f"Deep analysis found: {len(analysis['stocks'])} stocks, {len(analysis['key_movers'])} key movers")
        return analysis

    def _search_contextual_images(self, analysis: Dict[str, Any], max_images: int) -> List[Dict[str, Any]]:
        """
        Search for images using Serper API based on the actual content analysis.
        Prioritizes key movers for more targeted results.
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
        Enhanced to prioritize key movers.
        """
        queries = []

        # Prioritize key movers for more targeted searches
        for key_mover in analysis["key_movers"][:2]:  # Top 2 key movers
            symbol = key_mover.get("symbol")
            company = key_mover.get("company")
            
            if symbol:
                queries.extend([
                    f"{symbol} stock chart performance today key movers",
                    f"{symbol} {company} stock price chart today",
                    f"{symbol} financial chart analysis"
                ])
            elif company:
                queries.extend([
                    f"{company} stock chart performance today",
                    f"{company} stock price analysis chart"
                ])

        # Stock-specific queries for non-key-movers
        remaining_stocks = [s for s in analysis["stocks"] 
                          if not any(km.get("symbol") == s for km in analysis["key_movers"])]
        
        for stock in remaining_stocks[:2]:  # Max 2 additional stocks
            company = next((comp for comp in analysis["companies"] 
                           if any(stock_name in comp for stock_name in [stock])), stock)
            
            if "earnings" in analysis["key_events"]:
                queries.append(f"{stock} {company} earnings chart stock price")
            elif "market_movement" in analysis["key_events"]:
                queries.append(f"{stock} {company} stock chart performance")
            else:
                queries.append(f"{stock} stock chart financial graph")

        # Event-specific queries
        if "fed_policy" in analysis["key_events"]:
            queries.append("federal reserve interest rates market impact chart")
        
        if "economic_data" in analysis["key_events"]:
            queries.append("economic data market chart inflation jobs report")

        # General market queries if specific content found
        if analysis["stocks"] or analysis["companies"]:
            queries.append("stock market chart today financial news trading")

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
        Enhanced filtering for higher quality results.
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

        # Enhanced content-specific relevance check
        content_relevance = False
        
        # Check against key movers first (highest priority)
        for key_mover in analysis.get("key_movers", []):
            symbol = key_mover.get("symbol")
            company = key_mover.get("company")
            
            if symbol and symbol.lower() in text_to_check:
                content_relevance = True
                break
            if company and company.lower() in text_to_check:
                content_relevance = True
                break
        
        # Check against other stocks/companies
        if not content_relevance:
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
            "investing.com", "tradingview", "finviz", "wsj", "barrons"
        ]
        
        from_quality_source = any(source_name in text_to_check for source_name in quality_sources)

        # Exclude low-quality content
        exclude_indicators = [
            "meme", "joke", "cartoon", "logo", "icon", "template",
            "wallpaper", "avatar", "profile", "social", "thumbnail"
        ]
        
        is_low_quality = any(indicator in text_to_check for indicator in exclude_indicators)

        return (has_financial_content and not is_low_quality and 
                (content_relevance or from_quality_source))

    def _calculate_relevance(self, title: str, source: str, analysis: Dict[str, Any]) -> int:
        """
        Calculate relevance score for ranking images.
        Enhanced scoring for key movers.
        """
        score = 0
        text = f"{title} {source}".lower()

        # Higher points for key movers
        for key_mover in analysis.get("key_movers", []):
            symbol = key_mover.get("symbol")
            company = key_mover.get("company")
            
            if symbol and symbol.lower() in text:
                score += 15  # Higher score for key movers
            if company and company.lower() in text:
                score += 12

        # Points for other stock mentions
        for stock in analysis["stocks"]:
            if stock.lower() in text and not any(km.get("symbol") == stock for km in analysis.get("key_movers", [])):
                score += 8

        # Points for other company mentions
        for company in analysis["companies"]:
            if company.lower() in text and not any(km.get("company") == company for km in analysis.get("key_movers", [])):
                score += 6

        # Points for key events
        for event in analysis["key_events"]:
            if event in text:
                score += 5

        # Points for quality sources
        quality_sources = ["yahoo", "bloomberg", "reuters", "cnbc", "marketwatch", "finviz", "tradingview"]
        for source_name in quality_sources:
            if source_name in text:
                score += 4

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

    def get_formatted_results(self, search_content: str, max_images: int = 3) -> str:
        """
        Alternative method that returns formatted string results (for backward compatibility).
        """
        try:
            # Get JSON results first
            json_results = self._run(search_content, max_images)
            images = json.loads(json_results) if json_results else []
            
            if not images:
                return "No contextually relevant financial images found for the provided content."

            # Analyze content for context
            content_analysis = self._analyze_content_deeply(search_content)
            
            result_parts = ["=== CONTEXTUAL FINANCIAL IMAGES FOUND ===\n"]
            
            # Add context information
            if content_analysis["stocks"]:
                result_parts.append(f"📈 Related to stocks: {', '.join(content_analysis['stocks'][:5])}")
            if content_analysis["key_movers"]:
                key_movers_str = ", ".join([km.get('symbol') or km.get('company') for km in content_analysis['key_movers'][:3]])
                result_parts.append(f"🎯 Key movers: {key_movers_str}")
            if content_analysis["companies"]:
                result_parts.append(f"🏢 Companies mentioned: {', '.join(content_analysis['companies'][:5])}")
            
            result_parts.append("")

            for i, img in enumerate(images, 1):
                image_info = f"""
Image {i}:
- URL: {img['url']}
- Type: contextual_search
- Description: {img.get('title', 'Financial Chart')}
- Source: {img.get('source', 'web_search')}
- Relevance Score: {img.get('relevance_score', 0)}
---
"""
                result_parts.append(image_info)

            result_parts.append(f"Total contextual images: {len(images)}")
            result_parts.append(f"Content analysis confidence: {'High' if content_analysis['has_financial_content'] else 'Low'}")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Formatted results error: {e}")
            return f"Error generating formatted results: {e}"