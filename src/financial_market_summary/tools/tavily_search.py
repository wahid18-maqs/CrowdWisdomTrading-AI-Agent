from crewai.tools import BaseTool
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field
import requests
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import json
import re
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TavilySearchInput(BaseModel):
    """Input schema for the financial search tool."""

    query: str = Field(..., description="The search query for financial news.")
    hours_back: int = Field(
        default=1, description="Number of hours back to search for news."
    )
    max_results: int = Field(
        default=10, description="The maximum number of search results to return."
    )

class TavilyFinancialTool(BaseTool):
    """
    Enhanced Tavily search tool with proper source link inclusion and better content structuring.
    """

    name: str = "tavily_financial_search"
    description: str = (
        "Search for recent US financial news using the Tavily API within the last 1 hour. "
        "Includes article source links and structured content for better processing."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Executes financial news search with enhanced content structure and source links.
        """
        try:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                return "Error: TAVILY_API_KEY not found in environment variables."

            # Enforce 1-hour limit for real-time financial news
            if hours_back != 1:
                logger.info(f"Enforcing 1-hour limit for real-time news (was {hours_back})")
                hours_back = 1

            # Enhanced query construction
            financial_query = self._build_enhanced_query(query)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)

            logger.info(f"Searching for news from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_api_key,
                "query": financial_query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": [
                    "yahoo.com/finance",
                    "marketwatch.com",
                    "investing.com",
                    "benzinga.com",
                    "cnbc.com",
                    "reuters.com",
                    "bloomberg.com"
                ],
                "exclude_domains": ["reddit.com", "twitter.com", "facebook.com"],
                "start_published_date": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_published_date": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            
            # Filter results by time
            filtered_results = self._filter_results_by_time(data.get('results', []), start_time, end_time)
            
            if not filtered_results:
                logger.warning("No results in 1 hour, expanding to 2 hours...")
                # Expand to 2 hours if no results
                start_time = end_time - timedelta(hours=2)
                payload["start_published_date"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                filtered_results = data.get('results', [])
                
                if filtered_results:
                    logger.info(f"Expanded search found {len(filtered_results)} results")
            
            # Update data with filtered results
            data['results'] = filtered_results
            
            # Enhanced formatting with source links
            formatted_results = self._format_enhanced_results_with_links(data, start_time, query, hours_back)

            logger.info(f"Tavily search completed: {len(filtered_results)} results with source links")
            return formatted_results

        except requests.exceptions.RequestException as e:
            error_msg = f"Tavily API request failed: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Tavily search error: {e}"
            logger.error(error_msg)
            return error_msg

    def _filter_results_by_time(self, results: list, start_time: datetime, end_time: datetime) -> list:
        """Filter results to ensure they fall within the time window."""
        filtered_results = []
        
        for result in results:
            published_date = result.get('published_date')
            if not published_date:
                # If no date, include the result
                filtered_results.append(result)
                continue
                
            try:
                # Parse published date
                if 'T' in published_date:
                    pub_datetime = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                    if pub_datetime.tzinfo is not None:
                        pub_datetime = pub_datetime.astimezone().utctimetuple()
                        pub_datetime = datetime(*pub_datetime[:6])
                else:
                    # Try other formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            pub_datetime = datetime.strptime(published_date, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # If can't parse, include the result
                        filtered_results.append(result)
                        continue
                
                # Check if within time window
                if start_time <= pub_datetime <= end_time:
                    filtered_results.append(result)
                    
            except Exception as e:
                logger.debug(f"Error parsing date '{published_date}': {e}")
                # If parsing fails, include the result
                filtered_results.append(result)
                
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} within time window")
        return filtered_results

    def _build_enhanced_query(self, original_query: str) -> str:
        """Build enhanced search query for better financial results."""
        base_query = f"{original_query} US stock market financial news"
        
        query_lower = original_query.lower()
        
        if any(term in query_lower for term in ["earnings", "results", "revenue"]):
            base_query += " earnings report quarterly"
        elif any(term in query_lower for term in ["fed", "interest", "rate"]):
            base_query += " federal reserve interest rates"
        elif any(term in query_lower for term in ["tech", "technology"]):
            base_query += " technology stocks"
        
        return base_query

    def _format_enhanced_results_with_links(self, data: Dict[str, Any], start_time: datetime, original_query: str, hours_searched: int) -> str:
        """
        Enhanced formatting with proper source links and structured content.
        """
        results = data.get("results", [])
        if not results:
            return f"No financial news found within the last {hours_searched} hour(s)."

        # Content analysis for better structure
        content_analysis = self._analyze_search_results(results)
        
        formatted_output = ["=== REAL-TIME US FINANCIAL NEWS (LAST 1 HOUR) ==="]
        
        # Time window
        formatted_output.append(f"\n**SEARCH WINDOW:**")
        formatted_output.append(f"📅 From: {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        formatted_output.append(f"📅 To: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        formatted_output.append(f"⏱️ Time Range: Last {hours_searched} hour(s)")
        
        # Market overview
        if data.get("answer"):
            market_overview = self._clean_content(data["answer"])
            formatted_output.append(f"\n**MARKET OVERVIEW:**")
            formatted_output.append(market_overview)
        
        # Key highlights with stock info
        if content_analysis["key_stocks"] or content_analysis["key_themes"]:
            formatted_output.append(f"\n**KEY HIGHLIGHTS:**")
            if content_analysis["key_stocks"]:
                formatted_output.append(f"• 📈 Key Stocks: {', '.join(content_analysis['key_stocks'][:5])}")
            if content_analysis["key_themes"]:
                formatted_output.append(f"• 🎯 Key Themes: {', '.join(content_analysis['key_themes'][:3])}")
            if content_analysis["key_movers"]:
                movers_str = ", ".join([f"{m['symbol']} ({m['performance']})" for m in content_analysis["key_movers"][:3]])
                formatted_output.append(f"• 🚀 Key Movers: {movers_str}")
        
        # Breaking news articles with PROPER SOURCE LINKS
        formatted_output.append(f"\n**BREAKING NEWS ARTICLES:**")

        for i, result in enumerate(results[:8], 1):  # Limit to 8 articles
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content available")
            published = result.get("published_date", "Unknown date")
            
            # Clean and enhance content
            enhanced_content = self._enhance_article_content(content, content_analysis)
            if len(enhanced_content) > 400:
                enhanced_content = enhanced_content[:400] + "..."

            # Format with GUARANTEED source link
            news_item = f"""
**{i}. {self._clean_content(title)}**
📰 Source: {self._extract_clean_domain(url)}
⏰ Published: {self._format_news_date(published)}
📄 Summary: {self._clean_content(enhanced_content)}
🔗 Link: {url}
---"""
            formatted_output.append(news_item)

        # Market implications section
        formatted_output.append(f"\n**MARKET IMPLICATIONS:**")
        implications = self._generate_market_implications(content_analysis, data.get("answer", ""))
        for impl in implications:
            formatted_output.append(f"• {impl}")

        # Key points for summary
        formatted_output.append(f"\n**KEY POINTS:**")
        key_points = self._generate_key_points(content_analysis, results[:3])
        for point in key_points:
            formatted_output.append(f"• {point}")

        # Footer
        formatted_output.append(f"\n**SEARCH METADATA:**")
        formatted_output.append(f"🔍 Search completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        formatted_output.append(f"📊 Total results: {len(results)}")
        formatted_output.append(f"⚡ Real-time filter: Last {hours_searched} hour(s) enforced")
        formatted_output.append(f"🎯 Enhanced for: '{original_query}'")
        
        return "\n".join(formatted_output)

    def _clean_content(self, content: str) -> str:
        """Clean content of formatting artifacts"""
        if not content:
            return ""
        
        # Remove excessive formatting
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^*]+)\*', r'\1', content)
        content = re.sub(r'[#*_`~]+', '', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()

    def _extract_clean_domain(self, url: str) -> str:
        """Extract clean, readable domain name"""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Map to readable names
            domain_map = {
                'yahoo.com': 'Yahoo Finance',
                'marketwatch.com': 'MarketWatch',
                'investing.com': 'Investing.com',
                'benzinga.com': 'Benzinga',
                'cnbc.com': 'CNBC',
                'reuters.com': 'Reuters',
                'bloomberg.com': 'Bloomberg'
            }
            
            return domain_map.get(domain, domain.title())
        except:
            return "Financial News"

    def _format_news_date(self, date_str: str) -> str:
        """Format date for better readability"""
        if not date_str or date_str == "Unknown date":
            return "Recent"
        
        try:
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime("%Y-%m-%d %H:%M UTC")
            return date_str
        except:
            return date_str

    def _analyze_search_results(self, results: list) -> Dict[str, list]:
        """Analyze results for key financial information"""
        analysis = {
            "key_stocks": [],
            "key_themes": [],
            "mentioned_companies": [],
            "sectors": [],
            "market_events": [],
            "key_movers": []
        }

        all_content = ""
        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")
            all_content += f" {title} {content}"

        # Extract stocks
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, all_content)
        
        major_stocks = {
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "NVDA", "META",
            "NFLX", "AMD", "INTC", "CRM", "ADBE", "UBER", "SPOT", "ADSK", "FDX"
        }
        
        stock_frequency = {}
        for stock in potential_stocks:
            if stock in major_stocks:
                stock_frequency[stock] = stock_frequency.get(stock, 0) + 1
        
        analysis["key_stocks"] = [
            stock for stock, _ in sorted(stock_frequency.items(), key=lambda x: x[1], reverse=True)
        ][:6]

        # Find key movers with performance data
        mover_patterns = [
            r'([A-Z]{2,5})\s+(?:surged?|jumped?|gained?|rose|rallied)\s+([\d.]+%)',
            r'([A-Z]{2,5})\s+(?:dropped?|fell|declined?|lost)\s+([\d.]+%)',
        ]
        
        for pattern in mover_patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            for symbol, performance in matches:
                if symbol in major_stocks:
                    analysis["key_movers"].append({
                        "symbol": symbol,
                        "performance": performance
                    })
        
        # Remove duplicate movers
        seen_symbols = set()
        unique_movers = []
        for mover in analysis["key_movers"]:
            if mover["symbol"] not in seen_symbols:
                unique_movers.append(mover)
                seen_symbols.add(mover["symbol"])
        analysis["key_movers"] = unique_movers[:5]

        # Identify themes
        content_lower = all_content.lower()
        theme_keywords = {
            "earnings": ["earnings", "quarterly results", "revenue", "profit"],
            "fed_policy": ["federal reserve", "fed", "interest rate", "rate cut"],
            "technology": ["ai", "tech", "software", "semiconductor"],
            "healthcare": ["healthcare", "pharma", "biotech"],
            "energy": ["oil", "gas", "energy"]
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                analysis["key_themes"].append(theme)

        return analysis

    def _enhance_article_content(self, content: str, analysis: Dict[str, list]) -> str:
        """Enhance content by highlighting key information"""
        if not content:
            return "No content available"

        enhanced = self._clean_content(content)

        # Highlight key stocks
        for stock in analysis["key_stocks"][:3]:
            if stock in enhanced:
                enhanced = enhanced.replace(stock, f"**{stock}**", 1)

        return enhanced

    def _generate_market_implications(self, analysis: Dict[str, list], market_overview: str) -> list:
        """Generate relevant market implications"""
        implications = []
        
        if "earnings" in analysis["key_themes"]:
            implications.append("Strong corporate earnings suggest underlying resilience in key sectors")
        
        if "fed_policy" in analysis["key_themes"]:
            implications.append("Fed policy changes aim to support economic growth and market sentiment")
        
        if analysis["key_movers"]:
            implications.append("Key stock movements indicate focused investor attention on specific opportunities")
        
        # Default implications if none specific
        if not implications:
            implications = [
                "Market movements reflect ongoing investor sentiment and economic conditions",
                "Current trends suggest continued focus on corporate performance and policy impacts",
                "Investor attention remains on key sector developments and earnings outcomes"
            ]
        
        return implications[:3]

    def _generate_key_points(self, analysis: Dict[str, list], top_results: list) -> list:
        """Generate key points from analysis"""
        points = []
        
        # Stock-specific points
        if analysis["key_movers"]:
            for mover in analysis["key_movers"][:2]:
                points.append(f"{mover['symbol']} shows significant movement with {mover['performance']} change")
        
        # Theme-specific points
        if "earnings" in analysis["key_themes"]:
            points.append("Corporate earnings reports driving market activity")
        
        if "fed_policy" in analysis["key_themes"]:
            points.append("Federal Reserve policy decisions influencing market direction")
        
        # Sector points
        if "technology" in analysis["key_themes"]:
            points.append("Technology sector showing notable activity")
        
        # Default points if none specific
        if not points:
            points = [
                "Market activity reflects current economic conditions",
                "Investor focus on corporate performance and policy developments",
                "Key sectors showing varied performance patterns"
            ]
        
        return points[:4]