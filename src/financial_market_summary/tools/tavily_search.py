from crewai.tools import BaseTool
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field
import requests
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import json
import re  # Moved to top for efficiency
from urllib.parse import urlparse  # Moved to top

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


# Tavily Search Tools 
class TavilySearchTool(BaseTool):
    """A generic Tavily search tool that delegates to the specialized financial tool."""

    name: str = "tavily_search_tool"
    description: str = "A generic Tavily search tool for financial queries."
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Runs a generic Tavily search by delegating to TavilyFinancialTool.
        """
        return TavilyFinancialTool()._run(
            query, hours_back=hours_back, max_results=max_results
        )


class TavilyFinancialTool(BaseTool):
    """
    Enhanced Tavily search tool for recent US financial news with better content structuring
    for image finder integration.
    """

    name: str = "tavily_financial_search"
    description: str = (
        "Search for recent US financial news using the Tavily API. "
        "Enhanced to provide structured content that works well with image finding."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Executes a time-sensitive financial news search with enhanced content structuring.
        Default to 1 hour for real-time data.
        """
        try:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                return "Error: TAVILY_API_KEY not found in environment variables."

            # Enhanced query construction for better results
            financial_query = self._build_enhanced_query(query)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)  # Default 1 hour

            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_api_key,
                "query": financial_query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": [
                    "bloomberg.com",
                    "reuters.com", 
                    "cnbc.com",
                    "marketwatch.com",
                    "yahoo.com/finance",
                    "investing.com",
                    "benzinga.com",
                    "seekingalpha.com",
                    "wsj.com",
                    "forbes.com",
                    "ft.com"
                ],
                "exclude_domains": ["reddit.com", "twitter.com", "facebook.com"],
                "start_published_date": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_published_date": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            
            # Enhanced formatting for better image context extraction
            formatted_results = self._format_enhanced_results(data, start_time, query)

            logger.info(
                f"Enhanced Tavily search completed: {len(data.get('results', []))} results found for last {hours_back} hour(s)."
            )
            return formatted_results

        except requests.exceptions.RequestException as e:
            error_msg = f"Tavily API request failed: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Tavily search error: {e}"
            logger.error(error_msg)
            return error_msg

    def _build_enhanced_query(self, original_query: str) -> str:
        """Build enhanced search query for better financial results."""
        base_query = f"{original_query} US stock market trading financial news"
        
        # Add contextual terms based on query content
        query_lower = original_query.lower()
        
        if any(term in query_lower for term in ["earnings", "results", "revenue"]):
            base_query += " earnings report quarterly results"
        elif any(term in query_lower for term in ["merger", "acquisition", "deal"]):
            base_query += " corporate merger acquisition deal"
        elif any(term in query_lower for term in ["fed", "interest", "rate"]):
            base_query += " federal reserve interest rates monetary policy"
        elif any(term in query_lower for term in ["tech", "technology", "ai"]):
            base_query += " technology sector artificial intelligence"
        
        return base_query

    def _format_enhanced_results(self, data: Dict[str, Any], start_time: datetime, original_query: str) -> str:
        """
        Enhanced formatting that provides better structure for image finding and content analysis.
        """
        results = data.get("results", [])
        if not results:
            return "No financial news found for the specified time period."

        # Analyze content for key themes and stocks
        content_analysis = self._analyze_search_results(results)
        
        formatted_output = ["=== ENHANCED US FINANCIAL NEWS SUMMARY ==="]
        
        # Market overview section
        if data.get("answer"):
            formatted_output.append(f"\n**MARKET OVERVIEW:**\n{data['answer']}")
        
        # Key highlights section
        if content_analysis["key_stocks"] or content_analysis["key_themes"]:
            highlights = []
            if content_analysis["key_stocks"]:
                highlights.append(f"📈 Key Stocks: {', '.join(content_analysis['key_stocks'][:5])}")
            if content_analysis["key_themes"]:
                highlights.append(f"🎯 Key Themes: {', '.join(content_analysis['key_themes'][:3])}")
            
            formatted_output.append(f"\n**KEY HIGHLIGHTS:**")
            formatted_output.extend([f"• {highlight}" for highlight in highlights])
        
        # Detailed news sources
        formatted_output.append(f"\n**DETAILED NEWS SOURCES:**")

        for i, result in enumerate(results[:10], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content available")
            published = result.get("published_date", "Unknown date")

            # Enhanced content processing
            enhanced_content = self._enhance_article_content(content, content_analysis)
            
            if len(enhanced_content) > 400:
                enhanced_content = enhanced_content[:400] + "..."

            news_item = f"""
**{i}. {title}**
📰 Source: {self._extract_domain(url)}
📅 Published: {published}
📄 Summary: {enhanced_content}
🔗 Link: {url}
---
"""
            formatted_output.append(news_item)

        # Content analysis summary for image finder
        formatted_output.append(f"\n**CONTENT ANALYSIS FOR CHARTS:**")
        if content_analysis["mentioned_companies"]:
            formatted_output.append(f"Companies mentioned: {', '.join(content_analysis['mentioned_companies'][:8])}")
        if content_analysis["sectors"]:
            formatted_output.append(f"Sectors involved: {', '.join(content_analysis['sectors'])}")
        if content_analysis["market_events"]:
            formatted_output.append(f"Market events: {', '.join(content_analysis['market_events'])}")

        # Footer with metadata
        formatted_output.append(f"\n**SEARCH METADATA:**")
        formatted_output.append(f"Search completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        formatted_output.append(f"Total results: {len(results)}")
        formatted_output.append(f"Query optimization: Enhanced for '{original_query}'")
        
        return "\n".join(formatted_output)

    def _analyze_search_results(self, results: list) -> Dict[str, list]:
        """
        Analyze search results to extract key financial information for better image targeting.
        """
        analysis = {
            "key_stocks": [],
            "key_themes": [],
            "mentioned_companies": [],
            "sectors": [],
            "market_events": []
        }

        all_content = ""
        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")
            all_content += f" {title} {content}"

        content_lower = all_content.lower()

        # Extract stock symbols more intelligently
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, all_content)
        
        # Major stocks to prioritize
        major_stocks = {
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "NVDA", "META",
            "NFLX", "AMD", "INTC", "CRM", "ADBE", "PYPL", "UBER", "SPOT",
            "ZOOM", "JPM", "BAC", "WFC", "GS", "MS", "C", "JNJ", "PFE", 
            "UNH", "CVS", "XOM", "CVX", "WMT", "HD", "DIS", "V", "MA"
        }
        
        # Filter and prioritize stocks
        stock_frequency = {}
        for stock in potential_stocks:
            if stock in major_stocks:
                stock_frequency[stock] = stock_frequency.get(stock, 0) + 1
        
        # Sort by frequency and take top stocks
        analysis["key_stocks"] = [
            stock for stock, _ in sorted(stock_frequency.items(), key=lambda x: x[1], reverse=True)
        ][:6]

        # Identify key themes
        theme_keywords = {
            "earnings": ["earnings", "quarterly results", "revenue", "profit", "guidance"],
            "mergers": ["merger", "acquisition", "deal", "buyout", "takeover"],
            "fed_policy": ["federal reserve", "fed", "interest rate", "monetary policy"],
            "technology": ["ai", "artificial intelligence", "tech", "software", "semiconductor"],
            "healthcare": ["healthcare", "pharma", "biotech", "medical", "drug"],
            "energy": ["oil", "gas", "energy", "renewable", "solar"],
            "inflation": ["inflation", "cpi", "ppi", "prices", "costs"]
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                analysis["key_themes"].append(theme)

        # Extract company names (more comprehensive)
        company_patterns = [
            r'\b(Apple|Microsoft|Google|Amazon|Tesla|Meta|Netflix|Nvidia)\b',
            r'\b(JPMorgan|Goldman Sachs|Bank of America|Wells Fargo)\b',
            r'\b(Johnson & Johnson|Pfizer|Moderna|UnitedHealth)\b'
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            analysis["mentioned_companies"].extend([match.title() for match in matches])

        # Identify sectors
        sector_keywords = {
            "technology": ["tech", "software", "semiconductor", "ai", "cloud"],
            "healthcare": ["healthcare", "pharma", "biotech", "medical"],
            "financial": ["bank", "financial", "insurance", "lending"],
            "energy": ["oil", "gas", "energy", "renewable"],
            "consumer": ["retail", "consumer goods", "e-commerce"],
            "automotive": ["automotive", "car", "electric vehicle", "ev"]
        }

        for sector, keywords in sector_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                analysis["sectors"].append(sector)

        # Identify market events
        event_keywords = [
            "earnings report", "ipo", "stock split", "dividend", "buyback",
            "merger announcement", "acquisition deal", "partnership",
            "fed meeting", "interest rate decision", "inflation data",
            "jobs report", "gdp", "unemployment"
        ]

        for event in event_keywords:
            if event in content_lower:
                analysis["market_events"].append(event)

        # Remove duplicates
        for key in analysis:
            analysis[key] = list(set(analysis[key]))

        return analysis

    def _enhance_article_content(self, content: str, analysis: Dict[str, list]) -> str:
        """
        Enhance article content by highlighting important financial terms.
        """
        if not content:
            return "No content available"

        enhanced = content

        # Highlight key stocks mentioned
        for stock in analysis["key_stocks"]:
            if stock in enhanced:
                enhanced = enhanced.replace(stock, f"**{stock}**")

        # Highlight important financial terms
        important_terms = [
            "earnings", "revenue", "profit", "loss", "merger", "acquisition",
            "federal reserve", "interest rate", "inflation", "GDP"
        ]

        for term in important_terms:
            if term.lower() in enhanced.lower():
                # Case-insensitive replacement while preserving original case
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                enhanced = pattern.sub(f"**{term.title()}**", enhanced, count=1)

        return enhanced

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain name from URL."""
        try:
            domain = urlparse(url).netloc
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "Unknown source"


# Serper Search Tool with enhancements
class SerperFinancialTool(BaseTool):
    """Enhanced Serper search tool for financial news with better image integration support."""

    name: str = "serper_financial_search"
    description: str = "Search for financial news using the Serper Google Search API with enhanced content structuring."
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Enhanced Serper search with better content formatting for image integration.
        """
        try:
            serper_api_key = os.getenv("SERPER_API_KEY")
            if not serper_api_key:
                return "Error: SERPER_API_KEY not found."

            # Enhanced query construction
            enhanced_query = self._build_serper_query(query)
            
            url = "https://google.serper.dev/search"
            payload = {
                "q": enhanced_query,
                "num": max_results,
                "tbm": "nws",
                "tbs": f"qdr:h{hours_back}",
                "gl": "us",
                "hl": "en",
            }
            headers = {
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json",
            }

            response = requests.post(
                url, json=payload, headers=headers, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            return self._format_enhanced_serper_results(data, query)

        except Exception as e:
            error_msg = f"Serper search error: {e}"
            logger.error(error_msg)
            return error_msg

    def _build_serper_query(self, original_query: str) -> str:
        """Build enhanced Serper query."""
        sites = "site:bloomberg.com OR site:reuters.com OR site:cnbc.com OR site:marketwatch.com OR site:yahoo.com OR site:investing.com"
        return f"{original_query} {sites} financial news stock market today"

    def _format_enhanced_serper_results(self, data: Dict[str, Any], original_query: str) -> str:
        """
        Enhanced Serper results formatting with content analysis for image integration.
        """
        articles = data.get("news", [])
        if not articles:
            return "No recent financial news found via Serper search."

        # Quick content analysis
        all_content = " ".join([
            f"{article.get('title', '')} {article.get('snippet', '')}" 
            for article in articles
        ])
        
        content_analysis = self._quick_content_analysis(all_content)

        formatted_output = ["=== SERPER FINANCIAL NEWS RESULTS ==="]
        
        # Add content highlights
        if content_analysis["stocks"] or content_analysis["themes"]:
            formatted_output.append("\n**CONTENT HIGHLIGHTS:**")
            if content_analysis["stocks"]:
                formatted_output.append(f"📈 Stocks mentioned: {', '.join(content_analysis['stocks'][:5])}")
            if content_analysis["themes"]:
                formatted_output.append(f"🎯 Key themes: {', '.join(content_analysis['themes'][:3])}")

        formatted_output.append("\n**NEWS ARTICLES:**")
        
        for i, article in enumerate(articles[:10], 1):
            title = article.get("title", "No title")
            snippet = article.get("snippet", "No snippet")
            source = article.get("source", "Unknown source")
            date = article.get("date", "Unknown date")
            link = article.get("link", "")

            # Enhance snippet with key term highlighting
            enhanced_snippet = self._enhance_snippet(snippet, content_analysis)

            news_item = f"""
**{i}. {title}**
📰 {source} | 📅 {date}
📄 {enhanced_snippet}
🔗 {link}
---
"""
            formatted_output.append(news_item)

        # Add content analysis for image finder
        formatted_output.append(f"\n**ANALYSIS FOR CHART INTEGRATION:**")
        if content_analysis["companies"]:
            formatted_output.append(f"Companies: {', '.join(content_analysis['companies'][:6])}")
        if content_analysis["sectors"]:
            formatted_output.append(f"Sectors: {', '.join(content_analysis['sectors'])}")

        return "\n".join(formatted_output)

    def _quick_content_analysis(self, content: str) -> Dict[str, list]:
        """Quick analysis for Serper content."""
        
        analysis = {
            "stocks": [],
            "themes": [],
            "companies": [],
            "sectors": []
        }

        content_lower = content.lower()

        # Extract stocks
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, content)
        major_stocks = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"}
        analysis["stocks"] = [s for s in set(potential_stocks) if s in major_stocks][:5]

        # Quick theme detection
        if any(term in content_lower for term in ["earnings", "revenue", "profit"]):
            analysis["themes"].append("earnings")
        if any(term in content_lower for term in ["merger", "acquisition"]):
            analysis["themes"].append("M&A")
        if any(term in content_lower for term in ["fed", "interest rate"]):
            analysis["themes"].append("monetary_policy")

        # Quick company extraction
        companies = ["Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta", "Netflix"]
        for company in companies:
            if company.lower() in content_lower:
                analysis["companies"].append(company)

        return analysis

    def _enhance_snippet(self, snippet: str, analysis: Dict[str, list]) -> str:
        """Enhance snippet with key term highlighting."""
        if not snippet:
            return "No snippet available"

        enhanced = snippet
        
        # Highlight key stocks
        for stock in analysis["stocks"]:
            if stock in enhanced:
                enhanced = enhanced.replace(stock, f"**{stock}**")

        return enhanced