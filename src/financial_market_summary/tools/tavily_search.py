from crewai.tools import BaseTool
from typing import Type, Any, Optional
from pydantic import BaseModel, Field
import requests
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(r"C:\Users\wahid\Desktop\financial_market_summary\.env")

class TavilySearchInput(BaseModel):
    """Input schema for Tavily search tool."""
    query: str = Field(..., description="Search query for financial news")
    hours_back: int = Field(default=1, description="Number of hours back to search")
    max_results: int = Field(default=10, description="Maximum number of results")

class TavilyFinancialTool(BaseTool):
    name: str = "tavily_financial_search"
    description: str = (
        "Search for recent US financial news using Tavily API. "
        "Focuses on market activity, trading news, and financial updates from the last hour."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Search for financial news using Tavily API
        """
        try:
            tavily_api_key = os.getenv('TAVILY_API_KEY')
            if not tavily_api_key:
                return "Error: TAVILY_API_KEY not found in environment variables"

            # Enhance query for financial context
            financial_query = f"{query} US stock market trading financial news"
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Tavily API endpoint
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
                    "seekingalpha.com"
                ],
                "exclude_domains": ["reddit.com", "twitter.com"]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Format results
            formatted_results = self._format_search_results(data, start_time)
            
            logger.info(f"Tavily search completed: {len(data.get('results', []))} results found")
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Tavily API request failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"Tavily search error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_search_results(self, data: dict, start_time: datetime) -> str:
        """Format Tavily search results for financial analysis"""
        
        if not data.get('results'):
            return "No financial news found for the specified time period."
        
        formatted_output = []
        formatted_output.append("=== RECENT US FINANCIAL NEWS ===\n")
        
        # Add answer if available
        if data.get('answer'):
            formatted_output.append(f"MARKET OVERVIEW:\n{data['answer']}\n")
        
        formatted_output.append("DETAILED NEWS SOURCES:")
        
        for i, result in enumerate(data['results'][:10], 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            content = result.get('content', 'No content available')
            published = result.get('published_date', 'Unknown date')
            
            # Truncate content for readability
            if len(content) > 300:
                content = content[:300] + "..."
            
            news_item = f"""
{i}. {title}
   Source: {url}
   Published: {published}
   Summary: {content}
   ---
"""
            formatted_output.append(news_item)
        
        formatted_output.append(f"\nSearch completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        formatted_output.append(f"Total results: {len(data['results'])}")
        
        return "\n".join(formatted_output)


class SerperFinancialTool(BaseTool):
    """Alternative search tool using Serper API"""
    name: str = "serper_financial_search"
    description: str = "Search for financial news using Serper Google Search API"
    args_schema: Type[BaseModel] = TavilySearchInput
    
    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        try:
            serper_api_key = os.getenv('SERPER_API_KEY')
            if not serper_api_key:
                return "Error: SERPER_API_KEY not found"
            
            # Enhanced financial query
            financial_query = f"{query} site:bloomberg.com OR site:reuters.com OR site:cnbc.com OR site:marketwatch.com financial news today"
            
            url = "https://google.serper.dev/search"
            
            payload = {
                "q": financial_query,
                "num": max_results,
                "tbm": "nws",  # News search
                "tbs": f"qdr:h{hours_back}",  # Time range
                "gl": "us",
                "hl": "en"
            }
            
            headers = {
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return self._format_serper_results(data)
            
        except Exception as e:
            error_msg = f"Serper search error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_serper_results(self, data: dict) -> str:
        """Format Serper search results"""
        if not data.get('news'):
            return "No recent financial news found."
        
        formatted_output = ["=== SERPER FINANCIAL NEWS RESULTS ===\n"]
        
        for i, article in enumerate(data['news'][:10], 1):
            title = article.get('title', 'No title')
            snippet = article.get('snippet', 'No snippet')
            source = article.get('source', 'Unknown source')
            date = article.get('date', 'Unknown date')
            link = article.get('link', '')
            
            news_item = f"""
{i}. {title}
   Source: {source} | Date: {date}
   Summary: {snippet}
   Link: {link}
   ---
"""
            formatted_output.append(news_item)
        
        return "\n".join(formatted_output)