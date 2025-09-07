from crewai.tools import BaseTool
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field
import requests
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import json

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
    Searches for recent US financial news using the Tavily API.
    Focuses on market activity and financial updates from the last hour.
    """

    name: str = "tavily_financial_search"
    description: str = (
        "Search for recent US financial news using the Tavily API. "
        "The search is strictly time-bound to the most recent updates."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Executes a time-sensitive financial news search.
        """
        try:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                return "Error: TAVILY_API_KEY not found in environment variables."

            financial_query = f"{query} US stock market trading financial news"
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)

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
                ],
                "exclude_domains": ["reddit.com", "twitter.com"],
                "start_published_date": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_published_date": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            formatted_results = self._format_search_results(data, start_time)

            logger.info(
                f"Tavily search completed: {len(data.get('results', []))} results found."
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

    def _format_search_results(self, data: Dict[str, Any], start_time: datetime) -> str:
        """
        Formats the Tavily search results into a readable string.
        """
        results = data.get("results", [])
        if not results:
            return "No financial news found for the specified time period."

        formatted_output = ["=== RECENT US FINANCIAL NEWS ==="]
        if data.get("answer"):
            formatted_output.append(f"MARKET OVERVIEW:\n{data['answer']}")
        formatted_output.append("DETAILED NEWS SOURCES:")

        for i, result in enumerate(results[:10], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content available")
            published = result.get("published_date", "Unknown date")

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

        formatted_output.append(
            f"\nSearch completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        formatted_output.append(f"Total results: {len(results)}")
        return "\n".join(formatted_output)


#Serper Search Tool
class SerperFinancialTool(BaseTool):
    """An alternative search tool using the Serper Google Search API for financial news."""

    name: str = "serper_financial_search"
    description: str = "Search for financial news using the Serper Google Search API."
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Executes a financial news search using the Serper API.
        """
        try:
            serper_api_key = os.getenv("SERPER_API_KEY")
            if not serper_api_key:
                return "Error: SERPER_API_KEY not found."

            financial_query = f"{query} site:bloomberg.com OR site:reuters.com OR site:cnbc.com OR site:marketwatch.com financial news today"
            url = "https://google.serper.dev/search"
            payload = {
                "q": financial_query,
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
            return self._format_serper_results(data)

        except Exception as e:
            error_msg = f"Serper search error: {e}"
            logger.error(error_msg)
            return error_msg

    def _format_serper_results(self, data: Dict[str, Any]) -> str:
        """
        Formats the Serper search results into a readable string.
        """
        articles = data.get("news", [])
        if not articles:
            return "No recent financial news found."

        formatted_output = ["=== SERPER FINANCIAL NEWS RESULTS ==="]
        for i, article in enumerate(articles[:10], 1):
            title = article.get("title", "No title")
            snippet = article.get("snippet", "No snippet")
            source = article.get("source", "Unknown source")
            date = article.get("date", "Unknown date")
            link = article.get("link", "")

            news_item = f"""
{i}. {title}
   Source: {source} | Date: {date}
   Summary: {snippet}
   Link: {link}
   ---
"""
            formatted_output.append(news_item)

        return "\n".join(formatted_output)
    
