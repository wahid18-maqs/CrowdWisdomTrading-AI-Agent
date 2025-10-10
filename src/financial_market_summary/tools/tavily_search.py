from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List
from datetime import datetime, timedelta
from urllib.parse import urlparse
from pathlib import Path
import requests
import os
import json
import logging
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SkipValidation

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TavilySearchInput(BaseModel):
    """Input schema for the Tavily search tool."""
    query: str = Field(..., description="Search query for financial news.")
    hours_back: int = Field(default=1, description="Search news from the past number of hours.")
    max_results: int = Field(default=10, description="Maximum number of results to return.")

class TavilyFinancialTool(BaseTool):
    """
    Tavily search tool that returns raw financial news search results from the past 1 hour.
    """

    name: str = "tavily_financial_search"
    description: str = (
        "Search for financial news across all domains using Tavily API. "
        "Returns RAW results for the Summary Agent and stores results in JSON."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Execute Tavily search for the past `hours_back` hours (default 1 hour)
        """
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return "Error: TAVILY_API_KEY not found in environment variables."

            # Calculate time window
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)

            payload = {
                "api_key": api_key,
                "query": self._enhance_query(query),
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": True,
                "max_results": max(max_results, 20),
                "start_published_date": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_published_date": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "exclude_domains": ["reddit.com", "twitter.com", "facebook.com", "youtube.com", "instagram.com"]
            }

            response = requests.post("https://api.tavily.com/search", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = self._filter_results_by_time(data.get("results", []), start_time, end_time)

            # Store results
            results_file = self._store_results(results, query, start_time, end_time)

            # Format raw results for Summary Agent
            return self._format_results(results, results_file)

        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily API request failed: {e}")
            return f"Error: Tavily API request failed: {e}"
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return f"Error: Tavily search failed: {e}"

    def _enhance_query(self, query: str) -> str:
        """Add financial context to query if missing."""
        financial_terms = ['stock', 'market', 'trading', 'finance', 'nasdaq', 'dow', 's&p']
        if any(term in query.lower() for term in financial_terms):
            return query
        return f"{query} stock market financial news"

    def _filter_results_by_time(self, results: List[dict], start: datetime, end: datetime) -> List[dict]:
        """Filter articles to ensure they are within the search window."""
        filtered = []
        for article in results:
            date_str = article.get("published_date")
            if not date_str:
                # Include breaking news if no date but trusted source
                domain = urlparse(article.get("url", "")).netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
                if domain in self._trusted_domains() or self._is_breaking_news(article):
                    filtered.append(article)
                continue

            parsed_date = self._parse_date(date_str)
            if parsed_date and start <= parsed_date <= end:
                filtered.append(article)
        return filtered

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse ISO or relative published dates."""
        if not date_str:
            return None
        try:
            if 'T' in date_str:
                if date_str.endswith('Z'):
                    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
                return datetime.strptime(date_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                return datetime.strptime(date_str, "%Y-%m-%d")
            # Handle relative formats like "2 hours ago"
            match = re.search(r'(\d+)\s*(minute|hour|day|week)s?\s*ago', date_str.lower())
            if match:
                value, unit = int(match.group(1)), match.group(2)
                now = datetime.utcnow()
                if unit == 'minute':
                    return now - timedelta(minutes=value)
                elif unit == 'hour':
                    return now - timedelta(hours=value)
                elif unit == 'day':
                    return now - timedelta(days=value)
                elif unit == 'week':
                    return now - timedelta(weeks=value)
        except:
            return None
        return None

    def _trusted_domains(self) -> List[str]:
        """List of trusted financial news domains."""
        return [
            "finance.yahoo.com", "yahoo.com", "investing.com", "benzinga.com", "cnbc.com",
            "reuters.com", "bloomberg.com", "nasdaq.com", "seekingalpha.com", "fool.com",
            "thestreet.com", "wsj.com", "ft.com", "morningstar.com", "zacks.com", "financialpost.com",
            "barrons.com", "sec.gov", "federalreserve.gov", "bls.gov", "treasury.gov"
        ]

    def _is_breaking_news(self, article: dict) -> bool:
        """Determine if article is breaking news based on title/content."""
        text = (article.get("title", "") + " " + article.get("content", "")).lower()
        keywords = ['breaking', 'live updates', 'just in', 'developing', 'today', 'now']
        return any(kw in text for kw in keywords)

    def _store_results(self, results: List[dict], query: str, start: datetime, end: datetime) -> str:
        """Store filtered results in JSON."""
        output_dir = Path("output/search_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r'[^\w\s-]', '', query).replace(' ', '-')[:50]
        filepath = output_dir / f"search_results_{timestamp}_{safe_query}.json"

        output_data = {
            "query": query,
            "search_time": datetime.utcnow().isoformat(),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "total_articles": len(results),
            "articles": results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(results)} results to {filepath}")
        return str(filepath)

    def _format_results(self, results: List[dict], results_file: str) -> str:
        """Format raw search results as text for the Summary Agent."""
        lines = [
            "=== FINANCIAL NEWS SEARCH RESULTS ===",
            f"Total Articles Found: {len(results)}",
            f"Search Results File: {results_file}\n"
        ]
        for i, article in enumerate(results, 1):
            lines.append(f"--- Article {i} ---")
            lines.append(f"Title: {article.get('title', 'No title')}")
            lines.append(f"Source: {urlparse(article.get('url', '')).netloc}")
            lines.append(f"URL: {article.get('url', 'No URL')}")
            lines.append(f"Published: {article.get('published_date', 'Recent')}")
            lines.append(f"Content: {article.get('content', '')[:500]}...\n")
        lines.append("=== END OF SEARCH RESULTS ===")
        lines.append("\nInstructions for Summary Agent:")
        lines.append("Please create a 'The Crowd Wisdom's summary' style financial summary from the above articles.")
        return "\n".join(lines)
