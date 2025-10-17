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
    max_results: int = Field(default=20, description="Maximum number of results to return.")

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
        """Filter articles to ensure they are within the search window - STRICT TIME ENFORCEMENT."""
        filtered = []
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)

        logger.info(f"🕒 Filtering articles - Time window: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")

        for article in results:
            date_str = article.get("published_date")

            # STRICT: Reject articles without dates
            if not date_str:
                logger.debug(f"❌ Rejected (no date): {article.get('title', 'No title')[:60]}")
                continue

            # Parse the date
            parsed_date = self._parse_date(date_str)

            # STRICT: Reject if date can't be parsed
            if not parsed_date:
                logger.debug(f"❌ Rejected (unparseable date '{date_str}'): {article.get('title', 'No title')[:60]}")
                continue

            # STRICT: Must be within the 1-hour window
            if start <= parsed_date <= end:
                age_minutes = (now - parsed_date).total_seconds() / 60
                logger.info(f"✅ Accepted ({age_minutes:.0f}min old): {article.get('title', 'No title')[:60]}")
                filtered.append(article)
            else:
                age_hours = (now - parsed_date).total_seconds() / 3600
                logger.debug(f"❌ Rejected ({age_hours:.1f}h old): {article.get('title', 'No title')[:60]}")

        logger.info(f"📊 Filter results: {len(filtered)}/{len(results)} articles within 1-hour window")
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
        """Store filtered results in JSON with human-readable timestamps."""
        output_dir = Path("output/search_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r'[^\w\s-]', '', query).replace(' ', '-')[:50]
        filepath = output_dir / f"search_results_{timestamp}_{safe_query}.json"

        # Format timestamps in human-readable format with AM/PM UTC
        now = datetime.utcnow()
        output_data = {
            "query": query,
            "search_time": now.strftime("%Y-%m-%d %I:%M:%S %p UTC"),
            "start_time": start.strftime("%Y-%m-%d %I:%M:%S %p UTC"),
            "end_time": end.strftime("%Y-%m-%d %I:%M:%S %p UTC"),
            "time_window_hours": 1,
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
