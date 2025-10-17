import os
import logging
import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from crewai.tools import BaseTool
from tavily import TavilyClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TavilyQuery(BaseModel):
    """Structure for individual Tavily search queries"""
    query: str = Field(description="web search query")
    topic: str = Field(
        description="type of search, should be 'general' or 'news'. Choose 'news' for financial markets",
        default="news"
    )
    days: int = Field(
        description="number of days back to run 'news' search",
        default=1
    )
    domains: Optional[List[str]] = Field(
        description="list of domains to search in",
        default=None
    )

class TavilySearchInput(BaseModel):
    """Structure for search input containing multiple sub-queries"""
    sub_queries: List[TavilyQuery] = Field(
        description="list of search queries to execute",
        default_factory=list
    )

class TavilyTools(BaseTool):
    """Tool for searching financial news using Tavily API"""

    name: str = "Tavily News Search"
    description: str = "Search for financial news and market information using Tavily API"
    api_key: str = Field(default=None, description="Tavily API key")
    tavily_client: Optional[TavilyClient] = Field(default=None, description="Tavily API client")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        # Load environment variables
        load_dotenv()

        # Get API key from environment if not provided
        if 'api_key' not in data:
            data['api_key'] = os.getenv('TAVILY_API_KEY')
            if not data['api_key']:
                raise ValueError("TAVILY_API_KEY environment variable is not set")

        # Initialize parent class
        super().__init__(**data)

        # Initialize Tavily client
        self.tavily_client = TavilyClient(api_key=self.api_key)
        logger.info("Tavily client initialized successfully")

    def _run(self, query: str, hours_back: int = 1, max_results: int = 20) -> str:
        """
        Execute the search operation
        Args:
            query: The search query string
            hours_back: Number of hours back (kept for compatibility, not used by TavilyClient)
            max_results: Maximum number of results (kept for compatibility, always uses 20)
        Returns:
            str: Formatted search results with file path
        """
        try:
            # Create a single query
            search_query = TavilyQuery(query=query)

            # Perform search
            search_result = self.tavily_client.search(
                query=search_query.query,
                search_depth="advanced",
                include_images=False,
                include_answer=True,
                max_results=20
            )

            # Store results in JSON
            results_file = self._store_results(search_result, query)

            # Format results
            formatted_result = "=== FINANCIAL NEWS SEARCH RESULTS ===\n\n"
            formatted_result += f"Query: {search_query.query}\n"
            formatted_result += f"Total Results: {len(search_result.get('results', []))}\n"
            formatted_result += f"**SEARCH_RESULTS_FILE_PATH**: {results_file}\n\n"

            if search_result.get('answer'):
                formatted_result += f"Answer: {search_result['answer']}\n\n"

            formatted_result += "Sources:\n\n"

            for idx, result in enumerate(search_result.get('results', []), 1):
                formatted_result += f"{idx}. {result.get('title', 'No title')}\n"
                formatted_result += f"   URL: {result.get('url', 'No URL')}\n"
                formatted_result += f"   Content: {result.get('content', 'No content')[:300]}...\n\n"

            formatted_result += "=== END OF SEARCH RESULTS ===\n"
            formatted_result += f"\n📁 Search results saved to: {results_file}\n"
            formatted_result += "Use this file path when calling enhanced_financial_image_finder tool.\n"

            return formatted_result

        except Exception as e:
            error_msg = f"Error in Tavily search: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _store_results(self, search_result: Dict[str, Any], query: str) -> str:
        """Store search results in JSON file."""
        output_dir = Path("output/search_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r'[^\w\s-]', '', query).replace(' ', '-')[:50]
        filepath = output_dir / f"search_results_{timestamp}_{safe_query}.json"

        now = datetime.utcnow()
        output_data = {
            "query": query,
            "search_time": now.strftime("%Y-%m-%d %I:%M:%S %p UTC"),
            "total_results": len(search_result.get('results', [])),
            "answer": search_result.get('answer', ''),
            "articles": search_result.get('results', [])  # Store as 'articles' for image_finder
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(search_result.get('results', []))} results to {filepath}")
        return str(filepath)


# Alias for backward compatibility
class TavilyFinancialTool(TavilyTools):
    """Alias for TavilyTools to maintain backward compatibility"""
    name: str = "tavily_financial_search"
    description: str = (
        "Search for financial news across all domains using Tavily API. "
        "Returns results with file path for image extraction."
    )
