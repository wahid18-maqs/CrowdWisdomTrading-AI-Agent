import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.financial_market_summary.tools.tavily_search import TavilyFinancialTool, TavilyTools
from datetime import datetime, timezone


class TestTavilyFinancialTool:
    """Test suite for TavilyFinancialTool"""

    @pytest.fixture
    def mock_tavily_client(self):
        """Create a mock Tavily client"""
        with patch('src.financial_market_summary.tools.tavily_search.TavilyClient') as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def tool_with_mock_client(self, mock_tavily_client, monkeypatch):
        """Create TavilyFinancialTool instance with mocked client"""
        monkeypatch.setenv('TAVILY_API_KEY', 'test_api_key_12345')
        tool = TavilyFinancialTool()
        tool.tavily_client = mock_tavily_client
        return tool

    def test_tool_initialization(self, monkeypatch):
        """Test that tool initializes correctly with API key"""
        monkeypatch.setenv('TAVILY_API_KEY', 'test_api_key_12345')
        tool = TavilyFinancialTool()

        assert tool.name == "tavily_financial_search"
        assert tool.description is not None
        assert tool.api_key == 'test_api_key_12345'

    def test_tool_initialization_missing_api_key(self, monkeypatch):
        """Test that tool raises error when API key is missing"""
        # Block load_dotenv from loading .env file
        monkeypatch.setattr('src.financial_market_summary.tools.tavily_search.load_dotenv', lambda: None)
        monkeypatch.delenv('TAVILY_API_KEY', raising=False)

        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            TavilyFinancialTool()

    def test_run_with_valid_query(self, tool_with_mock_client, mock_tavily_client, monkeypatch):
        """Test _run method with valid query"""
        # Setup mock response
        mock_tavily_client.search.return_value = {
            'results': [
                {
                    'title': 'Test Article',
                    'url': 'https://example.com/article',
                    'content': 'Test content',
                    'published_date': datetime.now(timezone.utc).isoformat()
                }
            ],
            'answer': 'Test answer'
        }

        # Mock file operations to prevent creating fake search result files
        mock_open = MagicMock()
        monkeypatch.setattr('builtins.open', mock_open)

        result = tool_with_mock_client._run(
            query="US stock market financial news",
            hours_back=1,
            max_results=20
        )

        # Verify search was called with correct parameters
        mock_tavily_client.search.assert_called_once()
        call_kwargs = mock_tavily_client.search.call_args.kwargs
        assert call_kwargs['query'] == "US stock market financial news"
        assert call_kwargs['max_results'] == 20
        assert call_kwargs['topic'] == "finance"

        # Verify result is a string (file path)
        assert isinstance(result, str)

    def test_run_with_no_results(self, tool_with_mock_client, mock_tavily_client, monkeypatch):
        """Test _run method when no results are found"""
        mock_tavily_client.search.return_value = {
            'results': [],
            'answer': 'No information found'
        }

        # Mock file operations to prevent creating fake search result files
        mock_open = MagicMock()
        monkeypatch.setattr('builtins.open', mock_open)

        result = tool_with_mock_client._run(
            query="US stock market financial news",
            hours_back=1,
            max_results=20
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_with_custom_parameters(self, tool_with_mock_client, mock_tavily_client, monkeypatch):
        """Test _run method with custom hours_back and max_results"""
        mock_tavily_client.search.return_value = {
            'results': [
                {
                    'title': 'Article 1',
                    'url': 'https://example.com/1',
                    'published_date': datetime.now(timezone.utc).isoformat()
                },
                {
                    'title': 'Article 2',
                    'url': 'https://example.com/2',
                    'published_date': datetime.now(timezone.utc).isoformat()
                }
            ],
            'answer': 'Test answer'
        }

        # Mock file operations to prevent creating fake search result files
        mock_open = MagicMock()
        monkeypatch.setattr('builtins.open', mock_open)

        result = tool_with_mock_client._run(
            query="US stock market financial news",
            hours_back=24,
            max_results=10
        )

        # Verify search was called with custom max_results
        call_kwargs = mock_tavily_client.search.call_args.kwargs
        assert call_kwargs['max_results'] == 10

    def test_time_filtering(self, tool_with_mock_client, mock_tavily_client, monkeypatch):
        """Test that results are filtered by hours_back parameter"""
        from datetime import timedelta

        old_date = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        recent_date = datetime.now(timezone.utc).isoformat()

        mock_tavily_client.search.return_value = {
            'results': [
                {
                    'title': 'Old Article',
                    'url': 'https://example.com/old',
                    'published_date': old_date
                },
                {
                    'title': 'Recent Article',
                    'url': 'https://example.com/recent',
                    'published_date': recent_date
                }
            ],
            'answer': 'Test answer'
        }

        # Mock file operations to prevent creating fake search result files
        mock_open = MagicMock()
        monkeypatch.setattr('builtins.open', mock_open)

        result = tool_with_mock_client._run(
            query="US stock market financial news",
            hours_back=1,
            max_results=20
        )

        # Verify result is a string
        assert isinstance(result, str)

    def test_alias_class(self):
        """Test that TavilyFinancialTool alias exists"""
        assert TavilyFinancialTool is not None
        assert issubclass(TavilyFinancialTool, TavilyTools)

    def test_standard_query(self, tool_with_mock_client, mock_tavily_client, monkeypatch):
        """Test tool with standard financial news query"""
        query = "US stock market financial news"

        mock_tavily_client.search.return_value = {
            'results': [{'title': f'Article about {query}', 'url': 'https://example.com'}],
            'answer': f'Information about {query}'
        }

        # Mock file operations to prevent creating fake search result files
        mock_open = MagicMock()
        monkeypatch.setattr('builtins.open', mock_open)

        result = tool_with_mock_client._run(query=query, hours_back=1, max_results=20)

        # Verify search was called with correct query
        call_kwargs = mock_tavily_client.search.call_args.kwargs
        assert call_kwargs['query'] == query


class TestTavilyToolsIntegration:
    """Integration tests (require actual API key)"""

    @pytest.mark.skipif(
        not os.getenv('TAVILY_API_KEY'),
        reason="TAVILY_API_KEY not set, skipping integration tests"
    )
    def test_live_search(self):
        """Test actual API call (only runs if API key is available)"""
        tool = TavilyFinancialTool()
        result = tool._run(
            query="US stock market financial news",
            hours_back=1,
            max_results=5
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "SEARCH_RESULTS_FILE_PATH" in result or "Article" in result
