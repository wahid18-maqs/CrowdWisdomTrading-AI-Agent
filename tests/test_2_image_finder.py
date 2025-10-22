import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.financial_market_summary.tools.image_finder import EnhancedImageFinder


class TestEnhancedImageFinder:
    """Test suite for EnhancedImageFinder"""

    @pytest.fixture
    def image_finder(self):
        """Create EnhancedImageFinder instance"""
        return EnhancedImageFinder()

    @pytest.fixture
    def sample_search_results_file(self, tmp_path):
        """Create a sample search results JSON file"""
        search_data = {
            "query": "US stock market news",
            "articles": [
                {
                    "title": "Market Rally",
                    "url": "https://cnbc.com/article1",
                    "content": "Stock market rises today"
                },
                {
                    "title": "Tech Stocks Surge",
                    "url": "https://bloomberg.com/article2",
                    "content": "Technology stocks gain momentum"
                }
            ]
        }

        file_path = tmp_path / "test_search_results.json"
        with open(file_path, 'w') as f:
            json.dump(search_data, f)

        return str(file_path)

    def test_tool_initialization(self, image_finder):
        """Test that image finder initializes correctly"""
        assert image_finder.name == "enhanced_financial_image_finder"
        assert image_finder.description is not None

    def test_extract_urls_from_search_results(self, image_finder, sample_search_results_file):
        """Test URL extraction from search results file"""
        urls = image_finder._extract_urls_from_search_results(sample_search_results_file)

        assert isinstance(urls, list)
        assert len(urls) == 2
        assert "https://cnbc.com/article1" in urls
        assert "https://bloomberg.com/article2" in urls

    def test_extract_urls_prioritizes_cnbc(self, image_finder, tmp_path):
        """Test that CNBC URLs are prioritized"""
        search_data = {
            "articles": [
                {"url": "https://bloomberg.com/article1"},
                {"url": "https://cnbc.com/article2"},
                {"url": "https://reuters.com/article3"},
                {"url": "https://cnbc.com/article4"}
            ]
        }

        file_path = tmp_path / "test_results.json"
        with open(file_path, 'w') as f:
            json.dump(search_data, f)

        urls = image_finder._extract_urls_from_search_results(str(file_path))

        assert urls[0] == "https://cnbc.com/article2"
        assert urls[1] == "https://cnbc.com/article4"

    def test_extract_urls_missing_file(self, image_finder):
        """Test behavior when search results file is missing"""
        urls = image_finder._extract_urls_from_search_results("nonexistent_file.json")

        assert isinstance(urls, list)
        assert len(urls) == 0

    def test_extract_h1_from_url_success(self, image_finder):
        """Test extracting h1 heading from URL"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<html><h1>Test Article Title</h1></html>'
            mock_get.return_value = mock_response

            title = image_finder._extract_h1_from_url("https://example.com")

            assert title == "Test Article Title"

    def test_extract_h1_from_url_no_h1(self, image_finder):
        """Test fallback when no h1 tag is found"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<html><p>No heading here</p></html>'
            mock_get.return_value = mock_response

            title = image_finder._extract_h1_from_url("https://example.com")

            assert "Financial chart from example.com" in title

    def test_extract_h1_from_url_http_error(self, image_finder):
        """Test handling of HTTP errors"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            title = image_finder._extract_h1_from_url("https://example.com")

            assert "Financial chart from example.com" in title

    @patch('src.financial_market_summary.tools.image_finder.sync_playwright')
    def test_capture_chart_screenshot_success(self, mock_playwright, image_finder, tmp_path):
        """Test successful chart screenshot capture"""
        # Mock Playwright components
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_element = MagicMock()

        mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.query_selector_all.return_value = [mock_element]
        mock_element.bounding_box.return_value = {'width': 800, 'height': 600}
        mock_element.screenshot.return_value = b'fake_image_data'

        # Temporarily change output directory
        with patch.object(Path, 'resolve') as mock_resolve:
            mock_resolve.return_value = tmp_path
            result = image_finder._capture_chart_screenshot("https://cnbc.com/article")

        if result:
            assert isinstance(result, dict)
            assert 'url' in result
            assert 'title' in result
            assert result['type'] == 'screenshot'

    def test_run_with_missing_search_file(self, image_finder):
        """Test _run with missing search results file"""
        result = image_finder._run(
            search_content="test content",
            max_images=1,
            search_results_file=""
        )

        result_data = json.loads(result)
        assert isinstance(result_data, list)
        assert len(result_data) == 0

    def test_run_with_no_urls(self, image_finder, tmp_path):
        """Test _run when no article URLs are found"""
        
        search_data = {"articles": []}
        file_path = tmp_path / "empty_results.json"
        with open(file_path, 'w') as f:
            json.dump(search_data, f)

        result = image_finder._run(
            search_content="test content",
            max_images=1,
            search_results_file=str(file_path)
        )

        result_data = json.loads(result)
        assert isinstance(result_data, list)
        assert len(result_data) == 0

    @pytest.mark.parametrize("max_images", [1, 2, 3])
    def test_max_images_parameter(self, image_finder, sample_search_results_file, max_images):
        """Test that max_images parameter is respected"""
        with patch.object(image_finder, '_extract_screenshots_from_urls', return_value=[]):
            result = image_finder._run(
                search_content="test",
                max_images=max_images,
                search_results_file=sample_search_results_file
            )

            result_data = json.loads(result)
            assert isinstance(result_data, list)

    def test_crawl_article_with_crawl4ai_success(self, image_finder):
        """Test Crawl4AI article text extraction"""
        with patch('crawl4ai.AsyncWebCrawler') as mock_crawler:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.cleaned_html = '<html><p>Stock market gains today with strong performance.</p></html>'

            async def mock_arun(url):
                return mock_result

            mock_crawler.return_value.__aenter__.return_value.arun = mock_arun

            paragraphs = image_finder._crawl_article_with_crawl4ai("https://example.com")

            
            assert isinstance(paragraphs, list)

    def test_save_image_results(self, image_finder, tmp_path):
        """Test saving image results to JSON"""
        with patch.object(Path, 'resolve') as mock_resolve:
            mock_resolve.return_value = tmp_path

            extracted_images = [
                {
                    'url': '/path/to/image.png',
                    'title': 'Test Chart',
                    'source': 'cnbc.com',
                    'telegram_compatible': True
                }
            ]

            result_path = image_finder._save_image_results(extracted_images, "test search")

           
            assert isinstance(result_path, str)

    def test_extract_screenshots_from_urls(self, image_finder):
        """Test extracting screenshots from URLs"""
        article_urls = [
            "https://cnbc.com/article1",
            "https://bloomberg.com/article2"
        ]

        with patch.object(image_finder, '_capture_chart_screenshot') as mock_capture:
            mock_capture.return_value = {
                'url': '/path/to/screenshot.png',
                'title': 'Test Chart',
                'source': 'cnbc.com'
            }

            result = image_finder._extract_screenshots_from_urls(article_urls, "test content", max_images=1)

            assert isinstance(result, list)
            assert len(result) <= 1  
            assert mock_capture.called

    def test_extract_screenshots_from_urls_no_results(self, image_finder):
        """Test screenshot extraction when no charts found"""
        article_urls = ["https://example.com/article"]

        with patch.object(image_finder, '_capture_chart_screenshot', return_value=None):
            result = image_finder._extract_screenshots_from_urls(article_urls, "test content", max_images=1)

            assert isinstance(result, list)
            assert len(result) == 0

    def test_generate_ai_descriptions(self, image_finder, tmp_path):
        """Test AI description generation for images"""
       
        image_path = tmp_path / "test_chart.png"
        image_path.write_bytes(b'fake_image_data')

        extracted_images = [
            {
                'url': str(image_path),
                'source_article': 'https://cnbc.com/article',
                'image_description': ''
            }
        ]

        with patch.object(image_finder, '_crawl_article_with_crawl4ai', return_value=['Test paragraph about stocks']):
            with patch.object(image_finder, '_ai_extract_chart_description', return_value='Test description'):
                result = image_finder._generate_ai_descriptions(extracted_images, "test content")

                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0]['image_description'] == 'Test description'

    def test_generate_ai_descriptions_no_source_article(self, image_finder):
        """Test AI description generation without source article"""
        extracted_images = [
            {
                'url': '/path/to/image.png',
                'source_article': '',
                'image_description': ''
            }
        ]

        result = image_finder._generate_ai_descriptions(extracted_images, "test content")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['image_description'] == ''

    def test_ai_extract_chart_description(self, image_finder, tmp_path):
        """Test AI chart description extraction"""
        image_path = tmp_path / "test_chart.png"
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)

        paragraphs = [
            "The S&P 500 rose 2.5% today, reaching new highs.",
            "Tech stocks led the gains with strong trading volume."
        ]

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                mock_response = MagicMock()
                mock_response.text = "The S&P 500 rose 2.5% today, reaching new highs."
                mock_model.return_value.generate_content.return_value = mock_response

                result = image_finder._ai_extract_chart_description(str(image_path), paragraphs)

                assert isinstance(result, str)

    def test_ai_extract_chart_description_no_paragraphs(self, image_finder, tmp_path):
        """Test AI description extraction with no paragraphs"""
        image_path = tmp_path / "test_chart.png"
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)

        result = image_finder._ai_extract_chart_description(str(image_path), [])

        assert result == ""

    def test_verify_description_matches_image(self, image_finder, tmp_path):
        """Test image description verification"""
        image_path = tmp_path / "test_chart.png"
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)

        description = "The S&P 500 rose 2.5% today"

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                mock_response = MagicMock()
                mock_response.text = "YES"
                mock_model.return_value.generate_content.return_value = mock_response

                result = image_finder._verify_description_matches_image(str(image_path), description)

                assert isinstance(result, bool)

    def test_extract_description_from_text(self, image_finder):
        """Test extracting description from article text"""
        article_text = """
        Breaking News

        The stock market rose sharply today with the S&P 500 gaining 125 points or 2.5 percent.

        Investors cheered strong economic data and corporate earnings reports.
        """

        result = image_finder._extract_description_from_text(article_text, "stock market")

        assert isinstance(result, str)
        
        if result:
            assert "stock" in result.lower() or "market" in result.lower()

    def test_extract_description_from_text_no_match(self, image_finder):
        """Test description extraction when no financial content found"""
        article_text = "This is not a financial article. It contains no market data."

        result = image_finder._extract_description_from_text(article_text, "stock market")

        assert result == ""

    def test_ai_select_best_paragraph(self, image_finder):
        """Test AI paragraph selection"""
        paragraphs = [
            "The company announced new products today.",
            "The S&P 500 rose 2.5% with strong trading volume of 12 billion shares.",
            "Weather forecast shows sunny conditions ahead."
        ]

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                mock_response = MagicMock()
                mock_response.text = "2"
                mock_model.return_value.generate_content.return_value = mock_response

                result = image_finder._ai_select_best_paragraph(paragraphs, "stock market")

                assert isinstance(result, str)
                assert "S&P 500" in result or result in paragraphs

    def test_ai_select_best_paragraph_invalid_selection(self, image_finder):
        """Test AI paragraph selection with invalid response"""
        paragraphs = ["Paragraph 1", "Paragraph 2"]

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                mock_response = MagicMock()
                mock_response.text = "invalid"
                mock_model.return_value.generate_content.return_value = mock_response

                result = image_finder._ai_select_best_paragraph(paragraphs, "test")

                assert result == ""

    def test_extract_keywords(self, image_finder):
        """Test keyword extraction from content"""
        content = "US stock market financial news for today and tomorrow"

        result = image_finder._extract_keywords(content)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should extract meaningful words, excluding common words
        assert "stock" in result or "market" in result or "financial" in result


class TestImageFinderIntegration:
    """Integration tests (require actual environment)"""

    @pytest.mark.skipif(
        not os.getenv('GOOGLE_API_KEY'),
        reason="GOOGLE_API_KEY not set, skipping integration tests"
    )
    def test_live_image_extraction(self):
        """Test actual image extraction using real Tavily search results from test_1"""
       
        project_root = Path(__file__).resolve().parent.parent
        search_results_dir = project_root / "output" / "search_results"

        if not search_results_dir.exists():
            pytest.skip("No search results directory found - run test_1_tavily_search.py first")

        
        search_files = list(search_results_dir.glob("search_results_*.json"))
        if not search_files:
            pytest.skip("No search results files found - run test_1_tavily_search.py first")

        
        latest_file = max(search_files, key=lambda p: p.stat().st_mtime)

        
        with open(latest_file, 'r', encoding='utf-8') as f:
            search_data = json.load(f)

        if not search_data.get('articles'):
            pytest.skip("Search results file has no articles")

        image_finder = EnhancedImageFinder()
        result = image_finder._run(
            search_content="US stock market financial news",
            max_images=1,
            search_results_file=str(latest_file)
        )

        result_data = json.loads(result)
        assert isinstance(result_data, list)

        
        if len(result_data) > 0:
            assert 'url' in result_data[0]
            assert 'source' in result_data[0]
            print(f"\nImage extracted from: {result_data[0]['source']}")
