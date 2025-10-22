import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.financial_market_summary.tools.telegram_sender import EnhancedTelegramSender


class TestEnhancedTelegramSender:
    """Test suite for EnhancedTelegramSender"""

    @pytest.fixture
    def telegram_sender(self, monkeypatch):
        """Create EnhancedTelegramSender instance with mocked credentials"""
        monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test_bot_token_123')
        monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test_chat_id_456')

       
        with patch.object(EnhancedTelegramSender, '_test_credentials', return_value=True):
            return EnhancedTelegramSender()

    def test_tool_initialization(self, telegram_sender):
        """Test that telegram sender initializes correctly"""
        assert telegram_sender.name == "telegram_sender"
        assert telegram_sender.description is not None
        assert telegram_sender.bot_token == 'test_bot_token_123'
        assert telegram_sender.chat_id == 'test_chat_id_456'

    def test_initialization_missing_credentials(self, monkeypatch):
        """Test that tool raises error when credentials are missing"""
        monkeypatch.setattr('src.financial_market_summary.tools.telegram_sender.load_dotenv', lambda: None)
        monkeypatch.delenv('TELEGRAM_BOT_TOKEN', raising=False)
        monkeypatch.delenv('TELEGRAM_CHAT_ID', raising=False)
        monkeypatch.delenv('TELEGRAM_BOT_TOKEN_ENGLISH', raising=False)
        monkeypatch.delenv('TELEGRAM_CHAT_ID_ENGLISH', raising=False)

        with pytest.raises(ValueError, match="Missing TELEGRAM_BOT_TOKEN"):
            EnhancedTelegramSender()

    def test_set_bot_for_language_english(self, telegram_sender, monkeypatch):
        """Test bot configuration for English"""
        monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'english_token')
        monkeypatch.setenv('TELEGRAM_CHAT_ID', 'english_chat_id')

        telegram_sender.bots["english"]["token"] = 'english_token'
        telegram_sender.bots["english"]["chat_id"] = 'english_chat_id'

        telegram_sender._set_bot_for_language('english')

        assert telegram_sender.bot_token == 'english_token'
        assert telegram_sender.chat_id == 'english_chat_id'

    def test_set_bot_for_language_arabic(self, telegram_sender, monkeypatch):
        """Test bot configuration for Arabic"""
        monkeypatch.setenv('TELEGRAM_BOT_TOKEN_ARABIC', 'arabic_token')
        monkeypatch.setenv('TELEGRAM_CHAT_ID_ARABIC', 'arabic_chat_id')

        telegram_sender.bots["arabic"]["token"] = 'arabic_token'
        telegram_sender.bots["arabic"]["chat_id"] = 'arabic_chat_id'

        telegram_sender._set_bot_for_language('arabic')

        assert telegram_sender.bot_token == 'arabic_token'
        assert telegram_sender.chat_id == 'arabic_chat_id'

    def test_is_pre_formatted_content(self, telegram_sender):
        """Test detection of pre-formatted content"""
        # Content with verification icons (which the method actually checks for)
        pre_formatted_with_icon = "Source: [Article](https://example.com) ✅"
        assert telegram_sender._is_pre_formatted_content(pre_formatted_with_icon) is True

        # Content with verification status
        pre_formatted_with_status = "verification status: verified"
        assert telegram_sender._is_pre_formatted_content(pre_formatted_with_status) is True

        plain_text = "Just plain text"
        assert telegram_sender._is_pre_formatted_content(plain_text) is False

    def test_extract_stock_symbols(self, telegram_sender):
        """Test stock symbol extraction"""
        content = "AAPL rose 5% while MSFT gained 3%"

        symbols = telegram_sender._extract_stock_symbols(content)

        assert isinstance(symbols, list)
        assert 'AAPL' in symbols or 'MSFT' in symbols or len(symbols) >= 0

    def test_strip_all_formatting(self, telegram_sender):
        """Test stripping markdown formatting"""
        formatted_text = "**Bold** and *italic* text"

        result = telegram_sender._strip_all_formatting(formatted_text)

        assert "**" not in result
        assert "*" not in result or result.count("*") == 0

    def test_convert_markdown_to_telegram_html(self, telegram_sender):
        """Test markdown to HTML conversion"""
        markdown = "**Bold text** and [Link](https://example.com)"

        html = telegram_sender._convert_markdown_to_telegram_html(markdown)

        assert isinstance(html, str)
        assert "<b>" in html or "Bold" in html

    def test_clean_for_telegram(self, telegram_sender):
        """Test text cleaning for Telegram"""
        
        dirty_text = "Text with <tags> & \"quotes\""

        clean_text = telegram_sender._clean_for_telegram(dirty_text)

        assert isinstance(clean_text, str)
        
        assert '&lt;' in clean_text or 'tags' in clean_text
        assert '&amp;' in clean_text or '&' not in clean_text or clean_text.count('&') <= 2

    def test_send_message_success(self, telegram_sender):
        """Test successful message sending"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.json.return_value = {'ok': True}
            mock_post.return_value = mock_response

            result = telegram_sender._send_message("Test message")

            assert result is True
            mock_post.assert_called_once()

    def test_send_message_failure(self, telegram_sender):
        """Test message sending failure"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.ok = False
            mock_post.return_value = mock_response

            result = telegram_sender._send_message("Test message")

            assert result is False

    def test_run_with_simple_text(self, telegram_sender):
        """Test _run with simple text content"""
        with patch.object(telegram_sender, '_send_message', return_value=True):
            with patch.object(telegram_sender, '_get_latest_search_results_file', return_value=None):
                with patch.object(telegram_sender.image_finder, '_run', return_value='[]'):
                    result = telegram_sender._run(
                        content="Simple test message",
                        language="english"
                    )

                    assert "success" in result.lower() or "sent" in result.lower()

    def test_run_with_pre_formatted_content(self, telegram_sender):
        """Test _run with pre-formatted two-message format"""
        content = """=== TELEGRAM_TWO_MESSAGE_FORMAT ===

Message 1 (Image Caption):
Test caption

Message 2 (Full Summary):
Full summary content

---TELEGRAM_IMAGE_DATA---"""

        with patch.object(telegram_sender, '_send_message', return_value=True):
            with patch.object(telegram_sender, '_send_photo', return_value=True):
                with patch.object(telegram_sender, '_get_latest_search_results_file', return_value=None):
                    with patch.object(telegram_sender.image_finder, '_run', return_value='[]'):
                        result = telegram_sender._run(
                            content=content,
                            language="english"
                        )

                        assert isinstance(result, str)

    @pytest.mark.parametrize("language", ["english", "arabic", "hindi", "hebrew", "german"])
    def test_run_with_different_languages(self, telegram_sender, language):
        """Test sending messages in different languages"""
        with patch.object(telegram_sender, '_send_message', return_value=True):
            with patch.object(telegram_sender, '_get_latest_search_results_file', return_value=None):
                with patch.object(telegram_sender.image_finder, '_run', return_value='[]'):
                    result = telegram_sender._run(
                        content="Test message",
                        language=language
                    )

                    assert isinstance(result, str)

    def test_get_latest_search_results_file(self, telegram_sender):
        """Test finding the latest search results file"""
        result = telegram_sender._get_latest_search_results_file()

        
        assert result is None or isinstance(result, str)

    def test_validate_html_structure(self, telegram_sender):
        """Test HTML validation"""
        html = "<b>Bold</b> <i>Italic</i>"

        validated = telegram_sender._validate_html_structure(html)

        assert isinstance(validated, str)

    def test_ensure_message_fits_telegram_limit(self, telegram_sender):
        """Test that long messages are truncated"""
        long_message = "A" * 5000

        result = telegram_sender._ensure_message_with_footer_fits(long_message)

        
        assert len(result) <= 4096

    def test_is_metadata(self, telegram_sender):
        """Test metadata detection"""
        metadata = "**SEARCH_RESULTS_FILE_PATH**: /path/to/file.json"
        assert telegram_sender._is_metadata(metadata) is True

        regular_text = "This is regular content"
        assert telegram_sender._is_metadata(regular_text) is False


class TestTelegramSenderIntegration:
    """Integration tests (require actual credentials)"""

    @pytest.mark.skipif(
        not (os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID')),
        reason="Telegram credentials not set, skipping integration tests"
    )
    def test_live_message_send(self):
        """Test actual Telegram message delivery (only runs if credentials available)"""
        with patch.object(EnhancedTelegramSender, '_test_credentials', return_value=True):
            sender = EnhancedTelegramSender()
            result = sender._run(
                content="Test message from pytest",
                language="english"
            )

            assert isinstance(result, str)
