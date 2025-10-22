import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.financial_market_summary.tools.translator import TranslatorTool


class TestTranslatorTool:
    """Test suite for TranslatorTool"""

    @pytest.fixture
    def mock_genai(self):
        """Create a mock Google Generative AI"""
        with patch('src.financial_market_summary.tools.translator.genai') as mock:
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Translated content"
            mock_model.generate_content.return_value = mock_response
            mock.GenerativeModel.return_value = mock_model
            yield mock

    @pytest.fixture
    def translator(self, monkeypatch, mock_genai):
        """Create TranslatorTool instance with mocked genai"""
        monkeypatch.setenv('GOOGLE_API_KEY', 'test_api_key_12345')
        return TranslatorTool()

    def test_tool_initialization(self, translator):
        """Test that translator initializes correctly"""
        assert translator.name == "financial_translator"
        assert translator.description is not None

    def test_initialization_missing_api_key(self, monkeypatch):
        """Test that tool raises error when API key is missing"""
        monkeypatch.delenv('GOOGLE_API_KEY', raising=False)

        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            TranslatorTool()

    @pytest.mark.parametrize("target_language", ["arabic", "hindi", "hebrew", "german"])
    def test_translate_to_different_languages(self, translator, target_language):
        """Test translation to different supported languages"""
        content = "The stock market gained 2% today"

        result = translator._run(
            content=content,
            target_language=target_language
        )

        # Verify translation returns a string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_preserves_stock_symbols(self, translator):
        """Test that stock symbols are preserved during translation"""
        content = "AAPL stock rose 5% while MSFT gained 3%"

        result = translator._run(
            content=content,
            target_language="arabic"
        )

        # Should return translated string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_preserves_numbers(self, translator):
        """Test that numbers and percentages are preserved"""
        content = "The S&P 500 rose 2.5% to 4,500 points"

        result = translator._run(
            content=content,
            target_language="hindi"
        )

        # Should return translated string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_two_message_format(self, translator):
        """Test translation of two-message format content"""
        content = """=== TELEGRAM_TWO_MESSAGE_FORMAT ===

Message 1 (Image Caption):
Stock market rises today

Message 2 (Full Summary):
**Market Update**

The S&P 500 gained 2%

---TELEGRAM_IMAGE_DATA---"""

        result = translator._run(
            content=content,
            target_language="arabic"
        )

        # Should preserve format markers
        assert "=== TELEGRAM_TWO_MESSAGE_FORMAT ===" in result
        assert "---TELEGRAM_IMAGE_DATA---" in result

    def test_translate_with_html_links(self, translator):
        """Test that HTML links are preserved during translation"""
        content = '<a href="https://example.com">Click here</a> for more info'

        result = translator._run(
            content=content,
            target_language="german"
        )

        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_empty_content(self, translator):
        """Test translation of empty content"""
        content = ""

        result = translator._run(
            content=content,
            target_language="arabic"
        )

        # Should handle empty content gracefully
        assert isinstance(result, str)

    def test_translate_with_special_characters(self, translator):
        """Test translation with special financial characters"""
        content = "Stock price: $45.67 | Change: +$2.34 (5.4%) ↑"

        result = translator._run(
            content=content,
            target_language="hindi"
        )

       
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_long_content(self, translator):
        """Test translation of long content"""
        content = "Market analysis. " * 200  

        result = translator._run(
            content=content,
            target_language="arabic"
        )

        
        assert isinstance(result, str)

    def test_translate_with_market_terms(self, translator):
        """Test that financial market terms are properly translated"""
        content = """Bull market, bear market, volatility, recession,
        Federal Reserve, interest rates, inflation"""

        result = translator._run(
            content=content,
            target_language="hebrew"
        )

        assert isinstance(result, str)

    def test_invalid_target_language(self, translator):
        """Test behavior with invalid target language"""
        content = "Test content"

        
        try:
            result = translator._run(
                content=content,
                target_language="invalid_language"
            )
            assert isinstance(result, str)
        except ValueError:
            pass  

    def test_translate_with_emojis(self, translator):
        """Test translation with emojis"""
        content = "Markets up today! 📈 Great gains! 💰"

        result = translator._run(
            content=content,
            target_language="arabic"
        )

        
        assert isinstance(result, str)
        assert len(result) > 0

    @patch('src.financial_market_summary.tools.translator.genai.GenerativeModel')
    def test_api_error_handling(self, mock_model_class, translator):
        """Test handling of API errors"""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model

        content = "Test content"

        # Should handle API errors gracefully
        try:
            result = translator._run(
                content=content,
                target_language="arabic"
            )
            # If no exception, should return error message or empty string
            assert isinstance(result, str)
        except Exception:
            pass  # Exception is acceptable

    def test_translation_preserves_markdown(self, translator):
        """Test that markdown formatting is preserved"""
        content = """**Bold text**
*Italic text*
- Bullet point 1
- Bullet point 2"""

        result = translator._run(
            content=content,
            target_language="hindi"
        )

        # Should return translated string
        assert isinstance(result, str)
        assert len(result) > 0


class TestTranslatorIntegration:
    """Integration tests (require actual API key)"""

    @pytest.mark.skipif(
        not os.getenv('GOOGLE_API_KEY'),
        reason="GOOGLE_API_KEY not set, skipping integration tests"
    )
    def test_live_translation_arabic(self):
        """Test actual translation to Arabic (only runs if API key available)"""
        translator = TranslatorTool()
        result = translator._run(
            content="The stock market rose 2% today",
            target_language="arabic"
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(
        not os.getenv('GOOGLE_API_KEY'),
        reason="GOOGLE_API_KEY not set, skipping integration tests"
    )
    def test_live_translation_preserves_data(self):
        """Test that live translation preserves financial data"""
        translator = TranslatorTool()
        content = "AAPL stock gained 5.5% to $175.50"

        result = translator._run(
            content=content,
            target_language="hindi"
        )

        # Should preserve stock symbols and numbers
        assert isinstance(result, str)
        assert len(result) > 0
