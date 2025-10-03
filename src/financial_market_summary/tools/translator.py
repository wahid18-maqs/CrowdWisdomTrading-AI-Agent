import os
import logging
from typing import Optional
from crewai.tools import BaseTool
from pydantic import Field
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TranslatorTool(BaseTool):
    name: str = "financial_translator"
    description: str = """Translates financial market summaries to Arabic, Hindi, or Hebrew.

    CRITICAL: Preserves all financial data unchanged:
    - Stock symbols (AAPL, MSFT, TSLA, etc.)
    - Numbers (percentages, prices, dates)
    - HTML formatting (<b>, <a>, etc.)
    - URLs and links
    - Two-message format structure

    Input: English content in two-message format
    Language: 'arabic', 'hindi', or 'hebrew'
    Output: Translated content maintaining exact same structure"""

    api_key: Optional[str] = Field(default=None)
    model: Optional[any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("✅ Translator tool initialized with Gemini")

    def _run(self, content: str, target_language: str) -> str:
        """
        Translate financial content to target language

        Args:
            content: English content in two-message format
            target_language: 'arabic', 'hindi', or 'hebrew'

        Returns:
            Translated content in same format
        """
        try:
            logger.info(f"🌍 Starting translation to {target_language}")

            # Validate language
            supported_languages = ['arabic', 'hindi', 'hebrew']
            if target_language.lower() not in supported_languages:
                logger.error(f"❌ Unsupported language: {target_language}")
                return f"Error: Unsupported language. Use: {', '.join(supported_languages)}"

            # Extract Message 1 and Message 2
            if "=== TELEGRAM_TWO_MESSAGE_FORMAT ===" not in content:
                logger.warning("⚠️ Content not in two-message format, translating as-is")
                return self._translate_text(content, target_language)

            # Split content into sections
            parts = content.split("=== TELEGRAM_TWO_MESSAGE_FORMAT ===")
            if len(parts) < 2:
                return self._translate_text(content, target_language)

            message_section = parts[1].strip()

            # Extract Message 1 and Message 2
            lines = message_section.split('\n')
            message1 = []
            message2 = []
            current_section = None

            for line in lines:
                if line.startswith("Message 1 (Image Caption):"):
                    current_section = "message1"
                    continue
                elif line.startswith("Message 2 (Full Summary):"):
                    current_section = "message2"
                    continue
                elif line.startswith("---TELEGRAM_IMAGE_DATA---"):
                    break

                if current_section == "message1":
                    message1.append(line)
                elif current_section == "message2":
                    message2.append(line)

            message1_text = '\n'.join(message1).strip()
            message2_text = '\n'.join(message2).strip()

            logger.info(f"📝 Extracted Message 1: {len(message1_text)} chars")
            logger.info(f"📝 Extracted Message 2: {len(message2_text)} chars")

            # Translate each message
            logger.info(f"🔄 Translating Message 1...")
            translated_message1 = self._translate_text(message1_text, target_language)

            logger.info(f"🔄 Translating Message 2...")
            translated_message2 = self._translate_text(message2_text, target_language)

            # Reconstruct in same format
            translated_content = "=== TELEGRAM_TWO_MESSAGE_FORMAT ===\n"
            translated_content += f"Message 1 (Image Caption):\n{translated_message1}\n\n"
            translated_content += f"Message 2 (Full Summary):\n{translated_message2}\n"

            logger.info(f"✅ Translation to {target_language} completed")
            return translated_content

        except Exception as e:
            logger.error(f"❌ Translation failed: {e}")
            return f"Translation error: {str(e)}"

    def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text using Gemini while preserving financial data"""
        try:
            # Language-specific instructions
            language_instructions = {
                'arabic': """Translate to Modern Standard Arabic (العربية الفصحى).
                Use right-to-left (RTL) text formatting.
                Preserve all English stock symbols and numbers exactly as they appear.""",

                'hindi': """Translate to Hindi (हिन्दी) using Devanagari script.
                Keep formal tone suitable for financial news.
                Preserve all English stock symbols and numbers exactly as they appear.""",

                'hebrew': """Translate to Modern Hebrew (עברית).
                Use right-to-left (RTL) text formatting.
                Preserve all English stock symbols and numbers exactly as they appear."""
            }

            prompt = f"""You are a professional financial translator. Translate the following financial content to {target_language.upper()}.

CRITICAL RULES:
1. DO NOT translate stock symbols (AAPL, MSFT, TSLA, etc.) - keep them in English
2. DO NOT change any numbers, percentages, or dates
3. PRESERVE all HTML tags exactly: <b>, </b>, <a>, </a>, etc.
4. PRESERVE all URLs and links unchanged
5. Keep emojis in their original positions
6. Maintain the exact same structure and formatting

{language_instructions.get(target_language, '')}

CONTENT TO TRANSLATE:
{text}

IMPORTANT: Return ONLY the translated text. Do NOT add explanations or notes."""

            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,  # Low temperature for consistent translation
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 2000,
                }
            )

            translated = response.text.strip()

            # Verify translation preserved key elements
            if not self._verify_translation(text, translated):
                logger.warning("⚠️ Translation may have lost some financial data")

            return translated

        except Exception as e:
            logger.error(f"❌ Text translation failed: {e}")
            return f"Translation error: {str(e)}"

    def _verify_translation(self, original: str, translated: str) -> bool:
        """Verify that translation preserved critical elements"""
        import re

        # Extract stock symbols from original
        stock_symbols = re.findall(r'\b[A-Z]{2,5}\b', original)

        # Check if major symbols are preserved
        preserved_count = sum(1 for symbol in stock_symbols if symbol in translated)

        if stock_symbols and preserved_count == 0:
            logger.warning("⚠️ No stock symbols found in translation")
            return False

        return True


# For backward compatibility
class Translator(TranslatorTool):
    """Alias for TranslatorTool"""
    pass
