import os
import logging
from typing import Optional, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TranslatorInput(BaseModel):
    """Input schema for financial translator tool"""
    content: str = Field(..., description="The English financial content to translate")
    target_language: str = Field(..., description="Target language: 'arabic', 'hindi', 'hebrew', or 'german'")


class TranslatorTool(BaseTool):
    name: str = "financial_translator"
    description: str = """Translates financial market summaries to Arabic, Hindi, Hebrew, or German.

    CRITICAL: Preserves all financial data unchanged:
    - Stock symbols (AAPL, MSFT, TSLA, etc.)
    - Numbers (percentages, prices, dates)
    - HTML formatting (<b>, <a>, etc.)
    - URLs and links
    - Two-message format structure

    Input: English content in two-message format
    Language: 'arabic', 'hindi', 'hebrew', or 'german'
    Output: Translated content maintaining exact same structure"""

    args_schema: Type[BaseModel] = TranslatorInput

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
            supported_languages = ['arabic', 'hindi', 'hebrew', 'german']
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

            # Validate extraction
            if not message2_text:
                logger.error(f"❌ Message 2 extraction failed for {target_language}!")
                return f"Translation error: Could not extract Message 2 content"

            # Translate each message
            logger.info(f"🔄 Translating Message 1...")
            translated_message1 = self._translate_text(message1_text, target_language) if message1_text else ""

            logger.info(f"🔄 Translating Message 2...")
            translated_message2 = self._translate_text(message2_text, target_language)

            # Reconstruct in same format - ALWAYS include format markers
            translated_content = "=== TELEGRAM_TWO_MESSAGE_FORMAT ===\n"
            translated_content += f"Message 1 (Image Caption):\n{translated_message1}\n\n"
            translated_content += f"Message 2 (Full Summary):\n{translated_message2}\n"
            translated_content += "\n---TELEGRAM_IMAGE_DATA---\n"

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
                'arabic': """Translate ALL text to Modern Standard Arabic (العربية الفصحى).
                Write in Arabic script from right-to-left.
                ONLY keep stock symbols (AAPL, MSFT, etc.) and numbers in English.
                Example: "Stock Market Today" becomes "سوق الأسهم اليوم" """,

                'hindi': """Translate ALL text to Hindi (हिन्दी) using Devanagari script.
                Write in Hindi script.
                ONLY keep stock symbols (AAPL, MSFT, etc.) and numbers in English.
                Example: "Stock Market Today" becomes "आज का शेयर बाजार" """,

                'hebrew': """Translate ALL text to Modern Hebrew (עברית).
                Write in Hebrew script from right-to-left.
                ONLY keep stock symbols (AAPL, MSFT, etc.) and numbers in English.
                Example: "Stock Market Today" becomes "שוק המניות היום"
                Use modern Hebrew, not biblical Hebrew.""",

                'german': """Translate ALL text to German (Deutsch).
                Use formal business German suitable for financial news.
                ONLY keep stock symbols (AAPL, MSFT, etc.) and numbers in English.
                Example: "Stock Market Today" becomes "Börse Heute"
                Use proper German capitalization for nouns."""
            }

            prompt = f"""You are a professional financial translator. Your job is to translate English financial content to {target_language.upper()}.

WHAT TO TRANSLATE (convert to {target_language} script):
✅ ALL regular English words and sentences
✅ Financial terms (market, stock, trading, etc.)
✅ News headlines and summaries

WHAT TO KEEP IN ENGLISH (do NOT translate):
❌ Stock ticker symbols: AAPL, MSFT, TSLA, GOOGL, etc.
❌ Numbers: 1.5%, $100, 5,000, etc.
❌ HTML tags: <b>, </b>, <a href="...">, </a>, <div>, etc.
❌ URLs: https://... links
❌ COMPLETE HTML LINK STRUCTURE: <a href="URL">Link Text</a> - The ENTIRE structure must be preserved EXACTLY, including the href attribute

CRITICAL HTML LINK PRESERVATION:
- You MUST keep the complete HTML link format: <a href="URL">Text</a>
- NEVER change it to: <a>URL</a>Text
- NEVER remove the href attribute
- Example: <a href="https://finance.yahoo.com/quote/%5EGSPC/chart/">S&P 500</a> must stay EXACTLY as-is
- Only translate the link text between > and </a>, NOT the URL or href attribute

{language_instructions.get(target_language, '')}

CONTENT TO TRANSLATE:
{text}

IMPORTANT: You MUST translate the text to {target_language} language and script. Return ONLY the translated text in {target_language}, no explanations."""

            # Add retry logic for quota errors
            max_retries = 3
            retry_delay = 15  # seconds
            translated = ""

            for attempt in range(max_retries):
                try:
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
                    break  # Success, exit retry loop

                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "quota" in error_str.lower() or "resource" in error_str.lower():
                        if attempt < max_retries - 1:
                            logger.warning(f"⚠️ Quota/rate limit error (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s...")
                            import time
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff (15s, 30s, 60s)
                        else:
                            logger.error(f"❌ Max retries reached for {target_language}, quota still exceeded")
                            raise
                    else:
                        raise  # Re-raise non-quota errors immediately

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
