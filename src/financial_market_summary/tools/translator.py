from crewai.tools import BaseTool
from typing import Dict, Type
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import logging
import os
import re
import time
import requests

logger = logging.getLogger(__name__)

class TranslatorInput(BaseModel):
    """Input schema for the financial translator tool."""
    text: str = Field(..., description="The text content to be translated.")
    target_language: str = Field(
        ...,
        description="The target language for translation (arabic, hindi, hebrew).",
    )

class MultiLanguageTranslator(BaseTool):
    """
    A CrewAI tool for translating financial content while preserving key terminology.
    """
    name: str = "financial_translator"
    description: str = "Translate financial content to Arabic, Hindi, or Hebrew while preserving financial terminology."
    args_schema: Type[BaseModel] = TranslatorInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gemini_llm = None

    def _init_gemini_with_retry(self, max_retries: int = 3):
        """Initializes the Gemini LLM with retry logic."""
        for attempt in range(max_retries):
            try:
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    logger.error("GOOGLE_API_KEY not found.")
                    return
                self._gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.2,
                    google_api_key=google_api_key,
                    request_timeout=60,
                    max_retries=2,
                )
                self._gemini_llm.invoke([HumanMessage(content="Hello")])
                logger.info("Gemini LLM initialized successfully.")
                return
            except Exception as e:
                logger.warning(
                    f"Gemini init attempt {attempt + 1} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    logger.error("Failed to initialize Gemini LLM after all attempts.")

    def _run(self, text: str, target_language: str) -> str:
        """
        Main execution method to translate financial text.
        """
        try:
            if not self._gemini_llm:
                self._init_gemini_with_retry()
            if not self._gemini_llm:
                return f"Error: Gemini LLM not available. Cannot translate to {target_language}."

            supported_languages = ["arabic", "hindi", "hebrew"]
            if target_language.lower() not in supported_languages:
                return f"Error: Unsupported language '{target_language}'. Supported languages are: {', '.join(supported_languages)}"

            processed_text, _ = self._preprocess_text(text)
            translated_result = self._translate_with_retry(processed_text, target_language)

            if not translated_result.startswith("Error:"):
                logger.info(f"Successfully translated to {target_language}.")
                return translated_result
            else:
                return translated_result

        except Exception as e:
            error_msg = f"Translation error for {target_language}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _preprocess_text(self, text: str) -> tuple[str, Dict[str, str]]:
        """Simple preprocessing - no complex Unicode handling."""
        return text, {}

    def _translate_with_retry(self, text: str, target_language: str) -> str:
        """
        Handles the translation request with a retry mechanism.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 5 + (attempt * 2)
                    logger.info(f"Translation retry attempt {attempt + 1}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)

                result = self._translate_with_gemini(text, target_language)

                if "Error:" not in result:
                    return result
                else:
                    logger.warning(f"Translation attempt {attempt + 1} failed: {result}")
                    if attempt == max_retries - 1:
                        return result
            except requests.exceptions.HTTPError as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning(f"Rate limit or quota error detected: {e}. Retrying...")
                    if attempt == max_retries - 1:
                        return f"Error: Translation failed after {max_retries} attempts due to rate limit/quota."
                    time.sleep(10 + (attempt * 5))
                else:
                    logger.error(f"HTTP error during translation: {e}")
                    return f"Error: HTTP translation failed - {e}"
            except Exception as e:
                logger.error(f"Unexpected error during translation: {e}")
                return f"Error: Unexpected translation failed - {e}"
        return f"Error: Translation failed after {max_retries} attempts."

    def _translate_with_gemini(self, text: str, target_language: str) -> str:
        """
        Constructs and sends the translation request to the Gemini LLM.
        """
        try:
            language_codes = {
                "arabic": "Arabic (العربية)",
                "hindi": "Hindi (हिन्दी)",
                "hebrew": "Hebrew (עברית)",
            }
            target_lang_name = language_codes[target_language.lower()]
            system_prompt = f"""You are a professional financial translator. Translate to {target_lang_name}.
RULES:
1. Translate the text simply and directly.
2. Use professional financial terms.
3. Keep stock symbols in English (AAPL, MSFT, etc.).
4. Keep numbers and percentages unchanged.
5. Keep the translation clear and easy to understand."""
            user_prompt = f"Translate this financial content to {target_lang_name}:\n\n{text}"
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = self._gemini_llm.invoke(messages)
            translated_text = response.content.strip()
            if len(translated_text) < 10:
                return f"Error: Translation is too short for {target_language}."
            return translated_text
        except Exception as e:
            return f"Error: Gemini translation failed - {e}"