# Fixed translator.py with proper Gemini integration
from crewai.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
import os
import logging
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class TranslatorInput(BaseModel):
    """Input schema for translator tool."""
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language (arabic, hindi, hebrew)")

class MultiLanguageTranslator(BaseTool):
    name: str = "financial_translator"
    description: str = "Translate financial content to Arabic, Hindi, or Hebrew while preserving financial terminology"
    args_schema: Type[BaseModel] = TranslatorInput

    def __init__(self):
        super().__init__()
        try:
            self._gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            logger.info("Gemini LLM initialized successfully for translation")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            self._gemini_llm = None
    
    def _run(self, text: str, target_language: str) -> str:
        """Translate text to target language with financial context"""
        try:
            if not self._gemini_llm:
                return f"Error: Gemini LLM not initialized. Cannot translate to {target_language}"
            
            # Validate target language
            supported_languages = ['arabic', 'hindi', 'hebrew']
            if target_language.lower() not in supported_languages:
                return f"Error: Unsupported language '{target_language}'. Supported: {supported_languages}"
            
            result = self._translate_with_gemini(text, target_language)
            
            if not result.startswith("Error:"):
                logger.info(f"Successfully translated to {target_language}")
                return result
            else:
                return result
            
        except Exception as e:
            error_msg = f"Translation error for {target_language}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _translate_with_gemini(self, text: str, target_language: str) -> str:
        """Translate using Gemini with financial context"""
        try:
            language_codes = {
                'arabic': 'Arabic',
                'hindi': 'Hindi', 
                'hebrew': 'Hebrew'
            }
            
            target_lang_name = language_codes[target_language.lower()]
            
            system_prompt = f"""You are a professional financial translator specializing in {target_lang_name}.

CRITICAL REQUIREMENTS:
1. Translate financial content accurately to {target_lang_name}
2. Keep stock symbols (AAPL, MSFT, etc.) UNCHANGED
3. Preserve all numbers, percentages, and currency values EXACTLY
4. Maintain markdown formatting (**, *, headers)
5. Use proper financial terminology in {target_lang_name}
6. If unsure about financial terms, keep English term in parentheses

This is a financial market summary for professional traders and investors."""

            user_prompt = f"""Translate this financial content to {target_lang_name}:

{text}

Remember:
- Keep stock symbols unchanged 
- Preserve numerical data exactly
- Use professional financial terminology
- Maintain markdown formatting"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self._gemini_llm.invoke(messages)
            translated_text = response.content.strip()
            
            # Post-process to ensure accuracy
            processed_text = self._post_process_translation(translated_text, text, target_language)
            
            return processed_text
            
        except Exception as e:
            return f"Error: Gemini translation failed - {str(e)}"
    
    def _post_process_translation(self, translated: str, original: str, target_language: str) -> str:
        """Post-process translation to preserve financial terms"""
        try:
            # Preserve stock symbols
            stock_symbols = re.findall(r'\b[A-Z]{2,5}\b', original)
            for symbol in stock_symbols:
                # Ensure symbols aren't altered
                translated = re.sub(f'\\b{re.escape(symbol)}\\w*\\b', symbol, translated, flags=re.IGNORECASE)
            
            # Preserve currency and percentages
            currency_patterns = [r'\$[\d,]+\.?\d*', r'[\d,]+\.?\d*%', r'[\d]+\.?\d*[MBK]?']
            for pattern in currency_patterns:
                original_values = re.findall(pattern, original)
                for value in original_values:
                    if value not in translated:
                        # Attempt to replace number without comma with the original value with comma
                        translated = translated.replace(value.replace(',', ''), value)
            
            return translated
            
        except Exception as e:
            logger.warning(f"Post-processing error: {e}")
            return translated