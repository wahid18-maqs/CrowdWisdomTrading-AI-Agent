# Fixed translator.py with better error handling and rate limiting
from crewai.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
import os
import logging
import re
import time
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
        self._gemini_llm = None
        self._init_gemini_with_retry()
    
    def _init_gemini_with_retry(self, max_retries=3):
        """Initialize Gemini with retry logic"""
        for attempt in range(max_retries):
            try:
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    logger.error("GOOGLE_API_KEY not found in environment variables")
                    return
                
                self._gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.2,  # Lower temperature for more consistent translations
                    google_api_key=google_api_key,
                    request_timeout=60,
                    # Add retry configuration
                    max_retries=2
                )
                
                # Test the connection with a simple call
                test_response = self._gemini_llm.invoke([HumanMessage(content="Hello")])
                logger.info("Gemini LLM initialized and tested successfully for translation")
                return
                
            except Exception as e:
                logger.warning(f"Gemini initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Failed to initialize Gemini LLM after all attempts")
    
    def _run(self, text: str, target_language: str) -> str:
        """Translate text to target language with financial context"""
        try:
            if not self._gemini_llm:
                # Try to reinitialize
                self._init_gemini_with_retry()
                if not self._gemini_llm:
                    return f"Error: Gemini LLM not available. Cannot translate to {target_language}"
            
            # Validate target language
            supported_languages = ['arabic', 'hindi', 'hebrew']
            if target_language.lower() not in supported_languages:
                return f"Error: Unsupported language '{target_language}'. Supported: {supported_languages}"
            
            # Preprocess text to preserve important elements
            processed_text, preserved_elements = self._preprocess_text(text)
            
            # Perform translation with retry logic
            result = self._translate_with_retry(processed_text, target_language, max_retries=3)
            
            if not result.startswith("Error:"):
                # Restore preserved elements
                final_result = self._restore_preserved_elements(result, preserved_elements)
                logger.info(f"Successfully translated to {target_language}")
                return final_result
            else:
                return result
            
        except Exception as e:
            error_msg = f"Translation error for {target_language}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _preprocess_text(self, text: str) -> tuple[str, Dict[str, str]]:
        """Preprocess text to preserve financial elements"""
        preserved_elements = {}
        processed_text = text
        
        # Preserve stock symbols
        stock_symbols = re.findall(r'\b[A-Z]{2,5}\b', text)
        for i, symbol in enumerate(stock_symbols):
            placeholder = f"__STOCK_SYMBOL_{i}__"
            preserved_elements[placeholder] = symbol
            processed_text = processed_text.replace(symbol, placeholder, 1)
        
        # Preserve currency values
        currency_values = re.findall(r'\$[\d,]+\.?\d*', text)
        for i, value in enumerate(currency_values):
            placeholder = f"__CURRENCY_{i}__"
            preserved_elements[placeholder] = value
            processed_text = processed_text.replace(value, placeholder, 1)
        
        # Preserve percentages
        percentages = re.findall(r'[\d,]+\.?\d*%', text)
        for i, percent in enumerate(percentages):
            placeholder = f"__PERCENT_{i}__"
            preserved_elements[placeholder] = percent
            processed_text = processed_text.replace(percent, placeholder, 1)
        
        return processed_text, preserved_elements
    
    def _restore_preserved_elements(self, translated_text: str, preserved_elements: Dict[str, str]) -> str:
        """Restore preserved elements after translation"""
        result = translated_text
        for placeholder, original_value in preserved_elements.items():
            result = result.replace(placeholder, original_value)
        return result
    
    def _translate_with_retry(self, text: str, target_language: str, max_retries: int = 3) -> str:
        """Translate using Gemini with retry logic"""
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 5 + (attempt * 2)  # Increasing wait time
                    logger.info(f"Translation retry attempt {attempt + 1} for {target_language}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                
                result = self._translate_with_gemini(text, target_language)
                
                if not result.startswith("Error:"):
                    return result
                else:
                    logger.warning(f"Translation attempt {attempt + 1} failed: {result}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return result
                        
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Translation attempt {attempt + 1} exception: {error_msg}")
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = 10 + (attempt * 5)  # Longer wait for rate limits
                        logger.info(f"Rate limit detected, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                if attempt == max_retries - 1:
                    return f"Error: Translation failed after {max_retries} attempts - {error_msg}"
        
        return f"Error: Translation failed after {max_retries} attempts"
    
    def _translate_with_gemini(self, text: str, target_language: str) -> str:
        """Translate using Gemini with financial context"""
        try:
            language_codes = {
                'arabic': 'Arabic (العربية)',
                'hindi': 'Hindi (हिन्दी)', 
                'hebrew': 'Hebrew (עברית)'
            }
            
            target_lang_name = language_codes[target_language.lower()]
            
            # Shorter, more focused prompt to reduce token usage
            system_prompt = f"""You are a professional financial translator. Translate to {target_lang_name}.

RULES:
1. Keep stock symbols unchanged (AAPL, MSFT, etc.)
2. Preserve numbers, percentages, currency exactly
3. Keep markdown formatting (**, *, #)
4. Use professional financial terms
5. Keep English financial terms in () if needed"""

            user_prompt = f"Translate this financial content to {target_lang_name}:\n\n{text}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Make the API call
            response = self._gemini_llm.invoke(messages)
            translated_text = response.content.strip()
            
            # Quick validation
            if len(translated_text) < 10:
                return f"Error: Translation too short for {target_language}"
            
            return translated_text
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                return f"Error: Rate limit exceeded - {error_msg}"
            elif "quota" in error_msg.lower():
                return f"Error: API quota exceeded - {error_msg}"
            else:
                return f"Error: Gemini translation failed - {error_msg}"