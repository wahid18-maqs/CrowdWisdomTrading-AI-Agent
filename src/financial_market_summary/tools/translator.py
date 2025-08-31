from crewai.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import os
import logging
import requests
import json
import re

# New imports for Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class TranslatorInput(BaseModel):
    """Input schema for translator tool."""
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language (arabic, hindi, hebrew)")
    preserve_formatting: bool = Field(default=True, description="Preserve markdown formatting")

class MultiLanguageTranslator(BaseTool):
    name: str = "financial_translator"
    description: str = (
        "Translate financial content to Arabic, Hindi, or Hebrew while preserving "
        "financial terminology and formatting. Specialized for financial market content."
    )
    args_schema: Type[BaseModel] = TranslatorInput
    financial_glossary: Dict[str, Dict[str, str]] = {}
    
    # We no longer need this, as we're using a private attribute now.
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self):
        super().__init__()
        self.financial_glossary = self._load_financial_glossary()
        # Store the LLM as a private attribute to avoid Pydantic validation issues
        self._gemini_llm = ChatGoogleGenerativeAI( # Change 1: Use ChatGoogleGenerativeAI
            model="gemini-1.5-flash", # Change 2: Specify the Gemini model
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY") # Change 3: Use the correct environment variable
        )
    
    def _run(self, text: str, target_language: str, preserve_formatting: bool = True) -> str:
        """
        Translate text to target language with financial context
        """
        try:
            # Validate target language
            supported_languages = ['arabic', 'hindi', 'hebrew']
            if target_language.lower() not in supported_languages:
                return f"Error: Unsupported language '{target_language}'. Supported: {supported_languages}"
            
            # Use primary translation method (Gemini) only
            result = self._translate_with_gemini(text, target_language, preserve_formatting) # Change 4: Call the new method
            
            if "Error:" not in result:
                logger.info(f"Successfully translated to {target_language} using Gemini")
                return result
            
            # Return error if Gemini translation fails
            return f"Translation to {target_language} unavailable. Original text:\n\n{text}"
            
        except Exception as e:
            error_msg = f"Translation error for {target_language}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _translate_with_gemini(self, text: str, target_language: str, preserve_formatting: bool) -> str:
        """Translate using Gemini with financial context"""
        try:
            # Language mapping for API
            language_codes = {
                'arabic': 'Arabic',
                'hindi': 'Hindi', 
                'hebrew': 'Hebrew'
            }
            
            target_lang_name = language_codes[target_language.lower()]
            
            # Create specialized prompt for financial translation
            system_prompt = f"""You are a professional financial translator specializing in translating financial market content to {target_lang_name}.

CRITICAL REQUIREMENTS:
1. Translate the financial summary while preserving all financial terminology accuracy.
2. Keep stock symbols (like AAPL, MSFT) unchanged.
3. Preserve numbers, percentages, and currency values exactly.
4. Maintain the structure and formatting of the text.
5. Use appropriate financial terminology in {target_lang_name}.
6. If unsure about financial terms, provide the English term in parentheses.

Financial context: This is a daily financial market summary containing stock market news, trading activity, and market analysis."""

            user_prompt = f"""Translate this financial content to {target_lang_name}:

{text}

Requirements:
- Maintain financial accuracy
- Keep stock symbols unchanged 
- Preserve numerical data exactly
- Use proper financial terminology
- Preserve any markdown formatting (**, *, headers, etc.)
- Provide a natural, professional translation suitable for financial professionals"""

            # Make API call using ChatGoogleGenerativeAI
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self._gemini_llm.invoke(messages) # Change 5: Use the new LLM object
            
            translated_text = response.content.strip()
            
            # Post-process to ensure financial terms are preserved
            processed_text = self._post_process_translation(translated_text, text, target_language)
            
            return processed_text
            
        except Exception as e:
            error_msg = f"Gemini translation error: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def _post_process_translation(self, translated: str, original: str, target_language: str) -> str:
        """Post-process translation to fix financial terms and formatting"""
        try:
            
            # Extract stock symbols from original text (2-5 uppercase letters)
            stock_symbols = re.findall(r'\b[A-Z]{2,5}\b', original)
            
            # Ensure stock symbols remain unchanged in translation
            for symbol in stock_symbols:
                # Replace any translated version back to original symbol
                translated = re.sub(f'\\b{re.escape(symbol)}\\b', symbol, translated, flags=re.IGNORECASE)
            
            # Preserve currency symbols and numbers
            currency_patterns = [r'\$[\d,]+\.?\d*', r'[\d,]+\.?\d*%', r'[\d,]+\.?\d*[MBK]']
            for pattern in currency_patterns:
                original_values = re.findall(pattern, original)
                for value in original_values:
                    if value not in translated:
                        # Try to find where this value should be and ensure it's preserved
                        translated = translated.replace(value.replace('$', ''), value)
            
            # Add language header
            language_headers = {
                'arabic': 'ملخص السوق المالي اليومي',
                'hindi': 'दैनिक वित्तीय बाजार सारांश', 
                'hebrew': 'סיכום שוק הכספים היומי'
            }
            
            header = language_headers.get(target_language.lower(), '')
            if header and header not in translated:
                translated = f"📊 {header}\n{'='*30}\n\n{translated}"
            
            return translated
            
        except Exception as e:
            logger.warning(f"Post-processing error: {e}")
            return translated
    
    def _load_financial_glossary(self) -> Dict[str, Dict[str, str]]:
        """Load financial terminology glossary for better translations"""
        return {
            'arabic': {
                'stock market': 'سوق الأسهم',
                'trading': 'التداول', 
                'investment': 'الاستثمار',
                'portfolio': 'المحفظة',
                'earnings': 'الأرباح',
                'revenue': 'الإيرادات',
                'dividend': 'الأرباح الموزعة',
                'market cap': 'القيمة السوقية',
                'bull market': 'السوق الصاعد',
                'bear market': 'السوق الهابط'
            },
            'hindi': {
                'stock market': 'शेयर बाज़ार',
                'trading': 'व्यापार',
                'investment': 'निवेश',
                'portfolio': 'पोर्टफोलियो',
                'earnings': 'कमाई',
                'revenue': 'राजस्व',
                'dividend': 'लाभांश',
                'market cap': 'बाज़ार पूंजीकरण',
                'bull market': 'तेज़ी का बाज़ार',
                'bear market': 'मंदी का बाज़ार'
            },
            'hebrew': {
                'stock market': 'שוק המניות',
                'trading': 'מסחר',
                'investment': 'השקעה', 
                'portfolio': 'תיק השקעות',
                'earnings': 'רווחים',
                'revenue': 'הכנסות',
                'dividend': 'דיבידנד',
                'market cap': 'שווי שוק',
                'bull market': 'שוק עולה',
                'bear market': 'שוק יורד'
            }
        }
    
    def batch_translate(self, content_dict: Dict[str, str], target_languages: list) -> Dict[str, Dict[str, str]]:
        """Translate content to multiple languages"""
        results = {}
        
        for lang in target_languages:
            results[lang] = {}
            for key, text in content_dict.items():
                if text and text.strip():
                    translation = self._run(text, lang)
                    results[lang][key] = translation
                else:
                    results[lang][key] = f"No content to translate for {key}"
        
        return results
    
    def validate_translation(self, original: str, translated: str, target_language: str) -> Dict[str, Any]:
        """Basic validation of translation quality"""
        validation_results = {
            'length_ratio': len(translated) / len(original) if original else 0,
            'has_financial_terms': False,
            'preserved_numbers': True,
            'quality_score': 0
        }
        
        # Check if financial terms are present
        financial_keywords = ['market', 'trading', 'stock', 'investment', 'portfolio']
        validation_results['has_financial_terms'] = any(
            term in original.lower() for term in financial_keywords
        )
        
        # Check if numbers are preserved (basic check)
        original_numbers = set(re.findall(r'\d+\.?\d*', original))
        translated_numbers = set(re.findall(r'\d+\.?\d*', translated))
        validation_results['preserved_numbers'] = len(original_numbers & translated_numbers) > 0
        
        # Calculate basic quality score
        score = 0
        if 0.5 <= validation_results['length_ratio'] <= 2.0:  # Reasonable length
            score += 30
        if validation_results['has_financial_terms']:
            score += 30
        if validation_results['preserved_numbers']:
            score += 40
        
        validation_results['quality_score'] = score
        
        return validation_results