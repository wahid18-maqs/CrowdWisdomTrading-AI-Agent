from crewai.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
import os
import logging
from litellm import completion
import requests
import json

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

    def __init__(self):
        super().__init__()
        self.financial_glossary = self._load_financial_glossary()
    
    def _run(self, text: str, target_language: str, preserve_formatting: bool = True) -> str:
        """
        Translate text to target language with financial context
        """
        try:
            # Validate target language
            supported_languages = ['arabic', 'hindi', 'hebrew']
            if target_language.lower() not in supported_languages:
                return f"Error: Unsupported language '{target_language}'. Supported: {supported_languages}"
            
            # Try primary translation method (LiteLLM)
            result = self._translate_with_litellm(text, target_language, preserve_formatting)
            
            if "Error:" not in result:
                logger.info(f"Successfully translated to {target_language} using LiteLLM")
                return result
            
            # Fallback to Google Translate API if available
            logger.warning("LiteLLM translation failed, trying Google Translate")
            fallback_result = self._translate_with_google(text, target_language)
            
            if fallback_result:
                return fallback_result
            
            # Final fallback - return original with note
            return f"Translation to {target_language} unavailable. Original text:\n\n{text}"
            
        except Exception as e:
            error_msg = f"Translation error for {target_language}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _translate_with_litellm(self, text: str, target_language: str, preserve_formatting: bool) -> str:
        """Translate using LiteLLM with financial context"""
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
1. Translate the financial summary while preserving all financial terminology accuracy
2. Keep stock symbols (like AAPL, MSFT) unchanged
3. Preserve numbers, percentages, and currency values exactly
4. Maintain the structure and formatting of the text
5. Use appropriate financial terminology in {target_lang_name}
6. If unsure about financial terms, provide the English term in parentheses

Financial context: This is a daily financial market summary containing stock market news, trading activity, and market analysis."""

            formatting_instruction = ""
            if preserve_formatting:
                formatting_instruction = "\n7. Preserve any markdown formatting (**, *, headers, etc.)"
            
            user_prompt = f"""Translate this financial content to {target_lang_name}:

{text}

Requirements:
- Maintain financial accuracy
- Keep stock symbols unchanged  
- Preserve numerical data exactly
- Use proper financial terminology{formatting_instruction}
- Provide a natural, professional translation suitable for financial professionals"""

            # Make API call
            response = completion(
                model="gpt-3.5-turbo",  # You can change this to any model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=2000
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Post-process to ensure financial terms are preserved
            processed_text = self._post_process_translation(translated_text, text, target_language)
            
            return processed_text
            
        except Exception as e:
            error_msg = f"LiteLLM translation error: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def _translate_with_google(self, text: str, target_language: str) -> str:
        """Fallback translation using Google Translate API"""
        try:
            google_api_key = os.getenv('GOOGLE_TRANSLATE_API_KEY')
            if not google_api_key:
                return None
            
            # Language codes for Google Translate
            lang_codes = {
                'arabic': 'ar',
                'hindi': 'hi',
                'hebrew': 'he'
            }
            
            target_code = lang_codes.get(target_language.lower())
            if not target_code:
                return None
            
            url = f"https://translation.googleapis.com/language/translate/v2?key={google_api_key}"
            
            payload = {
                'q': text,
                'target': target_code,
                'source': 'en',
                'format': 'text'
            }
            
            response = requests.post(url, data=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if 'data' in result and 'translations' in result['data']:
                translated = result['data']['translations'][0]['translatedText']
                return self._post_process_translation(translated, text, target_language)
            
            return None
            
        except Exception as e:
            logger.error(f"Google Translate error: {e}")
            return None
    
    def _post_process_translation(self, translated: str, original: str, target_language: str) -> str:
        """Post-process translation to fix financial terms and formatting"""
        try:
            import re
            
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
        import re
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