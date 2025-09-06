import os
import logging
import time
import re
from typing import Dict, List, Optional
import google.generativeai as genai
from datetime import datetime

class FinancialContentTranslator:
    def __init__(self):
        """Initialize the financial content translator"""
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Initialize the model for translation
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Language configurations
        self.target_languages = {
            'Arabic': {
                'code': 'ar',
                'name': 'Arabic',
                'direction': 'rtl',
                'currency_note': 'USD amounts in US Dollars'
            },
            'Hindi': {
                'code': 'hi', 
                'name': 'Hindi',
                'direction': 'ltr',
                'currency_note': 'USD amounts in US Dollars'
            },
            'Hebrew': {
                'code': 'he',
                'name': 'Hebrew', 
                'direction': 'rtl',
                'currency_note': 'USD amounts in US Dollars'
            }
        }
        
        # Financial terms that should be preserved
        self.preserve_terms = [
            # Stock symbols
            r'\b[A-Z]{1,5}\b',  # Stock tickers like AAPL, TSLA
            # Exchanges and indices  
            r'\bNYSE\b', r'\bNASDAQ\b', r'\bS&P 500\b', r'\bDow Jones\b',
            # Percentages and numbers
            r'\d+\.?\d*%', r'\$\d+\.?\d*[KMB]?', r'\d+\.?\d*[KMB]?',
            # Financial metrics
            r'\bEPS\b', r'\bP/E\b', r'\bROE\b', r'\bROI\b',
            # Company names (major ones)
            r'\bApple\b', r'\bMicrosoft\b', r'\bGoogle\b', r'\bTesla\b', 
            r'\bAmazon\b', r'\bMeta\b', r'\bNvidia\b'
        ]
        
    def translate_financial_content(self, 
                                  english_content: str, 
                                  target_languages: List[str] = None,
                                  include_images_captions: bool = True) -> Dict:
        """
        Translate financial content to multiple languages
        
        Args:
            english_content: Original English content to translate
            target_languages: List of target languages (default: all supported)
            include_images_captions: Whether to translate image captions
            
        Returns:
            Dictionary with translations for each language
        """
        if target_languages is None:
            target_languages = list(self.target_languages.keys())
        
        results = {
            'success': True,
            'translations': {},
            'errors': [],
            'translation_metadata': {
                'source_language': 'English',
                'target_languages': target_languages,
                'translation_timestamp': datetime.now().isoformat(),
                'content_length': len(english_content)
            }
        }
        
        logging.info(f"Starting translation to {len(target_languages)} languages")
        
        for language in target_languages:
            try:
                logging.info(f"Translating to {language}...")
                
                translation_result = self._translate_to_language(
                    english_content, 
                    language
                )
                
                if translation_result['success']:
                    results['translations'][language] = translation_result
                    logging.info(f"Successfully translated to {language}")
                else:
                    results['errors'].append(f"Translation to {language} failed: {translation_result['error']}")
                    logging.error(f"Translation to {language} failed")
                
                # Rate limiting between translations
                time.sleep(2)
                
            except Exception as e:
                error_msg = f"Error translating to {language}: {str(e)}"
                results['errors'].append(error_msg)
                logging.error(error_msg)
        
        # Update overall success status
        results['success'] = len(results['translations']) > 0
        results['translation_metadata']['successful_languages'] = len(results['translations'])
        results['translation_metadata']['failed_languages'] = len(results['errors'])
        
        return results
    
    def _translate_to_language(self, content: str, target_language: str) -> Dict:
        """Translate content to a specific language"""
        try:
            lang_config = self.target_languages[target_language]
            
            # Create translation prompt
            prompt = self._create_translation_prompt(content, target_language, lang_config)
            
            # Execute translation using Gemini
            response = self.model.generate_content(prompt)
            
            if not response.text:
                return {
                    'success': False,
                    'error': 'Empty response from translation model'
                }
            
            # Post-process the translation
            translated_content = self._post_process_translation(response.text, target_language)
            
            return {
                'success': True,
                'summary': translated_content,
                'language': target_language,
                'language_code': lang_config['code'],
                'text_direction': lang_config['direction'],
                'word_count': len(translated_content.split()),
                'character_count': len(translated_content)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': target_language
            }
    
    def _create_translation_prompt(self, content: str, target_language: str, lang_config: Dict) -> str:
        """Create a comprehensive translation prompt"""
        
        return f"""
You are a professional financial translator specializing in translating US market analysis to {target_language}.

TRANSLATION REQUIREMENTS:
1. Translate the following English financial market summary to {target_language}
2. Maintain professional financial analysis tone
3. Preserve all technical accuracy
4. Keep financial terminology precise

PRESERVATION RULES:
- Keep ALL stock symbols unchanged (AAPL, TSLA, MSFT, etc.)
- Keep ALL numerical data exactly as provided (percentages, dollar amounts, etc.)
- Keep exchange names in English: NYSE, NASDAQ, S&P 500, Dow Jones
- Keep company names in English but you may add local pronunciation in parentheses
- Preserve all financial metrics (EPS, P/E, ROE, etc.)

LANGUAGE-SPECIFIC GUIDELINES:
- Use professional investment terminology in {target_language}
- Ensure cultural appropriateness for {target_language} speakers
- Maintain readability and flow in {target_language}
- Add brief explanations for complex US market concepts if needed

FORMATTING:
- Maintain the markdown structure (headers, bold text, etc.)
- Keep the same section organization
- Ensure proper {lang_config['direction']} text formatting considerations

FINANCIAL CONTEXT:
- This is US market analysis for international {target_language} speakers
- Focus on accuracy over literal word-for-word translation  
- Maintain the professional tone expected by financial market participants

CONTENT TO TRANSLATE:
{content}

Please provide only the translated content without any additional commentary or explanations.
"""
    
    def _post_process_translation(self, translated_text: str, target_language: str) -> str:
        """Post-process the translation to ensure quality"""
        
        # Remove any unwanted prefixes or suffixes that the model might add
        translated_text = translated_text.strip()
        
        # Remove common AI response patterns
        unwanted_patterns = [
            r'^Here is the translation:?\s*',
            r'^Translation:?\s*',
            r'^Translated content:?\s*',
            r'\*\*Translation\*\*:?\s*'
        ]
        
        for pattern in unwanted_patterns:
            translated_text = re.sub(pattern, '', translated_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Ensure proper formatting
        translated_text = translated_text.strip()
        
        # Validate that essential elements are preserved
        if not self._validate_translation(translated_text):
            logging.warning(f"Translation validation failed for {target_language}")
        
        return translated_text
    
    def _validate_translation(self, translated_text: str) -> bool:
        """Validate that the translation preserves essential elements"""
        
        # Check that some financial terms are present
        financial_indicators = [
            '%', '$', 'NYSE', 'NASDAQ', 'S&P', 'Dow'
        ]
        
        found_indicators = sum(1 for indicator in financial_indicators if indicator in translated_text)
        
        # Should have at least some financial indicators
        if found_indicators < 2:
            return False
        
        # Check reasonable length (shouldn't be too short or too long compared to original)
        if len(translated_text) < 100:  # Too short
            return False
        
        return True
    
    def translate_image_caption(self, caption: str, target_language: str, source_domain: str) -> str:
        """Translate an image caption to target language"""
        try:
            prompt = f"""
Translate this financial chart/image caption to {target_language}:

"{caption}"

Requirements:
- Keep it concise and professional
- Preserve any numbers, percentages, or financial data exactly
- Maintain source attribution
- Make it suitable for a financial news context

Source: {source_domain}

Provide only the translated caption, nothing else.
"""
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                # Clean up the response
                translated_caption = response.text.strip()
                
                # Ensure source attribution is included
                if source_domain and source_domain.lower() not in translated_caption.lower():
                    translated_caption += f" - {source_domain}"
                
                return translated_caption
            else:
                # Fallback: return original with language note
                return f"{caption} (Source: {source_domain})"
                
        except Exception as e:
            logging.warning(f"Failed to translate caption to {target_language}: {str(e)}")
            # Return original caption as fallback
            return f"{caption} (Source: {source_domain})"
    
    def test_translation_service(self) -> bool:
        """Test if the translation service is working"""
        try:
            test_text = "Apple stock rose 2.5% after strong quarterly earnings report."
            
            response = self.model.generate_content(f"Translate to Spanish: {test_text}")
            
            if response.text and len(response.text) > 10:
                logging.info("Translation service test passed")
                return True
            else:
                logging.error("Translation service test failed - empty response")
                return False
                
        except Exception as e:
            logging.error(f"Translation service test failed: {str(e)}")
            return False


def translate_financial_content(content: str, 
                              target_languages: List[str] = None,
                              include_image_captions: bool = True) -> Dict:
    """
    Main function to translate financial content
    
    Args:
        content: English financial content to translate
        target_languages: List of target languages (default: Arabic, Hindi, Hebrew)
        include_image_captions: Whether to handle image caption translation
        
    Returns:
        Dictionary with translation results
    """
    translator = FinancialContentTranslator()
    
    if target_languages is None:
        target_languages = ['Arabic', 'Hindi', 'Hebrew']
    
    return translator.translate_financial_content(
        content, 
        target_languages, 
        include_image_captions
    )


def test_translator() -> bool:
    """Test the translator functionality"""
    translator = FinancialContentTranslator()
    return translator.test_translation_service()