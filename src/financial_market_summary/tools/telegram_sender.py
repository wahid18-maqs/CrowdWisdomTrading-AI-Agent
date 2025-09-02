# Enhanced telegram_sender.py with better error handling and image support
import os
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv
from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field
import re

logger = logging.getLogger(__name__)

class TelegramSenderInput(BaseModel):
    """Input schema for the Telegram sender tool."""
    content: str = Field(..., description="The main text content to be sent to the Telegram channel.")
    language: str = Field(default="english", description="The language of the content (e.g., 'english', 'arabic'). Used for formatting the header.")

class TelegramSender(BaseTool):
    """
    Enhanced CrewAI tool to send messages to a specific Telegram channel with image support.
    """
    name: str = "telegram_sender"
    description: str = "Sends a formatted financial summary with images to a designated Telegram channel."
    args_schema: Type[BaseModel] = TelegramSenderInput

    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    base_url: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv()
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the .env file.")
            self.base_url = None
        else:
            logger.info("Telegram credentials loaded successfully.")
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def _run(self, content: str, language: str = "english") -> str:
        """Main execution method that handles both text and images"""
        if not self.base_url:
            return "Error: Telegram credentials are not configured. Cannot send message."
        
        try:
            # Process content and extract images
            processed_content, image_urls = self._process_content_and_images(content)
            
            # Create formatted message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            language_headers = {
                'english': f"📈 **US Financial Market Summary**\n🕐 {timestamp}\n\n",
                'arabic': f"📊 **ملخص السوق المالي الأمريكي**\n🕐 {timestamp}\n\n",
                'hindi': f"📈 **अमेरिकी वित्तीय बाजार सारांश**\n🕐 {timestamp}\n\n",
                'hebrew': f"📊 **סיכום שוק הכספים האמריקאי**\n🕐 {timestamp}\n\n"
            }
            
            header = language_headers.get(language.lower(), language_headers['english'])
            full_message = header + processed_content
            
            # Add footer
            footer = self._get_language_footer(language)
            full_message += f"\n\n{footer}"
            
            results = []
            
            # Send main message
            if len(full_message) > 4096:
                chunks = self._split_message(full_message, 4096)
                for i, chunk in enumerate(chunks):
                    success = self._send_single_message(chunk)
                    results.append(f"Text chunk {i+1}: {'✅' if success else '❌'}")
            else:
                success = self._send_single_message(full_message)
                results.append(f"Main message: {'✅' if success else '❌'}")
            
            # Send images if found
            if image_urls:
                for i, img_url in enumerate(image_urls[:3]):  # Limit to 3 images
                    img_success = self._send_image(img_url, f"Financial Chart {i+1}")
                    results.append(f"Image {i+1}: {'✅' if img_success else '❌'}")
            
            # Generate final status report
            success_count = len([r for r in results if '✅' in r])
            total_count = len(results)
            
            final_status = f"Telegram delivery for '{language}': {success_count}/{total_count} items sent successfully. Details: {', '.join(results)}"
            logger.info(final_status)
            return final_status

        except Exception as e:
            error_msg = f"Unexpected error sending to Telegram ({language}): {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _process_content_and_images(self, content: str) -> tuple[str, list[str]]:
        """Extract images from markdown content and return clean text + image URLs"""
        image_urls = []
        
        # Find markdown images: ![alt](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(image_pattern, content)
        
        for alt_text, url in matches:
            if self._is_valid_image_url(url):
                image_urls.append(url)
        
        # Remove image markdown from text content
        clean_content = re.sub(image_pattern, '', content)
        
        # Clean up extra whitespace
        clean_content = re.sub(r'\n{3,}', '\n\n', clean_content).strip()
        
        return clean_content, image_urls
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL"""
        try:
            # Basic URL validation
            if not url.startswith(('http://', 'https://')):
                return False
            
            # Check for image extensions
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            url_lower = url.lower()
            
            # Direct extension check or known image domains
            if any(ext in url_lower for ext in image_extensions):
                return True
            
            # Known financial image domains
            financial_domains = ['chart.yahoo.com', 'tradingview.com', 'finviz.com', 'images.unsplash.com']
            if any(domain in url_lower for domain in financial_domains):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _send_image(self, image_url: str, caption: str = "") -> bool:
        """Send an image to Telegram"""
        try:
            url = f"{self.base_url}/sendPhoto"
            payload = {
                "chat_id": self.chat_id,
                "photo": image_url,
                "caption": caption,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if result.get("ok"):
                logger.info(f"Image sent successfully: {caption}")
                return True
            else:
                logger.warning(f"Failed to send image: {result.get('description')}")
                return False
                
        except Exception as e:
            logger.warning(f"Error sending image to Telegram: {e}")
            return False

    def _send_single_message(self, message: str, max_retries: int = 3) -> bool:
        """Send a single message with retry logic"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Telegram send retry {attempt + 1}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                
                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                }
                
                response = requests.post(url, json=payload, timeout=20)
                response.raise_for_status()
                
                result = response.json()
                if result.get("ok"):
                    logger.info("Message successfully sent to Telegram")
                    return True
                else:
                    error_desc = result.get('description', 'Unknown error')
                    logger.warning(f"Telegram API error: {error_desc}")
                    if attempt < max_retries - 1:
                        continue
                    return False
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error sending to Telegram (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
                return False
            except Exception as e:
                logger.error(f"Unexpected error sending to Telegram: {e}")
                return False
        
        return False

    def _split_message(self, message: str, max_length: int) -> list[str]:
        """Split long messages while preserving formatting"""
        if len(message) <= max_length:
            return [message]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = message.split('\n\n')
        
        for paragraph in paragraphs:
            # If paragraph itself is too long, split by sentences
            if len(paragraph) > max_length:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if not sentence.endswith('.'):
                        sentence += '.'
                    
                    if len(current_chunk) + len(sentence) + 2 <= max_length:
                        current_chunk += sentence + ' '
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ' '
            else:
                # Normal paragraph processing
                if len(current_chunk) + len(paragraph) + 2 <= max_length:
                    current_chunk += paragraph + '\n\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + '\n\n'
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [message[:max_length]]
    
    def _get_language_footer(self, language: str) -> str:
        """Get appropriate footer for each language"""
        footers = {
            'english': "📊 *Powered by AI Financial Analysis*",
            'arabic': "📊 *مدعوم بتحليل مالي بالذكاء الاصطناعي*",
            'hindi': "📊 *एआई वित्तीय विश्लेषण द्वारा संचालित*",
            'hebrew': "📊 *מופעל על ידי ניתוח פיננסי AI*"
        }
        return footers.get(language.lower(), footers['english'])

import time  # Add missing import