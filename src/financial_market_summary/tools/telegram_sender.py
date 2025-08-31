from crewai.tools import BaseTool
from typing import Type, Any, Dict
from pydantic import BaseModel, Field
import requests
import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

load_dotenv()

class TelegramSendInput(BaseModel):
    """Input schema for Telegram sender tool."""
    content: str = Field(..., description="Content to send to Telegram channel")
    language: str = Field(default="english", description="Language of the content")
    include_timestamp: bool = Field(default=True, description="Include timestamp in message")

class TelegramSender(BaseTool):
    name: str = "telegram_sender"
    description: str = (
        "Send financial market summaries to Telegram channel. "
        "Supports multiple languages and formatting for financial content."
    )
    args_schema: Type[BaseModel] = TelegramSendInput
    bot_token: str = None
    chat_id: str = None
    base_url: str = None

    def __init__(self):
        super().__init__()
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def _run(self, content: str, language: str = "english", include_timestamp: bool = True) -> str:
        """
        Send content to Telegram channel
        """
        try:
            if not self.bot_token or not self.chat_id:
                return "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured"
            
            # Format message
            formatted_message = self._format_message(content, language, include_timestamp)
            
            # Send message
            result = self._send_message(formatted_message)
            
            if result:
                logger.info(f"Successfully sent {language} summary to Telegram")
                return f"Successfully sent {language} summary to Telegram channel"
            else:
                return f"Failed to send {language} summary to Telegram"
                
        except Exception as e:
            error_msg = f"Telegram send error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_message(self, content: str, language: str, include_timestamp: bool) -> str:
        """Format message for Telegram with appropriate styling"""
        
        # Language emojis
        language_emojis = {
            'english': '🇺🇸',
            'arabic': '🇸🇦', 
            'hindi': '🇮🇳',
            'hebrew': '🇮🇱'
        }
        
        emoji = language_emojis.get(language.lower(), '📊')
        
        # Header
        header_parts = [
            f"{emoji} **DAILY FINANCIAL MARKET SUMMARY**",
            f"Language: {language.title()}"
        ]
        
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            header_parts.append(f"Generated: {timestamp}")
        
        header = "\n".join(header_parts)
        
        # Format content
        formatted_content = f"""
{header}
{'='*50}

{content}

{'='*50}
📈 *CrowdWisdomTrading AI Agent*
🤖 *Automated Financial Analysis*
        """
        
        # Ensure message length doesn't exceed Telegram's limit (4096 characters)
        if len(formatted_content) > 4000:
            # Truncate content but keep header and footer
            available_space = 4000 - len(header) - 200  # Reserve space for footer
            truncated_content = content[:available_space] + "\n\n... (truncated)"
            
            formatted_content = f"""
{header}
{'='*50}

{truncated_content}

{'='*50}
📈 *CrowdWisdomTrading AI Agent*
🤖 *Automated Financial Analysis*
            """
        
        return formatted_content.strip()

    def _send_message(self, message: str) -> bool:
        """Send message to Telegram channel"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            result = response.json()
            
            # 🔥 Always log the full response
            print("🔍 Telegram API response:", result)
            logger.info(f"Telegram API response: {result}")

            if result.get('ok'):
                logger.info("Message sent successfully to Telegram")
                return True
            else:
                logger.error(f"Telegram API error: {result.get('description')}")
                return False

        except Exception as e:
            error_msg = f"Network/Unexpected error sending to Telegram: {str(e)}"
            print("❌ Telegram exception:", error_msg)   # force print even if logger hides it
            logger.error(error_msg)
            return False

    def send_multiple_languages(self, content_dict: Dict[str, str]) -> str:
        """Send content in multiple languages"""
        results = []
        
        for language, content in content_dict.items():
            if content and content.strip():
                # Add delay between messages to avoid rate limits
                if results:  # Not the first message
                    time.sleep(1)
                
                result = self._run(content, language)
                results.append(f"{language}: {result}")
            else:
                results.append(f"{language}: Skipped (empty content)")
        
        return "; ".join(results)
    
    def test_connection(self) -> str:
        """Test Telegram bot connection"""
        try:
            # Ensure bot_token is properly initialized
            if not hasattr(self, 'bot_token') or not self.bot_token:
                return "❌ Error: Bot token not properly initialized"
            
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('ok'):
                bot_info = result.get('result', {})
                return f"✅ Connected to bot: {bot_info.get('first_name', 'Unknown')} (@{bot_info.get('username', 'unknown')})"
            else:
                return f"❌ Bot connection failed: {result.get('description')}"
                
        except Exception as e:
            return f"❌ Connection test failed: {str(e)}"


class TelegramImageSender(BaseTool):
    """Tool for sending images to Telegram (for charts/graphs)"""
    name: str = "telegram_image_sender"
    description: str = "Send financial charts and images to Telegram channel"
    args_schema: Type[BaseModel] = TelegramSendInput
    
    def __init__(self):
        super().__init__()
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def _run(self, image_url: str, caption: str = "") -> str:
        """Send image to Telegram channel"""
        try:
            url = f"{self.base_url}/sendPhoto"
            
            payload = {
                'chat_id': self.chat_id,
                'photo': image_url,
                'caption': caption,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('ok'):
                return "Image sent successfully to Telegram"
            else:
                return f"Failed to send image: {result.get('description')}"
                
        except Exception as e:
            error_msg = f"Error sending image to Telegram: {str(e)}"
            logger.error(error_msg)
            return error_msg