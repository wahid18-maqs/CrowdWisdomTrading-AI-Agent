# src/financial_market_summary/tools/telegram_sender.py

import os
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv
from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field

# Set up basic logging
logger = logging.getLogger(__name__)

class TelegramSenderInput(BaseModel):
    """Input schema for the Telegram sender tool."""
    content: str = Field(..., description="The main text content to be sent to the Telegram channel.")
    language: str = Field(default="english", description="The language of the content (e.g., 'english', 'arabic'). Used for formatting the header.")

class TelegramSender(BaseTool):
    """
    A CrewAI tool to send messages to a specific Telegram channel.
    """
    name: str = "telegram_sender"
    description: str = "Sends a formatted financial summary to a designated Telegram channel."
    args_schema: Type[BaseModel] = TelegramSenderInput

    # ✅ Declare fields so Pydantic knows them
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
        """The main execution method for the tool. It formats and sends the message."""
        if not self.base_url:
            return "Error: Telegram credentials are not configured. Cannot send message."
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            language_headers = {
                'english': f"📈 **Daily Financial Market Summary**\n🕐 {timestamp}\n\n",
                'arabic': f"📊 **ملخص السوق المالي اليومي**\n🕐 {timestamp}\n\n",
                'hindi': f"📈 **दैनिक वित्तीय बाजार सारांश**\n🕐 {timestamp}\n\n",
                'hebrew': f"📊 **סיכום שוק הכספים היומי**\n🕐 {timestamp}\n\n"
            }
            header = language_headers.get(language.lower(), language_headers['english'])
            full_message = header + content

            if len(full_message) > 4096:
                chunks = self._split_message(full_message, 4096)
                results = []
                for i, chunk in enumerate(chunks):
                    success = self._send_single_message(chunk)
                    results.append(f"Chunk {i+1}: {'Success' if success else 'Failed'}")
                return f"Message was too long. Sent in {len(chunks)} parts: " + ", ".join(results)
            else:
                success = self._send_single_message(full_message)
                return f"Message in '{language}' sent successfully." if success else f"Failed to send message in '{language}'."

        except Exception as e:
            error_msg = f"An unexpected error occurred while sending to Telegram: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _send_single_message(self, message: str) -> bool:
        """Sends a single message chunk to the Telegram API."""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        try:
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            if result.get("ok"):
                logger.info("Message successfully sent to Telegram.")
                return True
            else:
                logger.error(f"Telegram API returned an error: {result.get('description')}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Telegram API: {e}")
            return False

    def _split_message(self, message: str, max_length: int) -> list[str]:
        """Splits a long message into smaller chunks that respect the max_length limit."""
        chunks, current_chunk = [], ""
        for line in message.split('\n'):
            if len(current_chunk) + len(line) + 1 <= max_length:
                current_chunk += line + '\n'
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
