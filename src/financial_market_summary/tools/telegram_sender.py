# telegram_sender.py
import os
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load .env explicitly
load_dotenv(r"C:\Users\wahid\Desktop\financial_market_summary\.env")

class TelegramSender:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot token or chat ID is missing! Check your .env file.")
        else:
            logger.info(f"Loaded Telegram bot token and chat ID successfully.")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, message: str) -> bool:
        """Send a single message to Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.error("Cannot send message: bot_token or chat_id not set.")
            return False
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            ok = resp.json().get("ok", False)
            if not ok:
                logger.error(f"Failed to send Telegram message: {resp.text}")
            return ok
        except Exception as e:
            logger.error(f"Telegram send exception: {e}")
            return False

# Test sending a message
if __name__ == "__main__":
    sender = TelegramSender()
    test_message = "✅ Test message from Telegram bot"
    result = sender.send_message(test_message)
    print("Message sent:", result)
