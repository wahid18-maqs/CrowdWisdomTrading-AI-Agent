from crewai.tools import BaseTool
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
import requests
import re
import time
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

class TelegramSenderInput(BaseModel):
    """Input schema for the Telegram sender tool."""
    content: str = Field(
        ..., description="The main text content to be sent to the Telegram channel."
    )
    language: str = Field(
        default="english",
        description="The language of the content (e.g., 'english', 'arabic'). Used for formatting the header.",
    )

class TelegramSender(BaseTool):
    """
    An enhanced CrewAI tool to send formatted financial summaries, including text and images,
    to a designated Telegram channel.
    This class correctly inherits from BaseTool and uses Pydantic to manage configuration,
    ensuring it integrates smoothly with the CrewAI framework.
    """
    name: str = "telegram_sender"
    description: str = "Sends a formatted financial summary with images to a designated Telegram channel."
    args_schema: Type[BaseModel] = TelegramSenderInput
    bot_token: Optional[str] = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    chat_id: Optional[str] = Field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    base_url: Optional[str] = None

    def __init__(self, **kwargs):
        """Initializes the TelegramSender tool with credentials."""
        super().__init__(**kwargs)
        if self.bot_token and self.chat_id:
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
            logger.info("Telegram credentials loaded successfully.")
        else:
            logger.error(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the .env file."
            )

    def _run(self, content: str, language: str = "english") -> str:
        """
        Main execution method that sends the formatted financial summary to Telegram.
        Args:
            content (str): The main text content, which may contain image URLs.
            language (str): The language of the content.
        Returns:
            str: A status report of the delivery attempt.
        """
        if not self.base_url:
            return "Error: Telegram credentials are not configured. Cannot send message."

        try:
            
            processed_content, image_urls = self._process_content_and_images(content)
            full_message = self._format_message(processed_content, language)
            results = []
            if len(full_message) > 4096:
                chunks = self._split_message(full_message, 4096)
                for i, chunk in enumerate(chunks):
                    success = self._send_single_message(chunk)
                    status = "Message sent successfully" if success else "Message failed"
                    results.append(f"Text chunk {i+1}: {status}")
            else:
                success = self._send_single_message(full_message)
                status = "Message sent successfully" if success else "Message failed"
                results.append(f"Main message: {status}")
            for i, img_url in enumerate(image_urls[:3]):
                img_success = self._send_image(img_url, f"Financial Chart {i+1}")
                status = "Image sent successfully" if img_success else "Image failed"
                results.append(f"Image {i+1}: {status}")
                time.sleep(1)
            return self._generate_status_report(results, language)
        except Exception as e:
            error_msg = f"Unexpected error sending to Telegram ({language}): {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg
        #helper functions
    def _process_content_and_images(self, content: str) -> tuple[str, List[str]]:
        """
        Extracts image URLs from markdown content and returns the cleaned text
        and a list of the extracted URLs.
        """
        image_urls = []
        image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        matches = re.findall(image_pattern, content)

        for _, url in matches:
            if self._is_valid_image_url(url):
                image_urls.append(url)

        clean_content = re.sub(image_pattern, "", content)
        clean_content = re.sub(r"\n{3,}", "\n\n", clean_content).strip()

        return clean_content, image_urls

    def _is_valid_image_url(self, url: str) -> bool:
        """
        Performs a basic check to validate if a URL is a likely image link.
        """
        if not url or not url.startswith(("http://", "https://")):
            return False

        url_lower = url.lower()
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        financial_domains = [
            "chart.yahoo.com",
            "tradingview.com",
            "finviz.com",
            "images.unsplash.com",
        ]

        return any(ext in url_lower for ext in image_extensions) or any(
            domain in url_lower for domain in financial_domains
        )

    def _format_message(self, content: str, language: str) -> str:
        """
        Adds a language-specific header and footer to the main message content.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        language_headers = {
            "english": f"📈 **US Financial Market Summary**\n🕐 {timestamp}\n\n",
            "arabic": f"📊 **ملخص السوق المالي الأمريكي**\n🕐 {timestamp}\n\n",
            "hindi": f"📈 **अमेरिकी वित्तीय बाजार सारांश**\n🕐 {timestamp}\n\n",
            "hebrew": f"📊 **סיכום שוק הכספים האמריקאי**\n🕐 {timestamp}\n\n",
        }
        header = language_headers.get(language.lower(), language_headers["english"])
        footer = self._get_language_footer(language)

        return f"{header}{content}\n\n{footer}"

    def _get_language_footer(self, language: str) -> str:
        """
        Returns a language-specific footer.
        """
        footers = {
            "english": "📊 *Powered by CrowdWisdomTrading*",
            "arabic": "📊 *مدعوم من CrowdWisdomTrading*",
            "hindi": "📊 *CrowdWisdomTrading द्वारा संचालित*",
            "hebrew": "📊 *מופעל על ידי CrowdWisdomTrading*",
        }
        return footers.get(language.lower(), footers["english"])

    def _send_image(self, image_url: str, caption: str = "") -> bool:
        """
        Sends a single image to the Telegram channel.
        """
        try:
            url = f"{self.base_url}/sendPhoto"
            payload = {
                "chat_id": self.chat_id,
                "photo": image_url,
                "caption": caption,
                "parse_mode": "Markdown",
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            if result.get("ok"):
                logger.info(f"Image sent successfully: {caption}")
                return True
            else:
                logger.warning(
                    f"Failed to send image with caption '{caption}': {result.get('description')}"
                )
                return False
        except Exception as e:
            logger.warning(f"Error sending image '{image_url}': {e}")
            return False

    def _send_single_message(self, message: str, max_retries: int = 3) -> bool:
        """
        Sends a single text message with retry logic.
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2**attempt
                    logger.info(
                        f"Telegram send retry {attempt + 1}, waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)

                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                }
                response = requests.post(url, json=payload, timeout=20)
                response.raise_for_status()

                result = response.json()
                if result.get("ok"):
                    logger.info("Message successfully sent to Telegram.")
                    return True
                else:
                    error_desc = result.get("description", "Unknown error")
                    logger.warning(f"Telegram API error: {error_desc}")
                    if "Too Many Requests" in error_desc:
                        continue
                    return False
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Network error sending to Telegram (attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    continue
                return False
            except Exception as e:
                logger.error(f"Unexpected error during message send: {e}")
                return False
        return False

    def _split_message(self, message: str, max_length: int) -> List[str]:
        """
        Splits a long message into chunks, attempting to preserve paragraph and sentence
        structure for better readability.
        """
        if len(message) <= max_length:
            return [message]

        chunks = []
        current_chunk = ""
        paragraphs = message.split("\n\n")

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

                if len(current_chunk) > max_length:
                    sentences = re.split(r"(?<=[.!?])\s+", current_chunk)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 <= max_length:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [message[:max_length]]

    def _generate_status_report(self, results: List[str], language: str) -> str:
        """
        Generates the final status report string from the list of results.
        """
        success_count = sum(1 for r in results if "successfully" in r)
        total_count = len(results)
        status_string = ", ".join(results)
        return f"Telegram delivery for '{language}': {success_count}/{total_count} items sent successfully. Details: {status_string}"
    

