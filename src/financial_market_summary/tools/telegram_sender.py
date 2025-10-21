import json
import re
import requests
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai.tools import BaseTool
from .image_finder import EnhancedImageFinderInput as ImageFinderInput
from .image_finder import EnhancedImageFinder as ImageFinder

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class TelegramSenderInput(BaseModel):
    content: str = Field(description="The full financial summary content")
    language: Optional[str] = Field("english", description="Language for formatting")

class EnhancedTelegramSender(BaseTool):
    name: str = "telegram_sender"
    description: str = "Sends formatted financial summaries with verified sources and Telegram-compatible images"
    args_schema: Type[BaseModel] = TelegramSenderInput

    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    base_url: Optional[str] = None
    image_finder: Optional[ImageFinder] = None
    bots: Optional[dict] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Language-specific bots configuration
        self.bots = {
            "english": {
                "token": os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN_ENGLISH"),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID_ENGLISH")
            },
            "arabic": {
                "token": os.getenv("TELEGRAM_BOT_TOKEN_ARABIC"),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID_ARABIC")
            },
            "hindi": {
                "token": os.getenv("TELEGRAM_BOT_TOKEN_HINDI"),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID_HINDI")
            },
            "hebrew": {
                "token": os.getenv("TELEGRAM_BOT_TOKEN_HEBREW"),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID_HEBREW")
            },
            "german": {
                "token": os.getenv("TELEGRAM_BOT_TOKEN_GERMAN"),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID_GERMAN")
            }
        }

        # Initialize with English bot for validation
        self.bot_token = self.bots["english"]["token"]
        self.chat_id = self.bots["english"]["chat_id"]

        if not self.bot_token or not self.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.image_finder = ImageFinder()

        if not self._test_credentials():
            raise ValueError("Invalid Telegram credentials")

    def _test_credentials(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            return response.ok
        except:
            return False

    def _set_bot_for_language(self, language: str) -> None:
        """Switch to language-specific bot - REQUIRED for each language"""
        language_lower = language.lower()

        logger.info(f"Setting bot for language: '{language_lower}'")

        # Get language-specific bot configuration
        bot_config = self.bots.get(language_lower)

        if not bot_config:
            raise ValueError(f"No bot configuration found for language: {language_lower}")

        if not bot_config.get('token') or not bot_config.get('chat_id'):
            raise ValueError(f"Missing token or chat_id for {language_lower} bot")

        # Switch to language-specific bot
        self.bot_token = bot_config['token']
        self.chat_id = bot_config['chat_id']
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        logger.info(f"Switched to {language_lower.upper()} bot (chat_id: {self.chat_id})")

    def _run(self, content: str, language: str = "english") -> str:
        try:
            # ALWAYS set the bot for the specified language - no defaults
            self._set_bot_for_language(language)

            # Check if content has two-message format OR embedded Telegram image data from search
            if "---TELEGRAM_IMAGE_DATA---" in content or "=== TELEGRAM_TWO_MESSAGE_FORMAT ===" in content:
                logger.info(f"Found Telegram-ready content with two-message format for {language}")
                return self._process_telegram_ready_content(content, language)

            # Check if content is already pre-formatted by content extractor agent
            elif self._is_pre_formatted_content(content):
                logger.info(f"Using pre-formatted agent content for {language}")
                message = self._prepare_pre_formatted_content(content, language)

                # Find image data for English only
                image_data = {}
                if language.lower() == "english":
                    # Create minimal structured data for image finding
                    structured_data = {"title": "Market Update"}
                    image_data = self._find_telegram_compatible_image(content, structured_data)

                result = self._send_content(message, image_data)
                return result

            # Fallback to original parsing method - treat all languages consistently
            else:
                logger.info(f"Content not pre-formatted, using fallback parsing for {language}")

                structured_data = self._extract_clean_content(content)

                # Find images
                image_data = self._find_telegram_compatible_image(content, structured_data)

            logger.info(f"Creating clean message for {language}")
            logger.info(f"Data: title='{structured_data.get('title', 'N/A')}'")

            message = self._create_clean_message(structured_data, language)

            logger.info(f"Created message length: {len(message)} chars")
            logger.info(f"Message preview: {message[:200]}...")

            # Apply language-specific formatting for all non-English languages
            if language.lower() in ["hindi", "arabic", "hebrew", "german"]:
                logger.info(f"Applying translation formatting for {language}")
                message = self._apply_translation_formatting(message, language)
                logger.info(f"Final formatted message length: {len(message)} chars")

            result = self._send_content(message, image_data)

            return result
        except Exception as e:
            logger.error(f"Telegram sender failed: {e}")
            return f"Error: {e}"

    def send_from_workflow_result(self, json_file_path: str, language: str = "english") -> str:
        try:
            logger.info(f"Processing workflow result from: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)

            if isinstance(workflow_data, dict) and 'content' in workflow_data:
                content = workflow_data['content']
            elif isinstance(workflow_data, str):
                content = workflow_data
            else:
                content = str(workflow_data)

            logger.info(f"Loaded {len(content)} characters for {language}")
            return self._run(content, language)

        except Exception as e:
            logger.error(f"Failed to send from workflow result: {e}")
            return f"Error processing workflow result: {e}"

    def _extract_clean_content(self, content: str) -> Dict[str, Any]:
        data = {
            "title": "",
            "source": "Financial News",
            "source_url": ""
        }

        clean_content = re.sub(r'<[^>]+>', '', content)
        lines = [line.strip() for line in clean_content.splitlines() if line.strip()]

        if not lines:
            return data

        title_found = False
        for line in lines[:25]:
            if any(skip in line.upper() for skip in [
                'SEARCH WINDOW', 'MARKET OVERVIEW', 'KEY HIGHLIGHTS', 'BREAKING NEWS',
                'SEARCH METADATA', 'REAL-TIME', 'UTC', '===', 'PHASE'
            ]):
                continue

            clean_line = self._strip_all_formatting(line)
            if 25 <= len(clean_line) <= 200 and not clean_line.startswith('•') and not title_found:
                market_terms = ['market', 'stock', 'trading', 'earning', 'fed', 'economic', 'financial', 'sector', 'rally', 'surge', 'high']
                if any(term in clean_line.lower() for term in market_terms):
                    data["title"] = clean_line
                    title_found = True
                    break

        data["title"] = data["title"] or "Market Update"

        return data

    def _extract_translated_content_structure(self, content: str) -> Dict[str, Any]:
        data = {
            "title": "",
            "source": "Financial News",
            "source_url": ""
        }

        if not content:
            return data

        logger.info(f"TRANSLATED CONTENT DEBUG:")
        logger.info(f"Content length: {len(content)} chars")
        logger.info(f"First 200 chars: {content[:200]}...")

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        logger.info(f"Total lines: {len(lines)}")

        for i, line in enumerate(lines[:10]):
            logger.info(f"  Line {i}: {line[:100]}...")

        if not lines:
            return data

        title_found = False
        for line in lines[:10]:
            if line.startswith('#'):
                clean_title = re.sub(r'^#+\s*', '', line).strip()
                if 20 <= len(clean_title) <= 200:
                    data["title"] = clean_title
                    title_found = True
                    break
            elif line.startswith('**') and line.endswith('**'):
                clean_title = line.strip('*').strip()
                if 20 <= len(clean_title) <= 200:
                    data["title"] = clean_title
                    title_found = True
                    break

        if not title_found:
            for line in lines[:5]:
                clean_line = re.sub(r'[#*`_]+', '', line).strip()
                if 20 <= len(clean_line) <= 200:
                    data["title"] = clean_line
                    break

        data["title"] = data["title"] or "Market Update"

        return data


    def _get_latest_search_results_file(self) -> str:
        """Get the most recent search results file from output/search_results directory"""
        try:
            from pathlib import Path
            import glob

            project_root = Path(__file__).resolve().parent.parent.parent.parent
            search_results_dir = project_root / "output" / "search_results"

            if not search_results_dir.exists():
                return ""

            # Get all search_results JSON files
            search_files = list(search_results_dir.glob("search_results_*.json"))

            if not search_files:
                return ""

            # Get the most recent file
            latest_file = max(search_files, key=lambda p: p.stat().st_mtime)
            return str(latest_file)

        except Exception as e:
            logger.error(f"Error finding search results file: {e}")
            return ""

    def _find_telegram_compatible_image(self, original_content: str, structured_data: Dict[str, Any]) -> Dict[str, str]:
        try:
            search_parts = []
            if structured_data.get("title"):
                search_parts.append(structured_data["title"])

            search_content = " ".join(search_parts)[:500] if search_parts else original_content[:500]
            stocks = self._extract_stock_symbols(original_content)

            # Get most recent search results file
            search_results_file = self._get_latest_search_results_file()

            if not search_results_file:
                logger.warning("No search results file found")
                return {}

            image_input = ImageFinderInput(
                search_content=search_content,
                max_images=3,
                search_results_file=search_results_file
            )
            results_json = self.image_finder._run(**image_input.dict())
            
            if results_json:
                images = json.loads(results_json) if isinstance(results_json, str) else results_json
                if isinstance(images, list) and images:
                    for image in images:
                        if image.get('telegram_compatible', False):
                            logger.info(f"Using Telegram-compatible image from finder: {image.get('title', 'Unknown')}")
                            # Use AI-generated description if available, otherwise use title
                            description = image.get("image_description", "") or image.get("title", "Financial Chart")
                            return {
                                "url": image["url"],
                                "title": description,
                                "source": "Enhanced Image Finder",
                                "telegram_compatible": True,
                                "type": "finder_result"
                            }
                    logger.warning("No Telegram-compatible images found in image finder results")
                    for image in images:
                        logger.debug(f"   - {image.get('title', 'Unknown')}: telegram_compatible={image.get('telegram_compatible', False)}")
        except Exception as e:
            logger.warning(f"Enhanced ImageFinder failed: {e}")

        logger.info("No image available")
        return {}

    def _is_pre_formatted_content(self, content: str) -> bool:
        """Check if content is already pre-formatted by content extractor agent."""
        # Look for indicators of pre-formatted content
        indicators = [
            # Markdown source links with verification icons
            r'\[.*\]\(https?://[^\)]+\)\s*[✅⚠️❌⏱️🔌]',
            # Arabic/Hebrew source patterns
            r'المصدر.*\[.*\]\(https?://[^\)]+\)',
            r'مصدر.*\[.*\]\(https?://[^\)]+\)',
            # Source patterns in different languages
            r'\*.*Source.*\*.*\[.*\]\(https?://[^\)]+\)',
            r'\*.*المصدر.*\*.*\[.*\]\(https?://[^\)]+\)',
            # Hindi source patterns
            r'स्रोत.*\[.*\]\(https?://[^\)]+\)',
            r'\*.*स्रोत.*\*.*\[.*\]\(https?://[^\)]+\)',
            # Hebrew source patterns
            r'מקור.*\[.*\]\(https?://[^\)]+\)',
            r'\*.*מקור.*\*.*\[.*\]\(https?://[^\)]+\)',
            # General patterns for any formatted content with links
            r'\*\*.*\*\*.*\[.*\]\(https?://[^\)]+\)',
            # Any content with verification icons
            r'[✅⚠️❌⏱️🔌]',
            # Technical details patterns (English)
            r'confidence.*score.*\d+/100',
            r'validation.*score.*\d+/100',
            r'url verified:.*[true|false]',
            r'source.*confidence.*\d+/100',
            r'verification.*status.*',
            r'will be attached.*message',
            # Hindi verification patterns
            r'स्रोत सत्यापन.*',
            r'छवि सत्यापन.*',
            r'\(स्रोत सत्यापन:.*\)',
            r'\(छवि सत्यापन:.*\)',
            r'असत्यापित.*❌',
            r'सत्यापित.*✅',
            r'.*शामिल की जाएगी.*',
            r'.*संदेश के साथ.*',
            r'\[वित्तीय समाचार.*\]\(\)',
            # Arabic verification patterns
            r'التحقق من المصدر.*',
            r'التحقق من الصورة.*',
            r'\(التحقق من المصدر:.*\)',
            r'\(التحقق من الصورة:.*\)',
            r'غير مؤكد.*❌',
            r'مؤكد.*✅',
            r'.*سيتم إرفاق.*',
            r'.*مع هذه الرسالة.*',
            # Hebrew verification patterns
            r'אימות מקור.*',
            r'אימות תמונה.*',
            r'\(אימות מקור:.*\)',
            r'\(אימות תמונה:.*\)',
            r'לא מאומת.*❌',
            r'מאומת.*✅',
            r'.*יצורף להודעה.*',
            # Specific verification status messages
            r'.*verification status.*',
            r'.*source link.*verified.*',
            r'.*image verified.*',
            r'.*telegram content distributor.*',
            r'.*verified by.*distributor.*',
            r'.*verified by telegram.*',
            r'.*link.*image.*verified.*',
            r'^verification.*',
            r'.*content distributor.*',
        ]

        for pattern in indicators:
            if re.search(pattern, content, re.IGNORECASE | re.UNICODE):
                logger.info(f"Detected pre-formatted content with pattern: {pattern}")
                return True

        return False

    def _prepare_pre_formatted_content(self, content: str, language: str) -> str:
        """Prepare pre-formatted content for Telegram delivery using structured format."""
        logger.info(f"Preparing pre-formatted {language} content with structured format")

        # Extract structured data from pre-formatted content
        structured_data = self._extract_clean_content(content)

        logger.info(f"Extracted data: title='{structured_data.get('title', 'N/A')}', points={len(structured_data.get('key_points', []))}, implications={len(structured_data.get('market_implications', []))}")

        # Use the consistent clean message format
        message = self._create_clean_message(structured_data, language)

        # Validate HTML structure
        message = self._validate_html_structure(message)

        logger.info(f"Pre-formatted content prepared with structured format: {len(message)} chars")
        return message

    def _convert_markdown_to_telegram_html(self, content: str) -> str:
        """Convert markdown formatting to Telegram HTML."""
        # Convert bold markdown to HTML
        content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
        content = re.sub(r'\*(.*?)\*', r'<b>\1</b>', content)

        # Convert markdown links to HTML
        content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', content)

        # Clean up any remaining markdown
        content = re.sub(r'#+\s*', '', content)

        return content

    def _add_live_charts_section(self, message: str) -> str:
        """Add live charts section to all messages."""
        # Only add if charts section doesn't already exist
        if "Live Charts:" not in message and "المخططات المباشرة:" not in message and "चार्ट:" not in message and "גרפים:" not in message:
            # Default to English charts (charts are universal and labeled in English on Yahoo Finance)
            charts_section = """

<b>Live Charts:</b>
🔗 📊 <a href="https://finance.yahoo.com/quote/%5EGSPC/chart/?guccounter=1">S&P 500 Chart</a>
🔗 📈 <a href="https://finance.yahoo.com/quote/%5EIXIC/chart/">NASDAQ Chart</a>
🔗 📉 <a href="https://finance.yahoo.com/quote/%5EDJI/chart/">Dow Jones Chart</a>
🔗 ⚡ <a href="https://finance.yahoo.com/quote/%5EVIX/chart/">VIX Chart</a>
🔗 🏛️ <a href="https://finance.yahoo.com/quote/%5ETNX/chart/">10-Year Chart</a>
🔗 💰 <a href="https://finance.yahoo.com/quote/GC%3DF/chart/">Gold Chart</a>"""
            message += charts_section

        return message

    def _extract_stock_symbols(self, content: str) -> List[str]:
        stocks = re.findall(r'\b([A-Z]{2,5})\b', content)
        major_stocks = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "FDX", "INTC", "AMD"}
        relevant_stocks = [s for s in stocks if s in major_stocks]
        return list(set(relevant_stocks))[:3]


    def _strip_all_formatting(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\*\*([^*]*)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]*)\*', r'\1', text)
        text = re.sub(r'^\*+|\*+$', '', text)
        text = re.sub(r'^#+\s*', '', text)
        text = re.sub(r'[_`~\[\]]+', '', text)
        text = re.sub(r'^\s*[🎯📈📊🔔⚡💡]+\s*', '', text)
        # Remove verification status patterns like "(Source Verified: ✅)"
        text = re.sub(r'\(.*Verified.*✅.*\)', '', text)
        text = re.sub(r'\(.*✅.*Verified.*\)', '', text)
        # Remove confidence indicators and technical details (English)
        text = re.sub(r'confidence.*score.*\d+/100', '', text, flags=re.IGNORECASE)
        text = re.sub(r'validation.*score.*\d+/100', '', text, flags=re.IGNORECASE)
        text = re.sub(r'url verified:.*[true|false]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'source.*confidence.*\d+/100', '', text, flags=re.IGNORECASE)
        text = re.sub(r'verification.*status.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'will be attached.*message', '', text, flags=re.IGNORECASE)


        # Remove any lines containing verification icons (catch-all pattern)
        text = re.sub(r'.*[✅❌⚠️⏱️🔌].*', '', text)

        # Remove specific verification status messages
        text = re.sub(r'.*verification status.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'.*source link.*verified.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'.*image verified.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'.*telegram content distributor.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'.*verified by.*distributor.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'.*verified by telegram.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'.*link.*image.*verified.*', '', text, flags=re.IGNORECASE)
        # Remove the exact message pattern that was reported
        text = re.sub(r'Verification Status:\s*Source link and image verified by Telegram Content Distributor\.?', '', text, flags=re.IGNORECASE)

        # Remove any line that starts with "Verification" or contains verification-related phrases
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if not (line_lower.startswith('verification') or
                   'verified by' in line_lower or
                   'verification status' in line_lower or
                   'content distributor' in line_lower):
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)

        # Clean up any remaining empty parentheses or brackets
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]\(\s*\)', '', text)
        # Convert asterisk bullet points to proper bullet points
        text = re.sub(r'^\*\s+', '• ', text)
        text = re.sub(r'^\d+\.\s*', '', text)

        # Final catch-all for any remaining verification patterns
        text = re.sub(r'.*[Vv]erification.*[Ss]tatus.*', '', text)
        text = re.sub(r'.*[Ss]ource.*[Ll]ink.*[Vv]erified.*', '', text)

        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _convert_asterisk_bullets(self, message: str) -> str:
        """Convert asterisk bullet points to proper bullet points (•)."""
        if not message:
            return message

        lines = message.split('\n')
        converted_lines = []

        for line in lines:
            # Convert lines that start with asterisk followed by space to bullet points
            if re.match(r'^\s*\*\s+', line):
                line = re.sub(r'^(\s*)\*\s+', r'\1• ', line)
            converted_lines.append(line)

        return '\n'.join(converted_lines)


    def _remove_asterisks(self, content: str) -> str:
        if not content:
            return content
        content = re.sub(r'\*+', '', content)
        content = re.sub(r'\s+', ' ', content)
        return content

    def _add_rtl_support(self, content: str) -> str:
        if not content:
            return content
        arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
        hebrew_pattern = r'[\u0590-\u05FF]'
        has_arabic = re.search(arabic_pattern, content)
        has_hebrew = re.search(hebrew_pattern, content)
        if has_arabic or has_hebrew:
            rtl_start = '\u202B'
            rtl_end = '\u202C'
            content = f"{rtl_start}{content}{rtl_end}"
            content = re.sub(r'<b>', '<b dir="rtl">', content)
            content = re.sub(r'<i>', '<i dir="rtl">', content)
        return content

    def _is_metadata(self, text: str) -> bool:
        metadata_terms = ['search', 'metadata', 'completed', 'results', 'hours', 'utc', 'total']
        return any(term in text.lower() for term in metadata_terms)

    def _create_clean_message(self, data: Dict[str, Any], language: str) -> str:
        title = self._clean_for_telegram(data.get("title", "Market Update"))

        message_parts = []
        message_parts.append(f"<b>{title}</b>")
        message_parts.append("")

        # Live Charts section
        message_parts.append("<b>Live Charts:</b>")
        message_parts.append('🔗 📊 <a href="https://finance.yahoo.com/quote/%5EGSPC/chart/?guccounter=1">S&P 500 Chart</a>')
        message_parts.append('🔗 📈 <a href="https://finance.yahoo.com/quote/%5EIXIC/chart/">NASDAQ Chart</a>')
        message_parts.append('🔗 📉 <a href="https://finance.yahoo.com/quote/%5EDJI/chart/">Dow Jones Chart</a>')
        message_parts.append('🔗 ⚡ <a href="https://finance.yahoo.com/quote/%5EVIX/chart/">VIX Chart</a>')
        message_parts.append('🔗 🏛️ <a href="https://finance.yahoo.com/quote/%5ETNX/chart/">10-Year Chart</a>')
        message_parts.append('🔗 💰 <a href="https://finance.yahoo.com/quote/GC%3DF/chart/">Gold Chart</a>')

        final_message = "\n".join(message_parts)
        final_message = self._validate_html_structure(final_message)

        return final_message


    def _clean_for_telegram(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = self._strip_all_formatting(text)
        text = (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;'))
        return text.strip()

    def _validate_html_structure(self, message: str) -> str:
        message = re.sub(r'<b>\s*</b>', '', message)
        message = re.sub(r'</?ul>', '', message)
        message = re.sub(r'</?li>', '', message)
        message = re.sub(r'\n\s*\n\s*\n', '\n\n', message)
        return message.strip()

    def _send_content(self, message: str, image_data: Dict[str, str]) -> str:
        if image_data and image_data.get("url") and image_data.get("telegram_compatible", False):
            image_url = image_data["url"]
            image_title = image_data.get("title", "Financial Chart")
            logger.info(f"Attempting to send Telegram-compatible image: {image_title}")
            if self._send_photo(image_url, message):
                logger.info(f"Photo sent successfully: {image_title}")
                return f"Message with verified image sent successfully - {image_title}"
            else:
                logger.warning(f"Photo send failed for: {image_title}")
        else:
            if image_data:
                logger.warning(f"Image not Telegram-compatible: {image_data.get('title', 'Unknown')} (Compatible: {image_data.get('telegram_compatible', False)})")
        
        if self._send_message(message):
            return "Text message sent successfully with verified source (image failed or not compatible)"
        return "Failed to send message"

    def _send_photo(self, image_url: str, caption: str) -> bool:
        """Send photo to Telegram - supports local files and URLs"""
        try:
            import os

            logger.info(f"Attempting to send photo: {image_url[:100]}")

            # Check if it's a local file - send as file upload
            if os.path.isfile(image_url):
                logger.info(f"Local file exists, uploading to Telegram")
                return self._send_photo_from_file(image_url, caption)
            else:
                logger.warning(f"File not found locally: {image_url}")

            # For remote URLs - send directly
            logger.info(f"Sending remote URL: {image_url[:100]}")

            if isinstance(caption, str):
                caption_encoded = caption.encode('utf-8').decode('utf-8')
            else:
                caption_encoded = str(caption)

            payload = {
                "chat_id": self.chat_id,
                "photo": image_url,
                "caption": caption_encoded[:1024],
                "parse_mode": "HTML"
            }

            response = requests.post(f"{self.base_url}/sendPhoto", json=payload, timeout=30,
                                   headers={'Content-Type': 'application/json; charset=utf-8'})

            if response.ok:
                response_data = response.json()
                if response_data.get("ok", False):
                    logger.info(f"Telegram photo sent successfully")
                    return True
                else:
                    error_description = response_data.get('description', 'Unknown error')
                    logger.error(f"Telegram photo send failed: {error_description}")
                    return False
            else:
                logger.error(f"Telegram photo send HTTP error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending photo: {e}")
            return False

    def _send_photo_from_file(self, file_path: str, caption: str) -> bool:
        """Send a photo from a local file to Telegram"""
        try:
            if isinstance(caption, str):
                caption_encoded = caption.encode('utf-8').decode('utf-8')
            else:
                caption_encoded = str(caption)

            with open(file_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption_encoded[:1024],
                    'parse_mode': 'HTML'
                }

                response = requests.post(f"{self.base_url}/sendPhoto", data=data, files=files, timeout=30)

            if response.ok:
                response_data = response.json()
                if response_data.get("ok", False):
                    logger.info(f"Telegram photo sent successfully from local file")
                    return True
                else:
                    error_description = response_data.get('description', 'Unknown error')
                    logger.error(f"Telegram photo send failed: {error_description}")
                    return False
            else:
                logger.error(f"Telegram photo send HTTP error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending photo from file: {e}")
            return False


    def _ensure_message_with_footer_fits(self, message: str) -> str:
        """Ensure message fits within 4096 characters while preserving live charts footer."""
        if len(message) <= 4096:
            return message

        # Check if message has live charts footer
        footer_patterns = [
            "<b>Live Charts:</b>",
            "المخططات المباشرة:",
            "चार्ट:",
            "גרפים:"
        ]

        footer_start_pos = -1
        for pattern in footer_patterns:
            pos = message.find(pattern)
            if pos != -1:
                footer_start_pos = pos
                break

        if footer_start_pos == -1:
            # No footer found, simple truncation
            logger.warning(f"Message truncated to 4096 chars (no footer found)")
            return message[:4096]

        # Extract footer
        footer = message[footer_start_pos:]
        footer_length = len(footer)

        # Calculate available space for main content
        available_space = 4096 - footer_length - 10  # 10 chars buffer for "...\n\n"

        if available_space < 100:
            # Footer too long, truncate footer itself
            logger.warning(f"Footer too long, truncating to fit")
            footer = footer[:300] + "..."
            footer_length = len(footer)
            available_space = 4096 - footer_length - 10

        # Truncate main content and add footer
        main_content = message[:footer_start_pos]
        if len(main_content) > available_space:
            main_content = main_content[:available_space].rsplit('\n', 1)[0]  # Cut at last complete line
            truncated_message = main_content + "...\n\n" + footer
        else:
            truncated_message = message

        logger.info(f"Message length: {len(message)} → {len(truncated_message)} chars (footer preserved)")
        return truncated_message

    def _send_message(self, message: str) -> bool:
        try:
            if isinstance(message, str):
                message_encoded = message.encode('utf-8').decode('utf-8')
            else:
                message_encoded = str(message)

            # Ensure message fits within Telegram limit while preserving footer
            message_encoded = self._ensure_message_with_footer_fits(message_encoded)

            payload = {
                "chat_id": self.chat_id,
                "text": message_encoded,
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            }
            
            response = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=30,
                                   headers={'Content-Type': 'application/json; charset=utf-8'})
            
            if response.ok:
                response_data = response.json()
                if response_data.get("ok", False):
                    logger.info(f"Telegram text message sent successfully")
                    return True
                else:
                    error_description = response_data.get('description', 'Unknown error')
                    logger.error(f"Telegram message send failed: {error_description}")
                    return False
            else:
                logger.error(f"Telegram message send HTTP error: {response.status_code} - {response.text}")
                return False

        except requests.exceptions.Timeout:
            logger.error(f"Telegram message send timeout")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"Telegram message send connection error")
            return False
        except Exception as e:
            logger.error(f"Telegram message send unexpected error: {e}")
            return False

    def _translate_image_description(self, english_description: str, target_language: str) -> str:
        """Translate English AI image description to target language using Gemini"""
        try:
            import google.generativeai as genai

            # Configure Gemini if not already done
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                logger.warning("No GOOGLE_API_KEY found, returning English description")
                return english_description

            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            # Language mapping
            language_names = {
                'arabic': 'Arabic (العربية)',
                'hindi': 'Hindi (हिन्दी)',
                'hebrew': 'Hebrew (עברית)',
                'german': 'German (Deutsch)'
            }

            target_lang_name = language_names.get(target_language.lower(), target_language)

            prompt = f"""Translate this financial chart description to {target_lang_name}.

ORIGINAL ENGLISH DESCRIPTION:
{english_description}

RULES:
1. Translate ALL text to {target_lang_name}
2. Keep stock symbols in English (AAPL, MSFT, S&P 500, NASDAQ, etc.)
3. Keep numbers exactly as they are (percentages, prices, values)
4. Maintain the professional financial news tone
5. Keep it concise (1-2 sentences maximum)

Return ONLY the translated description, nothing else."""

            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 300,
                }
            )

            translated = response.text.strip()
            logger.info(f"Translated AI description to {target_language}: {translated[:100]}...")
            return translated

        except Exception as e:
            logger.error(f"Failed to translate image description: {e}")
            # Fallback to English description
            return english_description

    def _process_telegram_ready_content(self, content: str, language: str) -> str:
        """Process content with two-message format and send both messages."""
        try:
            # Check if this is the new two-message format
            if "=== TELEGRAM_TWO_MESSAGE_FORMAT ===" in content:
                return self._handle_two_message_format(content, language)

            # Fallback to old single-message format
            return self._handle_single_message_format(content, language)

        except Exception as e:
            logger.error(f"Failed to process Telegram-ready content: {e}")
            # Ultimate fallback
            message_content = content.split("---TELEGRAM_IMAGE_DATA---")[0].strip()
            message = self._convert_markdown_to_telegram_html(message_content)
            return self._send_content(message, {})

    def _handle_two_message_format(self, content: str, language: str) -> str:
        """Handle the new two-message format for Telegram delivery."""
        try:
            logger.info("Processing two-message format for Telegram delivery")
            logger.info(f"Content length: {len(content)} chars")
            logger.info(f"Has TELEGRAM_IMAGE_DATA: {'---TELEGRAM_IMAGE_DATA---' in content}")

            # Extract the two messages
            parts = content.split("=== TELEGRAM_TWO_MESSAGE_FORMAT ===")[1]
            sections = parts.split("---TELEGRAM_IMAGE_DATA---")

            logger.info(f"Number of sections after split: {len(sections)}")

            message_section = sections[0].strip()
            image_data_raw = {}

            if len(sections) > 1:
                try:
                    image_data_raw = json.loads(sections[1].strip())
                    logger.info(f"Parsed image_data_raw successfully")
                    logger.info(f"image_data_raw keys: {list(image_data_raw.keys())}")
                except Exception as e:
                    logger.warning(f"Failed to parse image data: {e}")

            # Extract Message 1 (caption) and Message 2 (summary)
            lines = message_section.split('\n')
            translated_caption = ""
            full_summary = ""
            current_section = None

            # Debug: Log first few lines to see structure
            logger.info(f"First 5 lines of message_section for {language}:")
            for i, line in enumerate(lines[:5]):
                logger.info(f"   Line {i}: '{line}'")

            for line in lines:
                # Use .strip() to handle potential whitespace issues with RTL text
                line_stripped = line.strip()

                if line_stripped.startswith("Message 1") or "Message 1 (Image Caption)" in line:
                    current_section = "caption"
                    logger.info(f"Found Message 1 header in {language}")
                    continue
                elif line_stripped.startswith("Message 2") or "Message 2 (Full Summary)" in line:
                    current_section = "summary"
                    logger.info(f"Found Message 2 header in {language}")
                    continue

                if current_section == "caption":
                    translated_caption += line + "\n"
                elif current_section == "summary":
                    full_summary += line + "\n"

            translated_caption = translated_caption.strip()
            full_summary = full_summary.strip()

            logger.info(f"[{language}] Extracted Message 2 (Summary): {len(full_summary)} chars")
            logger.info(f"[{language}] Extracted Message 1 (Caption): {len(translated_caption)} chars")

            # CRITICAL VALIDATION: Ensure Message 2 is properly extracted
            if not full_summary or len(full_summary) < 100:
                logger.error(f"[{language}] Message 2 extraction FAILED! Length: {len(full_summary) if full_summary else 0}")
                logger.error(f"Attempting alternative extraction method...")

                # Alternative method 1: Split by "Message 2" directly
                if "Message 2" in message_section:
                    parts = message_section.split("Message 2", 1)
                    if len(parts) > 1:
                        # Skip the header line, get everything after
                        temp = parts[1].split('\n', 1)
                        if len(temp) > 1:
                            full_summary = temp[1].strip()
                            logger.info(f"Alternative extraction successful: {len(full_summary)} chars")

                # Alternative method 2: Use entire translated content if still empty
                if not full_summary or len(full_summary) < 100:
                    logger.warning(f"Alternative extraction also failed, reconstructing from original...")
                    # Get the entire content after the first section marker
                    full_summary = message_section
                    logger.info(f"Using entire message_section: {len(full_summary)} chars")

            # FINAL VALIDATION: Reject if contains format markers (means parsing failed)
            if "===" in full_summary or "TELEGRAM" in full_summary[:100]:
                logger.error(f"[{language}] Message 2 contains format markers! Cleaning...")
                # Remove format markers
                full_summary = full_summary.replace("=== TELEGRAM_TWO_MESSAGE_FORMAT ===", "")
                full_summary = full_summary.replace("---TELEGRAM_IMAGE_DATA---", "")
                full_summary = full_summary.replace("TELEGRAMTWOMESSAGEFORMAT", "")
                # Remove "Message 1" and "Message 2" headers and lines containing only ===
                lines = full_summary.split('\n')
                cleaned_lines = []
                for line in lines:
                    stripped = line.strip()
                    # Skip format marker lines, message headers, and lines with only ===
                    if (stripped.startswith("Message 1") or
                        stripped.startswith("Message 2") or
                        stripped == "===" or
                        "TELEGRAM" in stripped):
                        continue
                    cleaned_lines.append(line)
                full_summary = '\n'.join(cleaned_lines).strip()
                logger.info(f"Cleaned format markers: {len(full_summary)} chars")

            # Note: We keep the dynamic title generated by the AI agent
            # The title is now created based on the day's main market theme
            full_summary = full_summary.strip()

            # Find latest screenshot and AI description
            import os
            from pathlib import Path

            image_data = {}
            english_ai_description = None

            try:
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                screenshots_dir = project_root / "output" / "screenshots"

                if screenshots_dir.exists():
                    screenshots = list(screenshots_dir.glob("chart_*.png"))
                    if screenshots:
                        latest_screenshot = max(screenshots, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Found latest screenshot: {latest_screenshot.name}")

                        # ALWAYS read AI description from JSON (needed for all languages)
                        image_results_dir = project_root / "output" / "image_results"
                        if image_results_dir.exists():
                            image_jsons = list(image_results_dir.glob("image_results_*.json"))
                            if image_jsons:
                                latest_json = max(image_jsons, key=lambda p: p.stat().st_mtime)
                                logger.info(f"Reading AI description from: {latest_json.name}")
                                with open(latest_json, 'r', encoding='utf-8') as f:
                                    results = json.load(f)
                                    if results.get('extracted_images'):
                                        first_image = results['extracted_images'][0]
                                        english_ai_description = first_image.get('image_description', '')
                                        logger.info(f"Loaded English AI description: {english_ai_description[:150]}...")
                                    else:
                                        logger.warning("No extracted_images found in JSON")
                            else:
                                logger.warning("No image_results JSON files found")
                        else:
                            logger.warning("image_results directory doesn't exist")

                        image_data = {
                            "url": str(latest_screenshot),
                            "telegram_compatible": True
                        }
                        logger.info(f"Image ready: {latest_screenshot.name}")
                    else:
                        logger.warning("No screenshots found")
                else:
                    logger.warning("Screenshots directory doesn't exist")

            except Exception as e:
                logger.error(f"Error finding image: {e}")

            # Determine caption based on language - ONLY use AI description
            if english_ai_description:
                if language.lower() == 'english':
                    # For English: use AI-generated description from Gemini Vision
                    final_caption = english_ai_description
                    logger.info(f"English: Using AI description as caption")
                else:
                    # For translations: translate the English AI description
                    logger.info(f"Translating English AI description to {language}...")
                    final_caption = self._translate_image_description(english_ai_description, language)
                    logger.info(f"{language}: Using translated AI description as caption")

                logger.info(f"Final caption ({language}): {final_caption[:150]}...")
            else:
                # No AI description available - skip image message entirely
                logger.warning(f"No AI description available for {language}, will skip image message")
                final_caption = None

            # Send Message 1: Image with Caption
            message1_success = False
            has_valid_image = False

            if image_data and image_data.get("url") and final_caption:
                has_valid_image = True
                message1_success = self._send_photo(image_data["url"], final_caption)
                if message1_success:
                    logger.info("Message 1 (Image + Caption) sent successfully")
                    # Add delay between messages
                    import time
                    time.sleep(2)
                else:
                    logger.warning("Message 1 failed - image not accessible, skipping image message")
                    has_valid_image = False

            # Send Message 2: Full Summary (always send)
            logger.info("Sending Message 2: Full Summary")
            full_summary_formatted = self._convert_markdown_to_telegram_html(full_summary)
            full_summary_formatted = self._validate_html_structure(full_summary_formatted)

            message2_success = self._send_message(full_summary_formatted)

            # Results based on what actually happened
            if has_valid_image and message1_success and message2_success:
                return "Two messages sent successfully: Image with caption + Full summary"
            elif not has_valid_image and message2_success:
                logger.info("No image available - sent full summary only")
                return "Full summary sent successfully (no image available)"
            elif has_valid_image and not message1_success and message2_success:
                logger.info("Image failed but summary sent - single message mode")
                return "Full summary sent successfully (image failed to send)"
            elif message2_success:
                return "Full summary sent successfully"
            else:
                return "Failed to send messages"

        except Exception as e:
            logger.error(f"Error in two-message format: {e}")
            return f"Error processing two-message format: {e}"

    def _handle_single_message_format(self, content: str, language: str) -> str:
        """Handle the old single-message format (fallback)."""
        # Split content and image data
        content_parts = content.split("---TELEGRAM_IMAGE_DATA---")
        message_content = content_parts[0].strip()

        try:
            image_data_raw = json.loads(content_parts[1].strip()) if len(content_parts) > 1 else {}
            logger.info(f"Extracted image data: Primary image - {image_data_raw.get('primary_image', {}).get('description', 'None')}")
        except:
            logger.warning("Failed to parse embedded image data, sending without images")
            image_data_raw = None

        # Clean and format the message content
        message = self._convert_markdown_to_telegram_html(message_content)
        message = self._add_live_charts_section(message)
        message = self._validate_html_structure(message)

        # Prepare image data for Telegram
        image_data = {}
        if image_data_raw and image_data_raw.get('primary_image'):
            primary_image = image_data_raw['primary_image']

            # Verify image URL accessibility
            image_url = primary_image.get('url', '')
            if self._verify_image_url(image_url):
                image_data = {
                    "url": image_url,
                    "title": primary_image.get('description', 'Financial Chart'),
                    "telegram_compatible": True
                }
                logger.info(f"Primary image verified: {image_data['title']}")

        # Send content with image
        word_count = len(message.split())
        logger.info(f"Sending single message: {word_count} words with image: {image_data.get('title', 'None')}")

        result = self._send_content(message, image_data)
        return result

