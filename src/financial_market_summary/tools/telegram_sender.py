import json
import re
import time
import requests
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai.tools import BaseTool 
from .image_finder import ImageFinder, ImageFinderInput

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TelegramSenderInput(BaseModel):
    """Input schema for the EnhancedTelegramSender tool."""
    content: str = Field(description="The full financial summary content, including sections like title, key points, and market implications.")
    language: Optional[str] = Field("english", description="The language of the summary, used for formatting. Defaults to 'english'.")

class EnhancedTelegramSender(BaseTool): # Inherits from BaseTool
    """
    A CrewAI tool to send formatted financial summaries with relevant images to a Telegram channel.
    
    The tool extracts key information from a provided text, finds a suitable financial chart
    or image from various sources (ImageFinder, Yahoo Finance, Finviz), and sends a
    well-formatted message with the image to a specified Telegram channel.
    
    It handles:
    - HTML formatting for bolding, links, and bullet points.
    - Image validation and fallback mechanisms.
    - Rate limiting and error handling for network requests.
    - Detection and filtering of metadata from the input text.
    """
    name: str = "telegram_sender"
    description: str = (
        "Sends a clean, structured financial summary with a relevant chart to a Telegram channel. "
        "The summary is formatted using HTML, and includes a title, key points, market implications, "
        "and a relevant image (financial chart) to provide a complete and professional update."
    )
    args_schema: Type[BaseModel] = TelegramSenderInput
    
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    base_url: Optional[str] = None
    image_finder: Optional[Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Explicitly fetch environment variables to avoid FieldInfo issues
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env file")
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        logger.info(f"Telegram initialized: bot_token=****{self.bot_token[-4:]}, chat_id={self.chat_id}")
        
        if not self._test_credentials():
            logger.error("Telegram credentials invalid")
            raise ValueError("Invalid TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    def _test_credentials(self) -> bool:
        """Test Telegram credentials with a simple API call."""
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            if not response.ok or not response.json().get("ok"):
                logger.error(f"Bot token invalid: {response.text}")
                return False
            
            payload = {"chat_id": self.chat_id, "text": "Credential test"}
            response = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=10)
            if not response.ok or not response.json().get("ok"):
                logger.error(f"Chat access failed: {response.text}")
                return False
            
            logger.info("Telegram credentials validated successfully.")
            return True
        except Exception as e:
            logger.error(f"Credential test failed: {e}")
            return False

    def _run(self, content: str, language: str = "english") -> str:
        """Main execution method to process and send the summary."""
        try:
            logger.info(f"Processing content for {language}")
            
            structured_data = self._extract_content(content)
            if not self._has_valid_content(structured_data):
                return "No valid financial content found to send."
            
            image_data = self._find_image(structured_data)
            
            message = self._create_message(structured_data, language)
            result = self._send_content(message, image_data)
            
            logger.info(f"Workflow completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return f"Error: {e}"

    def _extract_content(self, content: str) -> Dict[str, Any]:
        """Extract clean, structured content from raw input."""
        logger.debug("Extracting structured content")
        
        data = {
            "title": "",
            "source": "",
            "source_url": "",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "key_points": [],
            "market_implications": [],
            "articles": []
        }
        
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            logger.warning("No content lines found")
            return data
        
        title_patterns = [
            r'^\*\*\d+\.\s*(.+?)\*\*$',    # **1. Title**
            r'^\*\*(.+?)\*\*$',            # **Title**
            r'^#\s+(.+?)$',                # # Title
            r'^\d+\.\s*(.+?)$',            # 1. Title
            r'^[^\n•*-].{20,100}$'         # Any non-bulleted line between 20-100 chars
        ]
        
        for line in lines[:10]:
            for pattern in title_patterns:
                match = re.match(pattern, line)
                if match:
                    potential_title = match.group(1).strip() if match.groups() else line.strip()
                    if (len(potential_title) > 20 and 
                        not self._is_metadata(potential_title) and
                        not potential_title.lower().startswith(("source:", "link:", "date:", "key ", "market "))):
                        data["title"] = potential_title[:100]
                        logger.debug(f"Extracted title: {data['title']}")
                        break
            if data["title"]:
                break
        
        if not data["title"]:
            for line in lines[:5]:
                if (len(line) > 30 and 
                    not self._is_metadata(line) and 
                    not line.lower().startswith(("source:", "date:", "link:", "-", "•", "key ", "market "))):
                    data["title"] = re.sub(r'^\*+|\*+$', '', line).strip()[:100]
                    logger.debug(f"Fallback title extracted: {data['title']}")
                    break
        
        for line in lines:
            if line.lower().startswith("source:"):
                data["source"] = re.sub(r'^Source:\s*', '', line, flags=re.IGNORECASE).strip()
            elif line.lower().startswith("link:"):
                url_match = re.search(r'https?://[^\s]+', line, re.IGNORECASE)
                if url_match:
                    data["source_url"] = url_match.group(0)
        
        date_match = re.search(r'Date:\s*([^\n]+)', content, re.IGNORECASE)
        if date_match:
            try:
                date_str = date_match.group(1).strip()
                for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d/%m/%Y"):
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        data["date"] = dt.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.debug(f"Date parsing failed: {e}")
        
        bullet_pattern = re.compile(r'^[-•*]\s*(.+)', re.IGNORECASE)
        section = None
        
        for line in lines:
            if line.lower().startswith(('key points:', 'key highlights:')):
                section = 'key_points'
                continue
            elif line.lower().startswith(('market implications:', 'market impact:')):
                section = 'market_implications'
                continue
            
            bullet_match = bullet_pattern.match(line)
            if bullet_match:
                point = bullet_match.group(1).strip()
                
                if self._is_metadata(point) or len(point) < 10:
                    continue
                
                clean_point = self._clean_text(point)
                
                if section == 'key_points' or (not section and not self._is_market_related(clean_point)):
                    data["key_points"].append(clean_point[:200])
                elif section == 'market_implications' or (not section and self._is_market_related(clean_point)):
                    data["market_implications"].append(clean_point[:200])
        
        data["key_points"] = data["key_points"][:5]
        data["market_implications"] = data["market_implications"][:3]
        
        data["title"] = data["title"] or "Financial Market Update"
        data["source"] = data["source"] or "Financial News"
        data["source_url"] = data["source_url"] or "https://www.reuters.com"
        
        logger.debug(f"Extracted: title='{data['title']}', {len(data['key_points'])} points, {len(data['market_implications'])} implications")
        return data

    def _is_metadata(self, text: str) -> bool:
        """Enhanced metadata detection to filter out non-content text."""
        clean_text = re.sub(r'^\*+|\*+$', '', text).strip()
        
        metadata_terms = [
            'KEY HIGHLIGHTS', 'MARKET OVERVIEW', 'CONTENT ANALYSIS',
            'DETAILED NEWS', 'SEARCH METADATA', 'Key Stocks', 'Key Themes',
            'ECONOMIC HIGHLIGHTS', 'SECTOR ANALYSIS', 'Key Movers'
        ]
        
        clean_upper = clean_text.upper()
        for term in metadata_terms:
            if term in clean_upper:
                return True
        
        article_patterns = [
            r'^\d+\.\s+.*(?:\.\.\.|after|ends|closes)',
            r'^Wall Street.*(?:after|ends|mixed)',
            r'^Dow.*(?:points|closes|up|down)',
            r'^S&P.*(?:inches|falls|rises)',
            r'^\w+\s+shares?\s+(?:rise|fall|gain|lose)',
            r'^\w+\s+stock.*(?:surge|drop|climb)',
            r'.*\s+(?:after|following)\s+.*(?:earnings|results|data|cut)',
        ]
        
        for pattern in article_patterns:
            if re.match(pattern, clean_text, re.IGNORECASE):
                return True
        
        if len(clean_text.split()) < 3:
            return True
        
        if re.match(r'^[A-Z]{2,5}(?:,\s*[A-Z]{2,5})*$', clean_text):
            return True
        
        return False

    def _is_market_related(self, text: str) -> bool:
        """Check if text is market-related content."""
        market_terms = ['market', 'trading', 'investors', 'fed', 'rate', 'policy', 'economic', 'inflation', 'stock', 'bond', 'currency']
        return any(term in text.lower() for term in market_terms)

    def _clean_text(self, text: str) -> str:
        """Clean text by removing formatting and metadata."""
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'^\*+|\*+$', '', text)
        
        metadata_prefixes = [
            r'^KEY HIGHLIGHTS?[:\s]*',
            r'^MARKET OVERVIEW[:\s]*', 
            r'^CONTENT ANALYSIS[:\s]*'
        ]
        for prefix in metadata_prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        
        return re.sub(r'\s+', ' ', text).strip()

    def _has_valid_content(self, data: Dict[str, Any]) -> bool:
        """Check if extracted data has valid content."""
        return bool(data.get("title") and (data.get("key_points") or data.get("market_implications")))

    def _find_image(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Find relevant image with ImageFinder as first priority."""
        logger.info("Searching for financial chart")
        
        image_sources = [
            self._use_original_image_finder,
            self._get_yahoo_finance_chart,
            self._get_finviz_chart,
            self._get_placeholder_image
        ]
        
        for source_func in image_sources:
            try:
                image_data = source_func(data)
                if image_data and image_data.get("url") and self._validate_image_thoroughly(image_data["url"]):
                    logger.info(f"Found valid image from {source_func.__name__}: {image_data.get('title', 'Unknown')}")
                    return image_data
            except Exception as e:
                logger.debug(f"{source_func.__name__} failed: {e}")
                continue
        
        logger.warning("All image sources failed, returning empty")
        return {}

    def _get_placeholder_image(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Get a reliable placeholder image."""
        placeholder_urls = [
            "https://via.placeholder.com/400x300/1565C0/ffffff?text=📊+Financial+Chart",
            "https://via.placeholder.com/400x300/2E7D32/ffffff?text=Market+Analysis"
        ]
        
        for url in placeholder_urls:
            if self._validate_image_thoroughly(url):
                return {
                    "url": url,
                    "title": "Market Analysis Chart",
                    "source": "Placeholder"
                }
        
        return {}

    def _get_yahoo_finance_chart(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Get a chart from Yahoo Finance."""
        stocks = self._extract_primary_stocks(data)
        if not stocks:
            return {}
        
        primary_stock = stocks[0]
        chart_urls = [
            f"https://chart.yahoo.com/z?s={primary_stock}&t=1d&q=l&l=on&z=s&p=s",
            f"https://chart.yahoo.com/z?s={primary_stock}&t=5d&q=l&l=on&z=m&p=s"
        ]
        
        for url in chart_urls:
            if self._validate_image_thoroughly(url):
                return {
                    "url": url,
                    "title": f"{primary_stock} Stock Chart",
                    "source": "Yahoo Finance"
                }
        return {}

    def _get_finviz_chart(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Get a chart from Finviz."""
        stocks = self._extract_primary_stocks(data)
        if not stocks:
            return {}
        
        primary_stock = stocks[0]
        chart_urls = [
            f"https://finviz.com/chart.ashx?t={primary_stock}&ty=c&ta=1&p=d&s=l",
            f"https://finviz.com/chart.ashx?t={primary_stock}&ty=c&ta=0&p=d&s=m"
        ]
        
        for url in chart_urls:
            if self._validate_image_thoroughly(url):
                return {
                    "url": url,
                    "title": f"{primary_stock} Technical Chart",
                    "source": "Finviz"
                }
        return {}

    def _use_original_image_finder(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Use the original image finder as a general web search fallback."""
        if not self.image_finder:
            return {}
        
        try:
            search_query = f"{data['title'][:50]} financial chart September 2025"
            image_input = ImageFinderInput(search_content=search_query, max_images=3)
            results_json = self.image_finder._run(**image_input.dict())
            results = json.loads(results_json) if results_json else []
            
            for image in results:
                url = image.get("url", "")
                if self._validate_image_thoroughly(url):
                    return {
                        "url": url,
                        "title": image.get("title", "Financial Chart"),
                        "source": image.get("source", "Web Search")
                    }
        except Exception as e:
            logger.debug(f"Original image finder failed: {e}")
        
        return {}

    def _extract_primary_stocks(self, data: Dict[str, Any]) -> List[str]:
        """Extract primary stock symbols from the content."""
        stocks = []
        text_to_search = f"{data.get('title', '')} {' '.join(data.get('key_points', []))}"
        
        major_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "AMD", "INTC", "CRM", "UBER", "PYPL", "SPOT", "JPM", "BAC", "WFC",
            "V", "MA", "HD", "WMT", "DIS", "KO", "PFE", "JNJ", "XOM", "CVX",
            "ULTA", "DKS", "GPS", "DELL", "BABA"
        ]
        
        stock_pattern = r'\b([A-Z]{2,5})\b'
        found_stocks = re.findall(stock_pattern, text_to_search)
        
        for stock in found_stocks:
            if stock in major_stocks and stock not in stocks:
                stocks.append(stock)
        
        return stocks[:3]

    def _validate_image_thoroughly(self, url: str) -> bool:
        """Thoroughly validate an image URL for Telegram compatibility."""
        if not url or not url.startswith("http"):
            return False
        
        if "via.placeholder.com" in url or "picsum.photos" in url:
            return True
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.head(url, headers=headers, timeout=8, allow_redirects=True)
            
            if response.status_code != 200:
                logger.debug(f"Image URL returned {response.status_code}: {url}")
                return False
            
            content_type = response.headers.get('content-type', '').lower()
            valid_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp']
            if any(img_type in content_type for img_type in valid_types):
                return True
            
            response = requests.get(url, headers=headers, timeout=8, stream=True, allow_redirects=True)
            if response.status_code != 200:
                return False
            
            chunk = next(response.iter_content(1024), b'')
            
            if chunk.startswith(b'<!DOCTYPE') or chunk.startswith(b'<html') or b'<title>' in chunk[:500]:
                logger.debug(f"URL returns an HTML page, not an image: {url}")
                return False
            
            image_signatures = [
                b'\xff\xd8\xff', b'\x89PNG\r\n\x1a\n', b'GIF87a', b'GIF89a', b'RIFF'
            ]
            
            if any(chunk.startswith(sig) for sig in image_signatures):
                logger.debug(f"Valid image signature found: {url}")
                return True
            
            logger.debug(f"No valid image signature found for: {url}")
            return False
            
        except Exception as e:
            logger.debug(f"Image validation failed for {url}: {e}")
            return False

    def _validate_image(self, url: str) -> bool:
        """Simple image validation (legacy method for compatibility)."""
        return self._validate_image_thoroughly(url)

    def _fallback_image(self) -> Dict[str, str]:
        """Legacy fallback method, uses placeholder."""
        return self._get_placeholder_image({})

    def _create_message(self, data: Dict[str, Any], language: str) -> str:
        """Create a clean HTML-formatted message for Telegram."""
        title = self._escape_html(data["title"])
        source = self._escape_html(data["source"])
        source_url = data["source_url"]
        date = data["date"]
        
        message = f"📢 <b>{title}</b>\n"
        message += f"📰 Source: <a href=\"{source_url}\">{source}</a>\n"
        message += f"📅 Date: {date}\n"
        
        if data["key_points"]:
            message += "\n🔑 <b>Key Points:</b>\n"
            for point in data["key_points"]:
                clean_point = self._escape_html(point)
                message += f"• {clean_point}\n"
        
        if data["market_implications"]:
            message += "\n📊 <b>Market Implications:</b>\n"
            for implication in data["market_implications"]:
                clean_implication = self._escape_html(implication)
                message += f"• {clean_implication}\n"
        
        if language.lower() in ["arabic", "ar"]:
            message = f"\u200F{message}"
        
        return message.strip()

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for safe rendering in Telegram."""
        if not isinstance(text, str):
            text = str(text)
        
        return (text.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                        .replace('"', '&quot;')
                        .replace("'", '&#x27;'))

    def _send_content(self, message: str, image_data: Dict[str, str]) -> str:
        """Send the formatted message with an image to Telegram."""
        if image_data and image_data.get("url"):
            if self._send_photo(image_data["url"], message):
                return "Message with image sent successfully"
        
        if self._send_message(message):
            return "Text message sent successfully"
        
        return "Failed to send message"

    def _send_photo(self, image_url: str, caption: str) -> bool:
        """Send a photo with a caption to Telegram."""
        try:
            payload = {
                "chat_id": self.chat_id,
                "photo": image_url,
                "caption": caption[:1024],
                "parse_mode": "HTML"
            }
            
            response = requests.post(f"{self.base_url}/sendPhoto", json=payload, timeout=30)
            if response.ok and response.json().get("ok"):
                logger.info("Photo sent successfully")
                return True
            else:
                logger.warning(f"Photo send failed: {response.text}")
                return False
                
        except Exception as e:
            logger.warning(f"Photo send error: {e}")
            return False

    def _send_message(self, message: str) -> bool:
        """Send a text-only message to Telegram."""
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4096],
                "parse_mode": "HTML"
            }
            
            response = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=30)
            if response.ok and response.json().get("ok"):
                logger.info("Message sent successfully")
                return True
            else:
                logger.warning(f"Message send failed: {response.text}")
                return False
                
        except Exception as e:
            logger.warning(f"Message send error: {e}")
            return False

    def _extract_key_stocks(self, content: str) -> List[str]:
        """Extract stock symbols."""
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, content)
        major_stocks = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "AMD", "INTC", "CRM", "ADBE", "PYPL", "UBER", "JPM", "BAC",
            "ULTA", "DKS", "GPS", "DELL", "QAN", "BABA", "PUM"
        }
        return [stock for stock in potential_stocks if stock in major_stocks][:5]

    def _extract_key_movers_with_performance(self, content: str) -> List[Dict[str, str]]:
        """Extract stock movers with performance data."""
        key_movers = []
        pattern = r'([A-Z]{2,5})\s+(?:surged?|jumped?|gained?|rose|dropped?|fell|declined?)\s+([\d.]+%)'
        
        matches = re.findall(pattern, content, re.IGNORECASE)
        for symbol, performance in matches:
            key_movers.append({
                "symbol": symbol,
                "performance": performance
            })
        return key_movers[:5]

    def _identify_primary_topic(self, content: str) -> str:
        """Identify the primary topic of the content."""
        content_lower = content.lower()
        if "fed" in content_lower or "federal reserve" in content_lower:
            return "federal reserve policy"
        elif "earnings" in content_lower:
            return "earnings report"
        elif any(term in content_lower for term in ["merger", "acquisition"]):
            return "corporate merger"
        else:
            return "financial news"