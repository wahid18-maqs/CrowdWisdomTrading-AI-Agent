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
from .image_finder import ImageFinder, ImageFinderInput

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class TelegramSenderInput(BaseModel):
    content: str = Field(description="The full financial summary content")
    language: Optional[str] = Field("english", description="Language for formatting")

class EnhancedTelegramSender(BaseTool):
    name: str = "telegram_sender"
    description: str = "Sends formatted financial summaries with relevant charts to Telegram"
    args_schema: Type[BaseModel] = TelegramSenderInput
    
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    base_url: Optional[str] = None
    image_finder: Optional[ImageFinder] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.image_finder = ImageFinder()
        
        if not self._test_credentials():
            raise ValueError("Invalid Telegram credentials")

    def _test_credentials(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            if not response.ok:
                return False
            
            test_payload = {"chat_id": self.chat_id, "text": "Test"}
            response = requests.post(f"{self.base_url}/sendMessage", json=test_payload, timeout=10)
            return response.ok
        except:
            return False

    def _run(self, content: str, language: str = "english") -> str:
        try:
            structured_data = self._extract_content(content)
            if not self._has_valid_content(structured_data):
                return "No valid financial content found"
            
            image_data = self._find_image(content, structured_data)
            message = self._create_message(structured_data, language)
            result = self._send_content(message, image_data)
            
            return result
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return f"Error: {e}"

    def _extract_content(self, content: str) -> Dict[str, Any]:
        """Extract structured content following the pattern"""
        data = {
            "title": "",
            "source": "",
            "source_url": "",
            "date": "",
            "key_points": [],
            "market_implications": []
        }
        
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return data

        # Simple title extraction
        for line in lines[:10]:
            if len(line) < 20:
                continue
            
            # Clean formatting
            clean_line = re.sub(r'[\*#🔔📊📈⚡🎯=\-]+|^\d+\.', '', line).strip()
            
            # Skip metadata
            if any(term in clean_line.upper() for term in ['HIGHLIGHTS', 'OVERVIEW', 'SEARCH', 'PHASE', '===']):
                continue
            
            # Skip prefixes
            if clean_line.lower().startswith(('source:', 'date:', 'link:')):
                continue
            
            if 20 <= len(clean_line) <= 150:
                data["title"] = clean_line
                break

        # Extract source and date
        for line in lines:
            if line.lower().startswith("source:"):
                data["source"] = re.sub(r'^Source:\s*', '', line, flags=re.IGNORECASE).strip()
            elif line.lower().startswith("link:"):
                url_match = re.search(r'https?://[^\s]+', line)
                if url_match:
                    data["source_url"] = url_match.group(0)

        # Extract date from content - DON'T override with current date
        date_match = re.search(r'Date:\s*([^\n]+)', content, re.IGNORECASE)
        if date_match:
            date_str = date_match.group(1).strip()
            try:
                if 'T' in date_str:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                data["date"] = dt.strftime("%Y-%m-%d")
            except:
                # Only use current date as last resort
                data["date"] = datetime.now().strftime("%Y-%m-%d")

        # Extract bullet points
        bullet_pattern = re.compile(r'^[-•*]\s*(.+)', re.IGNORECASE)
        section = None
        
        for line in lines:
            if 'key points' in line.lower():
                section = 'key_points'
                continue
            elif 'market implications' in line.lower():
                section = 'market_implications'
                continue
            
            bullet_match = bullet_pattern.match(line)
            if bullet_match:
                point = bullet_match.group(1).strip()
                if len(point) >= 10 and not self._is_metadata(point):
                    clean_point = self._clean_text(point)
                    
                    if section == 'key_points':
                        data["key_points"].append(clean_point[:200])
                    elif section == 'market_implications':
                        data["market_implications"].append(clean_point[:200])

        # Set defaults
        data["title"] = data["title"] or "Financial Market Update"
        data["source"] = data["source"] or "Financial News"
        data["source_url"] = data["source_url"] or "https://www.reuters.com"
        
        return data

    def _is_metadata(self, text: str) -> bool:
        """Simple metadata check"""
        text_upper = text.upper()
        metadata_keywords = ['HIGHLIGHTS', 'OVERVIEW', 'SEARCH', 'PHASE', 'METADATA', 'ANALYSIS']
        return any(keyword in text_upper for keyword in metadata_keywords)

    def _clean_text(self, text: str) -> str:
        """Clean formatting from text"""
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'^\*+|\*+$', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _has_valid_content(self, data: Dict[str, Any]) -> bool:
        """Check if we have valid content"""
        return bool(data.get("title") and (data.get("key_points") or data.get("market_implications")))

    def _find_image(self, original_content: str, structured_data: Dict[str, Any]) -> Dict[str, str]:
        """Find contextual image using ImageFinder first, then fallbacks"""
        try:
            # Prepare search content
            search_parts = []
            if structured_data.get("title"):
                search_parts.append(structured_data["title"])
            if structured_data.get("key_points"):
                search_parts.extend(structured_data["key_points"][:2])
            
            search_content = " ".join(search_parts)[:500]
            
            # Try ImageFinder
            image_input = ImageFinderInput(search_content=search_content, max_images=2)
            results_json = self.image_finder._run(**image_input.dict())
            
            if results_json:
                images = json.loads(results_json) if isinstance(results_json, str) else results_json
                if isinstance(images, list) and images:
                    best_image = self._select_best_image(images, structured_data)
                    if best_image and self._validate_image(best_image.get("url", "")):
                        return {
                            "url": best_image["url"],
                            "title": best_image.get("title", "Financial Chart"),
                            "source": "ImageFinder"
                        }
        except Exception as e:
            logger.debug(f"ImageFinder failed: {e}")

        # Fallback to other sources
        return self._get_fallback_image(structured_data)

    def _select_best_image(self, images: List[Dict], structured_data: Dict[str, Any]) -> Optional[Dict]:
        """Select best image based on relevance"""
        if not images:
            return None
        
        # Simple scoring based on title matches
        key_stocks = self._extract_stocks(structured_data)
        title_words = structured_data.get("title", "").lower().split()
        
        best_image = None
        best_score = 0
        
        for image in images:
            score = 0
            image_text = f"{image.get('title', '')} {image.get('source', '')}".lower()
            
            # Score stock matches
            for stock in key_stocks:
                if stock.lower() in image_text:
                    score += 10
            
            # Score title word matches
            for word in title_words:
                if len(word) > 3 and word in image_text:
                    score += 2
            
            if score > best_score:
                best_score = score
                best_image = image
        
        return best_image

    def _extract_stocks(self, data: Dict[str, Any]) -> List[str]:
        """Extract stock symbols from content"""
        text = f"{data.get('title', '')} {' '.join(data.get('key_points', []))}"
        major_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        found_stocks = re.findall(r'\b([A-Z]{2,5})\b', text)
        return [s for s in found_stocks if s in major_stocks][:3]

    def _get_fallback_image(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Get fallback image from Yahoo Finance or placeholder"""
        stocks = self._extract_stocks(data)
        
        if stocks:
            stock = stocks[0]
            url = f"https://chart.yahoo.com/z?s={stock}&t=1d&q=l&l=on&z=s&p=s"
            if self._validate_image(url):
                return {"url": url, "title": f"{stock} Chart", "source": "Yahoo Finance"}
        
        # Placeholder as last resort
        placeholder_url = "https://via.placeholder.com/400x300/1565C0/ffffff?text=Financial+Chart"
        return {"url": placeholder_url, "title": "Market Chart", "source": "Placeholder"}

    def _validate_image(self, url: str) -> bool:
        """Simple image validation"""
        if not url or not url.startswith("http"):
            return False
        
        if "via.placeholder.com" in url:
            return True
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.head(url, headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False

    def _create_message(self, data: Dict[str, Any], language: str) -> str:
        """Create simplified message: Catchy Title -> Key Points -> Market Implications ONLY"""
        
        title = self._escape_html(data.get("title", "Financial Market Update"))
        
        # Build ultra-simple message format
        message = f"<b>{title}</b>\n\n"
        
        # Key Points
        if data.get("key_points"):
            message += "<b>Key Points:</b>\n"
            for point in data["key_points"]:
                message += f"• {self._escape_html(point)}\n"
            message += "\n"
        
        # Market Implications
        if data.get("market_implications"):
            message += "<b>Market Implications:</b>\n"
            for impl in data["market_implications"]:
                message += f"• {self._escape_html(impl)}\n"
        
        # RTL support for Arabic
        if language.lower() in ["arabic", "ar"]:
            message = f"\u200F{message}"
        
        return message.strip()

    def _escape_html(self, text: str) -> str:
        """Escape HTML characters for Telegram"""
        if not isinstance(text, str):
            text = str(text)
        
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))

    def _send_content(self, message: str, image_data: Dict[str, str]) -> str:
        """Send message with image if available, otherwise send text-only"""
        # Try to send with image first
        if image_data and image_data.get("url"):
            if self._send_photo(image_data["url"], message):
                return f"Message with image sent successfully (source: {image_data.get('source', 'unknown')})"
            else:
                logger.warning("Photo send failed, falling back to text-only message")
        
        # Always try to send text message if image fails or no image available
        if self._send_message(message):
            return "Text message sent successfully (no image available)"
        
        return "Failed to send message"

    def _send_photo(self, image_url: str, caption: str) -> bool:
        """Send photo with caption to Telegram"""
        try:
            payload = {
                "chat_id": self.chat_id,
                "photo": image_url,
                "caption": caption[:1024],
                "parse_mode": "HTML"
            }
            
            response = requests.post(f"{self.base_url}/sendPhoto", json=payload, timeout=30)
            return response.ok and response.json().get("ok", False)
        except Exception as e:
            logger.warning(f"Photo send failed: {e}")
            return False

    def _send_message(self, message: str) -> bool:
        """Send text message to Telegram"""
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4096],
                "parse_mode": "HTML"
            }
            
            response = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=30)
            return response.ok and response.json().get("ok", False)
        except Exception as e:
            logger.warning(f"Message send failed: {e}")
            return False