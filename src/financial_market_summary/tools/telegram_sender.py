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
            response = requests.post(f"{self.base_url}/sendMessage", json=test_payload, timeout=30)
            return response.ok
        except:
            return False

    def _run(self, content: str, language: str = "english") -> str:
        try:
            # Parse content and create clean structure
            structured_data = self._extract_clean_content(content)
            
            # Debug: Show found article links
            self._debug_article_links(structured_data)
            
            if not self._has_valid_content(structured_data):
                return "No valid financial content found"
            
            # Find image
            image_data = self._find_image(content, structured_data)
            
            # Create clean message with proper structure
            message = self._create_clean_message(structured_data, language)
            
            # Send content
            result = self._send_content(message, image_data)
            
            return result
        except Exception as e:
            logger.error(f"Telegram sender failed: {e}")
            return f"Error: {e}"

    def _extract_clean_content(self, content: str) -> Dict[str, Any]:
        """Extract and clean content with robust parsing and article link validation"""
        data = {
            "title": "",
            "source": "Financial News",
            "source_url": "",
            "key_points": [],
            "market_implications": [],
            "article_links": [],
            "validation_score": 0
        }
        
        # Clean HTML but preserve structure for parsing
        clean_content = re.sub(r'<[^>]+>', '', content)
        lines = [line.strip() for line in clean_content.splitlines() if line.strip()]
        
        if not lines:
            return data

        # Extract title - enhanced search
        title_found = False
        for line in lines[:25]:
            # Skip metadata lines
            if any(skip in line.upper() for skip in [
                'SEARCH WINDOW', 'MARKET OVERVIEW', 'KEY HIGHLIGHTS', 'BREAKING NEWS',
                'SEARCH METADATA', 'REAL-TIME', 'UTC', '===', 'PHASE'
            ]):
                continue
            
            clean_line = self._strip_all_formatting(line)
            
            # Look for substantial content that could be a title
            if 25 <= len(clean_line) <= 200 and not clean_line.startswith('•') and not title_found:
                # Additional validation - should contain market-related terms
                market_terms = ['market', 'stock', 'trading', 'earning', 'fed', 'economic', 'financial', 'sector']
                if any(term in clean_line.lower() for term in market_terms):
                    data["title"] = clean_line
                    title_found = True
                    break

        # Extract article links with enhanced parsing
        article_links_data = self._extract_validated_article_links(lines)
        data["article_links"] = article_links_data["links"]
        data["validation_score"] = article_links_data["score"]

        # Select best source URL
        if data["article_links"]:
            best_article = max(data["article_links"], key=lambda x: x.get("score", 0))
            data["source_url"] = best_article["url"]
            data["source"] = best_article.get("title", best_article.get("source", "Financial News"))[:60] + "..."

        # Extract content sections with multiple approaches
        self._extract_content_sections(lines, data)

        # Fallback content generation
        if not data["key_points"]:
            data["key_points"] = self._generate_fallback_points(content)
        
        if not data["market_implications"]:
            data["market_implications"] = self._generate_fallback_implications(content)

        # Set defaults
        data["title"] = data["title"] or "US Market Update"
        if not data["source_url"]:
            data["source_url"] = "https://www.marketwatch.com"
        
        return data

    def _extract_validated_article_links(self, lines: List[str]) -> Dict[str, Any]:
        """Enhanced article link extraction with multiple patterns"""
        article_data = {"links": [], "score": 0}
        
        trusted_domains = {
            'reuters.com': 95, 'bloomberg.com': 95, 'cnbc.com': 90,
            'marketwatch.com': 85, 'investing.com': 80, 'benzinga.com': 75,
            'yahoo.com': 70, 'wsj.com': 95, 'ft.com': 90
        }
        
        current_article = None
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # Multiple patterns for article detection
            article_patterns = [
                r'\*\*\d+\.\s*(.+?)\*\*',  # **1. Title**
                r'^(\d+\.\s*.+?)$',        # 1. Title (without **)
                r'#{1,3}\s*(.+?)$'         # ### Title
            ]
            
            for pattern in article_patterns:
                match = re.search(pattern, line_clean)
                if match:
                    # Save previous article
                    if current_article and current_article.get("url"):
                        article_data["links"].append(current_article)
                    
                    # Start new article
                    title = self._strip_all_formatting(match.group(1))
                    current_article = {
                        "title": title,
                        "url": "",
                        "source": "",
                        "score": 0
                    }
                    break
            
            # Look for source and link information
            if current_article:
                if 'source:' in line_clean.lower():
                    source_match = re.search(r'source:\s*(.+)', line_clean, re.IGNORECASE)
                    if source_match:
                        current_article["source"] = source_match.group(1).strip()
                
                # Enhanced link detection
                link_patterns = [
                    r'🔗\s*link:\s*(https?://[^\s]+)',
                    r'🔗\s*(https?://[^\s]+)',
                    r'link:\s*(https?://[^\s]+)',
                    r'(https?://[^\s]+)'
                ]
                
                for link_pattern in link_patterns:
                    url_match = re.search(link_pattern, line_clean, re.IGNORECASE)
                    if url_match:
                        url = url_match.group(1)
                        current_article["url"] = url
                        
                        # Score the article
                        for domain, score in trusted_domains.items():
                            if domain in url.lower():
                                current_article["score"] = score
                                break
                        else:
                            current_article["score"] = 50
                        break
        
        # Don't forget the last article
        if current_article and current_article.get("url"):
            article_data["links"].append(current_article)
        
        # Calculate overall score
        if article_data["links"]:
            scores = [article.get("score", 0) for article in article_data["links"]]
            article_data["score"] = min(100, sum(scores) // len(scores))
        
        # Filter high-quality links
        article_data["links"] = [
            article for article in article_data["links"] 
            if article.get("score", 0) >= 60 and article.get("url")
        ]
        
        return article_data

    def _extract_content_sections(self, lines: List[str], data: Dict[str, Any]):
        """Extract key points and market implications with multiple approaches"""
        current_section = None
        
        for line in lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()
            
            # Section headers
            if any(phrase in line_lower for phrase in [
                'key points', 'key highlights', 'main points', 'highlights'
            ]):
                current_section = 'key_points'
                continue
            elif any(phrase in line_lower for phrase in [
                'market implications', 'implications', 'market impact', 'outlook'
            ]):
                current_section = 'market_implications'
                continue
            
            # Extract bullet points with multiple markers
            bullet_patterns = [r'^[•\-*]\s+(.+)', r'^[\d]+\.\s+(.+)']
            
            for pattern in bullet_patterns:
                match = re.match(pattern, line_clean)
                if match:
                    point_text = self._strip_all_formatting(match.group(1))
                    
                    if len(point_text) >= 20 and not self._is_metadata(point_text):
                        if current_section == 'key_points':
                            data["key_points"].append(point_text[:300])
                        elif current_section == 'market_implications':
                            data["market_implications"].append(point_text[:300])
                    break

    def _debug_article_links(self, data: Dict[str, Any]) -> None:
        """Debug method to show found article links"""
        article_links = data.get("article_links", [])
        validation_score = data.get("validation_score", 0)
        
        logger.info("=== ARTICLE LINKS DEBUG ===")
        logger.info(f"Validation Score: {validation_score}/100")
        logger.info(f"Articles Found: {len(article_links)}")
        
        for i, article in enumerate(article_links, 1):
            logger.info(f"  {i}. Title: {article.get('title', 'N/A')[:50]}...")
            logger.info(f"     Source: {article.get('source', 'N/A')}")
            logger.info(f"     URL: {article.get('url', 'N/A')}")
            logger.info(f"     Score: {article.get('score', 0)}/100")
            logger.info("     ---")
        
        if article_links:
            main_url = data.get("source_url", "")
            logger.info(f"Selected Main Source: {main_url}")
        
        logger.info("=== END DEBUG ===")

    def _strip_all_formatting(self, text: str) -> str:
        """Remove ALL formatting artifacts"""
        if not text:
            return ""
        
        # Remove markdown and HTML
        text = re.sub(r'\*\*([^*]*)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]*)\*', r'\1', text)      # *italic*
        text = re.sub(r'^\*+|\*+$', '', text)           # leading/trailing *
        text = re.sub(r'^#+\s*', '', text)              # headers
        text = re.sub(r'[_`~\[\]]+', '', text)          # other markdown
        text = re.sub(r'^\s*[🎯📈📊🔔⚡💡]+\s*', '', text)  # emoji prefixes
        text = re.sub(r'^\d+\.\s*', '', text)           # number prefixes
        text = re.sub(r'\s+', ' ', text)                # normalize whitespace
        
        return text.strip()

    def _is_metadata(self, text: str) -> bool:
        """Check if text is metadata"""
        metadata_terms = ['search', 'metadata', 'completed', 'results', 'hours', 'utc', 'total']
        return any(term in text.lower() for term in metadata_terms)

    def _generate_fallback_points(self, content: str) -> List[str]:
        """Generate fallback key points"""
        points = []
        
        # Extract stock mentions
        stocks = re.findall(r'\b([A-Z]{2,5})\b', content)
        major_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "FDX", "ADSK"]
        relevant_stocks = [s for s in stocks if s in major_stocks]
        
        if relevant_stocks:
            points.append(f"Key stocks in focus include {', '.join(relevant_stocks[:4])}")
        
        if 'earnings' in content.lower():
            points.append("Corporate earnings reports continue to drive market activity")
        
        if 'fed' in content.lower() or 'federal' in content.lower():
            points.append("Federal Reserve policy decisions remain a key market factor")
        
        # Default fallbacks
        if not points:
            points = [
                "US equity markets showing mixed performance across sectors",
                "Investor attention focused on corporate earnings and economic data",
                "Market sentiment reflecting current economic and policy conditions"
            ]
        
        return points[:4]

    def _generate_fallback_implications(self, content: str) -> List[str]:
        """Generate fallback market implications"""
        implications = []
        
        if 'earnings' in content.lower():
            implications.append("Earnings performance continues to be a primary driver of stock valuations")
        
        if 'fed' in content.lower():
            implications.append("Fed policy decisions remain crucial for market direction")
        
        if 'technology' in content.lower() or any(stock in content for stock in ['NVDA', 'AAPL', 'MSFT']):
            implications.append("Technology sector developments significantly impact broader market sentiment")
        
        # Default fallbacks
        if not implications:
            implications = [
                "Current market movements reflect ongoing economic assessment by investors",
                "Corporate performance and policy developments remain key focus areas",
                "Investor sentiment continues to be shaped by earnings results and economic indicators"
            ]
        
        return implications[:3]

    def _has_valid_content(self, data: Dict[str, Any]) -> bool:
        """Validate content quality"""
        has_title = bool(data.get("title") and len(data["title"]) > 15)
        has_points = bool(data.get("key_points"))
        has_implications = bool(data.get("market_implications"))
        
        logger.info(f"Content validation: title={has_title}, points={has_points}, implications={has_implications}")
        return has_title and has_points and has_implications

    def _create_clean_message(self, data: Dict[str, Any], language: str) -> str:
        """Create properly formatted Telegram message"""
        
        title = self._clean_for_telegram(data.get("title", "Market Update"))
        
        # Build message parts
        message_parts = []
        message_parts.append(f"<b>{title}</b>")
        message_parts.append("")
        
        # Source with validation score
        source = data.get("source", "Financial News")
        source_url = data.get("source_url", "")
        validation_score = data.get("validation_score", 0)
        
        if source_url and source_url.startswith("http"):
            if validation_score > 0:
                source_line = f"<b>Source:</b> <a href=\"{source_url}\">{self._clean_for_telegram(source)}</a> (Score: {validation_score}/100)"
            else:
                source_line = f"<b>Source:</b> <a href=\"{source_url}\">{self._clean_for_telegram(source)}</a>"
            message_parts.append(source_line)
        else:
            message_parts.append(f"<b>Source:</b> {self._clean_for_telegram(source)}")
        
        message_parts.append("")
        
        # Key Points - GUARANTEED
        key_points = data.get("key_points", [])
        if key_points:
            message_parts.append("<b>Key Points:</b>")
            for point in key_points[:4]:
                clean_point = self._clean_for_telegram(point)
                message_parts.append(f"• {clean_point}")
        
        message_parts.append("")
        
        # Market Implications - GUARANTEED
        implications = data.get("market_implications", [])
        if implications:
            message_parts.append("<b>Market Implications:</b>")
            for impl in implications[:3]:
                clean_impl = self._clean_for_telegram(impl)
                message_parts.append(f"• {clean_impl}")
        
        # Join message
        final_message = "\n".join(message_parts)
        
        # RTL support
        if language.lower() in ["arabic", "ar"]:
            final_message = f"\u200F{final_message}"
        
        # Final cleanup
        final_message = self._validate_html_structure(final_message)
        
        return final_message

    def _clean_for_telegram(self, text: str) -> str:
        """Clean text for Telegram while preserving structure"""
        if not isinstance(text, str):
            text = str(text)
        
        # Strip formatting but preserve content
        text = self._strip_all_formatting(text)
        
        # HTML escape (except for allowed tags)
        text = (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;'))
        
        return text.strip()

    def _validate_html_structure(self, message: str) -> str:
        """Final HTML validation"""
        # Remove malformed HTML
        message = re.sub(r'<b>\s*</b>', '', message)
        message = re.sub(r'</?ul>', '', message)
        message = re.sub(r'</?li>', '', message)
        
        # Clean whitespace
        message = re.sub(r'\n\s*\n\s*\n', '\n\n', message)
        message = message.strip()
        
        return message

    def _find_image(self, original_content: str, structured_data: Dict[str, Any]) -> Dict[str, str]:
        """Find contextual image"""
        try:
            search_parts = []
            if structured_data.get("title"):
                search_parts.append(structured_data["title"])
            if structured_data.get("key_points"):
                search_parts.extend(structured_data["key_points"][:2])
            
            search_content = " ".join(search_parts)[:500]
            
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

        return self._get_fallback_image(structured_data)

    def _select_best_image(self, images: List[Dict], structured_data: Dict[str, Any]) -> Optional[Dict]:
        """Select best image"""
        if not images:
            return None
        
        best_image = None
        best_score = 0
        
        for image in images:
            score = image.get('relevance_score', 0)
            if score > best_score:
                best_score = score
                best_image = image
        
        return best_image

    def _get_fallback_image(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Get fallback chart"""
        market_chart_url = "https://chart.yahoo.com/z?s=%5EGSPC&t=1d&q=l&l=on&z=s&p=s"
        return {
            "url": market_chart_url,
            "title": "S&P 500 Chart",
            "source": "Yahoo Finance"
        }

    def _validate_image(self, url: str) -> bool:
        """Validate image URL"""
        if not url or not url.startswith("http"):
            return False
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.head(url, headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False

    def _send_content(self, message: str, image_data: Dict[str, str]) -> str:
        """Send content to Telegram"""
        # Try image first
        if image_data and image_data.get("url"):
            if self._send_photo(image_data["url"], message):
                return f"Message with image sent successfully (source: {image_data.get('source', 'unknown')})"
            else:
                logger.warning("Photo send failed, trying text-only")
        
        # Send text
        if self._send_message(message):
            return "Text message sent successfully"
        
        return "Failed to send message"

    def _send_photo(self, image_url: str, caption: str) -> bool:
        """Send photo to Telegram"""
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
            success = response.ok and response.json().get("ok", False)
            
            if not success:
                logger.error(f"Telegram send failed: {response.text}")
            
            return success
        except Exception as e:
            logger.warning(f"Message send failed: {e}")
            return False