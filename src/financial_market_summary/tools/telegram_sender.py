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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            # Parse content and create clean structure with verified source priority
            structured_data = self._extract_clean_content(content)
            
            # Debug: Show found article links
            self._debug_article_links(structured_data)
            
            if not self._has_valid_content(structured_data):
                return "No valid financial content found"
            
            # Enhanced image finding with Telegram-compatible priority
            image_data = self._find_telegram_compatible_image(content, structured_data)
            
            # Create clean message with proper structure
            message = self._create_clean_message(structured_data, language)
            
            # Send content with verified image
            result = self._send_content(message, image_data)
            
            return result
        except Exception as e:
            logger.error(f"Telegram sender failed: {e}")
            return f"Error: {e}"

    def _extract_clean_content(self, content: str) -> Dict[str, Any]:
        """Extract and clean content with verified source priority"""
        data = {
            "title": "",
            "source": "Financial News",
            "source_url": "",
            "key_points": [],
            "market_implications": [],
            "verified_source": None,
            "verified_image": None,
            "validation_score": 0,
            "url_verified": False,
            "image_verified": False
        }
        
        # FIRST: Look for verified source information from web search
        verified_source = self._extract_verified_source_info(content)
        if verified_source:
            data["verified_source"] = verified_source
            data["source"] = verified_source.get("title", "Financial News")
            data["source_url"] = verified_source.get("url", "")
            data["url_verified"] = verified_source.get("url_verified", False)
            data["validation_score"] = verified_source.get("confidence_score", 0)
            logger.info(f"✅ Using verified web source: {data['source_url']} (Verified: {data['url_verified']})")
        
        # SECOND: Look for verified image information from enhanced image search
        verified_image = self._extract_verified_image_info(content)
        if verified_image:
            data["verified_image"] = verified_image
            data["image_verified"] = verified_image.get("telegram_compatible", False)
            logger.info(f"✅ Using verified image: {verified_image.get('title', 'Unknown')} (Telegram Compatible: {data['image_verified']})")
        
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
                'SEARCH METADATA', 'REAL-TIME', 'UTC', '===', 'PHASE', 'VERIFIED SOURCE',
                'WEB SOURCE', 'CONFIDENCE', 'IMAGE SEARCH', 'VERIFIED IMAGES'
            ]):
                continue
            
            clean_line = self._strip_all_formatting(line)
            
            # Look for substantial content that could be a title
            if 25 <= len(clean_line) <= 200 and not clean_line.startswith('•') and not title_found:
                # Additional validation - should contain market-related terms
                market_terms = ['market', 'stock', 'trading', 'earning', 'fed', 'economic', 'financial', 'sector', 'rally', 'surge', 'high']
                if any(term in clean_line.lower() for term in market_terms):
                    data["title"] = clean_line
                    title_found = True
                    break

        # Extract content sections with multiple approaches
        self._extract_content_sections(lines, data)

        # Fallback content generation
        if not data["key_points"]:
            data["key_points"] = self._generate_fallback_points(content)
        
        if not data["market_implications"]:
            data["market_implications"] = self._generate_fallback_implications(content)

        # Set defaults ONLY if no verified source found
        data["title"] = data["title"] or "US Market Update"
        if not data["source_url"] and not data["verified_source"]:
            data["source_url"] = "https://finance.yahoo.com"
            data["source"] = "Yahoo Finance"
        
        return data

    def _extract_verified_source_info(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract verified source information from web search results"""
        try:
            # Look for JSON structure containing verified source data
            json_pattern = r'\{[^}]*"main_source"[^}]*\}'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    # Try to parse the JSON
                    source_data = json.loads(json_str)
                    main_source = source_data.get("main_source", {})
                    
                    if main_source.get("url") and main_source.get("title"):
                        logger.info(f"🔍 Found verified source JSON: {main_source.get('title')[:50]}...")
                        return {
                            "title": main_source.get("title", ""),
                            "url": main_source.get("url", ""),
                            "source": main_source.get("source", ""),
                            "url_verified": main_source.get("url_verified", False),
                            "confidence_score": source_data.get("confidence_score", 0),
                            "verification_status": main_source.get("verification_status", "")
                        }
                except json.JSONDecodeError:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting verified source: {e}")
            return None

    def _extract_verified_image_info(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract verified image information from enhanced image search results"""
        try:
            # Look for JSON structure containing verified image data
            image_patterns = [
                r'\{[^}]*"verified_images"[^}]*\}',
                r'\{[^}]*"telegram_compatible"[^}]*true[^}]*\}',
                r'\{[^}]*"url"[^}]*"title"[^}]*\}'
            ]
            
            for pattern in image_patterns:
                json_matches = re.findall(pattern, content, re.DOTALL)
                
                for json_str in json_matches:
                    try:
                        # Try to parse the JSON
                        image_data = json.loads(json_str)
                        
                        # Check for verified_images array
                        if 'verified_images' in image_data:
                            verified_images = image_data.get('verified_images', [])
                            if verified_images:
                                # Get the first Telegram-compatible image
                                for image in verified_images:
                                    if image.get('telegram_compatible', False):
                                        logger.info(f"🖼️ Found Telegram-compatible image: {image.get('title', 'Unknown')[:50]}...")
                                        return image
                        
                        # Check for single image object
                        elif image_data.get('url') and image_data.get('telegram_compatible', False):
                            logger.info(f"🖼️ Found single Telegram-compatible image: {image_data.get('title')[:50]}...")
                            return image_data
                                
                    except json.JSONDecodeError:
                        continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting verified image: {e}")
            return None

    def _find_telegram_compatible_image(self, original_content: str, structured_data: Dict[str, Any]) -> Dict[str, str]:
        """Find Telegram-compatible image with priority system"""
        
        # FIRST PRIORITY: Use verified Telegram-compatible image if available
        verified_image = structured_data.get("verified_image")
        if verified_image and verified_image.get("telegram_compatible", False):
            logger.info(f"🖼️ Using verified Telegram-compatible image: {verified_image.get('title', 'Unknown')}")
            return {
                "url": verified_image.get("url", ""),
                "title": verified_image.get("title", "Financial Chart"),
                "source": f"Verified {verified_image.get('source', 'Source')}",
                "telegram_compatible": True,
                "stock_symbol": verified_image.get("stock_symbol", ""),
                "type": verified_image.get("type", "chart")
            }
        
        # SECOND PRIORITY: Use Enhanced Image Finder for Telegram-compatible images
        try:
            search_parts = []
            if structured_data.get("title"):
                search_parts.append(structured_data["title"])
            if structured_data.get("key_points"):
                search_parts.extend(structured_data["key_points"][:2])
            
            search_content = " ".join(search_parts)[:500]
            stocks = self._extract_stock_symbols(original_content)
            
            image_input = ImageFinderInput(
                search_content=search_content, 
                mentioned_stocks=stocks,
                max_images=3
            )
            results_json = self.image_finder._run(**image_input.dict())
            
            if results_json:
                images = json.loads(results_json) if isinstance(results_json, str) else results_json
                if isinstance(images, list) and images:
                    # Find first Telegram-compatible image
                    for image in images:
                        if image.get('telegram_compatible', False):
                            logger.info(f"🖼️ Using Telegram-compatible image from finder: {image.get('title', 'Unknown')}")
                            return {
                                "url": image["url"],
                                "title": image.get("title", "Financial Chart"),
                                "source": "Enhanced Image Finder",
                                "telegram_compatible": True,
                                "type": "finder_result"
                            }
                    
                    # If no Telegram-compatible images found, log this
                    logger.warning("❌ No Telegram-compatible images found in image finder results")
                    for image in images:
                        logger.debug(f"   - {image.get('title', 'Unknown')}: telegram_compatible={image.get('telegram_compatible', False)}")
        
        except Exception as e:
            logger.warning(f"Enhanced ImageFinder failed: {e}")

        # THIRD PRIORITY: Create stock-specific placeholder if stocks mentioned
        stocks = self._extract_stock_symbols(original_content)
        if stocks:
            primary_stock = stocks[0]
            stock_placeholder = self._create_stock_placeholder(primary_stock)
            logger.info(f"🖼️ Using stock-specific placeholder for {primary_stock}")
            return stock_placeholder

        # FALLBACK: Telegram-compatible general placeholder
        logger.info("🖼️ Using general market placeholder image")
        return self._get_telegram_compatible_fallback()

    def _create_stock_placeholder(self, stock_symbol: str) -> Dict[str, str]:
        """Create a stock-specific placeholder image"""
        placeholder_url = f"https://via.placeholder.com/800x600/1f77b4/ffffff?text={stock_symbol}+Stock+Chart"
        return {
            "url": placeholder_url,
            "title": f"{stock_symbol} Stock Chart",
            "source": "Stock Placeholder",
            "telegram_compatible": True,
            "stock_symbol": stock_symbol,
            "type": "stock_placeholder"
        }

    def _get_telegram_compatible_fallback(self) -> Dict[str, str]:
        """Get guaranteed Telegram-compatible fallback image"""
        placeholder_url = "https://via.placeholder.com/800x600/2ca02c/ffffff?text=Stock+Market+Update"
        return {
            "url": placeholder_url,
            "title": "Stock Market Update Chart",
            "source": "Market Placeholder",
            "telegram_compatible": True,
            "type": "market_placeholder"
        }

    def _extract_stock_symbols(self, content: str) -> List[str]:
        """Extract stock symbols from content"""
        stocks = re.findall(r'\b([A-Z]{2,5})\b', content)
        major_stocks = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "FDX", "INTC", "AMD"}
        relevant_stocks = [s for s in stocks if s in major_stocks]
        return list(set(relevant_stocks))[:3]

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
        """Debug method to show found article links and images"""
        verified_source = data.get("verified_source")
        verified_image = data.get("verified_image")
        validation_score = data.get("validation_score", 0)
        url_verified = data.get("url_verified", False)
        image_verified = data.get("image_verified", False)
        
        logger.info("=== SOURCE & IMAGE DEBUG ===")
        logger.info(f"Source Verification Status: {url_verified}")
        logger.info(f"Image Telegram Compatible: {image_verified}")
        logger.info(f"Validation Score: {validation_score}/100")
        
        if verified_source:
            logger.info("VERIFIED SOURCE FOUND:")
            logger.info(f"  Title: {verified_source.get('title', 'N/A')[:50]}...")
            logger.info(f"  Source: {verified_source.get('source', 'N/A')}")
            logger.info(f"  URL: {verified_source.get('url', 'N/A')}")
            logger.info(f"  Verified: {verified_source.get('url_verified', False)}")
            logger.info(f"  Confidence: {verified_source.get('confidence_score', 0)}/100")
        else:
            logger.info("NO VERIFIED SOURCE - Using fallback")
            logger.info(f"Fallback URL: {data.get('source_url', 'N/A')}")
        
        if verified_image:
            logger.info("VERIFIED IMAGE FOUND:")
            logger.info(f"  Title: {verified_image.get('title', 'N/A')}")
            logger.info(f"  URL: {verified_image.get('url', 'N/A')}")
            logger.info(f"  Source: {verified_image.get('source', 'N/A')}")
            logger.info(f"  Telegram Compatible: {verified_image.get('telegram_compatible', False)}")
            logger.info(f"  Stock: {verified_image.get('stock_symbol', 'N/A')}")
        else:
            logger.info("NO VERIFIED IMAGE - Will use Telegram-compatible placeholder")
        
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
        metadata_terms = ['search', 'metadata', 'completed', 'results', 'hours', 'utc', 'total', 'verification', 'confidence', 'image search']
        return any(term in text.lower() for term in metadata_terms)

    def _generate_fallback_points(self, content: str) -> List[str]:
        """Generate fallback key points"""
        points = []
        
        # Extract stock mentions
        stocks = re.findall(r'\b([A-Z]{2,5})\b', content)
        major_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "FDX", "INTC", "ADSK"]
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
        
        if 'technology' in content.lower() or any(stock in content for stock in ['NVDA', 'AAPL', 'MSFT', 'INTC']):
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
        """Create properly formatted Telegram message with verified source and live chart links."""
        
        title = self._clean_for_telegram(data.get("title", "Market Update"))
        
        message_parts = []
        message_parts.append(f"<b>{title}</b>")
        message_parts.append("")
        
        # Use verified source information
        verified_source = data.get("verified_source")
        if verified_source:
            source_title = verified_source.get("title", "Financial News")
            source_url = verified_source.get("url", "")
            source_name = verified_source.get("source", "")
            url_verified = verified_source.get("url_verified", False)
            
            # Create verification indicator
            if url_verified:
                verification_icon = " ✅"
            else:
                verification_icon = " ⚠️"
            
            if source_url and source_url.startswith("http"):
                # Format: [Title - Source](URL) ✅
                clean_title = self._clean_for_telegram(source_title)
                clean_source = self._clean_for_telegram(source_name)
                if clean_source and clean_source not in clean_title:
                    display_text = f"{clean_title} - {clean_source}"
                else:
                    display_text = clean_title
                
                source_line = f"<b>Source:</b> <a href=\"{source_url}\">{display_text}</a>{verification_icon}"
                message_parts.append(source_line)
            else:
                source_line = f"<b>Source:</b> {self._clean_for_telegram(source_title)}{verification_icon}"
                message_parts.append(source_line)
        else:
            # Fallback source handling
            source = data.get("source", "Financial News")
            source_url = data.get("source_url", "")
            
            if source_url and source_url.startswith("http"):
                source_line = f"<b>Source:</b> <a href=\"{source_url}\">{self._clean_for_telegram(source)}</a>"
                message_parts.append(source_line)
            else:
                message_parts.append(f"<b>Source:</b> {self._clean_for_telegram(source)}")
        
        message_parts.append("")
        
        key_points = data.get("key_points", [])
        if key_points:
            message_parts.append("<b>Key Points:</b>")
            for point in key_points[:4]:
                clean_point = self._clean_for_telegram(point)
                message_parts.append(f"• {clean_point}")
        
        message_parts.append("")
        
        implications = data.get("market_implications", [])
        if implications:
            message_parts.append("<b>Market Implications:</b>")
            for impl in implications[:3]:
                clean_impl = self._clean_for_telegram(impl)
                message_parts.append(f"• {clean_impl}")

        # --- LIVE CHARTS SECTION ---
        message_parts.append("")
        message_parts.append("<b>Live Charts:</b>")
        message_parts.append('🔗 📊 <a href="https://finance.yahoo.com/quote/%5EGSPC/chart/?guccounter=1">S&P 500 Chart</a>')
        message_parts.append('🔗 📈 <a href="https://finance.yahoo.com/quote/%5EIXIC/chart/">NASDAQ Chart</a>')
        message_parts.append('🔗 📉 <a href="https://finance.yahoo.com/quote/%5EDJI/chart/">Dow Jones Chart</a>')
        message_parts.append('🔗 ⚡ <a href="https://finance.yahoo.com/quote/%5EVIX/chart/">VIX Chart</a>')
        message_parts.append('🔗 🏛️ <a href="https://finance.yahoo.com/quote/%5ETNX/chart/">10-Year Chart</a>')
        message_parts.append('🔗 💰 <a href="https://finance.yahoo.com/quote/GC%3DF/chart/">Gold Chart</a>')
        
        # Add confidence footer with image verification status
        validation_score = data.get("validation_score", 0)
        url_verified = data.get("url_verified", False)
        image_verified = data.get("image_verified", False)
        
        if validation_score > 0 or url_verified or image_verified:
            message_parts.append("")
            footer_parts = []
            if validation_score > 0:
                footer_parts.append(f"📊 Confidence: {validation_score}/100")
            if url_verified:
                footer_parts.append("🔗 URL Verified ✅")
            else:
                footer_parts.append("🔗 Fallback Source ⚠️")
            if image_verified:
                footer_parts.append("📸 Image Compatible ✅")
            
            footer = " | ".join(footer_parts)
            message_parts.append(f"<b>{footer}</b>")
        
        # Join message
        final_message = "\n".join(message_parts)
        
        if language.lower() in ["arabic", "ar"]:
            final_message = f"\u200F{final_message}"
        
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

    def _send_content(self, message: str, image_data: Dict[str, str]) -> str:
        """Send content to Telegram with enhanced image handling"""
        # Try verified image first
        if image_data and image_data.get("url") and image_data.get("telegram_compatible", False):
            image_url = image_data["url"]
            image_title = image_data.get("title", "Financial Chart")
            
            logger.info(f"🖼️ Attempting to send Telegram-compatible image: {image_title}")
            
            if self._send_photo(image_url, message):
                logger.info(f"✅ Photo sent successfully: {image_title}")
                return f"Message with verified image sent successfully - {image_title}"
            else:
                logger.warning(f"❌ Photo send failed for: {image_title}")
        else:
            if image_data:
                logger.warning(f"❌ Image not Telegram-compatible: {image_data.get('title', 'Unknown')} (Compatible: {image_data.get('telegram_compatible', False)})")
        
        # Send text-only if image failed
        if self._send_message(message):
            return "Text message sent successfully with verified source (image failed or not compatible)"
        
        return "Failed to send message"

    def _send_photo(self, image_url: str, caption: str) -> bool:
        """Send photo to Telegram with enhanced error handling"""
        try:
            # Test if image URL is accessible first
            if not self._test_image_accessibility(image_url):
                logger.warning(f"Image URL not accessible, skipping photo send: {image_url}")
                return False
            
            payload = {
                "chat_id": self.chat_id,
                "photo": image_url,
                "caption": caption[:1024],  # Telegram caption limit
                "parse_mode": "HTML"
            }
            
            response = requests.post(f"{self.base_url}/sendPhoto", json=payload, timeout=30)
            
            if response.ok:
                response_data = response.json()
                if response_data.get("ok", False):
                    logger.info(f"✅ Telegram photo sent successfully")
                    return True
                else:
                    error_description = response_data.get('description', 'Unknown error')
                    logger.error(f"❌ Telegram photo send failed: {error_description}")
                    return False
            else:
                logger.error(f"❌ Telegram photo send HTTP error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Telegram photo send timeout for URL: {image_url}")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"🔌 Telegram photo send connection error for URL: {image_url}")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram photo send unexpected error: {e}")
            return False

    def _test_image_accessibility(self, url: str) -> bool:
        """Test if image URL is accessible"""
        try:
            headers = {
                'User-Agent': 'TelegramBot (like TwitterBot)',
                'Accept': 'image/*,*/*;q=0.8'
            }
            response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if any(img_type in content_type for img_type in ['image/', 'png', 'jpeg', 'jpg', 'gif']):
                    logger.info(f"✅ Image URL accessible and valid: {url}")
                    return True
                else:
                    logger.warning(f"❌ URL not an image: {url} (Content-Type: {content_type})")
                    return False
            else:
                logger.warning(f"❌ Image URL not accessible: {url} (Status: {response.status_code})")
                return False
                
        except Exception as e:
            logger.warning(f"❌ Error testing image accessibility: {url} - {e}")
            return False

    def _send_message(self, message: str) -> bool:
        """Send text message to Telegram with enhanced error handling"""
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4096],  # Telegram message limit
                "parse_mode": "HTML",
                "disable_web_page_preview": False  # Allow previews for source links
            }
            
            response = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=30)
            
            if response.ok:
                response_data = response.json()
                if response_data.get("ok", False):
                    logger.info(f"✅ Telegram text message sent successfully")
                    return True
                else:
                    error_description = response_data.get('description', 'Unknown error')
                    logger.error(f"❌ Telegram message send failed: {error_description}")
                    return False
            else:
                logger.error(f"❌ Telegram message send HTTP error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Telegram message send timeout")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"🔌 Telegram message send connection error")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram message send unexpected error: {e}")
            return False