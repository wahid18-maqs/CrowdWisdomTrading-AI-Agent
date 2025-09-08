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
    Structured Telegram sender that processes content, extracts images, and sends 
    well-formatted financial summaries with embedded images to Telegram.
    """
    name: str = "telegram_sender"
    description: str = "Sends structured financial summary with embedded images to Telegram channel."
    args_schema: Type[BaseModel] = TelegramSenderInput
    bot_token: Optional[str] = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    chat_id: Optional[str] = Field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    base_url: Optional[str] = None

    def __init__(self, **kwargs):
        """Initialize Telegram sender with credentials."""
        super().__init__(**kwargs)
        if self.bot_token and self.chat_id:
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
            logger.info("Telegram credentials loaded successfully.")
        else:
            logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env file.")

    def _run(self, content: str, language: str = "english") -> str:
        """Main execution method."""
        if not self.base_url:
            return "Error: Telegram credentials not configured."

        try:
            # Process content and extract images
            structured_content, image_urls, image_details = self._process_and_structure_content(content)
            
            # Create structured message
            formatted_message = self._create_structured_message(structured_content, language)
            
            results = []
            
            # Send main structured message
            success = self._send_message_with_retry(formatted_message)
            status = "sent successfully" if success else "failed"
            results.append(f"Main message: {status}")
            
            # Send images as embedded photos (not hyperlinks)
            if image_urls:
                time.sleep(1)  # Brief pause before images
                for i, (img_url, img_detail) in enumerate(zip(image_urls[:3], image_details[:3])):
                    caption = self._create_contextual_caption(img_detail, language, i+1)
                    img_success = self._send_photo_embedded(img_url, caption)
                    status = "sent successfully" if img_success else "failed"
                    results.append(f"Image {i+1}: {status}")
                    if img_success and i < len(image_urls[:3]) - 1:
                        time.sleep(1)

            return self._generate_status_report(results, language, len(image_urls))

        except Exception as e:
            error_msg = f"Telegram sender error ({language}): {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _process_and_structure_content(self, content: str) -> tuple[Dict[str, str], List[str], List[Dict[str, str]]]:
        """
        Process content and extract images, then structure content into sections.
        """
        # Extract images first
        image_urls = []
        image_details = []
        
        try:
            # Look for markdown image syntax and direct URLs
            markdown_image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            markdown_matches = re.findall(markdown_image_pattern, content)
            
            for alt_text, url in markdown_matches:
                if self._is_valid_image_url(url):
                    image_urls.append(url)
                    image_details.append({
                        "description": alt_text or "Financial Chart",
                        "type": "news_image",
                        "source": self._extract_source_from_url(url)
                    })
            
            # Look for image sections from search results
            image_section_pattern = r"=== (?:VERIFIED |CONTEXTUAL )?FINANCIAL IMAGES FOUND ===(.*?)(?=Total.*?images?:|$)"
            image_section = re.search(image_section_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if image_section:
                image_content = image_section.group(1)
                
                # Extract URLs and details from image section
                image_blocks = re.split(r"Image \d+:", image_content)[1:]
                
                for block in image_blocks:
                    url_match = re.search(r"- URL: (https?://[^\s]+)", block)
                    if url_match:
                        url = url_match.group(1)
                        if self._is_valid_image_url(url):
                            # Extract details
                            description_match = re.search(r"- Description: ([^\n]+)", block)
                            source_match = re.search(r"- Source: ([^\n]+)", block)
                            type_match = re.search(r"- Type: ([^\n]+)", block)
                            
                            image_urls.append(url)
                            image_details.append({
                                "description": description_match.group(1) if description_match else "Financial Chart",
                                "type": type_match.group(1) if type_match else "chart",
                                "source": source_match.group(1) if source_match else "unknown"
                            })

            # Remove image sections from content
            clean_content = re.sub(markdown_image_pattern, '', content)
            clean_content = re.sub(
                r"=== (?:VERIFIED |CONTEXTUAL )?FINANCIAL IMAGES FOUND ===.*?(?=\n\n|\Z)",
                "",
                clean_content,
                flags=re.DOTALL | re.IGNORECASE
            )
            
            # Structure the content into sections
            structured_content = self._structure_content_into_sections(clean_content)
            
            logger.info(f"Processed: {len(image_urls)} images, structured into {len(structured_content)} sections")
            return structured_content, image_urls, image_details
            
        except Exception as e:
            logger.warning(f"Content processing error: {e}")
            return {"content": content}, [], []

    def _structure_content_into_sections(self, content: str) -> Dict[str, str]:
        """
        Structure content into organized sections for better readability.
        """
        sections = {
            "market_overview": "",
            "key_movers": "",
            "sector_analysis": "",
            "economic_highlights": "",
            "outlook": "",
            "other_news": ""
        }

        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Categorize paragraphs based on content
            if any(term in paragraph_lower for term in ["market overview", "market summary", "broader market", "overall market"]):
                sections["market_overview"] += paragraph + "\n\n"
            elif any(term in paragraph_lower for term in ["earnings", "surge", "rally", "drop", "gain", "loss", "up ", "down ", "%"]):
                sections["key_movers"] += paragraph + "\n\n"
            elif any(term in paragraph_lower for term in ["sector", "industry", "technology", "healthcare", "financial", "energy"]):
                sections["sector_analysis"] += paragraph + "\n\n"
            elif any(term in paragraph_lower for term in ["fed", "federal reserve", "inflation", "unemployment", "gdp", "economic"]):
                sections["economic_highlights"] += paragraph + "\n\n"
            elif any(term in paragraph_lower for term in ["outlook", "forecast", "expect", "future", "next", "upcoming"]):
                sections["outlook"] += paragraph + "\n\n"
            else:
                # If no specific category, add to other news
                sections["other_news"] += paragraph + "\n\n"

        # Clean up empty sections
        sections = {k: v.strip() for k, v in sections.items() if v.strip()}
        
        # If no structured sections found, put everything in market_overview
        if not sections:
            sections["market_overview"] = content
            
        return sections

    def _create_structured_message(self, sections: Dict[str, str], language: str) -> str:
        """
        Create a well-structured message with proper sections.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Language-specific headers and section titles
        headers = {
            "english": f"📈 **US Financial Market Summary**\n🕐 {timestamp}",
            "arabic": f"📊 **ملخص السوق المالي الأمريكي**\n🕐 {timestamp}",
            "hindi": f"📈 **अमेरिकी वित्तीय बाजार सारांश**\n🕐 {timestamp}",
            "hebrew": f"📊 **סיכום שוק הכספים האמריקאי**\n🕐 {timestamp}"
        }
        
        section_titles = {
            "english": {
                "market_overview": "📊 **MARKET OVERVIEW**",
                "key_movers": "🎯 **KEY MOVERS**", 
                "sector_analysis": "🏢 **SECTOR ANALYSIS**",
                "economic_highlights": "📈 **ECONOMIC HIGHLIGHTS**",
                "outlook": "🔮 **MARKET OUTLOOK**",
                "other_news": "📰 **OTHER NEWS**"
            },
            "arabic": {
                "market_overview": "📊 **نظرة عامة على السوق**",
                "key_movers": "🎯 **المحركات الرئيسية**",
                "sector_analysis": "🏢 **تحليل القطاعات**", 
                "economic_highlights": "📈 **أبرز الأحداث الاقتصادية**",
                "outlook": "🔮 **توقعات السوق**",
                "other_news": "📰 **أخبار أخرى**"
            },
            "hindi": {
                "market_overview": "📊 **बाजार अवलोकन**",
                "key_movers": "🎯 **मुख्य चालक**",
                "sector_analysis": "🏢 **सेक्टर विश्लेषण**",
                "economic_highlights": "📈 **आर्थिक मुख्य बातें**",
                "outlook": "🔮 **बाजार दृष्टिकोण**", 
                "other_news": "📰 **अन्य समाचार**"
            },
            "hebrew": {
                "market_overview": "📊 **סקירת שוק**",
                "key_movers": "🎯 **מניעים עיקריים**",
                "sector_analysis": "🏢 **ניתוח סקטורים**",
                "economic_highlights": "📈 **דגשים כלכליים**",
                "outlook": "🔮 **תחזית שוק**",
                "other_news": "📰 **חדשות אחרות**"
            }
        }

        header = headers.get(language.lower(), headers["english"])
        titles = section_titles.get(language.lower(), section_titles["english"])
        
        # Build structured message
        message_parts = [header, ""]
        
        # Add sections in logical order
        section_order = ["market_overview", "key_movers", "sector_analysis", "economic_highlights", "outlook", "other_news"]
        
        for section_key in section_order:
            if section_key in sections and sections[section_key]:
                message_parts.append(titles[section_key])
                message_parts.append(sections[section_key])
                message_parts.append("")  # Add spacing between sections

        # Add footer
        footers = {
            "english": "📊 *Powered by CrowdWisdomTrading*",
            "arabic": "📊 *مدعوم من CrowdWisdomTrading*",
            "hindi": "📊 *CrowdWisdomTrading द्वारा संचालित*",
            "hebrew": "📊 *מופעל על ידי CrowdWisdomTrading*"
        }
        
        footer = footers.get(language.lower(), footers["english"])
        message_parts.append(footer)
        
        return "\n".join(message_parts)

    def _create_contextual_caption(self, img_detail: Dict[str, str], language: str, image_num: int) -> str:
        """
        Create contextual captions based on image details and language.
        """
        description = img_detail.get("description", "Financial Chart")
        img_type = img_detail.get("type", "chart")
        source = img_detail.get("source", "")
        
        caption_templates = {
            "english": {
                "news_image": f"📊 {description}",
                "chart": f"📈 Chart {image_num}: {description}",
                "stock_chart": f"📊 {description}",
                "default": f"📊 Financial Chart {image_num}: {description}"
            },
            "arabic": {
                "news_image": f"📊 {description}",
                "chart": f"📈 الرسم البياني {image_num}: {description}",
                "stock_chart": f"📊 {description}",
                "default": f"📊 الرسم البياني المالي {image_num}: {description}"
            },
            "hindi": {
                "news_image": f"📊 {description}",
                "chart": f"📈 चार्ट {image_num}: {description}",
                "stock_chart": f"📊 {description}",
                "default": f"📊 वित्तीय चार्ट {image_num}: {description}"
            },
            "hebrew": {
                "news_image": f"📊 {description}",
                "chart": f"📈 גרף {image_num}: {description}",
                "stock_chart": f"📊 {description}",
                "default": f"📊 גרף פיננסי {image_num}: {description}"
            }
        }

        templates = caption_templates.get(language.lower(), caption_templates["english"])
        template = templates.get(img_type, templates["default"])
        
        # Add source if available
        if source and source != "unknown":
            source_text = {
                "english": f" • Source: {source}",
                "arabic": f" • المصدر: {source}",
                "hindi": f" • स्रोत: {source}",
                "hebrew": f" • מקור: {source}"
            }
            template += source_text.get(language.lower(), source_text["english"])
        
        return template

    def _is_valid_image_url(self, url: str) -> bool:
        """Enhanced validation for image URLs."""
        if not url or not url.startswith(("http://", "https://")):
            return False
        
        url_lower = url.lower()
        
        # Check for image extensions or known chart domains
        indicators = [
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
            "chart", "graph", "finviz.com", "tradingview.com",
            "yahoo.com", "reuters.com", "bloomberg.com", "investing.com"
        ]
        
        return any(indicator in url_lower for indicator in indicators)

    def _extract_source_from_url(self, url: str) -> str:
        """Extract source domain from URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "web"

    def _send_message_with_retry(self, message: str, max_retries: int = 3) -> bool:
        """Send text message with retry logic and length handling."""
        
        # Check message length and split if needed
        if len(message) > 4096:
            logger.info(f"Message too long ({len(message)} chars), splitting...")
            chunks = self._split_long_message(message)
            
            all_success = True
            for i, chunk in enumerate(chunks):
                success = self._send_single_message_chunk(chunk, max_retries)
                if not success:
                    all_success = False
                if success and i < len(chunks) - 1:
                    time.sleep(0.5)  # Brief pause between chunks
            
            return all_success
        else:
            return self._send_single_message_chunk(message, max_retries)

    def _split_long_message(self, message: str) -> List[str]:
        """Split long message into chunks while preserving structure."""
        chunks = []
        current_chunk = ""
        
        # Split by double newlines (sections)
        parts = message.split('\n\n')
        
        for part in parts:
            # Check if adding this part would exceed limit
            if len(current_chunk) + len(part) + 2 > 4000:  # Leave some buffer
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part + '\n\n'
            else:
                current_chunk += part + '\n\n'
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split message into {len(chunks)} chunks")
        return chunks

    def _send_single_message_chunk(self, message: str, max_retries: int = 3) -> bool:
        """Send a single message chunk with retry logic."""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt
                    logger.info(f"Retry {attempt + 1}, waiting {wait_time}s...")
                    time.sleep(wait_time)

                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                }
                
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()

                result = response.json()
                if result.get("ok"):
                    logger.info(f"Message chunk sent successfully ({len(message)} chars)")
                    return True
                else:
                    error_desc = result.get("description", "Unknown error")
                    logger.warning(f"Telegram API error: {error_desc}")
                    
                    if "Too Many Requests" in error_desc:
                        retry_after = result.get("parameters", {}).get("retry_after", 5)
                        logger.info(f"Rate limited, waiting {retry_after}s...")
                        time.sleep(retry_after)
                        continue
                    elif "parse" in error_desc.lower() or "markdown" in error_desc.lower():
                        logger.info("Markdown parsing failed, trying fallback...")
                        return self._send_text_fallback(message)
                    
                    return False
                    
            except Exception as e:
                logger.warning(f"Message send error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
                return False
                
        return False

    def _send_photo_embedded(self, image_url: str, caption: str = "") -> bool:
        """
        Send image as embedded photo (not hyperlink) in Telegram.
        """
        try:
            url = f"{self.base_url}/sendPhoto"
            payload = {
                "chat_id": self.chat_id,
                "photo": image_url,  # This makes the image display embedded
                "caption": caption[:1024] if caption else "",  # Telegram caption limit
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            if result.get("ok"):
                logger.info(f"Embedded image sent: {caption[:50]}...")
                return True
            else:
                error_desc = result.get("description", "Unknown error")
                logger.warning(f"Image send failed: {error_desc}")
                
                # Try without parse_mode if that's the issue
                if "parse" in error_desc.lower():
                    return self._send_photo_fallback(image_url, caption)
                
                return False
                
        except Exception as e:
            logger.warning(f"Embedded image send error: {e}")
            return False

    def _send_photo_fallback(self, image_url: str, caption: str) -> bool:
        """Fallback to send photo without markdown in caption."""
        try:
            clean_caption = re.sub(r'[*_`]', '', caption)  # Remove markdown
            
            url = f"{self.base_url}/sendPhoto"
            payload = {
                "chat_id": self.chat_id,
                "photo": image_url,
                "caption": clean_caption[:1024] if clean_caption else "",
            }
            
            response = requests.post(url, json=payload, timeout=25)
            response.raise_for_status()
            
            result = response.json()
            if result.get("ok"):
                logger.info("Image sent (fallback mode)")
                return True
            return False
            
        except Exception as e:
            logger.warning(f"Photo fallback failed: {e}")
            return False

    def _send_text_fallback(self, message: str) -> bool:
        """Fallback to send text without markdown."""
        try:
            clean_message = re.sub(r'[*_`]', '', message)
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": clean_message,
                "disable_web_page_preview": True,
            }
            
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            
            result = response.json()
            if result.get("ok"):
                logger.info("Text sent (fallback mode)")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Text fallback failed: {e}")
            return False

    def _generate_status_report(self, results: List[str], language: str, image_count: int) -> str:
        """Generate comprehensive status report."""
        success_count = sum(1 for r in results if "successfully" in r)
        total_count = len(results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        status_report = (
            f"Structured Telegram delivery for '{language}':\n"
            f"✅ Success rate: {success_count}/{total_count} ({success_rate:.1f}%)\n"
            f"📊 Images processed: {image_count} (embedded as photos)\n"
            f"📝 Message structure: Market Overview → Key Movers → Sector Analysis → Economic Highlights → Outlook\n"
            f"🔍 Details: {' | '.join(results)}"
        )
        
        return status_report