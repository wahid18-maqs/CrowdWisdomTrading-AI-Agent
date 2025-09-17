import json
import re
import time
import requests
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from dotenv import load_dotenv
from .image_finder import ImageFinder
from .tavily_search import TavilyFinancialTool

logger = logging.getLogger(__name__)

load_dotenv()

class TelegramSenderInput(BaseModel):
    content: str
    language: Optional[str] = "english"

# Key changes to fix the workflow:

class EnhancedTelegramSender(BaseTool):
    """
    Fixed Telegram sender that follows the JSON workflow:
    1. Search summary for translation
    2. Find matching image/chart
    3. Verify image matches content
    4. Compose single message with verified image
    """
    name: str = "telegram_sender"
    description: str = "Sends verified financial summaries with matching images to Telegram."
    args_schema: Type[BaseModel] = TelegramSenderInput
    bot_token: Optional[str] = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    chat_id: Optional[str] = Field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    base_url: Optional[str] = None
    image_finder: Optional[Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.bot_token and self.chat_id:
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
            logger.info("Telegram credentials loaded successfully.")
        else:
            logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env file.")

    def _run(self, content: str, language: str = "english") -> str:
        """
        Fixed workflow following JSON instructions:
        1. Search summary content
        2. Find relevant image/chart
        3. Verify image matches summary
        4. Compose single Telegram message
        """
        try:
            # STEP 1: Search summary for key information
            search_results = self._search_summary_content(content, language)

            if not search_results["has_content"]:
                return "No financial content found to process."

            # STEP 2: Find image/chart related to search results
            verified_image = self._find_and_verify_image(search_results)

            # STEP 3 & 4: Compose and send single message with verified image
            message_result = self._compose_and_send_message(
                search_results, verified_image, language
            )

            return message_result

        except Exception as e:
            logger.error(f"Enhanced workflow error: {e}", exc_info=True)
            return f"Workflow failed: {e}"

    def _search_summary_content(self, content: str, language: str) -> Dict[str, Any]:
        """
        STEP 1: Search the summary for translation-specific content
        """
        logger.info(f"Step 1: Searching summary content for {language} translation")

        # Process and structure content
        structured_content, _, _ = self._process_and_structure_content(content)

        # Extract key information for image matching
        search_results = {
            "has_content": bool(structured_content),
            "structured_content": structured_content,
            "key_stocks": self._extract_key_stocks(content),
            "key_movers": self._extract_key_movers_with_performance(content),
            "primary_topic": self._identify_primary_topic(content),
            "language": language,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Search results: {len(search_results['key_stocks'])} stocks, "
                    f"{len(search_results['key_movers'])} key movers found")

        return search_results

    def _find_and_verify_image(self, search_results: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        STEP 2 & 3: Find image and verify it matches the summary content
        """
        logger.info("Step 2: Finding relevant image/chart")

        if not self.image_finder:
            logger.warning("No image_finder provided, skipping image search")
            return None

        # Create targeted search query based on key movers (highest priority)
        if search_results["key_movers"]:
            primary_mover = search_results["key_movers"][0]
            search_query = self._create_targeted_query(primary_mover)
        elif search_results["key_stocks"]:
            search_query = f"{search_results['key_stocks'][0]} stock chart performance financial"
        else:
            search_query = f"{search_results['primary_topic']} financial market chart"

        logger.info(f"Searching for images with query: {search_query}")

        try:
            # Use ImageFinder to get relevant images
            from .image_finder import ImageFinderInput
            image_input = ImageFinderInput(
                search_content=search_query,
                max_images=3
            )

            image_results_json = self.image_finder._run(**image_input.dict())
            image_results = json.loads(image_results_json) if image_results_json else []

            if not image_results:
                logger.warning("No images found from image finder")
                return None

            # STEP 3: Verify image matches content
            verified_image = self._verify_image_content_match(
                image_results, search_results
            )

            if verified_image:
                logger.info(f"Step 3: Image verified - {verified_image.get('title', 'Unknown')}")
            else:
                logger.warning("Step 3: No images passed verification")

            return verified_image

        except Exception as e:
            logger.error(f"Image finding/verification failed: {e}")
            return None

    def _verify_image_content_match(self, image_results: List[Dict], search_results: Dict) -> Optional[Dict[str, str]]:
        """
        STEP 3: Verify that image correctly represents the summary content
        """
        logger.info("Step 3: Verifying image matches summary content")

        for image in image_results:
            verification_score = self._calculate_content_match_score(image, search_results)

            # Only accept images with high verification scores
            if verification_score >= 10:  # Threshold for verification
                logger.info(f"Image verified with score {verification_score}: {image.get('title')}")
                return {
                    "image_url": image["url"],
                    "title": image.get("title", "Financial Chart"),
                    "source": image.get("source", "web_search"),
                    "verification_score": verification_score,
                    "matched_content": self._get_matched_content(image, search_results)
                }

        logger.warning("No images passed content verification threshold")
        return None

    def _calculate_content_match_score(self, image: Dict, search_results: Dict) -> int:
        """
        Calculate how well image matches the summary content
        """
        score = 0
        image_text = f"{image.get('title', '')} {image.get('source', '')}".lower()

        # High score for key mover matches (most important)
        for key_mover in search_results["key_movers"]:
            symbol = key_mover.get("symbol", "")
            company = key_mover.get("company", "")

            if symbol and symbol.lower() in image_text:
                score += 15
            if company and company.lower() in image_text:
                score += 12

        # Medium score for other stock matches
        for stock in search_results["key_stocks"]:
            if stock.lower() in image_text:
                score += 8

        # Score for financial content indicators
        financial_indicators = ["chart", "stock", "market", "trading", "performance", "financial"]
        for indicator in financial_indicators:
            if indicator in image_text:
                score += 2

        # Score for quality sources
        quality_sources = ["yahoo", "bloomberg", "reuters", "cnbc", "marketwatch", "finviz"]
        for source in quality_sources:
            if source in image_text:
                score += 5

        return score

    def _get_matched_content(self, image: Dict, search_results: Dict) -> str:
        """
        Identify what content the image matched
        """
        image_text = f"{image.get('title', '')} {image.get('source', '')}".lower()
        matches = []

        for key_mover in search_results["key_movers"]:
            symbol = key_mover.get("symbol", "")
            company = key_mover.get("company", "")

            if symbol and symbol.lower() in image_text:
                matches.append(f"Key mover: {symbol}")
            elif company and company.lower() in image_text:
                matches.append(f"Key mover: {company}")

        for stock in search_results["key_stocks"]:
            if stock.lower() in image_text and not any(stock in match for match in matches):
                matches.append(f"Stock: {stock}")

        return ", ".join(matches) if matches else "General market content"

    def _compose_and_send_message(self, search_results: Dict, verified_image: Optional[Dict], language: str) -> str:
        """
        STEP 4: Compose single Telegram message with verified image
        """
        logger.info("Step 4: Composing and sending Telegram message")

        # Create structured message
        structured_content = search_results["structured_content"]
        message_text = self._create_structured_message(structured_content, language)

        # Add image verification info to message if image is included
        if verified_image:
            image_info = f"\n📊 Chart: {verified_image['title']} (Score: {verified_image['verification_score']})"
            message_text += image_info

        # Send single message with or without image
        if verified_image:
            success = self._send_photo_with_retry(
                verified_image["image_url"],
                message_text[:1024]  # Telegram caption limit
            )
            result = "Message with verified image sent successfully" if success else "Failed to send message with image"
        else:
            success = self._send_message_with_retry(message_text)
            result = "Message sent successfully (no matching image found)" if success else "Failed to send message"

        # Log the workflow completion
        logger.info(f"Workflow completed for {language}: {result}")
        return result

    def _extract_key_movers_with_performance(self, content: str) -> List[Dict[str, str]]:
        """
        Extract stocks with performance indicators (key movers)
        """
        key_movers = []

        # Pattern for stock with performance: "AAPL surged 5.2%" or "Apple (AAPL) jumped 3.1%"
        patterns = [
            r'([A-Z]{2,5})\s+(?:surged?|jumped?|gained?|rose|rallied|climbed|dropped?|fell|declined?|lost|tumbled)\s+[\d.]+%',
            r'(\w+(?:\s+\w+)*?)\s+\(([A-Z]{2,5})\)\s+(?:surged?|jumped?|gained?|rose|rallied|climbed|dropped?|fell|declined?|lost|tumbled)\s+[\d.]+%',
            r'(\w+(?:\s+\w+)*?)\s+(?:surged?|jumped?|gained?|rose|rallied|climbed|dropped?|fell|declined?|lost|tumbled)\s+[\d.]+%'
        ]

        major_stocks = {
            "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon",
            "TSLA": "Tesla", "NVDA": "NVIDIA", "META": "Meta", "NFLX": "Netflix"
        }

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2 and match[1] in major_stocks:  # Company (SYMBOL) pattern
                        key_movers.append({
                            "company": match[0].strip(),
                            "symbol": match[1]
                        })
                    elif len(match) == 1:
                        if match[0] in major_stocks:  # SYMBOL pattern
                            key_movers.append({
                                "symbol": match[0],
                                "company": major_stocks[match[0]]
                            })
                        else:  # Company name pattern
                            key_movers.append({
                                "company": match[0].strip(),
                                "symbol": None
                            })
                else:
                    if match in major_stocks:
                        key_movers.append({
                            "symbol": match,
                            "company": major_stocks[match]
                        })

        return key_movers[:3]  # Return top 3 key movers

    def _extract_key_stocks(self, content: str) -> List[str]:
        """
        Extract all mentioned stock symbols
        """
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, content)

        major_stocks = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "AMD", "INTC", "CRM", "ADBE", "PYPL", "UBER", "JPM", "BAC"
        }

        return [stock for stock in potential_stocks if stock in major_stocks][:5]

    def _identify_primary_topic(self, content: str) -> str:
        """
        Identify the primary topic of the content
        """
        content_lower = content.lower()

        if "earnings" in content_lower:
            return "earnings report"
        elif any(term in content_lower for term in ["merger", "acquisition"]):
            return "corporate merger"
        elif any(term in content_lower for term in ["fed", "federal reserve"]):
            return "federal reserve policy"
        elif "market" in content_lower:
            return "stock market analysis"
        else:
            return "financial news"

    def _create_targeted_query(self, key_mover: Dict[str, str]) -> str:
        """
        Create targeted search query for key mover
        """
        symbol = key_mover.get("symbol", "")
        company = key_mover.get("company", "")

        if symbol and company:
            return f"{symbol} {company} stock chart performance today key movers"
        elif symbol:
            return f"{symbol} stock chart performance financial analysis"
        elif company:
            return f"{company} stock chart performance today"
        else:
            return "stock market key movers chart today"

    def _send_photo_with_retry(self, image_url: str, caption: str = "", max_retries: int = 3) -> bool:
        """
        Send photo with retry logic
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)

                url = f"{self.base_url}/sendPhoto"
                payload = {
                    "chat_id": self.chat_id,
                    "photo": image_url,
                    "caption": caption[:1024] if caption else "",
                    "parse_mode": "Markdown"
                }

                response = requests.post(url, data=payload, timeout=30)
                response.raise_for_status()

                result = response.json()
                if result.get("ok"):
                    logger.info("Photo sent successfully with verified image")
                    return True
                else:
                    logger.warning(f"Photo send failed: {result.get('description')}")

            except Exception as e:
                logger.warning(f"Photo send attempt {attempt + 1} failed: {e}")

        return False

    def _process_and_structure_content(self, content: str) -> tuple:
        """
        Improved processing:
        - Extract Title if a 'Title:' line exists (or use first meaningful line)
        - Extract Source if 'Source:' exists
        - Extract Date if 'Date:' or common date formats exist
        - Better bullet/paragraph parsing for Key Points and Market Implications
        Returns (sections, [], [])
        """
        sections: Dict[str, Any] = {}
        # Normalize and split non-empty lines
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        # 1) Title extraction (look for "Title:" first, otherwise choose first non-header line)
        title = None
        for idx, line in enumerate(lines[:6]):
            m = re.match(r'^(?:\d+\.\s*)?Title\s*[:\-]\s*(.+)', line)
            if m:
                title = m.group(1).strip()
                # remove that explicit title line from further parsing
                lines.pop(idx)
                break

        if not title:
            # pick first line that is not a known header
            for line in lines:
                if re.match(r'^(Key Points|Key Point|Market Implications|Source|Date)\s*[:\-]?', line, flags=re.IGNORECASE):
                    continue
                title = re.sub(r'^\d+\.\s*', '', line).strip()
                break

        sections["title"] = title or "Financial Update"

        # 2) Source extraction (look for "Source:" anywhere in first 10 lines)
        source = None
        for line in lines[:10]:
            m = re.search(r'Source\s*[:\-]\s*(.+)', line, flags=re.IGNORECASE)
            if m:
                source = m.group(1).strip()
                break
        sections["source"] = source or "Financial News"

        # 3) Date extraction (try explicit Date: then common "Month dd, yyyy")
        date_val = None
        m = re.search(r'Date\s*[:\-]\s*(.+)', content, flags=re.IGNORECASE)
        if m:
            date_str = m.group(1).strip()
            for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y"):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    date_val = dt.strftime("%Y-%m-%d")
                    break
                except Exception:
                    continue

        if not date_val:
            m2 = re.search(r'([A-Z][a-z]{2,8}\s+\d{1,2},\s*\d{4})', content)
            if m2:
                for fmt in ("%B %d, %Y", "%b %d, %Y"):
                    try:
                        dt = datetime.strptime(m2.group(1), fmt)
                        date_val = dt.strftime("%Y-%m-%d")
                        break
                    except Exception:
                        continue

        sections["date"] = date_val or datetime.now().strftime("%Y-%m-%d")

        # 4) Key points & market implications parsing
        key_points: List[str] = []
        market_implications: List[str] = []

        header_pattern = re.compile(r'^(Key Points|Key Point|Market Implications|Market Implication|Source|Date|Title)\s*[:\-]?', re.IGNORECASE)
        bullet_re = re.compile(r'^[-•*]\s*(.+)')
        numbered_re = re.compile(r'^\d+\.\s*(.+)')

        i = 0
        while i < len(lines):
            line = lines[i]
            # Skip explicit header lines and treat following lines as that section
            if header_pattern.match(line):
                hdr = line.lower()
                # determine the target list
                target = None
                if 'key' in hdr:
                    target = key_points
                elif 'market' in hdr:
                    target = market_implications

                # capture following lines until next header
                j = i + 1
                while j < len(lines) and not header_pattern.match(lines[j]):
                    ln = lines[j]
                    m_b = bullet_re.match(ln)
                    m_n = numbered_re.match(ln)
                    if m_b:
                        clean = m_b.group(1).strip()
                        if any(word in clean.lower() for word in ['market', 'trading', 'price', 'volume', 'bond', 'inflation', 'jobs', 'yield']):
                            market_implications.append(clean)
                        else:
                            key_points.append(clean)
                    elif m_n:
                        clean = m_n.group(1).strip()
                        key_points.append(clean)
                    else:
                        # plain line
                        if len(ln) > 10:
                            if any(word in ln.lower() for word in ['market', 'trading', 'price', 'volume', 'bond', 'inflation', 'jobs', 'yield']):
                                market_implications.append(ln)
                            else:
                                key_points.append(ln)
                    j += 1
                i = j
                continue

            # Non-header lines: check for bullets/numbering or treat as paragraph
            m_b = bullet_re.match(line)
            m_n = numbered_re.match(line)
            if m_b:
                clean = m_b.group(1).strip()
                if any(word in clean.lower() for word in ['market', 'trading', 'price', 'volume', 'bond', 'inflation', 'jobs', 'yield']):
                    market_implications.append(clean)
                else:
                    key_points.append(clean)
            elif m_n:
                clean = m_n.group(1).strip()
                key_points.append(clean)
            else:
                if len(line) > 20:
                    if any(word in line.lower() for word in ['market', 'trading', 'price', 'volume', 'bond', 'inflation', 'jobs', 'yield']):
                        market_implications.append(line)
                    else:
                        key_points.append(line)

            i += 1

        # Fallback: if still nothing structured, take first 1-3 paragraphs
        if not key_points and not market_implications:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for para in paragraphs[:3]:
                if len(para) > 20:
                    if any(word in para.lower() for word in ['market', 'trading', 'price', 'volume', 'bond', 'inflation']):
                        market_implications.append(para[:200] + '...' if len(para) > 200 else para)
                    else:
                        key_points.append(para[:200] + '...' if len(para) > 200 else para)

        sections["key_points"] = key_points[:5]
        sections["market_implications"] = market_implications[:3]
        sections["content"] = content

        return sections, [], []

    def _create_structured_message(self, sections: Dict, language: str) -> str:
        """
        Create structured Telegram message using financial news template.
        Fields are escaped to avoid breaking Markdown formatting.
        """
        title = self._escape_markdown(sections.get("title", "Financial Update"))
        source = self._escape_markdown(sections.get("source", "Financial News"))
        date = self._escape_markdown(sections.get("date", datetime.now().strftime("%Y-%m-%d")))
        key_points = sections.get("key_points", [])
        market_implications = sections.get("market_implications", [])

        # Build the message
        message = f"📢 *{title}*\n"
        message += f"📰 Source: {source}\n"
        message += f"📅 Date: {date}\n\n"

        if key_points:
            message += "🔑 *Key Points:*\n"
            for point in key_points:
                message += f"- {self._escape_markdown(point)}\n"

        if market_implications:
            message += "\n📊 *Market Implications:*\n"
            for implication in market_implications:
                message += f"- {self._escape_markdown(implication)}\n"

        # Add language-specific adjustments if needed
        if language.lower() in ["arabic", "ar"]:
            # Could add RTL markers or Arabic-specific formatting
            pass
        elif language.lower() in ["hindi", "hi"]:
            # Could add Devanagari-specific formatting
            pass

        return message.strip()

    def _escape_markdown(self, text: str) -> str:
        """
        Escape the most common Markdown characters so Telegram won't break the message.
        (Using basic Markdown escaping for characters: _ * ` [ )
        """
        if not isinstance(text, str):
            return str(text)
        # Escape these characters: _ * ` [
        return re.sub(r'([_\*\`\[])', r'\\\1', text)

    def _send_message_with_retry(self, message: str, max_retries: int = 3) -> bool:
        """
        Send message with retry logic
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)

                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message[:4096],  # Telegram message limit
                    "parse_mode": "Markdown"
                }

                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()

                result = response.json()
                if result.get("ok"):
                    logger.info("Message sent successfully")
                    return True
                else:
                    logger.warning(f"Message send failed: {result.get('description')}")

            except Exception as e:
                logger.warning(f"Message send attempt {attempt + 1} failed: {e}")

        return False


# Complete workflow integration example:

class TavilyToTelegramWorkflow:
    """
    Complete workflow that integrates Tavily search → Image verification → Telegram delivery
    Following the JSON requirements exactly
    """

    def __init__(self):
        self.tavily_tool = TavilyFinancialTool()
        self.image_finder = ImageFinder()
        self.telegram_sender = EnhancedTelegramSender()
        self.telegram_sender.image_finder = self.image_finder

    def run_complete_workflow(self, search_query: str, translations: List[str] = ["english"]) -> Dict[str, str]:
        """
        Run the complete workflow following JSON requirements:
        1. Get Tavily summary
        2. Process each translation with image verification
        3. Send single verified message per translation
        """
        results = {}

        try:
            # Get Tavily financial summary
            logger.info(f"Getting Tavily summary for query: {search_query}")
            tavily_summary = self.tavily_tool._run(
                query=search_query,
                hours_back=2,  # Last 2 hours
                max_results=10
            )

            if not tavily_summary or "Error:" in tavily_summary:
                return {"error": "Failed to get Tavily summary"}

            # Process each translation
            for language in translations:
                logger.info(f"Processing {language} translation")

                try:
                    # Run the enhanced workflow for this translation
                    result = self.telegram_sender._run(
                        content=tavily_summary,
                        language=language
                    )
                    results[language] = result

                    # Brief pause between translations
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Translation {language} failed: {e}")
                    results[language] = f"Failed: {e}"

            return results

        except Exception as e:
            logger.error(f"Complete workflow failed: {e}")
            return {"error": f"Workflow failed: {e}"}

# Usage example:
def main():
    """
    Example usage of the complete Tavily → Telegram workflow
    """
    workflow = TavilyToTelegramWorkflow()

    # Run for multiple translations
    results = workflow.run_complete_workflow(
        search_query="US stock market earnings key movers today",
        translations=["english", "arabic", "hindi"]
    )

    # Print results
    for language, result in results.items():
        print(f"{language.upper()}: {result}")

if __name__ == "__main__":
    main()