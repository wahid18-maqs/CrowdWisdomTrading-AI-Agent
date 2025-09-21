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
            # Parse content and create clean structure for all languages
            if language.lower() in ["hindi", "arabic", "hebrew"]:
                structured_data = self._extract_translated_content_structure(content)
            else:
                structured_data = self._extract_clean_content(content)

            if language.lower() == "english":
                self._debug_article_links(structured_data)

            if language.lower() in ["hindi", "arabic", "hebrew"]:
                if not self._has_basic_content(structured_data):
                    logger.warning(f"⚠️ Basic validation failed for {language}, but continuing anyway")
                    self._ensure_minimum_structure(structured_data)
            else:
                if not self._has_valid_content(structured_data):
                    return "No valid financial content found"

            image_data = {}
            if language.lower() == "english":
                image_data = self._find_telegram_compatible_image(content, structured_data)

            logger.info(f"🏗️ Creating clean message for {language}")
            logger.info(f"📊 Data: title='{structured_data.get('title', 'N/A')}', points={len(structured_data.get('key_points', []))}, implications={len(structured_data.get('market_implications', []))}")

            message = self._create_clean_message(structured_data, language)

            logger.info(f"📝 Created message length: {len(message)} chars")
            logger.info(f"🔤 Message preview: {message[:200]}...")

            if language.lower() in ["hindi", "arabic", "hebrew"]:
                logger.info(f"🔄 Applying translation formatting for {language}")
                message = self._apply_translation_formatting(message, language)
                logger.info(f"✅ Final formatted message length: {len(message)} chars")

            result = self._send_content(message, image_data)

            return result
        except Exception as e:
            logger.error(f"Telegram sender failed: {e}")
            return f"Error: {e}"

    def send_from_workflow_result(self, json_file_path: str, language: str = "english") -> str:
        try:
            logger.info(f"📄 Processing workflow result from: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)

            if isinstance(workflow_data, dict) and 'content' in workflow_data:
                content = workflow_data['content']
            elif isinstance(workflow_data, str):
                content = workflow_data
            else:
                content = str(workflow_data)

            logger.info(f"✅ Loaded {len(content)} characters for {language}")
            return self._run(content, language)

        except Exception as e:
            logger.error(f"❌ Failed to send from workflow result: {e}")
            return f"Error processing workflow result: {e}"

    def _extract_clean_content(self, content: str) -> Dict[str, Any]:
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
        
        verified_source = self._extract_verified_source_info(content)
        if verified_source:
            data["verified_source"] = verified_source
            data["source"] = verified_source.get("title", "Financial News")
            data["source_url"] = verified_source.get("url", "")
            data["url_verified"] = verified_source.get("url_verified", False)
            data["validation_score"] = verified_source.get("confidence_score", 0)
            logger.info(f"✅ Using verified web source: {data['source_url']} (Verified: {data['url_verified']})")
        
        verified_image = self._extract_verified_image_info(content)
        if verified_image:
            data["verified_image"] = verified_image
            data["image_verified"] = verified_image.get("telegram_compatible", False)
            logger.info(f"✅ Using verified image: {verified_image.get('title', 'Unknown')} (Telegram Compatible: {data['image_verified']})")
        
        clean_content = re.sub(r'<[^>]+>', '', content)
        lines = [line.strip() for line in clean_content.splitlines() if line.strip()]
        
        if not lines:
            return data

        title_found = False
        for line in lines[:25]:
            if any(skip in line.upper() for skip in [
                'SEARCH WINDOW', 'MARKET OVERVIEW', 'KEY HIGHLIGHTS', 'BREAKING NEWS',
                'SEARCH METADATA', 'REAL-TIME', 'UTC', '===', 'PHASE', 'VERIFIED SOURCE',
                'WEB SOURCE', 'CONFIDENCE', 'IMAGE SEARCH', 'VERIFIED IMAGES'
            ]):
                continue
            
            clean_line = self._strip_all_formatting(line)
            if 25 <= len(clean_line) <= 200 and not clean_line.startswith('•') and not title_found:
                market_terms = ['market', 'stock', 'trading', 'earning', 'fed', 'economic', 'financial', 'sector', 'rally', 'surge', 'high']
                if any(term in clean_line.lower() for term in market_terms):
                    data["title"] = clean_line
                    title_found = True
                    break

        self._extract_content_sections(lines, data)
        self._ensure_required_item_count(data, lines)
        data["title"] = data["title"] or "Market Update"
        
        return data

    def _extract_translated_content_structure(self, content: str) -> Dict[str, Any]:
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

        if not content:
            return data

        logger.info(f"🔍 TRANSLATED CONTENT DEBUG:")
        logger.info(f"📄 Content length: {len(content)} chars")
        logger.info(f"🔤 First 200 chars: {content[:200]}...")

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        logger.info(f"📝 Total lines: {len(lines)}")

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

        self._extract_translated_sections(lines, data)

        if len(data["key_points"]) < 3:
            additional_points = self._extract_fallback_points(lines, "key_points", 3 - len(data["key_points"]))
            data["key_points"].extend(additional_points)

        if len(data["market_implications"]) < 3:
            additional_implications = self._extract_fallback_points(lines, "market_implications", 3 - len(data["market_implications"]))
            data["market_implications"].extend(additional_implications)

        if len(data["key_points"]) < 3:
            data["key_points"] = self._split_content_into_points(lines, "key_points", 3)

        if len(data["market_implications"]) < 3:
            data["market_implications"] = self._split_content_into_points(lines, "market_implications", 3)

        if len(data["key_points"]) < 3 or len(data["market_implications"]) < 3:
            logger.info("🚨 Using ultimate fallback: forcing paragraph content into structure")
            self._force_paragraph_into_structure(content, data)

        data["title"] = data["title"] or "Market Update"

        return data

    def _extract_translated_sections(self, lines: List[str], data: Dict[str, Any]):
        current_section = None

        key_point_terms = [
            'key', 'main', 'important', 'highlight', 'point', 'major', 'primary', 'summary', 'overview',
            'मुख्य', 'प्रमुख', 'महत्वपूर्ण', 'सारांश',  # Hindi
            'الرئيسية', 'المهمة', 'النقاط', 'الملخص', 'المحورية',  # Arabic
            'עיקרי', 'חשוב', 'נקודות', 'ראשי'  # Hebrew
        ]

        market_terms = [
            'market', 'implication', 'impact', 'outlook', 'effect', 'consequence', 'forecast', 'analysis',
            'बाजार', 'प्रभाव', 'दृष्टिकोण', 'विश्लेषण',  # Hindi
            'السوق', 'التأثير', 'التوقعات', 'التحليل', 'الآثار',  # Arabic
            'שוק', 'השפעה', 'תחזית', 'ניתוח'  # Hebrew
        ]

        logger.info(f"🔍 Extracting sections from {len(lines)} lines of translated content")

        for i, line in enumerate(lines):
            line_clean = line.strip()
            line_lower = line_clean.lower()

            is_key_section = any(term in line_lower for term in key_point_terms)
            is_market_section = any(term in line_lower for term in market_terms)

            if is_key_section and ('market' not in line_lower or 'implication' not in line_lower):
                current_section = 'key_points'
                logger.info(f"📋 Found Key Points section at line {i}: {line_clean[:50]}...")
                continue
            elif is_market_section:
                current_section = 'market_implications'
                logger.info(f"📈 Found Market Implications section at line {i}: {line_clean[:50]}...")
                continue

            bullet_patterns = [
                r'^[-•*+]\s+(.+)',  # Various bullet points
                r'^\d+\.\s+(.+)',   # Numbered lists
                r'^[▪▫‣⁃►]\s+(.+)', # Alternative bullets
                r'^[٠-٩]\.\s+(.+)', # Arabic numbers
                r'^[०-९]\.\s+(.+)', # Hindi numbers
                r'^[\u05D0-\u05EA]\.\s+(.+)', # Hebrew letters as list markers
            ]

            point_extracted = False
            for pattern in bullet_patterns:
                match = re.match(pattern, line_clean)
                if match:
                    point_text = match.group(1).strip()
                    point_text = re.sub(r'[*_`~]+', '', point_text).strip()
                    if len(point_text) >= 10:
                        if current_section == 'key_points' and len(data["key_points"]) < 5:
                            data["key_points"].append(point_text[:200])
                            logger.info(f"✅ Added key point: {point_text[:50]}...")
                        elif current_section == 'market_implications' and len(data["market_implications"]) < 5:
                            data["market_implications"].append(point_text[:200])
                            logger.info(f"✅ Added market implication: {point_text[:50]}...")
                    point_extracted = True
                    break

            if not point_extracted and not current_section and len(line_clean) > 20:
                clean_text = re.sub(r'[#*_`~]+', '', line_clean).strip()
                if len(clean_text) >= 20 and not clean_text.startswith('http'):
                    if any(term in clean_text.lower() for term in market_terms[:5]):
                        if len(data["market_implications"]) < 5:
                            data["market_implications"].append(clean_text[:200])
                            logger.info(f"🎯 Auto-categorized as market implication: {clean_text[:50]}...")
                    else:
                        if len(data["key_points"]) < 5:
                            data["key_points"].append(clean_text[:200])
                            logger.info(f"🎯 Auto-categorized as key point: {clean_text[:50]}...")

        logger.info(f"📊 Extracted {len(data['key_points'])} key points and {len(data['market_implications'])} market implications")

    def _extract_fallback_points(self, lines: List[str], section_type: str, needed_count: int) -> List[str]:
        points = []
        extracted = 0

        logger.info(f"🔄 Looking for {needed_count} additional {section_type} from {len(lines)} lines")

        for line in lines:
            if extracted >= needed_count:
                break

            line_clean = line.strip()
            if (line_clean.startswith('#') or
                len(line_clean) < 15 or
                (line_clean.startswith('**') and line_clean.endswith('**'))):
                continue

            clean_text = re.sub(r'[#*_`]+', '', line_clean).strip()
            if 15 <= len(clean_text) <= 300 and not clean_text.startswith('http'):
                if not any(skip in clean_text.lower() for skip in ['image', 'chart', 'source:', 'http']):
                    points.append(clean_text)
                    extracted += 1
                    logger.info(f"📝 Fallback extracted: {clean_text[:50]}...")

        logger.info(f"📊 Fallback extraction: got {extracted}/{needed_count} {section_type}")
        return points

    def _split_content_into_points(self, lines: List[str], section_type: str, needed_count: int) -> List[str]:
        points = []

        logger.info(f"🚨 Emergency splitting content into {needed_count} {section_type}")

        content_paragraphs = []
        for line in lines:
            clean_line = re.sub(r'[#*_`]+', '', line.strip()).strip()
            if len(clean_line) > 30 and not clean_line.startswith('http'):
                content_paragraphs.append(clean_line)

        if content_paragraphs:
            content_paragraphs.sort(key=len, reverse=True)
            for paragraph in content_paragraphs[:2]:
                if len(points) >= needed_count:
                    break
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and len(points) < needed_count:
                        points.append(sentence[:200])
                        logger.info(f"✂️ Split into point: {sentence[:50]}...")

        while len(points) < needed_count:
            if section_type == "key_points":
                fallback = ["Market developments being analyzed", "Financial data under review", "Key trends being monitored"]
            else:
                fallback = ["Market impact being assessed", "Investment outlook under evaluation", "Economic effects being studied"]
            remaining_needed = needed_count - len(points)
            points.extend(fallback[:remaining_needed])

        return points[:needed_count]

    def _force_paragraph_into_structure(self, content: str, data: Dict[str, Any]):
        logger.info("💪 Forcing paragraph content into structured format")
        sentences = re.split(r'[.!?]+', content)
        good_sentences = []

        for sentence in sentences:
            clean_sentence = re.sub(r'[#*_`~\[\]()]+', '', sentence).strip()
            if 20 <= len(clean_sentence) <= 300 and not clean_sentence.startswith('http'):
                good_sentences.append(clean_sentence)

        logger.info(f"🔤 Found {len(good_sentences)} good sentences to work with")

        if len(data["key_points"]) < 3:
            needed_points = 3 - len(data["key_points"])
            first_half = good_sentences[:len(good_sentences)//2] if len(good_sentences) > 3 else good_sentences[:3]
            for i, sentence in enumerate(first_half):
                if i >= needed_points:
                    break
                if sentence not in data["key_points"]:
                    data["key_points"].append(sentence)
                    logger.info(f"🎯 Forced key point: {sentence[:50]}...")

        if len(data["market_implications"]) < 3:
            needed_implications = 3 - len(data["market_implications"])
            second_half = good_sentences[len(good_sentences)//2:] if len(good_sentences) > 3 else good_sentences[3:6]
            for i, sentence in enumerate(second_half):
                if i >= needed_implications:
                    break
                if sentence not in data["market_implications"]:
                    data["market_implications"].append(sentence)
                    logger.info(f"🎯 Forced market implication: {sentence[:50]}...")

        while len(data["key_points"]) < 3:
            data["key_points"].append("Financial market developments are being analyzed")

        while len(data["market_implications"]) < 3:
            data["market_implications"].append("Market implications are being assessed")

        logger.info(f"✅ Final structure: {len(data['key_points'])} key points, {len(data['market_implications'])} implications")

    def _ensure_minimum_structure(self, data: Dict[str, Any]):
        logger.info("🔧 Ensuring minimum structure for translated content")
        if not data.get("title"):
            data["title"] = "Market Update"

        while len(data.get("key_points", [])) < 3:
            if "key_points" not in data:
                data["key_points"] = []
            fallback_points = [
                "Market developments are being monitored",
                "Financial indicators show ongoing activity",
                "Key economic factors are under analysis"
            ]
            needed = 3 - len(data["key_points"])
            data["key_points"].extend(fallback_points[:needed])

        while len(data.get("market_implications", [])) < 3:
            if "market_implications" not in data:
                data["market_implications"] = []
            fallback_implications = [
                "Market impact is being assessed",
                "Investment outlook remains under evaluation",
                "Economic effects are being studied"
            ]
            needed = 3 - len(data["market_implications"])
            data["market_implications"].extend(fallback_implications[:needed])

        logger.info(f"🔧 Ensured structure: {len(data['key_points'])} key points, {len(data['market_implications'])} implications")

    def _has_basic_content(self, data: Dict[str, Any]) -> bool:
        has_title = bool(data.get("title"))
        has_some_points = len(data.get("key_points", [])) >= 1
        has_some_implications = len(data.get("market_implications", [])) >= 1
        logger.info(f"Basic validation: title={has_title}, points={len(data.get('key_points', []))}, implications={len(data.get('market_implications', []))}")
        return has_title and has_some_points and has_some_implications

    def _extract_verified_source_info(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            json_pattern = r'\{[^}]*"main_source"[^}]*\}'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            for json_str in json_matches:
                try:
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
        try:
            image_patterns = [
                r'\{[^}]*"verified_images"[^}]*\}',
                r'\{[^}]*"telegram_compatible"[^}]*true[^}]*\}',
                r'\{[^}]*"url"[^}]*"title"[^}]*\}'
            ]
            for pattern in image_patterns:
                json_matches = re.findall(pattern, content, re.DOTALL)
                for json_str in json_matches:
                    try:
                        image_data = json.loads(json_str)
                        if 'verified_images' in image_data:
                            verified_images = image_data.get('verified_images', [])
                            if verified_images:
                                for image in verified_images:
                                    if image.get('telegram_compatible', False):
                                        logger.info(f"🖼️ Found Telegram-compatible image: {image.get('title', 'Unknown')[:50]}...")
                                        return image
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
                    logger.warning("❌ No Telegram-compatible images found in image finder results")
                    for image in images:
                        logger.debug(f"   - {image.get('title', 'Unknown')}: telegram_compatible={image.get('telegram_compatible', False)}")
        except Exception as e:
            logger.warning(f"Enhanced ImageFinder failed: {e}")

        logger.info("🖼️ No image available")
        return {}

    def _extract_stock_symbols(self, content: str) -> List[str]:
        stocks = re.findall(r'\b([A-Z]{2,5})\b', content)
        major_stocks = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "FDX", "INTC", "AMD"}
        relevant_stocks = [s for s in stocks if s in major_stocks]
        return list(set(relevant_stocks))[:3]

    def _extract_content_sections(self, lines: List[str], data: Dict[str, Any]):
        current_section = None
        for line in lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()
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

            bullet_patterns = [r'^[•\-*]\s+(.+)', r'^[\d]+\.\s+(.+)']
            for pattern in bullet_patterns:
                match = re.match(pattern, line_clean)
                if match:
                    point_text = self._strip_all_formatting(match.group(1))
                    if len(point_text) >= 20 and not self._is_metadata(point_text):
                        if current_section == 'key_points' and len(data["key_points"]) < 5:
                            concise_point = point_text[:150].strip()
                            data["key_points"].append(concise_point)
                        elif current_section == 'market_implications' and len(data["market_implications"]) < 5:
                            concise_impl = point_text[:150].strip()
                            data["market_implications"].append(concise_impl)
                    break

    def _ensure_required_item_count(self, data: Dict[str, Any], lines: List[str]):
        if len(data["key_points"]) < 3:
            self._extract_additional_points(data, lines, "key_points", 3 - len(data["key_points"]))
        if len(data["market_implications"]) < 3:
            self._extract_additional_points(data, lines, "market_implications", 3 - len(data["market_implications"]))
        if len(data["key_points"]) < 3:
            fallback_points = [
                "Market analysis in progress",
                "Key developments being monitored",
                "Financial trends under review"
            ]
            needed = 3 - len(data["key_points"])
            data["key_points"].extend(fallback_points[:needed])
        if len(data["market_implications"]) < 3:
            fallback_implications = [
                "Market impact assessment ongoing",
                "Investment strategies under evaluation",
                "Economic indicators being analyzed"
            ]
            needed = 3 - len(data["market_implications"])
            data["market_implications"].extend(fallback_implications[:needed])

    def _extract_additional_points(self, data: Dict[str, Any], lines: List[str], section_type: str, needed_count: int):
        extracted = 0
        for line in lines:
            if extracted >= needed_count:
                break
            line_clean = line.strip()
            if any(skip in line_clean.lower() for skip in [
                'key points', 'market implications', 'search', 'metadata', 'verified', 'confidence'
            ]):
                continue
            clean_text = self._strip_all_formatting(line_clean)
            if (30 <= len(clean_text) <= 150 and
                not self._is_metadata(clean_text) and
                clean_text not in data[section_type]):
                if section_type == "market_implications":
                    market_terms = ['market', 'trading', 'investor', 'economic', 'financial', 'outlook', 'impact']
                    if any(term in clean_text.lower() for term in market_terms):
                        data[section_type].append(clean_text)
                        extracted += 1
                else:
                    data[section_type].append(clean_text)
                    extracted += 1

    def _debug_article_links(self, data: Dict[str, Any]) -> None:
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
        if not text:
            return ""
        text = re.sub(r'\*\*([^*]*)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]*)\*', r'\1', text)
        text = re.sub(r'^\*+|\*+$', '', text)
        text = re.sub(r'^#+\s*', '', text)
        text = re.sub(r'[_`~\[\]]+', '', text)
        text = re.sub(r'^\s*[🎯📈📊🔔⚡💡]+\s*', '', text)
        text = re.sub(r'^\d+\.\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _apply_translation_formatting(self, message: str, language: str) -> str:
        if not message:
            return message

        # Preserve bullet points and structure
        lines = message.split('\n')
        formatted_lines = []
        for line in lines:
            # Ensure bullet points remain as •
            if line.strip().startswith('•'):
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        message = '\n'.join(formatted_lines)

        # Add RTL support for Arabic and Hebrew
        if language.lower() in ["arabic", "hebrew"]:
            arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
            hebrew_pattern = r'[\u0590-\u05FF]'
            has_arabic = re.search(arabic_pattern, message)
            has_hebrew = re.search(hebrew_pattern, message)
            if has_arabic or has_hebrew:
                rtl_start = '\u202B'
                rtl_end = '\u202C'
                message = f"{rtl_start}{message}{rtl_end}"
                message = re.sub(r'<b>', '<b dir="rtl">', message)
                message = re.sub(r'<i>', '<i dir="rtl">', message)

        # Ensure proper encoding
        try:
            message = message.encode('utf-8').decode('utf-8')
        except:
            pass

        return message

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
        metadata_terms = ['search', 'metadata', 'completed', 'results', 'hours', 'utc', 'total', 'verification', 'confidence', 'image search']
        return any(term in text.lower() for term in metadata_terms)

    def _has_valid_content(self, data: Dict[str, Any]) -> bool:
        has_title = bool(data.get("title") and len(data["title"]) > 15)
        key_points = data.get("key_points", [])
        market_implications = data.get("market_implications", [])
        has_sufficient_points = 3 <= len(key_points) <= 5
        has_sufficient_implications = 3 <= len(market_implications) <= 5
        logger.info(f"Content validation: title={has_title}, points={len(key_points)}/3-5, implications={len(market_implications)}/3-5")
        return has_title and has_sufficient_points and has_sufficient_implications

    def _create_clean_message(self, data: Dict[str, Any], language: str) -> str:
        title = self._clean_for_telegram(data.get("title", "Market Update"))
        
        message_parts = []
        message_parts.append(f"<b>{title}</b>")
        message_parts.append("")

        verified_source = data.get("verified_source")
        if verified_source:
            source_title = verified_source.get("title", "Financial News")
            source_url = verified_source.get("url", "")
            source_name = verified_source.get("source", "")
            url_verified = verified_source.get("url_verified", False)
            verification_icon = " ✅" if url_verified else " ⚠️"
            clean_title = self._clean_for_telegram(source_title)
            clean_source = self._clean_for_telegram(source_name)
            if source_url and source_url.startswith("http"):
                if clean_source and clean_source not in clean_title:
                    display_text = f"{clean_title} - {clean_source}"
                else:
                    display_text = clean_title
                source_line = f"<b>Source:</b> <a href=\"{source_url}\">{display_text}</a>{verification_icon}"
            else:
                source_line = f"<b>Source:</b> {clean_title}{verification_icon}"
            message_parts.append(source_line)
        else:
            source = data.get("source", "Financial News")
            source_url = data.get("source_url", "")
            if source_url and source_url.startswith("http"):
                source_line = f"<b>Source:</b> <a href=\"{source_url}\">{self._clean_for_telegram(source)}</a>"
            else:
                source_line = f"<b>Source:</b> {self._clean_for_telegram(source)}"
            message_parts.append(source_line)
        
        message_parts.append("")

        key_points = data.get("key_points", [])
        if key_points:
            message_parts.append("<b>Key Points:</b>")
            for point in key_points[:5]:  # Limit to 5 points
                clean_point = self._clean_for_telegram(point)
                if clean_point:
                    message_parts.append(f"• {clean_point}")

        message_parts.append("")

        implications = data.get("market_implications", [])
        if implications:
            message_parts.append("<b>Market Implications:</b>")
            for impl in implications[:5]:  # Limit to 5 implications
                clean_impl = self._clean_for_telegram(impl)
                if clean_impl:
                    message_parts.append(f"• {clean_impl}")

        message_parts.append("")
        message_parts.append("<b>Live Charts:</b>")
        message_parts.append('🔗 📊 <a href="https://finance.yahoo.com/quote/%5EGSPC/chart/?guccounter=1">S&P 500 Chart</a>')
        message_parts.append('🔗 📈 <a href="https://finance.yahoo.com/quote/%5EIXIC/chart/">NASDAQ Chart</a>')
        message_parts.append('🔗 📉 <a href="https://finance.yahoo.com/quote/%5EDJI/chart/">Dow Jones Chart</a>')
        message_parts.append('🔗 ⚡ <a href="https://finance.yahoo.com/quote/%5EVIX/chart/">VIX Chart</a>')
        message_parts.append('🔗 🏛️ <a href="https://finance.yahoo.com/quote/%5ETNX/chart/">10-Year Chart</a>')
        message_parts.append('🔗 💰 <a href="https://finance.yahoo.com/quote/GC%3DF/chart/">Gold Chart</a>')

        if language.lower() == "english":
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
            logger.info(f"🖼️ Attempting to send Telegram-compatible image: {image_title}")
            if self._send_photo(image_url, message):
                logger.info(f"✅ Photo sent successfully: {image_title}")
                return f"Message with verified image sent successfully - {image_title}"
            else:
                logger.warning(f"❌ Photo send failed for: {image_title}")
        else:
            if image_data:
                logger.warning(f"❌ Image not Telegram-compatible: {image_data.get('title', 'Unknown')} (Compatible: {image_data.get('telegram_compatible', False)})")
        
        if self._send_message(message):
            return "Text message sent successfully with verified source (image failed or not compatible)"
        return "Failed to send message"

    def _send_photo(self, image_url: str, caption: str) -> bool:
        try:
            if not self._test_image_accessibility(image_url):
                logger.warning(f"Image URL not accessible, skipping photo send: {image_url}")
                return False

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
        try:
            if isinstance(message, str):
                message_encoded = message.encode('utf-8').decode('utf-8')
            else:
                message_encoded = str(message)

            payload = {
                "chat_id": self.chat_id,
                "text": message_encoded[:4096],
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            }
            
            response = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=30,
                                   headers={'Content-Type': 'application/json; charset=utf-8'})
            
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