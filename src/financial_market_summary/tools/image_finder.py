import json
import re
import requests
import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin, unquote
from bs4 import BeautifulSoup
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from PIL import Image
import io

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedImageFinderInput(BaseModel):
    search_content: str = Field(description="Financial content to find relevant images for")
    mentioned_stocks: Optional[List[str]] = Field(default=[], description="Stock symbols mentioned in content")
    max_images: Optional[int] = Field(default=1, description="Maximum number of images to find")
    search_results_file: str = Field(description="Path to Tavily search results JSON file")

class EnhancedImageFinder(BaseTool):
    name: str = "enhanced_financial_image_finder"
    description: str = "Finds relevant financial images by extracting from Tavily search results file"
    args_schema: Type[BaseModel] = EnhancedImageFinderInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, search_content: str, mentioned_stocks: List[str] = None, max_images: int = 1, search_results_file: str = "") -> str:
        try:
            logger.info("="*60)
            logger.info("🖼️ IMAGE FINDER STARTED")
            logger.info("="*60)

            if mentioned_stocks is None:
                mentioned_stocks = []

            # Extract stock symbols from content if not provided
            if not mentioned_stocks:
                mentioned_stocks = self._extract_stock_symbols(search_content)

            logger.info(f"📊 Mentioned stocks: {mentioned_stocks}")
            logger.info(f"🎯 Max images to find: {max_images}")
            logger.info(f"📁 Search results file: {search_results_file}")

            # Always extract URLs from search results file
            if not search_results_file:
                logger.error("❌ No search results file provided")
                return json.dumps([], indent=2)

            article_urls = self._extract_urls_from_search_results(search_results_file)
            logger.info(f"📄 Extracted {len(article_urls)} URLs from search results")

            if not article_urls:
                logger.warning("⚠️ No article URLs found in search results")
                return json.dumps([], indent=2)

            # Extract screenshots from article URLs
            logger.info(f"📸 Starting screenshot extraction from {len(article_urls)} URLs...")
            extracted_images = self._extract_screenshots_from_urls(article_urls, search_content, max_images)
            logger.info(f"📊 Screenshot extraction complete: {len(extracted_images)} images captured")

            if not extracted_images:
                logger.warning("⚠️ No screenshots were captured from any URLs")
                return json.dumps([], indent=2)

            # Generate AI descriptions for screenshots
            logger.info(f"🤖 Generating AI descriptions for {len(extracted_images)} images...")
            extracted_images = self._generate_ai_descriptions(extracted_images, search_content)
            logger.info(f"✅ AI descriptions generated")

            # Save and return
            logger.info(f"💾 Saving {len(extracted_images)} image results...")
            self._save_image_results(extracted_images, search_content)

            extracted_images.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            logger.info("="*60)
            logger.info(f"✅ IMAGE FINDER COMPLETED: {len(extracted_images)} images")
            logger.info("="*60)

            return json.dumps(extracted_images[:max_images], indent=2)

        except Exception as e:
            logger.error(f"❌ Image finder failed with exception: {e}", exc_info=True)
            return json.dumps([], indent=2)

    def _extract_stock_symbols(self, content: str) -> List[str]:
        """Extract stock symbols from content"""
        stocks = re.findall(r'\b([A-Z]{2,5})\b', content)
        major_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META',
            'FDX', 'INTC', 'AMD', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'IBM',
            'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA'
        }

        valid_stocks = [stock for stock in stocks if stock in major_stocks]
        return list(set(valid_stocks))[:5]

    def _extract_urls_from_search_results(self, search_results_file: str) -> List[str]:
        """Extract article URLs from Tavily search results JSON file"""
        try:
            with open(search_results_file, 'r', encoding='utf-8') as f:
                search_data = json.load(f)

            urls = []
            articles = search_data.get('articles', [])

            for article in articles:
                url = article.get('url')
                if url and url.startswith('http'):
                    urls.append(url)

            logger.info(f"📄 Extracted {len(urls)} URLs from search results file")
            return urls

        except Exception as e:
            logger.error(f"Failed to extract URLs from search results file: {e}")
            return []

    def _extract_screenshots_from_urls(self, article_urls: List[str], content: str, max_images: int) -> List[Dict[str, Any]]:
        """Extract screenshots from provided article URLs"""
        extracted_images = []

        # Limit attempts to avoid long timeouts
        max_attempts = min(5, len(article_urls))
        attempts = 0
        skipped_yahoo = 0

        logger.info(f"🔍 Starting screenshot extraction:")
        logger.info(f"   - Total URLs available: {len(article_urls)}")
        logger.info(f"   - Max attempts: {max_attempts}")
        logger.info(f"   - Target images: {max_images}")

        for idx, url in enumerate(article_urls, 1):
            if attempts >= max_attempts:
                logger.info(f"⚠️ Reached max attempts ({max_attempts}), stopping screenshot extraction")
                break

            # Allow Yahoo Finance for chart extraction
            # Yahoo Finance has good charts that we can capture

            attempts += 1

            try:
                logger.info(f"📸 [{idx}/{len(article_urls)}] Attempt {attempts}/{max_attempts}")
                logger.info(f"   URL: {url}")
                screenshot_data = self._capture_chart_screenshot(url)

                if screenshot_data:
                    extracted_images.append(screenshot_data)
                    logger.info(f"✅ Screenshot captured successfully!")
                    logger.info(f"   Saved to: {screenshot_data.get('url', 'unknown')}")
                else:
                    logger.warning(f"⚠️ Screenshot capture returned None for {url[:80]}...")

                if len(extracted_images) >= max_images:
                    logger.info(f"✅ Target images ({max_images}) reached, stopping")
                    break

            except Exception as e:
                logger.error(f"❌ Exception during screenshot from {url[:80]}...: {e}", exc_info=True)
                continue

        logger.info(f"📊 Screenshot extraction summary:")
        logger.info(f"   - URLs processed: {idx}")
        logger.info(f"   - Yahoo Finance skipped: {skipped_yahoo}")
        logger.info(f"   - Screenshot attempts: {attempts}")
        logger.info(f"   - Images captured: {len(extracted_images)}")

        return extracted_images

    def _capture_chart_screenshot(self, article_url: str) -> Optional[Dict[str, Any]]:
        """Capture a single chart screenshot from article page"""
        try:
            with sync_playwright() as p:
                # Launch browser
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-gpu',
                        '--disable-extensions'
                    ]
                )

                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                )

                page = context.new_page()
                page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

                # Load page with more aggressive timeout and fallback strategies
                logger.info(f"📖 Loading page: {article_url}")
                try:
                    # Try with domcontentloaded first (faster)
                    page.goto(article_url, wait_until='domcontentloaded', timeout=20000)
                    logger.info("✅ Page loaded (domcontentloaded)")
                    page.wait_for_timeout(3000)
                except Exception as e:
                    logger.warning(f"⚠️ domcontentloaded failed, trying load strategy: {e}")
                    try:
                        # Fallback to basic load
                        page.goto(article_url, wait_until='load', timeout=25000)
                        logger.info("✅ Page loaded (load)")
                        page.wait_for_timeout(2000)
                    except Exception as e2:
                        logger.error(f"❌ Both load strategies failed: {e2}")
                        context.close()
                        browser.close()
                        return None

                # Scroll to load dynamic content and wait for charts to render
                try:
                    # Scroll multiple times to trigger lazy loading
                    logger.info(f"📜 Scrolling page to trigger chart loading...")
                    page.evaluate("window.scrollTo(0, 800)")
                    page.wait_for_timeout(5000)  # Increased to 5s
                    page.evaluate("window.scrollTo(0, 1600)")
                    page.wait_for_timeout(5000)  # Increased to 5s
                    page.evaluate("window.scrollTo(0, 2400)")
                    page.wait_for_timeout(5000)  # Increased to 5s
                    logger.info(f"✅ Scrolled page to load dynamic content")
                except Exception as e:
                    logger.warning(f"⚠️ Scroll failed: {e}")

                # Wait significantly longer for charts to fully render (live market data)
                logger.info(f"⏳ Waiting 15 seconds for live market charts to fully render...")
                page.wait_for_timeout(15000)  # Increased from 8s to 15s for live data
                logger.info(f"✅ Chart rendering wait complete")

                screenshot_data = None

                # Search for chart elements on page
                logger.info(f"🔍 Searching for chart elements on page...")

                # Find chart elements - prioritize containers that include labels/legends
                chart_selectors = [
                    # Yahoo Finance specific
                    'section[data-testid*="chart"]',  # Yahoo Finance chart section
                    'div[data-testid*="chart"]',
                    'div[id*="chart-"]',
                    'section[class*="chart"]',

                    # Generic chart containers (better than raw canvas)
                    'div[class*="chart-container"]',
                    'div[class*="chart-wrapper"]',
                    'div[class*="Chart"]',
                    'article[class*="chart"]',
                    'figure[class*="chart"]',

                    # Canvas/SVG as fallback
                    'canvas',
                    'svg[class*="chart"]',
                    'svg[class*="highcharts"]',
                ]

                for selector in chart_selectors:
                    try:
                        elements = page.query_selector_all(selector)
                        logger.info(f"   Selector '{selector}': found {len(elements)} element(s)")

                        if not elements:
                            continue

                        # Try multiple elements (first few) in case some are hidden
                        elements_to_try = min(len(elements), 3)
                        logger.info(f"   Will try first {elements_to_try} element(s)")

                        for elem_idx in range(elements_to_try):
                            element = elements[elem_idx]
                            logger.info(f"   Trying element #{elem_idx + 1}/{elements_to_try}...")

                            # Get bounding box first (before scrolling)
                            box = element.bounding_box()
                            if not box:
                                logger.warning(f"   ⚠️ Element #{elem_idx + 1} has no bounding box, trying next...")
                                continue

                            if box['width'] < 200 or box['height'] < 100:
                                logger.warning(f"   ⚠️ Element #{elem_idx + 1} too small ({box['width']}x{box['height']}), trying next...")
                                continue

                            logger.info(f"   ✅ Element #{elem_idx + 1} has valid size: {box['width']}x{box['height']}")

                            # Try to scroll element into view (with timeout protection)
                            try:
                                element.scroll_into_view_if_needed(timeout=5000)
                                logger.info(f"   ✅ Scrolled element into view")
                                # Wait extra time after scrolling for chart to render
                                logger.info(f"   ⏳ Waiting 10 seconds for chart to render after scroll...")
                                page.wait_for_timeout(10000)  # Increased from 5s to 10s
                            except Exception as scroll_error:
                                logger.warning(f"   ⚠️ Scroll timeout/failed, continuing anyway: {scroll_error}")
                                # Continue anyway - element might already be visible
                                # Still wait a bit for rendering
                                page.wait_for_timeout(5000)  # Increased from 3s to 5s

                            # Use the element directly (we're already targeting containers)
                            target_element = element
                            logger.info(f"   📊 Using element: {box['width']}x{box['height']}")

                            # Take screenshot with timeout protection
                            try:
                                logger.info(f"   📸 Taking screenshot...")
                                screenshot_bytes = target_element.screenshot(
                                    timeout=15000,  # Increased back to 15s
                                    animations='disabled'
                                )
                                logger.info(f"   ✅ Screenshot captured: {len(screenshot_bytes)} bytes")
                            except Exception as screenshot_error:
                                logger.error(f"   ❌ Screenshot failed: {screenshot_error}")
                                continue  # Try next element

                            # Save screenshot
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            domain = re.sub(r'[^\w\s-]', '', urlparse(article_url).netloc)
                            filename = f"chart_{domain}_{timestamp}.png"

                            project_root = Path(__file__).resolve().parent.parent.parent.parent
                            screenshots_dir = project_root / "output" / "screenshots"
                            screenshots_dir.mkdir(parents=True, exist_ok=True)
                            filepath = screenshots_dir / filename

                            with open(filepath, 'wb') as f:
                                f.write(screenshot_bytes)

                            logger.info(f"💾 Screenshot saved: {filepath}")

                            screenshot_data = {
                                'url': str(filepath).replace('\\', '/'),
                                'title': f'Financial chart from {urlparse(article_url).netloc}',
                                'source': urlparse(article_url).netloc,
                                'type': 'screenshot',
                                'relevance_score': 95,
                                'telegram_compatible': True,
                                'file_type': 'png',
                                'trusted_source': True,
                                'extraction_method': f'screenshot_{selector}',
                                'source_article': article_url,
                                'image_description': '',  # Will be filled by AI
                                'content_type': 'image/png',
                                'file_size': str(len(screenshot_bytes)),
                            }

                            break  # Found chart, stop trying elements

                        # If we got a screenshot, stop trying selectors
                        if screenshot_data:
                            break

                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed with selector '{selector}': {e}")
                        continue

                # Log if no charts were found at all
                if not screenshot_data:
                    logger.warning(f"❌ No usable chart elements found on page")
                    logger.info(f"   Tried {len(chart_selectors)} different selectors")

                context.close()
                browser.close()

                return screenshot_data

        except Exception as e:
            logger.error(f"Screenshot capture failed for {article_url}: {e}")
            return None

    def _generate_ai_descriptions(self, extracted_images: List[Dict[str, Any]], search_content: str) -> List[Dict[str, Any]]:
        """Generate AI descriptions for screenshots using Gemini Vision API"""
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                logger.error("❌ GOOGLE_API_KEY not found in environment")
                return extracted_images

            # Configure Gemini
            genai.configure(api_key=google_api_key)

            # Use Gemini 2.5 Flash with vision
            logger.info(f"   🤖 Initializing Gemini 2.5 Flash for vision analysis...")
            model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',  # Gemini 2.5 Flash with vision support
                generation_config={
                    'temperature': 0.4,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 200,
                }
            )
            logger.info(f"   ✅ Model initialized: gemini-2.0-flash-exp")

            logger.info(f"🤖 Generating AI descriptions for {len(extracted_images)} images")

            for img in extracted_images:
                try:
                    image_path = img.get('url', '')
                    if not image_path or not os.path.exists(image_path):
                        logger.warning(f"⚠️ Image file not found: {image_path}")
                        img['image_description'] = 'Financial chart'
                        continue

                    logger.info(f"📸 Analyzing: {image_path}")

                    # Load image using PIL
                    pil_image = Image.open(image_path)

                    # Convert to RGB if needed
                    if pil_image.mode in ('RGBA', 'LA', 'P'):
                        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                        if pil_image.mode == 'P':
                            pil_image = pil_image.convert('RGBA')
                        rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                        pil_image = rgb_image

                    # Create analysis prompt
                    analysis_prompt = f"""You are a financial news analyst. Analyze this chart image and write a concise 1-2 sentence description.

Context: {search_content[:300]}

Requirements:
1. Identify what financial instrument or market is shown (stock, index, commodity, etc.)
2. Describe the trend or movement visible in the chart
3. Include specific numbers if clearly visible (prices, percentages, values)
4. Use past tense verbs (climbed, rose, fell, dropped, gained, declined)
5. Write in professional news style
6. Keep it to 1-2 sentences maximum

Examples:
- "S&P 500 index climbed 1.8% to 4,567 points showing strong market momentum."
- "Tesla stock fell 3.2% to $245.80 amid profit-taking after recent rally."
- "U.S. Treasury 10-year yields rose to 4.28% as investors assessed inflation data."

Write ONLY the description, no additional text."""

                    # Generate description with image
                    logger.info(f"   🤖 Sending to Gemini Vision API...")
                    try:
                        response = model.generate_content(
                            [analysis_prompt, pil_image],
                            safety_settings={
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                            }
                        )
                        logger.info(f"   ✅ Got response from Gemini Vision")
                    except Exception as api_error:
                        logger.error(f"   ❌ Gemini API call failed: {api_error}")
                        img['image_description'] = 'Financial market chart'
                        continue

                    # Extract description
                    if response and response.text:
                        raw_description = response.text.strip()
                        logger.info(f"   📝 Raw response: {raw_description[:100]}...")

                        # Clean up description
                        description = re.sub(r'^["\']+|["\']+$', '', raw_description)  # Remove quotes
                        description = re.sub(r'\s+', ' ', description)  # Normalize spaces

                        # Limit to 2 sentences
                        sentences = [s.strip() for s in re.split(r'[.!?]+', description) if s.strip()]
                        if len(sentences) > 2:
                            description = '. '.join(sentences[:2]) + '.'
                        elif sentences:
                            description = '. '.join(sentences)
                            if not description.endswith('.'):
                                description += '.'

                        # Truncate if too long
                        if len(description) > 300:
                            description = description[:297] + '...'

                        # Validate description is not too generic
                        generic_phrases = ['financial market chart', 'chart showing', 'graph displaying', 'unable to', 'cannot determine']
                        is_generic = any(phrase in description.lower() for phrase in generic_phrases) and len(description) < 50

                        if is_generic:
                            logger.warning(f"⚠️ Description too generic, retrying with simpler prompt...")

                            # Try again with a simpler, more direct prompt
                            simple_prompt = f"""Look at this financial chart and describe what you see in 1-2 sentences.

Include:
- What is being shown (stock/index name and current value if visible)
- The trend (up/down/stable)
- Any specific numbers you can read

Example: "S&P 500 index at 6,711.20 points, up 22.74 points, showing upward trend throughout the trading day."

Description:"""

                            retry_response = model.generate_content(
                                [simple_prompt, pil_image],
                                safety_settings={
                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                }
                            )

                            if retry_response and retry_response.text:
                                description = retry_response.text.strip()
                                description = re.sub(r'^["\']+|["\']+$', '', description)
                                description = re.sub(r'\s+', ' ', description)
                                logger.info(f"   🔄 Retry description: {description[:100]}...")

                        img['image_description'] = description
                        logger.info(f"✅ Final description: {description}")

                    else:
                        logger.warning(f"⚠️ Empty response from Gemini")
                        img['image_description'] = 'Financial market chart'

                    # Small delay to avoid rate limits
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"❌ AI description generation failed: {e}")
                    img['image_description'] = 'Financial market chart'
                    continue

            return extracted_images

        except Exception as e:
            logger.error(f"❌ AI description system failed: {e}")
            # Return images with default descriptions
            for img in extracted_images:
                if not img.get('image_description'):
                    img['image_description'] = 'Financial market chart'
            return extracted_images

    def _save_image_results(self, extracted_images: List[Dict[str, Any]], search_content: str) -> str:
        """Save scraped image details to output/image_results directory"""
        try:
            # Create output directory structure
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            image_results_dir = project_root / "output" / "image_results"
            image_results_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_content = re.sub(r'[^\w\s-]', '', search_content[:50]).strip()
            safe_content = re.sub(r'[-\s]+', '-', safe_content)

            filename = f"image_results_{timestamp}_{safe_content}.json"
            filepath = image_results_dir / filename

            # Prepare detailed image results data
            image_results_data = {
                "metadata": {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                    "search_content": search_content[:200] + "..." if len(search_content) > 200 else search_content,
                    "total_images_found": len(extracted_images),
                    "extraction_date": datetime.now().isoformat()
                },
                "extracted_images": []
            }

            # Process each image with detailed information
            for idx, img in enumerate(extracted_images, 1):
                image_detail = {
                    "image_number": idx,
                    "url": img.get('url', ''),
                    "title": img.get('title', 'Unknown Title'),
                    "source": img.get('source', 'Unknown Source'),
                    "source_article": img.get('source_article', ''),
                    "extraction_method": img.get('extraction_method', 'unknown'),
                    "file_type": img.get('file_type', 'unknown'),
                    "image_description": img.get('image_description', ''),
                    "relevance_score": img.get('relevance_score', 0),
                    "telegram_compatible": img.get('telegram_compatible', False),
                    "content_type": img.get('content_type'),
                    "file_size": img.get('file_size'),
                    "extraction_timestamp": datetime.now().isoformat()
                }
                image_results_data["extracted_images"].append(image_detail)

            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(image_results_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Saved {len(extracted_images)} image results to: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.warning(f"Failed to save image results: {e}")
            return ""
