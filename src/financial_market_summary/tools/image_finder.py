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
    max_images: Optional[int] = Field(default=1, description="Maximum number of images to find")
    search_results_file: str = Field(description="Path to Tavily search results JSON file")

class EnhancedImageFinder(BaseTool):
    name: str = "enhanced_financial_image_finder"
    description: str = "Finds relevant financial images by extracting from Tavily search results file"
    args_schema: Type[BaseModel] = EnhancedImageFinderInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, search_content: str, max_images: int = 1, search_results_file: str = "") -> str:
        try:
            logger.info("🔍 IMAGE FINDER STARTED")

            # Always extract URLs from search results file
            if not search_results_file:
                logger.error("❌ No search results file provided")
                return json.dumps([], indent=2)

            article_urls = self._extract_urls_from_search_results(search_results_file)

            if not article_urls:
                logger.warning("⚠️  No URLs found")
                return json.dumps([], indent=2)

            logger.info(f"📄 Processing {len(article_urls)} URL(s)")
            extracted_images = self._extract_screenshots_from_urls(article_urls, search_content, max_images)

            if not extracted_images:
                logger.warning("⚠️  No images captured")
                return json.dumps([], indent=2)

            # Verify descriptions
            extracted_images = self._generate_ai_descriptions(extracted_images, search_content)
            self._save_image_results(extracted_images, search_content)

            logger.info("="*60)
            logger.info(f" IMAGE FINDER COMPLETED: {len(extracted_images)} images")
            logger.info("="*60)

            return json.dumps(extracted_images[:max_images], indent=2)

        except Exception as e:
            logger.error(f" Image finder failed with exception: {e}", exc_info=True)
            return json.dumps([], indent=2)

    def _extract_h1_from_url(self, url: str) -> str:
        """Extract h1 heading from article URL"""
        try:
            logger.info(f"    Extracting h1 heading from: {url[:80]}...")

            response = requests.get(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
                },
                timeout=10
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                h1_tag = soup.find('h1')

                if h1_tag:
                    h1_text = h1_tag.get_text(strip=True)
                    logger.info(f"    Found h1: {h1_text[:80]}...")
                    return h1_text
                else:
                    logger.warning(f"    No h1 tag found")
                    return f'Financial chart from {urlparse(url).netloc}'
            else:
                logger.warning(f"    HTTP {response.status_code}")
                return f'Financial chart from {urlparse(url).netloc}'

        except Exception as e:
            logger.error(f"    H1 extraction failed: {e}")
            return f'Financial chart from {urlparse(url).netloc}'

    def _extract_urls_from_search_results(self, search_results_file: str) -> List[str]:
        """Extract article URLs from Tavily search results JSON file"""
        try:
            with open(search_results_file, 'r', encoding='utf-8') as f:
                search_data = json.load(f)

            cnbc_urls = []
            other_urls = []
            articles = search_data.get('articles', [])

            for article in articles:
                url = article.get('url')
                if url and url.startswith('http'):
                    # Prioritize CNBC URLs silently
                    if 'cnbc.com' in url.lower():
                        cnbc_urls.append(url)
                    else:
                        other_urls.append(url)

            # CNBC URLs first, then others
            urls = cnbc_urls + other_urls

            logger.info(f" Extracted {len(urls)} URLs from search results file")
            return urls

        except Exception as e:
            logger.error(f"Failed to extract URLs from search results file: {e}")
            return []

    def _ai_should_skip_image(self, image_bytes: bytes, img_alt: str) -> bool:
        """Use AI to determine if image is a generic photo (traders, buildings) or relevant content"""
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                logger.warning("    No GOOGLE_API_KEY, accepting image by default")
                return False

            # Configure Gemini
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config={
                    'temperature': 0.2,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 20,
                }
            )

            # Load image
            pil_image = Image.open(io.BytesIO(image_bytes))

            # AI prompt
            prompt = f"""Look at this image from a financial news article.

Alt text: "{img_alt}"

Question: Is this a CHART or GRAPH showing financial/market data?

SKIP if the image shows:
- People (traders, executives, workers, anyone)
- Buildings or offices
- Products or objects (robots, phones, cars, etc.)
- Factories or warehouses
- ANY photograph of real-world scenes
- Logos or company signs

KEEP ONLY if the image is:
- A financial chart (line/bar/candlestick chart)
- A graph with data
- A data visualization (pie chart, infographic with numbers)
- Stock market price charts
- Index performance charts

If you see people, buildings, products, or any real-world photography → say SKIP
If you see a chart/graph with financial data → say KEEP

Answer ONLY "SKIP" or "KEEP"."""

            response = model.generate_content(
                [prompt, pil_image],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            if response and response.text:
                answer = response.text.strip().upper()
                logger.info(f"    AI decision: {answer}")
                return "SKIP" in answer
            else:
                logger.warning(f"    Empty AI response, keeping image")
                return False

        except Exception as e:
            logger.error(f"    AI check failed: {e}, keeping image")
            return False

    def _extract_image_from_img_tag(self, article_url: str) -> Optional[Dict[str, Any]]:
        """Use existing chart screenshot method but look for <img> tags instead"""
        # This method now just returns None to use chart fallback
        # We'll modify the chart screenshot method to look for <img> first
        return None

    def _extract_screenshots_from_urls(self, article_urls: List[str], content: str, max_images: int) -> List[Dict[str, Any]]:
        """Extract images from provided article URLs - tries <img> tags first, screenshots as fallback"""
        extracted_images = []

        for idx, url in enumerate(article_urls, 1):
            if len(extracted_images) >= max_images:
                break

            try:
                logger.info(f"📸 [{idx}/{len(article_urls)}] {url[:60]}...")

                # Try <img> tag first
                img_tag_data = self._extract_image_from_img_tag(url)

                if img_tag_data:
                    extracted_images.append(img_tag_data)
                    logger.info(f"✅ Image captured from <img> tag")
                    continue

                # Fallback to chart screenshot
                screenshot_data = self._capture_chart_screenshot(url)

                if screenshot_data:
                    extracted_images.append(screenshot_data)
                    logger.info(f"✅ Chart screenshot captured")
                else:
                    logger.warning(f"⚠️  Failed to extract image")

            except Exception as e:
                logger.error(f"❌ Error: {e}")
                continue

        logger.info(f"✅ Extracted {len(extracted_images)} image(s)")

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

                # Load page
                try:
                    page.goto(article_url, wait_until='domcontentloaded', timeout=20000)
                    page.wait_for_timeout(3000)
                except Exception as e:
                    try:
                        page.goto(article_url, wait_until='load', timeout=25000)
                        page.wait_for_timeout(2000)
                    except Exception as e2:
                        logger.error(f"❌ Page load failed: {e2}")
                        context.close()
                        browser.close()
                        return None

                # Hide popups
                try:
                    page.add_style_tag(content="""
                        [role="dialog"], [class*="dialog"], [class*="modal"],
                        body > [class*="backdrop"], body > [class*="overlay"],
                        div[style*="z-index: 9999"], [class*="lightbox"] {
                            display: none !important;
                        }
                        body { overflow: visible !important; }
                    """)
                except:
                    pass

                # Scroll to load lazy images
                page_height = page.evaluate("document.body.scrollHeight")
                for scroll_position in range(0, int(page_height) + 800, 800):
                    page.evaluate(f"window.scrollTo(0, {scroll_position})")
                    page.wait_for_timeout(200)

                page.wait_for_timeout(2000)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(3000)

                screenshot_data = None

                # Try <img> tags (skip first, capture second)
                try:
                    img_elements = page.query_selector_all('img')
                    large_images_found = 0

                    for img_elem in img_elements:
                        try:
                            box = img_elem.bounding_box()

                            if box and box['width'] >= 400 and box['height'] >= 250:
                                large_images_found += 1

                                # Skip first large image
                                if large_images_found == 1:
                                    continue

                                # Capture second large image
                                logger.info(f"✅ Capturing image #{large_images_found} ({int(box['width'])}x{int(box['height'])})")

                                # Try to screenshot parent container
                                screenshot_elem = img_elem

                                # Check if parent is figure, picture, or container div
                                parent = img_elem.evaluate('el => el.parentElement')
                                if parent:
                                    parent_elem = img_elem.evaluate_handle('el => el.parentElement')
                                    parent_tag = parent_elem.evaluate('el => el.tagName.toLowerCase()')

                                    if parent_tag in ['figure', 'picture', 'article']:
                                        screenshot_elem = parent_elem

                                screenshot_elem.scroll_into_view_if_needed(timeout=5000)
                                page.wait_for_timeout(500)
                                screenshot_bytes = screenshot_elem.screenshot(timeout=10000)

                                # Extract rich description from multiple sources
                                fallback_description = ''
                                try:
                                    # Extract multiple <p> and <li> tags for rich context
                                    description_parts = img_elem.evaluate('''(element) => {
                                            let parts = [];

                                            // 1. Get image alt text
                                            const alt = element.getAttribute('alt');
                                            if (alt && alt.length > 10) {
                                                parts.push('IMAGE: ' + alt.trim());
                                            }

                                            // 2. Get figcaption if image is in a figure
                                            let figcaption = element.closest('figure')?.querySelector('figcaption');
                                            if (figcaption) {
                                                parts.push('CAPTION: ' + figcaption.textContent.trim());
                                            }

                                            // 3. Find multiple <p> tags after the image (up to 3)
                                            let current = element;
                                            let pCount = 0;
                                            const maxP = 3;

                                            while (current && pCount < maxP) {
                                                let nextSibling = current.nextElementSibling;
                                                if (nextSibling) {
                                                    if (nextSibling.tagName.toLowerCase() === 'p') {
                                                        const pText = nextSibling.textContent.trim();
                                                        if (pText.length > 20) {
                                                            parts.push(pText);
                                                            pCount++;
                                                        }
                                                    }
                                                    // Check if next sibling contains <p> tags
                                                    const nestedP = nextSibling.querySelectorAll('p');
                                                    nestedP.forEach(p => {
                                                        if (pCount < maxP) {
                                                            const pText = p.textContent.trim();
                                                            if (pText.length > 20) {
                                                                parts.push(pText);
                                                                pCount++;
                                                            }
                                                        }
                                                    });
                                                }
                                                current = current.parentElement;
                                                if (!current || current.tagName.toLowerCase() === 'body') break;
                                            }

                                            // 4. Find <li> items near the image (up to 5)
                                            let parent = element.parentElement;
                                            let searchDistance = 0;
                                            while (parent && searchDistance < 3) {
                                                const listItems = parent.querySelectorAll('li');
                                                if (listItems.length > 0) {
                                                    const liTexts = [];
                                                    listItems.forEach((li, idx) => {
                                                        if (idx < 5) {
                                                            const liText = li.textContent.trim();
                                                            if (liText.length > 10) {
                                                                liTexts.push('• ' + liText);
                                                            }
                                                        }
                                                    });
                                                    if (liTexts.length > 0) {
                                                        parts.push('KEY POINTS:\\n' + liTexts.join('\\n'));
                                                        break;
                                                    }
                                                }
                                                parent = parent.parentElement;
                                                searchDistance++;
                                            }

                                            return parts.join('\\n\\n');
                                    }''')

                                    if description_parts and len(description_parts) > 20:
                                        fallback_description = description_parts
                                        sources = []
                                        if 'IMAGE:' in description_parts:
                                            sources.append('alt')
                                        if 'CAPTION:' in description_parts:
                                            sources.append('figcaption')
                                        if 'KEY POINTS:' in description_parts:
                                            sources.append('li')
                                        p_count = description_parts.count('\n\n') - len(sources) + 1
                                        if p_count > 0:
                                            sources.append(f'{p_count}p')

                                        logger.info(f"📝 Description from: {', '.join(sources)}")

                                except Exception as p_error:
                                    logger.warning(f"⚠️  Description extraction failed: {p_error}")

                                # Save it
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                domain = re.sub(r'[^\w\s-]', '', urlparse(article_url).netloc)
                                filename = f"img_{domain}_{timestamp}.png"

                                project_root = Path(__file__).resolve().parent.parent.parent.parent
                                screenshots_dir = project_root / "output" / "screenshots"
                                screenshots_dir.mkdir(parents=True, exist_ok=True)
                                filepath = screenshots_dir / filename

                                with open(filepath, 'wb') as f:
                                    f.write(screenshot_bytes)

                                screenshot_data = {
                                    'url': str(filepath).replace('\\', '/'),
                                    'title': self._extract_h1_from_url(article_url),
                                    'source': urlparse(article_url).netloc,
                                    'type': 'img_tag_screenshot',
                                    'telegram_compatible': True,
                                    'file_type': 'png',
                                    'trusted_source': True,
                                    'extraction_method': 'img_tag_screenshot',
                                    'source_article': article_url,
                                    'image_description': fallback_description,
                                    'content_type': 'image/png',
                                    'file_size': str(len(screenshot_bytes)),
                                }

                                context.close()
                                browser.close()
                                return screenshot_data

                        except Exception as img_error:
                            logger.warning(f"⚠️  Image capture failed: {img_error}")
                            continue

                    logger.warning("⚠️  No <img> tags found, trying chart selectors...")

                except Exception as e:
                    logger.warning(f" <img> tag search failed: {e}, trying chart selectors...")

                # STEP 2: Fallback to chart selectors
                logger.info(f" STEP 2: Searching for chart elements...")

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
                                logger.warning(f"    Element #{elem_idx + 1} has no bounding box, trying next...")
                                continue

                            if box['width'] < 200 or box['height'] < 100:
                                logger.warning(f"    Element #{elem_idx + 1} too small ({box['width']}x{box['height']}), trying next...")
                                continue

                            logger.info(f"    Element #{elem_idx + 1} has valid size: {box['width']}x{box['height']}")

                            # Try to scroll element into view (with timeout protection)
                            try:
                                element.scroll_into_view_if_needed(timeout=5000)
                                logger.info(f"    Scrolled element into view")
                                # Wait extra time after scrolling for chart to render
                                logger.info(f"   ⏳ Waiting 10 seconds for chart to render after scroll...")
                                page.wait_for_timeout(10000)  # Increased from 5s to 10s
                            except Exception as scroll_error:
                                logger.warning(f"    Scroll timeout/failed, continuing anyway: {scroll_error}")
                                # Continue anyway - element might already be visible
                                # Still wait a bit for rendering
                                page.wait_for_timeout(5000)  # Increased from 3s to 5s

                            # Use the element directly (we're already targeting containers)
                            target_element = element
                            logger.info(f"    Using element: {box['width']}x{box['height']}")

                            # Take screenshot with timeout protection
                            try:
                                logger.info(f"    Taking screenshot...")
                                screenshot_bytes = target_element.screenshot(
                                    timeout=15000,  # Increased back to 15s
                                    animations='disabled'
                                )
                                logger.info(f"    Screenshot captured: {len(screenshot_bytes)} bytes")
                            except Exception as screenshot_error:
                                logger.error(f"    Screenshot failed: {screenshot_error}")
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

                            logger.info(f" Screenshot saved: {filepath}")

                            # Extract h1 heading for title
                            article_title = self._extract_h1_from_url(article_url)

                            screenshot_data = {
                                'url': str(filepath).replace('\\', '/'),
                                'title': article_title,
                                'source': urlparse(article_url).netloc,
                                'type': 'screenshot',
                                'telegram_compatible': True,
                                'file_type': 'png',
                                'trusted_source': True,
                                'extraction_method': f'screenshot_{selector}',
                                'source_article': article_url,
                                'image_description': '',  # Will be filled later using Crawl4AI
                                'content_type': 'image/png',
                                'file_size': str(len(screenshot_bytes)),
                            }

                            break  # Found chart, stop trying elements

                        # If we got a screenshot, stop trying selectors
                        if screenshot_data:
                            break

                    except Exception as e:
                        logger.warning(f"    Failed with selector '{selector}': {e}")
                        continue

                # Log if no charts were found at all
                if not screenshot_data:
                    logger.warning(f" No usable chart elements found on page")
                    logger.info(f"   Tried {len(chart_selectors)} different selectors")

                context.close()
                browser.close()

                return screenshot_data

        except Exception as e:
            logger.error(f"Screenshot capture failed for {article_url}: {e}")
            return None

    def _generate_ai_descriptions(self, extracted_images: List[Dict[str, Any]], search_content: str) -> List[Dict[str, Any]]:
        """Generate descriptions: Use <p> tags for img screenshots, AI analysis for chart screenshots"""
        try:
            logger.info(f" Processing descriptions for {len(extracted_images)} images")

            for idx, img in enumerate(extracted_images, 1):
                extraction_method = img.get('extraction_method', '')
                img_type = img.get('type', '')

                # Check if description already exists (from <p> tag during img capture)
                existing_description = img.get('image_description', '')

                if existing_description:
                    # Already has description from <p> tag (img_tag_screenshot)
                    logger.info(f" [{idx}/{len(extracted_images)}] ✓ Using <p> tag: {existing_description[:100]}...")
                    continue

                # No description yet - this is a chart screenshot fallback
                if img_type == 'screenshot' or 'chart' in extraction_method.lower():
                    logger.info(f" [{idx}/{len(extracted_images)}] Chart screenshot detected - using AI analysis")

                    source_article = img.get('source_article', '')
                    if not source_article:
                        logger.warning(f" [{idx}/{len(extracted_images)}] No source article URL")
                        img['image_description'] = ''
                        continue

                    try:
                        # Crawl article to get paragraphs
                        logger.info(f" [{idx}/{len(extracted_images)}] Crawling article for chart description...")
                        paragraphs = self._crawl_article_with_crawl4ai(source_article)

                        if not paragraphs:
                            logger.warning(f" [{idx}/{len(extracted_images)}] No paragraphs extracted")
                            img['image_description'] = ''
                            continue

                        logger.info(f" [{idx}/{len(extracted_images)}] Extracted {len(paragraphs)} paragraphs")

                        # Use AI to match chart to paragraph
                        image_path = img.get('url', '')
                        if image_path and os.path.exists(image_path):
                            description = self._ai_extract_chart_description(image_path, paragraphs)

                            if description:
                                img['image_description'] = description
                                logger.info(f" [{idx}/{len(extracted_images)}] ✓ AI description: {description[:100]}...")
                            else:
                                logger.warning(f" [{idx}/{len(extracted_images)}] ✗ AI could not extract description")
                                img['image_description'] = ''
                        else:
                            logger.warning(f" [{idx}/{len(extracted_images)}] Image file not found")
                            img['image_description'] = ''

                        time.sleep(0.5)  # Rate limiting

                    except Exception as e:
                        logger.error(f" [{idx}/{len(extracted_images)}] Chart description failed: {e}")
                        img['image_description'] = ''
                        continue
                else:
                    # Unknown type without description
                    logger.warning(f" [{idx}/{len(extracted_images)}] ✗ No description for type: {img_type}")
                    img['image_description'] = ''

            return extracted_images

        except Exception as e:
            logger.error(f" Description generation failed: {e}")
            return extracted_images

    def _crawl_article_with_crawl4ai(self, article_url: str) -> List[str]:
        """Use Crawl4AI to extract article paragraphs - only for chart fallback"""
        try:
            import asyncio
            from crawl4ai import AsyncWebCrawler

            logger.info(f"    Crawl4AI crawling URL for chart description...")

            async def crawl():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(url=article_url)
                    return result

            result = asyncio.run(crawl())

            if result.success:
                html_content = ""
                if hasattr(result, 'cleaned_html') and result.cleaned_html:
                    html_content = result.cleaned_html
                elif hasattr(result, 'html') and result.html:
                    html_content = result.html
                else:
                    logger.error(f"    No HTML content available")
                    return []

                logger.info(f"    Crawl4AI success: {len(html_content)} chars")

                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                paragraphs = soup.find_all('p')
                logger.info(f"    Found {len(paragraphs)} <p> tags")

                # Extract and filter paragraph texts
                paragraph_texts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) < 80:
                        continue
                    if text.count(' ') < 10:
                        continue

                    # Financial keywords filter
                    text_lower = text.lower()
                    financial_keywords = [
                        'stock', 'market', 'index', 'dow', 'nasdaq', 's&p', 'djia',
                        'rose', 'fell', 'gained', 'declined', 'points', 'percent',
                        'shares', 'trading', 'investors', 'wall street', 'treasury',
                        'futures', 'bond', 'yield', 'rally', 'price', 'earnings',
                        'revenue', 'profit', 'loss', 'up', 'down', 'higher', 'lower',
                        'popped', 'jumped', 'dropped', 'year', 'quarter'
                    ]

                    if any(keyword in text_lower for keyword in financial_keywords):
                        paragraph_texts.append(text)

                logger.info(f"    Extracted {len(paragraph_texts)} relevant paragraphs")
                return paragraph_texts

            else:
                logger.error(f"    Crawl4AI failed")
                return []

        except Exception as e:
            logger.error(f"    Crawl4AI exception: {e}")
            return []

    def _ai_extract_chart_description(self, image_path: str, paragraphs: List[str]) -> str:
        """Use AI to match chart image to article paragraphs - only for chart fallback"""
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            if not paragraphs or len(paragraphs) == 0:
                logger.error("    No paragraphs provided")
                return ""

            logger.info(f"    Analyzing {len(paragraphs)} paragraphs for chart...")

            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                logger.error("    GOOGLE_API_KEY not found")
                return ""

            # Configure Gemini
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config={
                    'temperature': 0.3,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 200,
                }
            )

            # Load image
            pil_image = Image.open(image_path)
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGBA')
                rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                pil_image = rgb_image

            # Prepare paragraphs
            numbered_paragraphs = []
            for i, para in enumerate(paragraphs[:20], 1):  # Limit to 20
                para_text = para[:300] + "..." if len(para) > 300 else para
                numbered_paragraphs.append(f"{i}. {para_text}")
            paragraphs_text = "\n\n".join(numbered_paragraphs)

            # STEP 1: Analyze chart
            logger.info(f"    Step 1: Analyzing chart image...")
            chart_analysis_prompt = """Analyze this financial chart and describe what it shows.

IDENTIFY:
1. Which stock/index? (S&P 500, Dow Jones, Nasdaq, specific stocks)
2. Chart type? (line, candlestick, bar)
3. Time period? (intraday, daily, weekly, monthly, yearly)
4. Trend? (up, down, sideways, volatile)
5. Price levels or percentages visible?

Provide 2-3 sentence analysis."""

            chart_response = model.generate_content(
                [chart_analysis_prompt, pil_image],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            chart_understanding = ""
            if chart_response and chart_response.text:
                chart_understanding = chart_response.text.strip()
                logger.info(f"    Chart analysis: {chart_understanding[:150]}...")

            # STEP 2: Match to paragraph
            logger.info(f"    Step 2: Matching chart to paragraphs...")
            extraction_prompt = f"""CHART SHOWS:
{chart_understanding}

ARTICLE PARAGRAPHS:
{paragraphs_text}

TASK:
Find and COPY the exact text that describes this chart.

INSTRUCTIONS:
1. Look for the stock/company/index shown in the chart
2. COPY the exact 1-2 sentences that mention it
3. Return ONLY the copied text
4. If nothing matches, return "NONE"

EXAMPLE:
Chart shows "Deere stock up 8%"
Paragraph 7: "Shares of Deere have popped 8% this year. The stock was last trading marginally higher."
Return EXACTLY: Shares of Deere have popped 8% this year. The stock was last trading marginally higher.

Your response (exact text):"""

            response = model.generate_content(
                [extraction_prompt, pil_image],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            if response and response.text:
                ai_response = response.text.strip()
                logger.info(f"    AI response: {ai_response[:200]}...")

                if ai_response.upper() == "NONE" or ai_response.upper().startswith("NONE"):
                    logger.warning(f"    AI could not find matching paragraph")
                    return ""

                description = ai_response.strip()
                if description and not description.endswith(('.', '!', '?')):
                    description += '.'

                logger.info(f"    ✓ AI extracted: {description[:150]}...")
                return description

            logger.warning(f"    Empty AI response")
            return ""

        except Exception as e:
            logger.error(f"    AI extraction failed: {e}")
            return ""

    def _verify_description_matches_image(self, image_path: str, description: str) -> bool:
        """No longer used - descriptions come from <p> tags during image capture"""
        return True

    def _extract_description_from_text(self, article_text: str, search_content: str) -> str:
        """No longer used - descriptions come from <p> tags during image capture"""
        return ""

    def _ai_select_best_paragraph(self, paragraphs: List[str], search_content: str) -> str:
        """No longer used - descriptions come from <p> tags during image capture"""
        return ""

    def _extract_keywords(self, content: str) -> List[str]:
        """No longer used - descriptions come from <p> tags during image capture"""
        return []

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
                    "telegram_compatible": img.get('telegram_compatible', False),
                    "content_type": img.get('content_type'),
                    "file_size": img.get('file_size'),
                    "extraction_timestamp": datetime.now().isoformat()
                }
                image_results_data["extracted_images"].append(image_detail)

            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(image_results_data, f, indent=2, ensure_ascii=False)

            logger.info(f" Saved {len(extracted_images)} image results to: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.warning(f"Failed to save image results: {e}")
            return ""