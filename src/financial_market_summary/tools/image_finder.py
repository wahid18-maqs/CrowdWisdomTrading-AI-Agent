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
            logger.info("="*60)
            logger.info(" IMAGE FINDER STARTED")
            logger.info("="*60)

            logger.info(f" Max images to find: {max_images}")
            logger.info(f" Search results file: {search_results_file}")

            # Always extract URLs from search results file
            if not search_results_file:
                logger.error(" No search results file provided")
                return json.dumps([], indent=2)

            article_urls = self._extract_urls_from_search_results(search_results_file)
            logger.info(f" Extracted {len(article_urls)} URLs from search results")

            if not article_urls:
                logger.warning(" No article URLs found in search results")
                return json.dumps([], indent=2)

            # Extract screenshots from article URLs
            logger.info(f" Starting screenshot extraction from {len(article_urls)} URLs...")
            extracted_images = self._extract_screenshots_from_urls(article_urls, search_content, max_images)
            logger.info(f" Screenshot extraction complete: {len(extracted_images)} images captured")

            if not extracted_images:
                logger.warning(" No screenshots were captured from any URLs")
                return json.dumps([], indent=2)

            # Extract descriptions from article URLs (web scraping, NO AI)
            logger.info(f" Extracting descriptions from article URLs for {len(extracted_images)} images...")
            extracted_images = self._generate_ai_descriptions(extracted_images, search_content)
            logger.info(f" Descriptions extracted from URLs")

            # Save and return
            logger.info(f" Saving {len(extracted_images)} image results...")
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

        logger.info(f" Starting image extraction:")
        logger.info(f"   - Total URLs available: {len(article_urls)}")
        logger.info(f"   - Target images: {max_images}")
        logger.info(f"   - Strategy: <img> tags first, screenshots as fallback")

        for idx, url in enumerate(article_urls, 1):
            if len(extracted_images) >= max_images:
                logger.info(f" Target images ({max_images}) reached, stopping")
                break

            try:
                logger.info(f" [{idx}/{len(article_urls)}] Processing URL: {url[:80]}...")

                # STEP 1: Try to extract image from <img> tags first
                logger.info(f"   Step 1: Attempting to extract <img> tag from URL...")
                img_tag_data = self._extract_image_from_img_tag(url)

                if img_tag_data:
                    extracted_images.append(img_tag_data)
                    logger.info(f" Image extracted from <img> tag successfully!")
                    logger.info(f"   Image URL: {img_tag_data.get('url', 'unknown')[:80]}...")
                    continue

                # STEP 2: Fallback to screenshot capture if no <img> tag found
                logger.info(f"   Step 2: No suitable <img> tag found, falling back to screenshot...")
                screenshot_data = self._capture_chart_screenshot(url)

                if screenshot_data:
                    extracted_images.append(screenshot_data)
                    logger.info(f" Screenshot captured successfully!")
                    logger.info(f"   Saved to: {screenshot_data.get('url', 'unknown')}")
                else:
                    logger.warning(f" Both <img> tag and screenshot extraction failed for {url[:80]}...")

            except Exception as e:
                logger.error(f" Exception during image extraction from {url[:80]}...: {e}", exc_info=True)
                continue

        logger.info(f" Image extraction summary:")
        logger.info(f"   - URLs processed: {idx}")
        logger.info(f"   - Images extracted: {len(extracted_images)}")

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
                logger.info(f" Loading page: {article_url}")
                try:
                    # Try with domcontentloaded first (faster)
                    page.goto(article_url, wait_until='domcontentloaded', timeout=20000)
                    logger.info(" Page loaded (domcontentloaded)")
                    page.wait_for_timeout(3000)
                except Exception as e:
                    logger.warning(f" domcontentloaded failed, trying load strategy: {e}")
                    try:
                        # Fallback to basic load
                        page.goto(article_url, wait_until='load', timeout=25000)
                        logger.info(" Page loaded (load)")
                        page.wait_for_timeout(2000)
                    except Exception as e2:
                        logger.error(f" Both load strategies failed: {e2}")
                        context.close()
                        browser.close()
                        return None

                # Scroll to load dynamic content and wait for charts to render
                try:
                    # Scroll multiple times to trigger lazy loading
                    logger.info(f" Scrolling page to trigger chart loading...")
                    page.evaluate("window.scrollTo(0, 800)")
                    page.wait_for_timeout(5000)  # Increased to 5s
                    page.evaluate("window.scrollTo(0, 1600)")
                    page.wait_for_timeout(5000)  # Increased to 5s
                    page.evaluate("window.scrollTo(0, 2400)")
                    page.wait_for_timeout(5000)  # Increased to 5s
                    logger.info(f" Scrolled page to load dynamic content")
                except Exception as e:
                    logger.warning(f" Scroll failed: {e}")

                # Hide all overlays, modals, and popups with CSS injection
                try:
                    logger.info(f" Injecting CSS to hide popups and overlays...")
                    page.add_style_tag(content="""
                        /* Hide dialogs and modals - EXCLUDE chart-related elements */
                        [role="dialog"]:not([class*="chart"]):not([id*="chart"]),
                        [class*="dialog"]:not([class*="chart"]):not([id*="chart"]),
                        [class*="Dialog"]:not([class*="chart"]):not([id*="chart"]),
                        [id*="dialog"]:not([class*="chart"]),

                        /* Hide modal backdrops and overlays that are children of body */
                        body > [class*="backdrop"],
                        body > [class*="Backdrop"],
                        body > [class*="overlay"]:not([class*="chart"]):not([id*="chart"]),
                        body > [class*="Overlay"]:not([class*="chart"]):not([id*="chart"]),

                        /* Hide specific high z-index overlays */
                        div[style*="z-index: 9999"],
                        div[style*="z-index: 999999"],
                        div[style*="z-index: 10000"],

                        /* Hide lightboxes */
                        [class*="lightbox"],
                        [class*="Lightbox"] {
                            display: none !important;
                            visibility: hidden !important;
                            opacity: 0 !important;
                            pointer-events: none !important;
                        }

                        /* Ensure body is scrollable (some modals disable scroll) */
                        body {
                            overflow: visible !important;
                        }
                    """)
                    logger.info(f" CSS injection complete - popups should be hidden")
                except Exception as e:
                    logger.warning(f" CSS injection failed (non-critical): {e}")

                # Wait significantly longer for charts to fully render (live market data)
                logger.info(f"⏳ Waiting 15 seconds for live market charts to fully render...")
                page.wait_for_timeout(15000)  # Increased from 8s to 15s for live data
                logger.info(f" Chart rendering wait complete")

                screenshot_data = None

                # FIRST: Try to find and screenshot <img> tags (skip first one)
                logger.info(f" STEP 1: Looking for <img> tags first...")
                try:
                    img_elements = page.query_selector_all('img')
                    logger.info(f"   Found {len(img_elements)} <img> elements")

                    large_images_found = 0
                    for idx, img_elem in enumerate(img_elements):
                        box = img_elem.bounding_box()
                        if box and box['width'] >= 400 and box['height'] >= 250:
                            large_images_found += 1

                            # Skip first large image
                            if large_images_found == 1:
                                logger.info(f"   Skipping first large image (usually header)")
                                continue

                            # Take second large image
                            logger.info(f"   Found second large image ({box['width']}x{box['height']}), capturing...")
                            try:
                                # Try to screenshot parent container to capture captions/logos
                                screenshot_elem = img_elem

                                # Check if parent is figure, picture, or container div
                                parent = img_elem.evaluate('el => el.parentElement')
                                if parent:
                                    parent_elem = img_elem.evaluate_handle('el => el.parentElement')
                                    parent_tag = parent_elem.evaluate('el => el.tagName.toLowerCase()')

                                    # Use parent if it's a semantic container
                                    if parent_tag in ['figure', 'picture', 'article']:
                                        screenshot_elem = parent_elem
                                        logger.info(f"   Capturing parent <{parent_tag}> container (includes captions/logos)")
                                    else:
                                        logger.info(f"   Capturing <img> element directly")

                                screenshot_elem.scroll_into_view_if_needed(timeout=5000)
                                page.wait_for_timeout(2000)
                                screenshot_bytes = screenshot_elem.screenshot(timeout=10000)

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

                                logger.info(f" ✅ Image captured successfully from <img> tag!")

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
                                    'image_description': '',
                                    'content_type': 'image/png',
                                    'file_size': str(len(screenshot_bytes)),
                                }

                                context.close()
                                browser.close()
                                return screenshot_data

                            except Exception as img_error:
                                logger.warning(f"   Failed to capture <img>: {img_error}")
                                continue

                    logger.info(f" No suitable <img> tags found, falling back to chart selectors...")

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
        """Extract descriptions using Crawl4AI + AI selection"""
        try:
            logger.info(f" Extracting descriptions using Crawl4AI for {len(extracted_images)} images")

            for img in extracted_images:
                try:
                    source_article = img.get('source_article', '')
                    if not source_article:
                        logger.error(f" No source article URL - skipping")
                        img['image_description'] = ''
                        continue

                    logger.info(f" Crawling article: {source_article[:80]}...")

                    # Use Crawl4AI to extract paragraphs from <p> tags
                    paragraphs = self._crawl_article_with_crawl4ai(source_article)

                    if not paragraphs:
                        logger.error(f" Crawl4AI could not extract paragraphs")
                        img['image_description'] = ''
                        continue

                    logger.info(f" Crawl4AI extracted {len(paragraphs)} paragraphs")

                    # Use AI to select which paragraph describes the chart
                    image_path = img.get('url', '')
                    if image_path and os.path.exists(image_path):
                        logger.info(f"    Using AI to select chart description from <p> tags...")
                        description = self._ai_extract_chart_description(image_path, paragraphs)

                        if description:
                            img['image_description'] = description
                            logger.info(f" AI extracted description: {description[:100]}...")
                        else:
                            logger.error(f" AI could not extract chart description")
                            img['image_description'] = ''
                    else:
                        logger.error(f" Image file not found")
                        img['image_description'] = ''

                    # Small delay to avoid rate limits
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f" Description extraction failed: {e}")
                    img['image_description'] = ''
                    continue

            return extracted_images

        except Exception as e:
            logger.error(f" Description extraction system failed: {e}")
            return extracted_images

    def _crawl_article_with_crawl4ai(self, article_url: str) -> List[str]:
        """
        Use Crawl4AI to extract article HTML, then parse <p> tags with BeautifulSoup.
        Returns list of paragraph texts from the article.
        """
        try:
            import asyncio
            from crawl4ai import AsyncWebCrawler

            logger.info(f"    Crawl4AI crawling URL...")

            # Define async function to crawl
            async def crawl():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(url=article_url)
                    return result

            # Run async function
            result = asyncio.run(crawl())

            if result.success:
                # Get HTML content (prefer cleaned_html, fallback to html)
                html_content = ""

                if hasattr(result, 'cleaned_html') and result.cleaned_html:
                    html_content = result.cleaned_html
                    logger.info(f"    Using cleaned_html")
                elif hasattr(result, 'html') and result.html:
                    html_content = result.html
                    logger.info(f"    Using raw html")
                else:
                    logger.error(f"    No HTML content available")
                    return []

                logger.info(f"    Crawl4AI success: {len(html_content)} chars")

                # Parse HTML with BeautifulSoup to extract <p> tags
                soup = BeautifulSoup(html_content, 'html.parser')

                # Extract all <p> tags
                paragraphs = soup.find_all('p')
                logger.info(f"    Found {len(paragraphs)} <p> tags")

                # Extract text from <p> tags and filter
                paragraph_texts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)

                    # Filter out short/irrelevant paragraphs
                    if len(text) < 80:  # Skip short paragraphs
                        continue

                    if text.count(' ') < 10:  # Skip if not enough words
                        continue

                    # Must contain financial keywords (relaxed filter)
                    text_lower = text.lower()
                    financial_keywords = [
                        'stock', 'market', 'index', 'dow', 'nasdaq', 's&p', 'djia',
                        'rose', 'fell', 'gained', 'declined', 'points', 'percent',
                        'shares', 'trading', 'investors', 'wall street', 'treasury',
                        'futures', 'bond', 'yield', 'rally', 'price', 'earnings',
                        'revenue', 'profit', 'loss', 'up', 'down', 'higher', 'lower',
                        'popped', 'jumped', 'dropped', 'year', 'quarter', 'friday'
                    ]

                    if any(keyword in text_lower for keyword in financial_keywords):
                        paragraph_texts.append(text)

                logger.info(f"    Extracted {len(paragraph_texts)} relevant paragraphs")

                if paragraph_texts:
                    logger.info(f"    First paragraph preview: {paragraph_texts[0][:150]}...")

                return paragraph_texts

            else:
                error_msg = result.error_message if hasattr(result, 'error_message') else 'Unknown error'
                logger.error(f"    Crawl4AI failed: {error_msg}")
                return []

        except Exception as e:
            logger.error(f"    Crawl4AI exception: {e}")
            return []

    def _ai_extract_chart_description(self, image_path: str, paragraphs: List[str]) -> str:
        """
        Use AI to look at the chart image and article paragraphs (from <p> tags),
        then select which paragraph (or 1-2 sentences from it) describes this chart.

        CRITICAL: AI ONLY SELECTS existing text from paragraphs - NO AI-generated content!
        Returns ACTUAL text from the article, not AI-written descriptions.
        """
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            # Check if we have paragraphs
            if not paragraphs or len(paragraphs) == 0:
                logger.error("    No paragraphs provided")
                return ""

            logger.info(f"    Analyzing {len(paragraphs)} paragraphs from <p> tags...")

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

            # Prepare paragraphs for AI - number them
            # Use all paragraphs (Gemini 2.0 can handle more context)
            paragraphs_to_analyze = paragraphs  # Use ALL paragraphs for better matching
            numbered_paragraphs = []
            for i, para in enumerate(paragraphs_to_analyze, 1):
                # Truncate if too long
                para_text = para[:300] + "..." if len(para) > 300 else para
                numbered_paragraphs.append(f"{i}. {para_text}")

            paragraphs_text = "\n\n".join(numbered_paragraphs)

            logger.info(f"    First paragraph: {paragraphs_to_analyze[0][:150]}...")

            # STEP 1: First, analyze the chart to understand what it shows
            logger.info(f"    Step 1: Analyzing chart image to identify content...")
            chart_analysis_prompt = """Analyze this financial chart image and describe what it shows.

WHAT TO IDENTIFY:
1. Which market indices or stocks are shown? (S&P 500, Dow Jones, Nasdaq, Russell 2000, specific stocks, etc.)
2. What is the chart type? (line chart, candlestick, bar chart, etc.)
3. What time period? (intraday, daily, weekly, monthly, yearly)
4. What is the trend? (up, down, sideways, volatile)
5. Are there any specific price levels or percentages visible?

Provide a brief 2-3 sentence analysis of what this chart displays."""

            chart_analysis_response = model.generate_content(
                [chart_analysis_prompt, pil_image],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            chart_understanding = ""
            if chart_analysis_response and chart_analysis_response.text:
                chart_understanding = chart_analysis_response.text.strip()
                logger.info(f"    Chart analysis: {chart_understanding[:150]}...")
            else:
                logger.warning(f"    Could not analyze chart, proceeding without analysis")

            # STEP 2: Pre-filter paragraphs that might mention the stock/company
            # Extract stock/company names dynamically from chart analysis
            stock_keywords = []

            # Extract potential stock symbols (2-5 uppercase letters)
            import re
            potential_symbols = re.findall(r'\b[A-Z]{2,5}\b', chart_understanding)
            stock_keywords.extend(potential_symbols)

            # Extract company names (capitalized words, excluding common words)
            words = chart_understanding.split()
            common_words = {'The', 'Chart', 'Stock', 'Company', 'Index', 'Line', 'YTD', 'Here', 'Type'}
            for word in words:
                # Capitalized word that's not a common word and longer than 2 chars
                if word and word[0].isupper() and word not in common_words and len(word) > 2:
                    # Remove punctuation
                    clean_word = re.sub(r'[^\w]', '', word)
                    if clean_word and clean_word not in stock_keywords:
                        stock_keywords.append(clean_word)

            # Also check for known market indices
            index_keywords = ['S&P', 'Dow', 'Nasdaq', 'Russell', 'DJIA']
            for idx in index_keywords:
                if idx.lower() in chart_understanding.lower():
                    stock_keywords.append(idx)

            if stock_keywords:
                logger.info(f"    Pre-filtering for keywords: {stock_keywords}")
                # Find paragraphs that mention these keywords
                relevant_indices = []
                for i, para in enumerate(paragraphs_to_analyze):
                    if any(keyword.lower() in para.lower() for keyword in stock_keywords):
                        relevant_indices.append(i)

                if relevant_indices:
                    logger.info(f"    Found {len(relevant_indices)} paragraphs mentioning {stock_keywords}")
                    # Prioritize these paragraphs - put them first
                    prioritized_paragraphs = [paragraphs_to_analyze[i] for i in relevant_indices]
                    other_paragraphs = [p for i, p in enumerate(paragraphs_to_analyze) if i not in relevant_indices]
                    paragraphs_to_analyze = prioritized_paragraphs + other_paragraphs[:10]  # Keep some others too
                    logger.info(f"    Using {len(prioritized_paragraphs)} relevant + {min(10, len(other_paragraphs))} other paragraphs")

            # STEP 3: Now use the chart understanding to select matching paragraph
            logger.info(f"    Step 3: Using chart analysis to select matching paragraph...")
            extraction_prompt = f"""CHART SHOWS:
{chart_understanding}

ARTICLE PARAGRAPHS:
{paragraphs_text}

TASK:
Find and COPY the exact text from the paragraphs that describes what's in the chart.

INSTRUCTIONS:
1. Look for mentions of the stock/company/index shown in the chart
2. When you find it, COPY the exact 1-2 sentences that mention it
3. Return ONLY the copied text - nothing else
4. If you find nothing, return "NONE"

EXAMPLE:
If chart shows "Deere stock" and you find paragraph 7 says "Shares of Deere have popped 8% this year. The stock was last trading marginally higher on Friday"

Then return EXACTLY:
Shares of Deere have popped 8% this year. The stock was last trading marginally higher on Friday

DO NOT return "PARAGRAPH_7" or "SENTENCES_7:" - just return the actual text.

Your response (exact text from article):"""

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

                # Check if AI said NONE
                if ai_response.upper() == "NONE" or ai_response.upper().startswith("NONE"):
                    logger.warning(f"    AI could not find matching paragraph")
                    return ""

                # AI should have returned the actual text directly
                # Just clean it up and return it
                description = ai_response.strip()

                # Ensure it ends with proper punctuation
                if description and not description.endswith(('.', '!', '?')):
                    description += '.'

                logger.info(f"    AI extracted text: {description[:150]}...")
                return description

            else:
                logger.warning(f"    Empty AI response")
                return ""

        except Exception as e:
            logger.error(f"    AI extraction failed: {e}")
            return ""

    def _verify_description_matches_image(self, image_path: str, description: str) -> bool:
        """
        Use AI to verify that the description actually matches what's shown in the chart image.
        Returns True if description is related to the image, False otherwise.
        """
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                logger.warning("    GOOGLE_API_KEY not found - skipping verification")
                return True  # If no API key, accept the description

            # Configure Gemini
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config={
                    'temperature': 0.2,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 50,
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

            # Verification prompt
            verification_prompt = f"""Look at this financial chart image and this description:

DESCRIPTION: "{description}"

QUESTION: Does this description accurately describe what is shown in this chart image?

Consider:
- Does it mention the correct financial instrument/index shown in the chart?
- Does it describe data/movement that could be from this chart?
- Is it talking about the same subject as the chart?

Answer ONLY "YES" or "NO"."""

            response = model.generate_content(
                [verification_prompt, pil_image],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            if response and response.text:
                answer = response.text.strip().upper()
                logger.info(f"    AI verification: {answer}")
                return "YES" in answer
            else:
                logger.warning(f"    Empty AI response, accepting description")
                return True

        except Exception as e:
            logger.error(f"    AI verification failed: {e}")
            return True  # If verification fails, accept the description

    def _extract_description_from_text(self, article_text: str, search_content: str) -> str:
        """
        Extract relevant description from article text.
        Simply finds first paragraph with financial keywords and returns 1-2 sentences.
        """
        try:
            logger.info(f"    Searching for financial content...")

            # Split text into lines
            lines = article_text.split('\n')

            # Financial keywords to look for
            financial_keywords = [
                'stock', 'market', 'index', 'dow', 'nasdaq', 's&p', 'djia',
                'rose', 'fell', 'gained', 'declined', 'climbed', 'dropped',
                'surged', 'plunged', 'rallied', 'tumbled',
                'up', 'down', 'higher', 'lower',
                'shares', 'trading', 'points', 'percent', '%',
                'wall street', 'investors', 'treasury', 'yield'
            ]

            # Words that indicate headings/titles (to skip)
            heading_indicators = [
                'live updates', 'breaking', 'watch', 'read more', 'subscribe',
                'sign up', 'follow', 'share', 'click here', 'menu', 'search'
            ]

            # Find first substantial line with financial content
            for line in lines:
                line = line.strip()

                # Skip short lines or very long lines
                if len(line) < 80 or len(line) > 500:
                    continue

                # Skip navigation/menu items (usually short phrases)
                if line.count(' ') < 8:  # Increased from 5 to 8 to skip short headings
                    continue

                # Skip if it looks like a heading
                line_lower = line.lower()
                if any(indicator in line_lower for indicator in heading_indicators):
                    continue

                # Skip if ALL CAPS (likely a heading)
                if line.isupper():
                    continue

                # Skip if ends with colon (likely a heading)
                if line.endswith(':'):
                    continue

                # Must end with proper punctuation (sentences end with . ! ?)
                if not line.endswith(('.', '!', '?')):
                    continue

                # Check if contains financial keywords
                if any(keyword in line_lower for keyword in financial_keywords):
                    # Check if contains numbers (likely data)
                    if re.search(r'\d+', line):
                        logger.info(f"    Found financial content")

                        # Extract first 1-2 sentences
                        sentences = [s.strip() for s in re.split(r'[.!?]+', line) if s.strip()]
                        description = '. '.join(sentences[:2])

                        if not description.endswith('.'):
                            description += '.'

                        # Limit length
                        if len(description) > 300:
                            description = description[:297] + '...'

                        logger.info(f"    Description: {description[:100]}...")
                        return description

            logger.warning(f"    No financial content found with strict filters")
            logger.info(f"    Text preview: {article_text[:500]}...")
            return ""

        except Exception as e:
            logger.error(f"    Text extraction failed: {e}")
            return ""

    def _ai_select_best_paragraph(self, paragraphs: List[str], search_content: str) -> str:
        """
        Use AI to intelligently select the most relevant paragraph from the article.
        AI only selects, does not generate - returns actual paragraph text from website.
        """
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                logger.error("    GOOGLE_API_KEY not found - cannot use AI selection")
                return ""

            # Configure Gemini
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config={
                    'temperature': 0.3,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 100,
                }
            )

            # Limit to first 10 paragraphs to avoid token limits
            paragraphs_to_analyze = paragraphs[:10]

            # Create numbered list of paragraphs for AI
            numbered_paragraphs = []
            for i, para in enumerate(paragraphs_to_analyze, 1):
                # Truncate very long paragraphs
                para_text = para[:400] + "..." if len(para) > 400 else para
                numbered_paragraphs.append(f"{i}. {para_text}")

            paragraphs_text = "\n\n".join(numbered_paragraphs)

            # AI prompt - asks AI to select paragraph number, not generate text
            prompt = f"""You are analyzing a financial news article. Select the BEST paragraph that describes the main financial data or market movement.

Search Context: {search_content[:200]}

Available Paragraphs:
{paragraphs_text}

INSTRUCTIONS:
- Select the paragraph number (1-{len(paragraphs_to_analyze)}) that best describes financial market data, stock movements, or trading activity
- Look for paragraphs with specific numbers, prices, percentages, or market trends
- Prefer paragraphs that mention indices (S&P 500, Dow, Nasdaq) or specific stocks
- Return ONLY the number (e.g., "3"), nothing else

Your selection:"""

            response = model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            if response and response.text:
                # Extract paragraph number from AI response
                selection = response.text.strip()
                logger.info(f"    AI selected: {selection}")

                # Parse the number
                match = re.search(r'\d+', selection)
                if match:
                    selected_index = int(match.group(0)) - 1  # Convert to 0-indexed
                    if 0 <= selected_index < len(paragraphs_to_analyze):
                        selected_paragraph = paragraphs_to_analyze[selected_index]
                        logger.info(f"    Returning paragraph {selected_index + 1}")
                        return selected_paragraph
                    else:
                        logger.warning(f"    AI selected invalid index: {selected_index + 1}")
                        return ""
                else:
                    logger.warning(f"    Could not parse number from AI response: {selection}")
                    return ""
            else:
                logger.warning(f"  Empty AI response")
                return ""

        except Exception as e:
            logger.error(f" AI selection failed: {e}")
            return ""

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from search content"""
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                       'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}

        # Split content and extract meaningful words
        words = re.findall(r'\b[A-Za-z]{3,}\b', content)
        keywords = [w for w in words if w.lower() not in common_words]

        # Return unique keywords, limited to first 10
        return list(dict.fromkeys(keywords))[:10]

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
