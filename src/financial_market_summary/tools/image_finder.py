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

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            if mentioned_stocks is None:
                mentioned_stocks = []

            # Extract stock symbols from content if not provided
            if not mentioned_stocks:
                mentioned_stocks = self._extract_stock_symbols(search_content)

            # Always extract URLs from search results file
            if not search_results_file:
                logger.error("❌ No search results file provided")
                return json.dumps([], indent=2)

            article_urls = self._extract_urls_from_search_results(search_results_file)

            # Extract images from article URLs
            if article_urls:
                logger.info(f"🔍 Extracting images from {len(article_urls)} article URLs from search results")
                extracted_images = self._extract_from_provided_urls(article_urls, search_content, mentioned_stocks, max_images)
            else:
                logger.warning("⚠️ No article URLs found in search results file")
                extracted_images = []

            # Verify Telegram compatibility
            extracted_images = self._verify_telegram_compatibility(extracted_images)

            # Analyze images for financial relevance using vision AI
            extracted_images = self._analyze_images_for_relevance(extracted_images, search_content)

            # Remove images with unknown sources
            extracted_images = [img for img in extracted_images if img.get('source', 'unknown') != 'unknown']

            # Filter out low-relevance images (score < 70)
            extracted_images = [img for img in extracted_images if img.get('vision_relevance_score', 0) >= 70]

            # Sort by vision relevance, telegram compatibility, and original relevance
            extracted_images.sort(key=lambda x: (
                x.get('telegram_compatible', False),
                x.get('vision_relevance_score', 0),
                x.get('relevance_score', 0)
            ), reverse=True)

            # Limit to requested number
            final_images = extracted_images[:max_images]

            logger.info(f"📊 Returning {len(final_images)} images from article extraction")

            # Save image results to output directory
            if extracted_images:
                self._save_image_results(extracted_images, search_content)

            # Return as JSON
            return json.dumps(final_images, indent=2)

        except Exception as e:
            logger.error(f"Image finder failed: {e}")
            # Return empty result if extraction fails
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

    def _extract_from_provided_urls(self, article_urls: List[str], content: str, stocks: List[str], max_images: int) -> List[Dict[str, Any]]:
        """Extract images from provided article URLs"""
        extracted_images = []

        for url in article_urls:  # Process all URLs from search results
            try:
                logger.info(f"🔍 Extracting images from: {url[:50]}...")
                article_images = self._extract_images_from_article(url)

                # Process and score the extracted images
                for img in article_images[:3]:  # Take top 3 from each article
                    source = urlparse(url).netloc

                    # Skip images with unknown sources
                    if source == 'unknown' or not source:
                        continue

                    processed_img = {
                        'url': img['url'],
                        'title': img.get('alt', 'Financial Chart from Article'),
                        'source': source,
                        'type': 'extracted_from_article',
                        'relevance_score': 100,
                        'telegram_compatible': False,  # Will be verified later
                        'file_type': self._get_file_extension(img['url']),
                        'trusted_source': self._is_trusted_financial_source(source),
                        'extraction_method': img['type'],
                        'source_article': url
                    }
                    extracted_images.append(processed_img)

                    # Stop if we have enough images
                    if len(extracted_images) >= max_images:
                        break

                if len(extracted_images) >= max_images:
                    break

            except Exception as e:
                logger.warning(f"Failed to extract images from {url}: {e}")
                continue

        logger.info(f"📸 Extracted {len(extracted_images)} images from provided URLs")
        return extracted_images

    def _determine_image_type(self, title: str, content: str) -> str:
        """Determine the type of financial image"""
        title_lower = title.lower()
        content_lower = content.lower()

        if any(term in title_lower for term in ['chart', 'graph', 'price']):
            return 'stock_chart'
        elif any(term in content_lower for term in ['fed', 'federal reserve', 'interest rate']):
            return 'economic_data'
        elif 'earnings' in content_lower:
            return 'earnings_chart'
        elif any(term in content_lower for term in ['nasdaq', 's&p', 'dow']):
            return 'market_index'
        else:
            return 'financial_chart'

    def _get_file_extension(self, url: str) -> str:
        """Get file extension from URL"""
        url_lower = url.lower()
        if '.png' in url_lower:
            return 'png'
        elif '.jpg' in url_lower or '.jpeg' in url_lower:
            return 'jpg'
        elif '.gif' in url_lower:
            return 'gif'
        elif '.webp' in url_lower:
            return 'webp'
        elif '.svg' in url_lower:
            return 'svg'
        else:
            return 'unknown'

    def _is_trusted_financial_source(self, source: str) -> bool:
        """Check if source is a trusted financial website"""
        trusted_sources = [
            'yahoo.com', 'finance.yahoo.com', 'marketwatch.com', 'bloomberg.com',
            'cnbc.com', 'reuters.com', 'wsj.com', 'ft.com', 'investing.com',
            'tradingview.com', 'benzinga.com', 'seekingalpha.com', 'morningstar.com',
            'fool.com', 'finviz.com', 'nasdaq.com', 'nyse.com'
        ]

        source_lower = source.lower()
        return any(trusted in source_lower for trusted in trusted_sources)


    def _extract_images_from_article(self, article_url: str) -> List[Dict[str, Any]]:
        """Extract all image URLs from an article page"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            response = requests.get(article_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            image_urls = []

            # Method 1: Find all img tags
            images = soup.find_all('img')
            for img in images:
                img_url = (img.get('src') or
                          img.get('data-src') or
                          img.get('data-lazy-src') or
                          img.get('data-original') or
                          img.get('data-srcset'))

                if img_url:
                    # Handle srcset (multiple URLs)
                    if 'srcset' in str(img_url):
                        img_url = img_url.split(',')[0].split()[0]

                    full_url = urljoin(article_url, img_url)
                    alt_text = img.get('alt', '')

                    image_urls.append({
                        'url': full_url,
                        'alt': alt_text,
                        'type': 'img_tag'
                    })

            # Method 2: Find images in meta tags (og:image, twitter:image)
            meta_images = soup.find_all('meta', property=re.compile(r'(og:image|twitter:image)'))
            for meta in meta_images:
                img_url = meta.get('content')
                if img_url:
                    full_url = urljoin(article_url, img_url)
                    image_urls.append({
                        'url': full_url,
                        'alt': 'Meta tag image',
                        'type': 'meta_tag'
                    })

            # Method 3: Find images in picture tags
            pictures = soup.find_all('picture')
            for picture in pictures:
                source = picture.find('source')
                if source:
                    img_url = source.get('srcset') or source.get('src')
                    if img_url:
                        if 'srcset' in str(img_url):
                            img_url = img_url.split(',')[0].split()[0]
                        full_url = urljoin(article_url, img_url)
                        image_urls.append({
                            'url': full_url,
                            'alt': 'Picture source',
                            'type': 'picture_tag'
                        })

            # Method 4: Find images in inline styles and backgrounds
            elements_with_style = soup.find_all(style=re.compile(r'background.*url'))
            for element in elements_with_style:
                style = element.get('style', '')
                urls = re.findall(r'url\(["\']?(.*?)["\']?\)', style)
                for url in urls:
                    full_url = urljoin(article_url, url)
                    image_urls.append({
                        'url': full_url,
                        'alt': 'Background image',
                        'type': 'css_background'
                    })

            # Remove duplicates
            seen = set()
            unique_images = []
            for img in image_urls:
                if img['url'] not in seen:
                    seen.add(img['url'])
                    unique_images.append(img)

            # Filter out tiny images (likely icons/logos)
            filtered_images = [
                img for img in unique_images
                if not any(skip in img['url'].lower() for skip in ['icon', 'logo', 'pixel', 'spacer', '1x1'])
                and img['url'].startswith('http')
                and self._is_likely_financial_image(img)
            ]

            logger.info(f"📸 Extracted {len(filtered_images)} relevant images from article")
            return filtered_images

        except Exception as e:
            logger.warning(f"Error extracting images from article: {e}")
            return []

    def _is_likely_financial_image(self, img: Dict[str, Any]) -> bool:
        """Check if image is likely financial/chart related"""
        url_lower = img['url'].lower()
        alt_lower = img.get('alt', '').lower()

        # Financial image indicators
        financial_indicators = [
            'chart', 'graph', 'stock', 'market', 'trading', 'financial',
            'price', 'data', 'analytics', 'report', 'earnings', 'revenue'
        ]

        # Check URL and alt text for financial indicators
        return any(indicator in url_lower or indicator in alt_lower for indicator in financial_indicators)

    def _verify_telegram_compatibility(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify that images are compatible with Telegram"""
        compatible_images = []

        for image in images:
            url = image.get('url', '')
            if not url:
                continue

            # Check if URL looks like a direct image file
            if self._is_direct_image_url(url):
                verification_result = self._test_telegram_compatibility(url)
                image.update(verification_result)
                compatible_images.append(image)
            else:
                # Mark as incompatible but keep for reference
                image['telegram_compatible'] = False
                image['verification_status'] = 'not_direct_image'
                compatible_images.append(image)

        return compatible_images

    def _is_direct_image_url(self, url: str) -> bool:
        """Check if URL points to a direct image file"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']
        url_lower = url.lower()

        # Check for direct image extensions
        if any(url_lower.endswith(ext) for ext in image_extensions):
            return True

        # Check for image-like patterns
        image_patterns = [
            r'\.png(\?|$)',
            r'\.jpe?g(\?|$)',
            r'\.gif(\?|$)',
            r'\.webp(\?|$)',
            r'\.svg(\?|$)',
            r'/images?/',
            r'/charts?/',
            r'chart.*\.(png|jpg)',
            r'snapshot.*\.(png|jpg)'
        ]

        return any(re.search(pattern, url_lower) for pattern in image_patterns)

    def _test_telegram_compatibility(self, url: str) -> Dict[str, Any]:
        """Test if image URL works with Telegram"""
        verification_data = {
            'telegram_compatible': False,
            'verification_status': 'not_tested',
            'content_type': None,
            'file_size': None,
            'response_time': None
        }

        try:
            headers = {
                'User-Agent': 'TelegramBot (like TwitterBot)',
                'Accept': 'image/*,image/svg+xml,*/*;q=0.8'
            }

            start_time = time.time()
            response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            response_time = time.time() - start_time

            verification_data['response_time'] = round(response_time, 2)

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                content_length = response.headers.get('content-length')

                verification_data['content_type'] = content_type
                verification_data['file_size'] = content_length

                # Check if it's actually an image
                if any(img_type in content_type for img_type in ['image/', 'png', 'jpeg', 'jpg', 'gif', 'svg']):
                    # Check file size (Telegram limit is 10MB for photos)
                    if content_length and int(content_length) < 10 * 1024 * 1024:  # 10MB
                        verification_data['telegram_compatible'] = True
                        verification_data['verification_status'] = 'compatible'
                        logger.info(f"✅ Telegram-compatible image: {url}")
                    else:
                        verification_data['verification_status'] = 'too_large'
                        logger.warning(f"⚠️ Image too large for Telegram: {url}")
                else:
                    verification_data['verification_status'] = 'not_image'
                    logger.warning(f"❌ Not an image file: {url}")
            else:
                verification_data['verification_status'] = f'http_{response.status_code}'
                logger.warning(f"❌ HTTP {response.status_code}: {url}")

        except requests.exceptions.Timeout:
            verification_data['verification_status'] = 'timeout'
            logger.warning(f"⏱️ Timeout testing: {url}")
        except requests.exceptions.ConnectionError:
            verification_data['verification_status'] = 'connection_error'
            logger.warning(f"🔌 Connection error: {url}")
        except Exception as e:
            verification_data['verification_status'] = f'error_{str(e)[:20]}'
            logger.warning(f"❌ Test error: {url} - {e}")

        return verification_data

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
                    "telegram_compatible_count": len([img for img in extracted_images if img.get('telegram_compatible', False)]),
                    "extraction_date": datetime.now().isoformat()
                },
                "extraction_summary": {
                    "sources_scraped": list(set([img.get('source', 'unknown') for img in extracted_images])),
                    "extraction_methods": list(set([img.get('extraction_method', 'unknown') for img in extracted_images])),
                    "file_types": list(set([img.get('file_type', 'unknown') for img in extracted_images])),
                    "trusted_sources_count": len([img for img in extracted_images if img.get('trusted_source', False)])
                },
                "extracted_images": []
            }

            # Process each image with detailed information
            for idx, img in enumerate(extracted_images, 1):
                image_detail = {
                    "image_number": idx,
                    "url": img.get('url', ''),
                    "title": img.get('title', 'Unknown Title'),
                    "alt_text": img.get('alt_text', ''),
                    "source": img.get('source', 'Unknown Source'),
                    "source_article": img.get('source_article', ''),
                    "extraction_method": img.get('extraction_method', 'unknown'),
                    "file_type": img.get('file_type', 'unknown'),
                    "relevance_score": img.get('relevance_score', 0),
                    "telegram_compatible": img.get('telegram_compatible', False),
                    "trusted_source": img.get('trusted_source', False),
                    "verification_status": img.get('verification_status', 'not_verified'),
                    "content_type": img.get('content_type'),
                    "file_size": img.get('file_size'),
                    "response_time": img.get('response_time'),
                    "vision_relevance_score": img.get('vision_relevance_score', 0),
                    "is_financial_content": img.get('is_financial_content', False),
                    "vision_content_type": img.get('vision_content_type', 'unknown'),
                    "vision_analysis_summary": img.get('vision_analysis_summary', ''),
                    "vision_confidence": img.get('vision_confidence', 0),
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

    def _analyze_images_for_relevance(self, extracted_images: List[Dict[str, Any]], search_content: str) -> List[Dict[str, Any]]:
        """Analyze images using Gemini Vision to determine financial relevance and filter out generic content"""
        logger.info(f"🔍 Analyzing {len(extracted_images)} images for financial relevance using Gemini Vision")

        analyzed_images = []

        for img in extracted_images:
            try:
                image_url = img.get('url', '')
                if not image_url:
                    continue

                # Analyze the image using Gemini Vision
                vision_analysis = self._analyze_single_image(image_url, search_content)

                # Add vision analysis data to image
                img.update({
                    'vision_relevance_score': vision_analysis.get('relevance_score', 0),
                    'is_financial_content': vision_analysis.get('is_financial', False),
                    'vision_content_type': vision_analysis.get('content_type', 'unknown'),
                    'vision_analysis_summary': vision_analysis.get('summary', ''),
                    'vision_confidence': vision_analysis.get('confidence', 0)
                })

                analyzed_images.append(img)

                # Add a small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to analyze image {image_url}: {e}")
                # Keep image with default low score if analysis fails
                img.update({
                    'vision_relevance_score': 30,
                    'is_financial_content': False,
                    'vision_content_type': 'analysis_failed',
                    'vision_analysis_summary': f'Analysis failed: {str(e)}',
                    'vision_confidence': 0
                })
                analyzed_images.append(img)
                continue

        logger.info(f"🎯 Vision analysis completed for {len(analyzed_images)} images")
        return analyzed_images

    def _analyze_single_image(self, image_url: str, context: str) -> Dict[str, Any]:
        """Analyze a single image using Gemini Vision to determine financial relevance"""
        try:
            from crewai import Agent, Task, Crew, Process
            from ..agents import FinancialAgents

            # Create image analysis agent
            agents_factory = FinancialAgents()
            analysis_agent = agents_factory.image_analysis_agent()

            # Create analysis task
            analysis_prompt = f"""
            Analyze this financial image URL: {image_url}

            Financial context: {context[:300]}

            Determine the following:
            1. Is this a relevant financial chart, graph, or data visualization? (Yes/No)
            2. Relevance score (0-100) for financial content
            3. Content type classification
            4. Brief summary of what the image shows

            SCORING GUIDELINES:
            - HIGH SCORE (80-100): Stock charts, market indices, earnings graphs, financial data visualizations, trading charts
            - MEDIUM SCORE (40-79): Business-related content but not pure financial data (company logos in context, financial news photos)
            - LOW SCORE (0-39): Generic images, ads, author photos, banners, unrelated content, website logos

            CONTENT TYPES:
            - stock_chart: Individual stock price charts
            - market_index: Market index charts (S&P 500, NASDAQ, etc.)
            - earnings_chart: Earnings or financial performance charts
            - trading_chart: Technical analysis or trading charts
            - financial_data: Other financial data visualizations
            - business_photo: Business-related but not financial data
            - logo: Company or website logos
            - generic_photo: Author photos, generic images
            - advertisement: Ad banners or promotional images
            - unknown: Cannot determine content

            Return ONLY a JSON response with this exact format:
            {{
                "relevance_score": 85,
                "is_financial": true,
                "content_type": "stock_chart",
                "summary": "Apple Inc stock price chart showing recent performance",
                "confidence": 90
            }}
            """

            analysis_task = Task(
                description=analysis_prompt,
                expected_output="JSON with image analysis results including relevance score, content type, and summary",
                agent=analysis_agent
            )

            crew = Crew(
                agents=[analysis_agent],
                tasks=[analysis_task],
                process=Process.sequential,
                verbose=False
            )

            result = crew.kickoff()

            # Parse the JSON response
            import json
            import re

            # Extract JSON from the result
            json_match = re.search(r'\{.*\}', str(result), re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())

                # Validate required fields
                required_fields = ['relevance_score', 'is_financial', 'content_type']
                if all(field in analysis_data for field in required_fields):
                    return analysis_data
                else:
                    logger.warning(f"Incomplete analysis data: {analysis_data}")
                    return self._get_default_analysis()
            else:
                logger.warning(f"No JSON found in analysis result: {result}")
                return self._get_default_analysis()

        except Exception as e:
            logger.warning(f"Image analysis failed for {image_url}: {e}")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when vision analysis fails"""
        return {
            'relevance_score': 30,
            'is_financial': False,
            'content_type': 'analysis_failed',
            'summary': 'Vision analysis failed',
            'confidence': 0
        }