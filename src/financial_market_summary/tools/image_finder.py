import json
import re
import requests
import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from datetime import datetime, timedelta
from urllib.parse import urlparse
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedImageFinderInput(BaseModel):
    search_content: str = Field(description="Financial content to find relevant images for")
    mentioned_stocks: Optional[List[str]] = Field(default=[], description="Stock symbols mentioned in content")
    max_images: Optional[int] = Field(default=3, description="Maximum number of images to find")

class EnhancedImageFinder(BaseTool):
    name: str = "enhanced_financial_image_finder"
    description: str = "Finds relevant financial images using Serper API based on summary content and mentioned stocks"
    args_schema: Type[BaseModel] = EnhancedImageFinderInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, search_content: str, mentioned_stocks: List[str] = None, max_images: int = 3) -> str:
        try:
            if mentioned_stocks is None:
                mentioned_stocks = []

            # Extract stock symbols from content if not provided
            if not mentioned_stocks:
                mentioned_stocks = self._extract_stock_symbols(search_content)

            # Find relevant images using Serper API
            verified_images = self._find_serper_images(search_content, mentioned_stocks, max_images)

            # Return as JSON
            return json.dumps(verified_images, indent=2)

        except Exception as e:
            logger.error(f"Serper image finder failed: {e}")
            # Fallback to static images if Serper fails
            fallback_images = self._get_fallback_images(search_content, mentioned_stocks, max_images)
            return json.dumps(fallback_images, indent=2)

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

    def _find_serper_images(self, content: str, stocks: List[str], max_images: int) -> List[Dict[str, Any]]:
        """Find relevant financial images using Serper API based on summary content"""
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            logger.warning("⚠️ No Serper API key available, using fallback images")
            return self._get_fallback_images(content, stocks, max_images)

        verified_images = []

        try:
            # Build search queries based on content and stocks
            search_queries = self._build_image_search_queries(content, stocks)

            for query in search_queries[:3]:  # Limit to 3 queries to avoid rate limits
                logger.info(f"🔍 Searching images with query: {query}")

                # Search for images using Serper API
                serper_images = self._search_serper_images(query, max_images, serper_api_key)

                if serper_images:
                    # Process and verify images
                    processed_images = self._process_serper_results(serper_images, content, stocks)
                    verified_images.extend(processed_images)

                    if len(verified_images) >= max_images:
                        break

            # Verify Telegram compatibility
            verified_images = self._verify_telegram_compatibility(verified_images)

            # Sort by relevance and compatibility
            verified_images.sort(key=lambda x: (
                x.get('telegram_compatible', False),
                x.get('relevance_score', 0)
            ), reverse=True)

            # Add fallback if no good images found - always include fallbacks for financial content
            if not any(img.get('telegram_compatible', False) for img in verified_images):
                logger.warning("⚠️ No Telegram-compatible images from Serper, adding fallbacks")
                fallback_images = self._get_fallback_images(content, stocks, 3)
                verified_images.extend(fallback_images)

            # Always ensure at least one compatible image - use reliable chart sources
            if not any(img.get('telegram_compatible', False) for img in verified_images):
                logger.info("📊 No compatible images found, forcing reliable chart inclusion")

                # Use most reliable financial chart sources
                reliable_charts = [
                    {
                        'url': 'https://finviz.com/chart.ashx?t=SPY&ty=c&ta=1&p=d&s=l',
                        'title': 'S&P 500 Chart',
                        'source': 'Finviz',
                        'type': 'market_chart',
                        'relevance_score': 90,
                        'telegram_compatible': True,
                        'file_type': 'png',
                        'trusted_source': True
                    },
                    {
                        'url': 'https://stockcharts.com/c-sc/sc?s=SPY&p=D&st=2023-01-01&en=(today)&i=t9999999999999&r=1',
                        'title': 'SPY Stock Chart',
                        'source': 'StockCharts',
                        'type': 'market_chart',
                        'relevance_score': 85,
                        'telegram_compatible': True,
                        'file_type': 'png',
                        'trusted_source': True
                    }
                ]

                # Add the first reliable chart
                verified_images.extend(reliable_charts[:1])
                logger.info(f"✅ Added reliable chart: {reliable_charts[0]['title']}")

            # If we have stocks mentioned, try to add a stock-specific chart
            if stocks and not any(img.get('stock_symbol') for img in verified_images):
                primary_stock = stocks[0]
                stock_chart = {
                    'url': f'https://finviz.com/chart.ashx?t={primary_stock}&ty=c&ta=1&p=d&s=l',
                    'title': f'{primary_stock} Stock Chart',
                    'source': 'Finviz',
                    'type': 'stock_chart',
                    'stock_symbol': primary_stock,
                    'relevance_score': 95,
                    'telegram_compatible': True,
                    'file_type': 'png',
                    'trusted_source': True
                }
                verified_images.append(stock_chart)
                logger.info(f"✅ Added stock-specific chart: {primary_stock}")

            # Prioritize Telegram-compatible images
            compatible_images = [img for img in verified_images if img.get('telegram_compatible', False)]
            non_compatible_images = [img for img in verified_images if not img.get('telegram_compatible', False)]

            # Return compatible images first, then non-compatible as backup
            final_images = compatible_images + non_compatible_images
            logger.info(f"📊 Returning {len(compatible_images)} compatible + {len(non_compatible_images)} non-compatible images")

            return final_images[:max_images]

        except Exception as e:
            logger.error(f"Error in Serper image search: {e}")
            return self._get_fallback_images(content, stocks, max_images)

    def _build_image_search_queries(self, content: str, stocks: List[str]) -> List[str]:
        """Build targeted search queries for financial images"""
        queries = []

        # Query 1: Stock-specific chart
        if stocks:
            primary_stock = stocks[0]
            queries.append(f"{primary_stock} stock chart price graph financial")

        # Query 2: Content-based financial search
        content_keywords = self._extract_content_keywords(content)
        if content_keywords:
            queries.append(f"{content_keywords} financial chart market graph")

        # Query 3: Market index based on content
        market_terms = ['federal reserve', 'fed', 'interest rate', 'inflation', 'gdp', 'jobs report']
        if any(term in content.lower() for term in market_terms):
            queries.append("federal reserve economic data chart graph")
        elif 'earnings' in content.lower():
            queries.append("corporate earnings chart financial results")
        else:
            queries.append("stock market chart financial graph today")

        return queries

    def _extract_content_keywords(self, content: str) -> str:
        """Extract key financial terms from content for search"""
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
            'federal reserve', 'fed', 'interest rate', 'inflation', 'gdp',
            'unemployment', 'jobs', 'manufacturing', 'retail', 'housing',
            'merger', 'acquisition', 'ipo', 'dividend', 'buyback',
            'nasdaq', 's&p 500', 'dow jones', 'russell', 'market'
        ]

        content_lower = content.lower()
        found_keywords = [kw for kw in financial_keywords if kw in content_lower]

        return ' '.join(found_keywords[:3])  # Top 3 relevant keywords

    def _search_serper_images(self, query: str, max_results: int = 10, api_key: str = None) -> List[Dict[str, Any]]:
        """Search for images using Serper API"""
        url = "https://google.serper.dev/images"

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }

        # Add time filter for recent images - expanded for financial content
        payload = {
            "q": query,
            "num": max_results,
            "gl": "us",
            "hl": "en",
            "tbm": "isch",
            "tbs": "qdr:m"  # Images from past month (more financial charts available)
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                images = data.get("images", [])
                logger.info(f"✅ Found {len(images)} images from Serper API")
                return images
            else:
                logger.error(f"❌ Serper API error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"❌ Error calling Serper API: {e}")
            return []

    def _process_serper_results(self, serper_images: List[Dict[str, Any]], content: str, stocks: List[str]) -> List[Dict[str, Any]]:
        """Process Serper API results into our format"""
        processed_images = []

        for i, img in enumerate(serper_images):
            try:
                # Extract image information
                image_url = img.get("imageUrl", "")
                title = img.get("title", "")
                source = img.get("source", "")

                if not image_url:
                    continue

                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(title, source, content, stocks)

                processed_image = {
                    'url': image_url,
                    'title': title or f"Financial Chart {i+1}",
                    'source': source or "Financial News",
                    'type': self._determine_image_type(title, content),
                    'relevance_score': relevance_score,
                    'telegram_compatible': False,  # Will be verified later
                    'file_type': self._get_file_extension(image_url),
                    'trusted_source': self._is_trusted_financial_source(source),
                    'from_serper': True,
                    'search_query': content[:50] + "..." if len(content) > 50 else content
                }

                # Add stock symbol if relevant
                for stock in stocks:
                    if stock.lower() in title.lower() or stock.upper() in title:
                        processed_image['stock_symbol'] = stock
                        break

                processed_images.append(processed_image)

            except Exception as e:
                logger.warning(f"Error processing Serper image result: {e}")
                continue

        return processed_images

    def _calculate_relevance_score(self, title: str, source: str, content: str, stocks: List[str]) -> int:
        """Calculate relevance score for an image"""
        score = 50  # Base score

        title_lower = title.lower()
        content_lower = content.lower()

        # Stock symbol relevance
        for stock in stocks:
            if stock.lower() in title_lower:
                score += 30
                break

        # Financial terms relevance
        financial_terms = ['chart', 'graph', 'stock', 'market', 'financial', 'trading', 'price']
        for term in financial_terms:
            if term in title_lower:
                score += 10

        # Content keyword matching
        content_words = set(content_lower.split())
        title_words = set(title_lower.split())
        common_words = content_words.intersection(title_words)
        score += min(len(common_words) * 5, 25)

        # Trusted source bonus
        if self._is_trusted_financial_source(source):
            score += 20

        # Recency bonus (assumed for images from Serper with time filter)
        score += 15

        return min(score, 100)

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

    def _get_fallback_images(self, content: str, stocks: List[str], max_images: int) -> List[Dict[str, Any]]:
        """Get fallback images when Serper API fails or returns no results"""
        fallback_images = []

        # Stock-specific TradingView charts
        for stock in stocks[:2]:
            fallback_images.extend(self._get_tradingview_snapshots(stock))

        # Market index charts
        fallback_images.extend(self._get_market_index_images(content))

        # Generic financial charts
        fallback_images.extend(self._get_alternative_chart_images(stocks))

        # Verify compatibility
        fallback_images = self._verify_telegram_compatibility(fallback_images)

        return fallback_images[:max_images]


    def _get_tradingview_snapshots(self, stock_symbol: str) -> List[Dict[str, Any]]:
        """Get TradingView chart snapshot images (actual PNG files)"""
        images = []

        # TradingView provides direct PNG snapshot URLs + more reliable alternatives
        tradingview_urls = [
            f"https://s3.tradingview.com/snapshots/u/{stock_symbol.lower()}.png",
            f"https://s3.tradingview.com/snapshots/s/{stock_symbol.upper()}.png",
            f"https://charts.tradingview.com/chart-images/{stock_symbol.lower()}_1D.png",
            # Add more reliable Yahoo Finance chart snapshots
            f"https://chart.yahoo.com/z?s={stock_symbol}&t=1d&q=l&l=on&z=s&p=s",
            f"https://chart.yahoo.com/z?s={stock_symbol}&t=5d&q=l&l=on&z=s&p=s"
        ]
        
        for i, url in enumerate(tradingview_urls):
            images.append({
                'url': url,
                'title': f'{stock_symbol} TradingView Chart',
                'source': 'TradingView',
                'type': 'stock_chart',
                'stock_symbol': stock_symbol,
                'relevance_score': 90 - (i * 10),
                'telegram_compatible': False,  # Will be verified
                'file_type': 'png',
                'trusted_source': True
            })
        
        return images

    def _get_financial_news_images(self, content: str, stocks: List[str]) -> List[Dict[str, Any]]:
        """Get chart images from financial news websites"""
        images = []
        
        # MarketWatch chart images (often PNG)
        if stocks:
            primary_stock = stocks[0]
            marketwatch_url = f"https://mw3.wsj.net/mw5/content/logos/mw_logo_social.png"
            
            images.append({
                'url': marketwatch_url,
                'title': f'MarketWatch Financial Chart',
                'source': 'MarketWatch',
                'type': 'financial_logo',
                'relevance_score': 70,
                'telegram_compatible': False,
                'file_type': 'png',
                'trusted_source': True
            })
        
        # Bloomberg chart images
        bloomberg_chart = "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iayWjE0qjPxU/v1/1200x-1.png"
        images.append({
            'url': bloomberg_chart,
            'title': 'Bloomberg Market Chart',
            'source': 'Bloomberg',
            'type': 'market_chart',
            'relevance_score': 80,
            'telegram_compatible': False,
            'file_type': 'png',
            'trusted_source': True
        })
        
        return images

    def _get_market_index_images(self, content: str) -> List[Dict[str, Any]]:
        """Get market index chart images as PNG files"""
        images = []
        
        # Determine relevant indices
        content_lower = content.lower()
        
        # Static market charts that are usually available as images
        index_mappings = [
            ('nasdaq', 'NASDAQ Chart', 85),
            ('s&p', 'S&P 500 Chart', 90),
            ('dow', 'Dow Jones Chart', 80)
        ]
        
        for keyword, title, score in index_mappings:
            if keyword in content_lower:
                # Use placeholder financial chart images that are typically available
                chart_url = f"https://cdn.corporatefinanceinstitute.com/assets/market-chart-{keyword}.png"
                images.append({
                    'url': chart_url,
                    'title': title,
                    'source': 'Financial Charts',
                    'type': 'market_index',
                    'relevance_score': score,
                    'telegram_compatible': False,
                    'file_type': 'png',
                    'trusted_source': True
                })
        
        # Generic market chart as fallback
        if not images:
            generic_chart = "https://media.istockphoto.com/id/1347652268/vector/stock-market-chart-graph-investment-trading.jpg"
            images.append({
                'url': generic_chart,
                'title': 'Stock Market Chart',
                'source': 'Stock Charts',
                'type': 'generic_market',
                'relevance_score': 60,
                'telegram_compatible': False,
                'file_type': 'jpg',
                'trusted_source': False
            })
        
        return images

    def _get_alternative_chart_images(self, stocks: List[str]) -> List[Dict[str, Any]]:
        """Get chart images from alternative sources"""
        images = []
        
        # Alpha Vantage chart images (if API available)
        if stocks:
            stock = stocks[0]
            alpha_url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={stock}&market=USD&apikey=demo&datatype=png"
            
            images.append({
                'url': alpha_url,
                'title': f'{stock} Alpha Vantage Chart',
                'source': 'Alpha Vantage',
                'type': 'stock_chart',
                'stock_symbol': stock,
                'relevance_score': 75,
                'telegram_compatible': False,
                'file_type': 'png',
                'trusted_source': True
            })
        
        # Quandl/World Bank chart images
        world_bank_chart = "https://data.worldbank.org/share/widget?indicators=NY.GDP.MKTP.CD&locations=US"
        images.append({
            'url': world_bank_chart,
            'title': 'Economic Data Chart',
            'source': 'World Bank',
            'type': 'economic_data',
            'relevance_score': 65,
            'telegram_compatible': False,
            'file_type': 'png',
            'trusted_source': True
        })
        
        return images

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
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
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
                'Accept': 'image/*,*/*;q=0.8'
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
                if any(img_type in content_type for img_type in ['image/', 'png', 'jpeg', 'jpg', 'gif']):
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

