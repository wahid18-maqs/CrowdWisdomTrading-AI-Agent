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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedImageFinderInput(BaseModel):
    search_content: str = Field(description="Financial content to find relevant images for")
    mentioned_stocks: Optional[List[str]] = Field(default=[], description="Stock symbols mentioned in content")
    max_images: Optional[int] = Field(default=3, description="Maximum number of images to find")

class EnhancedImageFinder(BaseTool):
    name: str = "enhanced_financial_image_finder"
    description: str = "Finds actual image files (PNG/JPG) for financial charts and stock images that work with Telegram"
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
            
            # Find Telegram-compatible images
            verified_images = self._find_telegram_compatible_images(search_content, mentioned_stocks, max_images)
            
            # Return as JSON
            return json.dumps(verified_images, indent=2)
            
        except Exception as e:
            logger.error(f"Telegram-compatible image finder failed: {e}")
            return json.dumps([])

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

    def _find_telegram_compatible_images(self, content: str, stocks: List[str], max_images: int) -> List[Dict[str, Any]]:
        """Find actual image files that work with Telegram"""
        verified_images = []
        
        # Strategy 1: TradingView snapshot images (actual PNG files)
        for stock in stocks[:2]:
            tradingview_images = self._get_tradingview_snapshots(stock)
            verified_images.extend(tradingview_images)
            if len(verified_images) >= max_images:
                break
        
        # Strategy 2: Financial news site chart images
        if len(verified_images) < max_images:
            news_images = self._get_financial_news_images(content, stocks)
            verified_images.extend(news_images)
        
        # Strategy 3: Market index PNG images
        if len(verified_images) < max_images:
            index_images = self._get_market_index_images(content)
            verified_images.extend(index_images)
        
        # Strategy 4: Alternative chart providers
        if len(verified_images) < max_images:
            alt_images = self._get_alternative_chart_images(stocks)
            verified_images.extend(alt_images)
        
        # Verify all image URLs work with Telegram
        verified_images = self._verify_telegram_compatibility(verified_images)
        
        # Sort by relevance and verification status
        verified_images.sort(key=lambda x: (x.get('telegram_compatible', False), x.get('relevance_score', 0)), reverse=True)
        
        return verified_images[:max_images]

    def _get_tradingview_snapshots(self, stock_symbol: str) -> List[Dict[str, Any]]:
        """Get TradingView chart snapshot images (actual PNG files)"""
        images = []
        
        # TradingView provides direct PNG snapshot URLs
        tradingview_urls = [
            f"https://s3.tradingview.com/snapshots/u/{stock_symbol.lower()}.png",
            f"https://s3.tradingview.com/snapshots/s/{stock_symbol.upper()}.png",
            f"https://charts.tradingview.com/chart-images/{stock_symbol.lower()}_1D.png"
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

