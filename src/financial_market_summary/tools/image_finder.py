# Fixed image_finder.py - Returns real financial charts and data
from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import requests
import re
import os
import logging
from urllib.parse import urljoin, urlparse
import json
import time

logger = logging.getLogger(__name__)

class ImageFinderInput(BaseModel):
    """Input schema for image finder tool."""
    search_context: str = Field(..., description="Context or content to find relevant images for")
    max_images: int = Field(default=2, description="Maximum number of images to find")
    image_types: List[str] = Field(default=["chart", "graph", "financial"], description="Types of images to look for")

class ImageFinder(BaseTool):
    name: str = "financial_image_finder"
    description: str = (
        "Find relevant financial charts, graphs, and images based on news content. "
        "Returns real, working URLs for stock charts, market graphs, and financial visualizations."
    )
    args_schema: Type[BaseModel] = ImageFinderInput

    def _run(self, search_context: str, max_images: int = 2, image_types: List[str] = None) -> str:
        """
        Find relevant financial images based on search context
        """
        try:
            if image_types is None:
                image_types = ["chart", "graph", "financial"]
            
            logger.info(f"Searching for {max_images} financial images based on context...")
            
            # Extract key financial terms from context
            financial_terms = self._extract_financial_terms(search_context)
            logger.info(f"Extracted financial terms: {financial_terms}")
            
            # Search for images using multiple methods
            images = []
            
            # Method 1: Get real stock charts for mentioned stocks
            if financial_terms['stocks']:
                logger.info(f"Finding charts for stocks: {financial_terms['stocks']}")
                stock_images = self._find_real_stock_charts(financial_terms['stocks'])
                images.extend(stock_images[:max_images//2])
            
            # Method 2: Get market index charts
            if len(images) < max_images:
                index_images = self._find_market_index_charts()
                images.extend(index_images[:max_images - len(images)])
            
            # Method 3: Search for real financial images using Serper
            if len(images) < max_images:
                search_images = self._search_real_financial_images(financial_terms, max_images - len(images))
                images.extend(search_images)
            
            # Method 4: Get real-time financial charts as fallback
            if len(images) < max_images:
                realtime_images = self._get_realtime_financial_charts(max_images - len(images))
                images.extend(realtime_images)
            
            # Verify all images are accessible
            verified_images = self._verify_and_filter_images(images)
            
            # Format results
            return self._format_image_results(verified_images, search_context)
            
        except Exception as e:
            error_msg = f"Image search error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _extract_financial_terms(self, content: str) -> Dict[str, List[str]]:
        """Extract financial terms and entities from content"""
        financial_data = {
            'stocks': [],
            'indices': [],
            'sectors': [],
            'keywords': []
        }
        
        # Enhanced stock symbols pattern - look for context clues
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, content)
        
        # Filter for likely stock symbols with better exclusion
        exclude_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 
            'OUR', 'HAD', 'HAS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'ITS', 'DID', 'GET', 
            'MAY', 'HIM', 'SHE', 'SUN', 'USE', 'WAR', 'FAR', 'OFF', 'BAD', 'OWN', 'SAY', 'TOO', 
            'ANY', 'DAY', 'END', 'WAY', 'OUT', 'MAN', 'TOP', 'PUT', 'SET', 'RUN', 'GOT', 'LET',
            'NEWS', 'SAID', 'ALSO', 'MADE', 'OVER', 'HERE', 'TIME', 'YEAR', 'WEEK', 'HOUR',
            'PLUS', 'THAN', 'ONLY', 'JUST', 'LIKE', 'INTO', 'MORE', 'SOME', 'VERY', 'WHAT',
            'FROM', 'THEY', 'KNOW', 'WANT', 'BEEN', 'GOOD', 'MUCH', 'COME', 'COULD', 'WOULD',
            'MARKET', 'STOCK', 'PRICE', 'DOWN', 'CLOSE', 'OPEN', 'HIGH', 'TRADE', 'SELL', 'BUY'
        }
        
        # Known major stocks to prioritize
        major_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD',
            'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'SPOT', 'ZOOM', 'SQ', 'ROKU',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
            'JNJ', 'PFE', 'UNH', 'CVS', 'ABBV', 'MRK', 'BMY', 'GILD', 'AMGN', 'BIIB'
        }
        
        # Prioritize major stocks found in content
        financial_data['stocks'] = [
            stock for stock in potential_stocks 
            if stock in major_stocks or (stock not in exclude_words and len(stock) <= 5)
        ][:5]  # Limit to top 5
        
        # Major indices
        indices = ['SPY', 'QQQ', 'DIA', 'VIX', 'IWM', 'VTI', 'VOO']
        financial_data['indices'] = [idx for idx in indices if idx in content.upper()]
        
        # Financial keywords for context
        financial_keywords = ['earnings', 'revenue', 'profit', 'growth', 'trading', 'market', 'nasdaq', 'dow']
        financial_data['keywords'] = [kw for kw in financial_keywords if kw.lower() in content.lower()]
        
        return financial_data
    
    def _find_real_stock_charts(self, stocks: List[str]) -> List[Dict[str, Any]]:
        """Find real stock charts for specific symbols"""
        images = []
        
        try:
            # Use multiple real chart sources
            for stock in stocks[:3]:  # Limit to prevent too many API calls
                logger.info(f"Looking for chart for {stock}")
                
                # Try different chart sources in order of reliability
                chart_sources = [
                    self._get_finviz_chart,
                    self._get_yahoo_finance_chart, 
                    self._get_tradingview_chart,
                    self._get_investing_com_chart
                ]
                
                for chart_source in chart_sources:
                    try:
                        chart_data = chart_source(stock)
                        if chart_data:
                            images.append(chart_data)
                            logger.info(f"Found chart for {stock} from {chart_source.__name__}")
                            break  # Found chart for this stock, move to next
                    except Exception as e:
                        logger.warning(f"Failed to get chart from {chart_source.__name__} for {stock}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error finding stock charts: {e}")
        
        return images
    
    def _get_finviz_chart(self, symbol: str) -> Dict[str, Any]:
        """Get real chart from Finviz (most reliable)"""
        try:
            # Finviz provides direct chart access
            chart_url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
            
            # Verify accessibility
            if self._verify_image_url(chart_url):
                return {
                    'url': chart_url,
                    'type': 'stock_chart',
                    'symbol': symbol,
                    'source': 'finviz',
                    'description': f'{symbol} daily stock chart from Finviz'
                }
        except Exception as e:
            logger.debug(f"Finviz chart error for {symbol}: {e}")
        
        return None
    
    def _get_yahoo_finance_chart(self, symbol: str) -> Dict[str, Any]:
        """Get chart from Yahoo Finance"""
        try:
            # Yahoo Finance chart URL with proper parameters
            chart_url = f"https://chart.yahoo.com/z?s={symbol}&t=1d&q=l&l=on&z=s&p=m50,m200&a=v&c="
            
            if self._verify_image_url(chart_url):
                return {
                    'url': chart_url,
                    'type': 'stock_chart',
                    'symbol': symbol,
                    'source': 'yahoo_finance',
                    'description': f'{symbol} stock chart from Yahoo Finance'
                }
        except Exception as e:
            logger.debug(f"Yahoo Finance chart error for {symbol}: {e}")
        
        return None
    
    def _get_tradingview_chart(self, symbol: str) -> Dict[str, Any]:
        """Get chart from TradingView"""
        try:
            # TradingView snapshot URL
            chart_url = f"https://s3.tradingview.com/snapshots/u/{symbol}.png"
            
            if self._verify_image_url(chart_url):
                return {
                    'url': chart_url,
                    'type': 'stock_chart',
                    'symbol': symbol,
                    'source': 'tradingview',
                    'description': f'{symbol} chart snapshot from TradingView'
                }
        except Exception as e:
            logger.debug(f"TradingView chart error for {symbol}: {e}")
        
        return None
    
    def _get_investing_com_chart(self, symbol: str) -> Dict[str, Any]:
        """Get chart from Investing.com"""
        try:
            # Investing.com chart format
            chart_url = f"https://i-invdn-com.investing.com/charts/us_stocks_{symbol.lower()}_1d.png"
            
            if self._verify_image_url(chart_url):
                return {
                    'url': chart_url,
                    'type': 'stock_chart',
                    'symbol': symbol,
                    'source': 'investing_com',
                    'description': f'{symbol} chart from Investing.com'
                }
        except Exception as e:
            logger.debug(f"Investing.com chart error for {symbol}: {e}")
        
        return None
    
    def _find_market_index_charts(self) -> List[Dict[str, Any]]:
        """Find real market index charts"""
        images = []
        
        # Major market indices with their chart sources
        indices = [
            {'symbol': 'SPY', 'name': 'S&P 500'},
            {'symbol': 'QQQ', 'name': 'NASDAQ'},
            {'symbol': 'DIA', 'name': 'Dow Jones'}
        ]
        
        for index in indices[:2]:  # Limit to 2 indices
            symbol = index['symbol']
            
            # Try to get chart from Finviz (most reliable)
            chart_url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
            
            if self._verify_image_url(chart_url):
                images.append({
                    'url': chart_url,
                    'type': 'index_chart',
                    'symbol': symbol,
                    'source': 'finviz',
                    'description': f"{index['name']} ({symbol}) index chart"
                })
                logger.info(f"Found index chart for {symbol}")
        
        return images
    
    def _search_real_financial_images(self, financial_terms: Dict[str, List[str]], max_images: int) -> List[Dict[str, Any]]:
        """Search for real financial images using Serper API"""
        images = []
        
        try:
            serper_key = os.getenv('SERPER_API_KEY')
            if not serper_key:
                logger.warning("SERPER_API_KEY not found, skipping image search")
                return images
            
            # Build search query from financial terms
            query_parts = ['financial chart']
            
            if financial_terms['stocks']:
                query_parts.append(f"{financial_terms['stocks'][0]} stock chart")
            if financial_terms['keywords']:
                query_parts.append(' '.join(financial_terms['keywords'][:2]))
            
            query = ' '.join(query_parts) + ' market graph today'
            
            logger.info(f"Searching for images with query: {query}")
            
            url = "https://google.serper.dev/images"
            payload = {
                "q": query,
                "num": max_images * 3,  # Get more to filter for quality
                "gl": "us",
                "safe": "active"
            }
            headers = {
                "X-API-KEY": serper_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            
            # Process and filter results
            for img_data in data.get('images', []):
                if len(images) >= max_images:
                    break
                
                img_url = img_data.get('imageUrl') or img_data.get('link')
                title = img_data.get('title', 'Financial Chart')
                
                if self._is_quality_financial_image(title, img_url):
                    if self._verify_image_url(img_url, quick_check=True):
                        images.append({
                            'url': img_url,
                            'type': 'financial_search',
                            'title': title,
                            'source': 'serper_search',
                            'description': title
                        })
                        logger.info(f"Found quality financial image: {title[:50]}...")
        
        except Exception as e:
            logger.error(f"Serper image search error: {e}")
        
        return images
    
    def _get_realtime_financial_charts(self, max_images: int) -> List[Dict[str, Any]]:
        """Get real-time financial charts and data visualizations"""
        images = []
        
        # Real financial data sources that provide chart APIs
        try:
            # Yahoo Finance real-time charts
            major_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
            
            for symbol in major_symbols[:max_images]:
                # Yahoo Finance chart API
                yahoo_chart = f"https://chart.yahoo.com/z?s={symbol}&t=1d&q=l&l=on&z=s&p=m50,m200"
                
                if self._verify_image_url(yahoo_chart, quick_check=True):
                    images.append({
                        'url': yahoo_chart,
                        'type': 'realtime_chart',
                        'symbol': symbol,
                        'source': 'yahoo_finance',
                        'description': f'{symbol} real-time chart from Yahoo Finance'
                    })
                    logger.info(f"Added real-time chart for {symbol}")
        
        except Exception as e:
            logger.error(f"Error getting real-time charts: {e}")
        
        # If still need more images, try market overview charts
        if len(images) < max_images:
            try:
                # Market heat map and overview charts
                market_charts = [
                    {
                        'url': 'https://finviz.com/grp_image.ashx?bar_sector_t.png',
                        'type': 'sector_performance',
                        'source': 'finviz',
                        'description': 'Sector Performance Heatmap'
                    },
                    {
                        'url': 'https://finviz.com/grp_image.ashx?bar_industry_d.png',
                        'type': 'industry_performance',
                        'source': 'finviz',
                        'description': 'Industry Performance Overview'
                    }
                ]
                
                for chart in market_charts:
                    if len(images) >= max_images:
                        break
                    
                    if self._verify_image_url(chart['url'], quick_check=True):
                        images.append(chart)
                        logger.info(f"Added market overview: {chart['description']}")
            
            except Exception as e:
                logger.error(f"Error getting market overview charts: {e}")
        
        return images
    
    def _is_quality_financial_image(self, title: str, url: str) -> bool:
        """Check if image is high-quality financial content"""
        if not title or not url:
            return False
        
        # Quality indicators in title
        quality_indicators = [
            'chart', 'graph', 'stock', 'market', 'trading', 'financial',
            'price', 'volume', 'earnings', 'revenue', 'index', 'nasdaq',
            'dow', 'analysis', 'performance', 'data', 'trend'
        ]
        
        # Exclude low-quality indicators
        low_quality_indicators = [
            'meme', 'joke', 'funny', 'cartoon', 'logo', 'icon', 'avatar',
            'wallpaper', 'background', 'template', 'generic', 'stock photo'
        ]
        
        title_lower = title.lower()
        url_lower = url.lower()
        text_to_check = title_lower + ' ' + url_lower
        
        # Must have quality indicators
        has_quality = any(indicator in text_to_check for indicator in quality_indicators)
        
        # Must not have low-quality indicators
        has_low_quality = any(indicator in text_to_check for indicator in low_quality_indicators)
        
        # Prefer images from known financial domains
        trusted_domains = [
            'finviz.com', 'yahoo.com', 'tradingview.com', 'investing.com',
            'bloomberg.com', 'marketwatch.com', 'cnbc.com', 'reuters.com'
        ]
        
        from_trusted_domain = any(domain in url_lower for domain in trusted_domains)
        
        return has_quality and not has_low_quality and (from_trusted_domain or len(title) > 10)
    
    def _verify_image_url(self, image_url: str, quick_check: bool = False) -> bool:
        """Verify that an image URL is accessible and returns an image"""
        try:
            if quick_check:
                # Quick check - just verify URL format
                return image_url.startswith(('http://', 'https://')) and len(image_url) > 10
            
            # Full verification with HTTP request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.head(image_url, timeout=5, headers=headers, allow_redirects=True)
            
            # Check if response is successful
            if response.status_code != 200:
                return False
            
            # Check content type if available
            content_type = response.headers.get('content-type', '').lower()
            if content_type and not content_type.startswith('image/'):
                # Some chart APIs don't set proper content-type, so allow chart domains
                trusted_chart_domains = ['finviz.com', 'chart.yahoo.com', 'tradingview.com']
                return any(domain in image_url.lower() for domain in trusted_chart_domains)
            
            return True
            
        except Exception as e:
            logger.debug(f"Image verification failed for {image_url}: {e}")
            return False
    
    def _verify_and_filter_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify all images are accessible and filter out broken ones"""
        verified_images = []
        
        for img in images:
            url = img.get('url')
            if url and self._verify_image_url(url):
                verified_images.append(img)
                logger.info(f"✅ Verified image: {img.get('description', 'Unknown')}")
            else:
                logger.warning(f"❌ Failed verification: {img.get('description', 'Unknown')}")
        
        return verified_images
    
    def _format_image_results(self, images: List[Dict[str, Any]], context: str) -> str:
        """Format image search results for the agent"""
        if not images:
            return "No accessible financial images found. Please proceed without images."
        
        result_parts = ["=== VERIFIED FINANCIAL IMAGES FOUND ===\n"]
        
        for i, img in enumerate(images, 1):
            # Verify URL one more time before adding
            url = img['url']
            if self._verify_image_url(url, quick_check=True):
                image_info = f"""
Image {i}:
- URL: {url}
- Type: {img.get('type', 'unknown')}
- Title: {img.get('title', 'Financial Chart')}
- Description: {img.get('description', 'Financial visualization')}
- Source: {img.get('source', 'unknown')}
- Symbol: {img.get('symbol', 'N/A')}
---
"""
                result_parts.append(image_info)
        
        result_parts.append(f"\nTotal verified images: {len(images)}")
        result_parts.append("\n=== MARKDOWN USAGE INSTRUCTIONS ===")
        result_parts.append("Use these verified images in your markdown format:")
        
        for i, img in enumerate(images, 1):
            title = img.get('title', f"Financial Chart {i}")
            result_parts.append(f"![{title}]({img['url']})")
        
        result_parts.append("\nAll URLs have been verified and are accessible.")
        
        return "\n".join(result_parts)