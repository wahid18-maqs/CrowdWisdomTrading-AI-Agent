from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import requests
import re
import os
import logging
from urllib.parse import urljoin, urlparse
import json

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
        "Searches for stock charts, market graphs, and financial visualizations."
    )
    args_schema: Type[BaseModel] = ImageFinderInput

    def _run(self, search_context: str, max_images: int = 2, image_types: List[str] = None) -> str:
        """
        Find relevant financial images based on search context
        """
        try:
            if image_types is None:
                image_types = ["chart", "graph", "financial"]
            
            # Extract key financial terms from context
            financial_terms = self._extract_financial_terms(search_context)
            
            # Search for images using multiple methods
            images = []
            
            # Method 1: Search for stock charts using financial APIs
            if financial_terms['stocks']:
                stock_images = self._find_stock_charts(financial_terms['stocks'])
                images.extend(stock_images[:max_images//2])
            
            # Method 2: Search for general financial images
            if len(images) < max_images:
                general_images = self._find_financial_images(financial_terms, max_images - len(images))
                images.extend(general_images)
            
            # Format results
            return self._format_image_results(images, search_context)
            
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
        
        # Common stock symbols pattern (2-5 uppercase letters)
        stock_pattern = r'\b[A-Z]{2,5}\b'
        potential_stocks = re.findall(stock_pattern, content)
        
        # Filter for likely stock symbols (exclude common words)
        exclude_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'HAS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'ITS', 'DID', 'GET', 'MAY', 'HIM', 'SHE', 'SUN', 'USE', 'WAR', 'FAR', 'OFF', 'BAD', 'OWN', 'SAY', 'TOO', 'ANY', 'DAY', 'END', 'WAY'}
        financial_data['stocks'] = [stock for stock in potential_stocks if stock not in exclude_words]
        
        # Major indices
        indices = ['SPY', 'QQQ', 'DIA', 'VIX', 'NASDAQ', 'DJIA', 'S&P']
        financial_data['indices'] = [idx for idx in indices if idx.upper() in content.upper()]
        
        # Financial keywords
        financial_keywords = ['earnings', 'revenue', 'profit', 'loss', 'growth', 'market', 'trading', 'investment', 'portfolio', 'bonds', 'commodities']
        financial_data['keywords'] = [kw for kw in financial_keywords if kw.lower() in content.lower()]
        
        return financial_data
    
    def _find_stock_charts(self, stocks: List[str]) -> List[Dict[str, Any]]:
        """Find stock charts for specific symbols"""
        images = []
        
        try:
            # Use multiple chart sources
            chart_sources = [
                self._get_yahoo_chart,
                self._get_trading_view_chart,
                self._get_finviz_chart
            ]
            
            for stock in stocks[:3]:  # Limit to first 3 stocks
                for chart_source in chart_sources:
                    try:
                        chart_url = chart_source(stock)
                        if chart_url:
                            images.append({
                                'url': chart_url,
                                'type': 'stock_chart',
                                'symbol': stock,
                                'source': chart_source.__name__,
                                'description': f'{stock} stock chart'
                            })
                            break  # Found chart for this stock, move to next
                    except Exception as e:
                        logger.warning(f"Failed to get chart from {chart_source.__name__}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error finding stock charts: {e}")
        
        return images
    
    def _get_yahoo_chart(self, symbol: str) -> str:
        """Get chart URL from Yahoo Finance"""
        try:
            # Yahoo Finance chart URL format
            chart_url = f"https://chart.yahoo.com/z?s={symbol}&t=1d&q=l&l=on&z=s&p=m50,m200"
            
            # Verify the URL is accessible
            response = requests.head(chart_url, timeout=10)
            if response.status_code == 200:
                return chart_url
        except Exception:
            pass
        return None
    
    def _get_trading_view_chart(self, symbol: str) -> str:
        """Get chart URL from TradingView"""
        try:
            # TradingView widget chart URL
            chart_url = f"https://s3.tradingview.com/snapshots/u/{symbol}.png"
            
            # Alternative: TradingView chart widget
            widget_url = f"https://www.tradingview.com/widgetembed/?frameElementId=tradingview&symbol={symbol}&interval=D&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=F1F3F6&studies=[]&hideideas=1&theme=White&timezone=Etc%2FUTC&studies_overrides=%7B%7D&overrides=%7B%7D&enabled_features=[]&disabled_features=[]&showpopupbutton=1&locale=en&utm_source=www.tradingview.com&utm_medium=widget&utm_campaign=chart&utm_term={symbol}"
            
            return widget_url
        except Exception:
            pass
        return None
    
    def _get_finviz_chart(self, symbol: str) -> str:
        """Get chart URL from Finviz"""
        try:
            chart_url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
            return chart_url
        except Exception:
            pass
        return None
    
    def _find_financial_images(self, financial_terms: Dict[str, List[str]], max_images: int) -> List[Dict[str, Any]]:
        """Find general financial images using search APIs"""
        images = []
        
        try:
            # Use Serper for image search if available
            serper_key = os.getenv('SERPER_API_KEY')
            if serper_key:
                images.extend(self._serper_image_search(financial_terms, max_images))
            
            # If we still need more images, use backup sources
            if len(images) < max_images:
                images.extend(self._get_generic_financial_images(max_images - len(images)))
        
        except Exception as e:
            logger.error(f"Error finding financial images: {e}")
        
        return images
    
    def _serper_image_search(self, financial_terms: Dict[str, List[str]], max_images: int) -> List[Dict[str, Any]]:
        """Search for financial images using Serper API"""
        images = []
        
        try:
            serper_key = os.getenv('SERPER_API_KEY')
            if not serper_key:
                return images
            
            # Build search query from financial terms
            query_parts = []
            if financial_terms['stocks']:
                query_parts.append(' '.join(financial_terms['stocks'][:2]))
            if financial_terms['keywords']:
                query_parts.append(' '.join(financial_terms['keywords'][:2]))
            
            query = ' '.join(query_parts) + ' financial chart graph market'
            
            url = "https://google.serper.dev/images"
            payload = {
                "q": query,
                "num": max_images * 2,  # Get more to filter
                "gl": "us"
            }
            headers = {
                "X-API-KEY": serper_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            
            # Process results
            for img_data in data.get('images', [])[:max_images]:
                if self._is_financial_image(img_data.get('title', ''), img_data.get('link', '')):
                    images.append({
                        'url': img_data.get('imageUrl') or img_data.get('link'),
                        'type': 'financial_chart',
                        'title': img_data.get('title', 'Financial Chart'),
                        'source': 'serper_search',
                        'description': img_data.get('title', 'Financial market visualization')
                    })
        
        except Exception as e:
            logger.error(f"Serper image search error: {e}")
        
        return images
    
    def _is_financial_image(self, title: str, url: str) -> bool:
        """Check if image is likely to be financial/market related"""
        financial_indicators = [
            'chart', 'graph', 'stock', 'market', 'trading', 'financial',
            'price', 'volume', 'earnings', 'revenue', 'index', 'portfolio'
        ]
        
        text_to_check = (title + ' ' + url).lower()
        return any(indicator in text_to_check for indicator in financial_indicators)
    
    def _get_generic_financial_images(self, max_images: int) -> List[Dict[str, Any]]:
        """Get generic financial images as fallback"""
        generic_images = [
            {
                'url': 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=800',
                'type': 'generic_financial',
                'title': 'Stock Market Graph',
                'source': 'unsplash',
                'description': 'Generic stock market visualization'
            },
            {
                'url': 'https://images.unsplash.com/photo-1642790106117-e829e14a795f?w=800',
                'type': 'generic_financial', 
                'title': 'Financial Data Visualization',
                'source': 'unsplash',
                'description': 'Financial market data analysis'
            },
            {
                'url': 'https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?w=800',
                'type': 'generic_financial',
                'title': 'Market Trading Screen',
                'source': 'unsplash', 
                'description': 'Stock market trading interface'
            }
        ]
        
        return generic_images[:max_images]
    
    def _format_image_results(self, images: List[Dict[str, Any]], context: str) -> str:
        """Format image search results for the agent"""
        if not images:
            return "No relevant financial images found for the given context."
        
        result_parts = ["=== FINANCIAL IMAGES FOUND ===\n"]
        
        for i, img in enumerate(images, 1):
            image_info = f"""
Image {i}:
- URL: {img['url']}
- Type: {img.get('type', 'unknown')}
- Title: {img.get('title', 'No title')}
- Description: {img.get('description', 'No description')}
- Source: {img.get('source', 'unknown')}
- Symbol: {img.get('symbol', 'N/A')}
---
"""
            result_parts.append(image_info)
        
        result_parts.append(f"\nTotal images found: {len(images)}")
        result_parts.append("Usage instructions: These images can be embedded in the financial summary using markdown syntax:")
        
        for i, img in enumerate(images, 1):
            result_parts.append(f"![{img.get('title', f'Financial Chart {i}')}]({img['url']})")
        
        return "\n".join(result_parts)
    
    def verify_image_accessibility(self, image_url: str) -> bool:
        """Verify that an image URL is accessible"""
        try:
            response = requests.head(image_url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False