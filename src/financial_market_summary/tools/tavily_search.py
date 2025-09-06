import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import json
from urllib.parse import urlparse

class TavilyFinancialSearch:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"
        self.session = requests.Session()
        
        # Search configuration
        self.max_results = 8  # Number of articles to return
        self.search_depth = "advanced"  # Use advanced search for better results
        self.include_images = True  # Enable image search
        self.include_raw_content = True  # Get full article content
        
        # Financial news sources (prioritized)
        self.trusted_financial_sources = [
            'bloomberg.com', 'reuters.com', 'wsj.com', 'cnbc.com',
            'marketwatch.com', 'finance.yahoo.com', 'ft.com',
            'seekingalpha.com', 'investopedia.com', 'fool.com',
            'morningstar.com', 'benzinga.com', 'zacks.com'
        ]
        
        # Keywords for US financial market search
        self.market_keywords = [
            'US stock market', 'Wall Street', 'NYSE', 'NASDAQ', 'S&P 500',
            'Dow Jones', 'American stocks', 'US equities', 'US market news',
            'Federal Reserve', 'US economy', 'earnings report', 'IPO',
            'US financial markets', 'American companies'
        ]
    
    def search_us_financial_news(self, hours_back: int = 2, custom_query: Optional[str] = None) -> Dict:
        """
        Search for recent US financial market news
        
        Args:
            hours_back: How many hours back to search (default: 2)
            custom_query: Optional custom search query
            
        Returns:
            Dictionary containing search results with articles and metadata
        """
        try:
            # Build search query
            if custom_query:
                query = custom_query
            else:
                # Create dynamic query based on current time and market keywords
                base_query = "US stock market news today Wall Street NYSE NASDAQ"
                query = f"{base_query} latest financial news"
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            logging.info(f"Searching for financial news: '{query}' from {start_time} to {end_time}")
            
            # Perform the search
            search_results = self._execute_search(
                query=query,
                max_results=self.max_results,
                include_domains=self.trusted_financial_sources,
                time_range=(start_time, end_time)
            )
            
            if not search_results['success']:
                logging.error(f"Search failed: {search_results['error']}")
                return search_results
            
            # Process and enhance results
            processed_results = self._process_search_results(search_results['results'])
            
            # Add metadata
            processed_results.update({
                'search_metadata': {
                    'query_used': query,
                    'search_time_range': {
                        'start': start_time.isoformat(),
                        'end': end_time.isoformat(),
                        'hours_back': hours_back
                    },
                    'total_articles_found': len(processed_results.get('articles', [])),
                    'total_urls_for_image_extraction': len(processed_results.get('article_urls', [])),
                    'search_timestamp': datetime.now().isoformat()
                }
            })
            
            logging.info(f"Successfully found {len(processed_results.get('articles', []))} financial articles")
            return processed_results
            
        except Exception as e:
            error_msg = f"Error in search_us_financial_news: {str(e)}"
            logging.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'articles': [],
                'article_urls': []
            }
    
    def _execute_search(self, 
                       query: str, 
                       max_results: int,
                       include_domains: List[str] = None,
                       time_range: tuple = None) -> Dict:
        """Execute the actual Tavily API search"""
        try:
            # Prepare search parameters
            search_params = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": self.search_depth,
                "include_images": self.include_images,
                "include_raw_content": self.include_raw_content,
                "max_results": max_results,
                "include_answer": False  # We don't need AI-generated answers
            }
            
            # Add domain filtering if specified
            if include_domains:
                # Tavily uses include_domains parameter
                search_params["include_domains"] = include_domains[:5]  # Limit to top 5 domains
            
            # Add time filtering if specified
            if time_range:
                start_time, end_time = time_range
                # Tavily might support date filtering - check their latest API docs
                search_params["published_after"] = start_time.strftime("%Y-%m-%d")
            
            # Make the API request
            response = self.session.post(
                self.base_url,
                json=search_params,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Check if search was successful
            if 'results' not in data:
                return {
                    'success': False,
                    'error': 'No results returned from Tavily API',
                    'results': []
                }
            
            return {
                'success': True,
                'results': data['results'],
                'query_used': query
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP error during Tavily search: {str(e)}"
            logging.error(error_msg)
            return {'success': False, 'error': error_msg, 'results': []}
        
        except Exception as e:
            error_msg = f"Unexpected error during Tavily search: {str(e)}"
            logging.error(error_msg)
            return {'success': False, 'error': error_msg, 'results': []}
    
    def _process_search_results(self, raw_results: List[Dict]) -> Dict:
        """Process and structure the search results"""
        articles = []
        article_urls = []
        
        for result in raw_results:
            try:
                # Extract basic information
                title = result.get('title', 'Untitled')
                url = result.get('url', '')
                content = result.get('content', '')
                raw_content = result.get('raw_content', content)  # Full article text if available
                
                # Skip if no URL (can't extract images later)
                if not url:
                    continue
                
                # Parse URL to get domain info
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                
                # Skip if not from trusted financial sources
                if not any(trusted_domain in domain for trusted_domain in self.trusted_financial_sources):
                    logging.debug(f"Skipping article from untrusted domain: {domain}")
                    continue
                
                # Extract publication date if available
                published_date = result.get('published_date') or result.get('date')
                
                # Check for financial relevance
                relevance_score = self._calculate_financial_relevance(title, content)
                
                # Only include articles with reasonable financial relevance
                if relevance_score < 2.0:
                    logging.debug(f"Skipping low-relevance article: {title[:50]}...")
                    continue
                
                # Structure article data
                article_data = {
                    'title': title,
                    'url': url,
                    'domain': domain,
                    'content': content,
                    'raw_content': raw_content,  # Full content for better summarization
                    'published_date': published_date,
                    'relevance_score': relevance_score,
                    'word_count': len(content.split()) if content else 0,
                    'has_full_content': bool(raw_content and len(raw_content) > len(content))
                }
                
                # Extract any image information provided by Tavily
                if 'images' in result and result['images']:
                    article_data['tavily_images'] = result['images'][:3]  # Keep top 3 images
                
                articles.append(article_data)
                article_urls.append(url)
                
                logging.debug(f"Processed article: {title[:50]}... (Score: {relevance_score})")
                
            except Exception as e:
                logging.warning(f"Error processing search result: {str(e)}")
                continue
        
        # Sort articles by relevance score (descending)
        articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Prepare final results
        processed_data = {
            'success': True,
            'articles': articles,
            'article_urls': article_urls,  # For image extraction tool
            'total_found': len(articles),
            'content_summary': self._generate_content_summary(articles)
        }
        
        return processed_data
    
    def _calculate_financial_relevance(self, title: str, content: str) -> float:
        """Calculate how relevant an article is to US financial markets"""
        score = 0.0
        text_to_analyze = f"{title} {content}".lower()
        
        # US market specific terms
        us_market_terms = [
            'wall street', 'nasdaq', 'nyse', 's&p 500', 'dow jones',
            'federal reserve', 'fed', 'us stock', 'american market',
            'us economy', 'treasury', 'sec'
        ]
        
        # General financial terms
        financial_terms = [
            'stock', 'share', 'market', 'trading', 'investor', 'earnings',
            'revenue', 'profit', 'growth', 'dividend', 'ipo', 'merger',
            'acquisition', 'analyst', 'upgrade', 'downgrade', 'buy rating',
            'sell rating', 'price target', 'market cap', 'volume'
        ]
        
        # Company and sector terms
        sector_terms = [
            'technology', 'healthcare', 'financial', 'energy', 'retail',
            'manufacturing', 'pharmaceutical', 'biotech', 'semiconductor',
            'bank', 'insurance', 'reit'
        ]
        
        # Count US market terms (high weight)
        us_matches = sum(1 for term in us_market_terms if term in text_to_analyze)
        score += us_matches * 3.0
        
        # Count general financial terms (medium weight)
        financial_matches = sum(1 for term in financial_terms if term in text_to_analyze)
        score += financial_matches * 1.5
        
        # Count sector terms (lower weight)
        sector_matches = sum(1 for term in sector_terms if term in text_to_analyze)
        score += sector_matches * 1.0
        
        # Bonus for earnings/results keywords
        earnings_terms = ['earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4', 'guidance']
        earnings_matches = sum(1 for term in earnings_terms if term in text_to_analyze)
        score += earnings_matches * 2.0
        
        # Bonus for market movement terms
        movement_terms = ['surge', 'plunge', 'rally', 'decline', 'volatility', 'gain', 'loss']
        movement_matches = sum(1 for term in movement_terms if term in text_to_analyze)
        score += movement_matches * 1.0
        
        return round(score, 2)
    
    def _generate_content_summary(self, articles: List[Dict]) -> Dict:
        """Generate a summary of the content found"""
        if not articles:
            return {'total_articles': 0}
        
        total_words = sum(article['word_count'] for article in articles)
        domains = list(set(article['domain'] for article in articles))
        avg_relevance = sum(article['relevance_score'] for article in articles) / len(articles)
        
        # Find top themes/topics
        all_titles = ' '.join(article['title'] for article in articles).lower()
        
        return {
            'total_articles': len(articles),
            'total_word_count': total_words,
            'unique_sources': len(domains),
            'source_domains': domains,
            'average_relevance_score': round(avg_relevance, 2),
            'articles_with_full_content': len([a for a in articles if a['has_full_content']]),
            'top_article_titles': [article['title'] for article in articles[:3]]
        }
    
    def test_api_connection(self) -> bool:
        """Test if the Tavily API key is working"""
        try:
            test_params = {
                "api_key": self.api_key,
                "query": "test financial news",
                "max_results": 1,
                "search_depth": "basic"
            }
            
            response = self.session.post(self.base_url, json=test_params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' in data:
                logging.info("Tavily API connection test successful")
                return True
            else:
                logging.error(f"Tavily API test failed: {data}")
                return False
                
        except Exception as e:
            logging.error(f"Tavily API connection test failed: {str(e)}")
            return False
    
    def search_specific_topic(self, topic: str, max_results: int = 5) -> Dict:
        """Search for news about a specific financial topic"""
        try:
            enhanced_query = f"{topic} US stock market financial news"
            
            search_results = self._execute_search(
                query=enhanced_query,
                max_results=max_results,
                include_domains=self.trusted_financial_sources
            )
            
            if search_results['success']:
                processed_results = self._process_search_results(search_results['results'])
                processed_results['search_topic'] = topic
                return processed_results
            else:
                return search_results
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error searching for topic '{topic}': {str(e)}",
                'articles': [],
                'article_urls': []
            }


def search_financial_news(api_key: str, hours_back: int = 2, custom_query: Optional[str] = None) -> Dict:
    """
    Main function to search for US financial market news
    
    Args:
        api_key: Tavily API key
        hours_back: How many hours back to search
        custom_query: Optional custom search query
        
    Returns:
        Dictionary containing articles and URLs for image extraction
    """
    searcher = TavilyFinancialSearch(api_key)
    return searcher.search_us_financial_news(hours_back, custom_query)


def test_tavily_connection(api_key: str) -> bool:
    """
    Test Tavily API connection
    
    Args:
        api_key: Tavily API key
        
    Returns:
        Boolean indicating if connection is successful
    """
    searcher = TavilyFinancialSearch(api_key)
    return searcher.test_api_connection()