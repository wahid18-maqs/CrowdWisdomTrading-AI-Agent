import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Dict, Optional
import time
import os
from PIL import Image
import io

class ArticleImageExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Keywords that indicate financial charts/graphs
        self.financial_image_keywords = [
            'chart', 'graph', 'data', 'market', 'stock', 'trading', 'financial',
            'analysis', 'performance', 'earnings', 'revenue', 'profit', 'growth',
            'trend', 'volatility', 'index', 'sector', 'portfolio', 'investment',
            'economic', 'gdp', 'inflation', 'yield', 'price', 'volume'
        ]
        
        # Image extensions to accept
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}
        
        # Max image size (5MB for Telegram)
        self.max_image_size = 5 * 1024 * 1024
        
    def extract_images_from_articles(self, article_urls: List[str]) -> List[Dict]:
        """
        Extract relevant financial images from a list of article URLs
        
        Args:
            article_urls: List of news article URLs to scrape
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        all_images = []
        
        for url in article_urls:
            try:
                logging.info(f"Extracting images from: {url}")
                images = self._extract_images_from_single_article(url)
                all_images.extend(images)
                
                # Add delay between requests to be respectful
                time.sleep(1)
                
            except Exception as e:
                logging.warning(f"Failed to extract images from {url}: {str(e)}")
                continue
        
        # Filter and rank images
        relevant_images = self._filter_and_rank_images(all_images)
        
        # Validate and download top images
        validated_images = self._validate_images(relevant_images[:5])  # Top 5 candidates
        
        return validated_images[:2]  # Return top 2 images
    
    def _extract_images_from_single_article(self, url: str) -> List[Dict]:
        """Extract images from a single article URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            images = []
            
            # Find all img tags
            img_tags = soup.find_all('img')
            
            for img in img_tags:
                img_data = self._process_img_tag(img, url, soup)
                if img_data:
                    images.append(img_data)
            
            # Also check for figure tags which often contain financial charts
            figure_tags = soup.find_all('figure')
            for figure in figure_tags:
                img = figure.find('img')
                if img:
                    img_data = self._process_img_tag(img, url, soup, figure)
                    if img_data:
                        images.append(img_data)
            
            return images
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return []
    
    def _process_img_tag(self, img_tag, article_url: str, soup, parent_figure=None) -> Optional[Dict]:
        """Process individual img tag and extract metadata"""
        try:
            # Get image URL
            img_url = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('data-lazy-src')
            if not img_url:
                return None
            
            # Convert relative URLs to absolute
            img_url = urljoin(article_url, img_url)
            
            # Check if URL has valid extension
            parsed_url = urlparse(img_url)
            path = parsed_url.path.lower()
            if not any(path.endswith(ext) for ext in self.valid_extensions):
                return None
            
            # Extract metadata
            alt_text = img_tag.get('alt', '').lower()
            title = img_tag.get('title', '').lower()
            
            # Get caption from parent figure if available
            caption = ''
            if parent_figure:
                figcaption = parent_figure.find('figcaption')
                if figcaption:
                    caption = figcaption.get_text(strip=True)
            
            # Get surrounding context
            context = self._get_surrounding_context(img_tag, soup)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(alt_text, title, caption, context)
            
            return {
                'url': img_url,
                'source_article': article_url,
                'alt_text': alt_text,
                'title': title,
                'caption': caption,
                'context': context,
                'relevance_score': relevance_score,
                'source_domain': urlparse(article_url).netloc
            }
            
        except Exception as e:
            logging.warning(f"Error processing img tag: {str(e)}")
            return None
    
    def _get_surrounding_context(self, img_tag, soup) -> str:
        """Get text context around the image"""
        try:
            # Get parent container
            parent = img_tag.parent
            context_text = ""
            
            # Try to get text from parent and siblings
            if parent:
                # Get previous sibling text
                prev_sibling = img_tag.find_previous_sibling(['p', 'div', 'span'])
                if prev_sibling:
                    context_text += prev_sibling.get_text(strip=True)[:200]
                
                # Get next sibling text
                next_sibling = img_tag.find_next_sibling(['p', 'div', 'span'])
                if next_sibling:
                    context_text += " " + next_sibling.get_text(strip=True)[:200]
            
            return context_text.lower()
            
        except Exception:
            return ""
    
    def _calculate_relevance_score(self, alt_text: str, title: str, caption: str, context: str) -> float:
        """Calculate how relevant an image is to financial content"""
        score = 0.0
        all_text = f"{alt_text} {title} {caption} {context}".lower()
        
        # Check for financial keywords
        keyword_matches = sum(1 for keyword in self.financial_image_keywords if keyword in all_text)
        score += keyword_matches * 2
        
        # Bonus for specific chart/graph indicators
        if any(word in all_text for word in ['chart', 'graph', 'data visualization']):
            score += 5
        
        # Penalty for common non-financial images
        penalty_words = ['author', 'logo', 'advertisement', 'banner', 'profile', 'headshot']
        penalties = sum(1 for word in penalty_words if word in all_text)
        score -= penalties * 3
        
        # Bonus for having meaningful alt text or caption
        if len(alt_text) > 10 or len(caption) > 10:
            score += 2
        
        return max(0.0, score)  # Don't allow negative scores
    
    def _filter_and_rank_images(self, images: List[Dict]) -> List[Dict]:
        """Filter and rank images by relevance"""
        # Filter out images with very low relevance scores
        filtered_images = [img for img in images if img['relevance_score'] > 1.0]
        
        # Sort by relevance score (descending)
        filtered_images.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_images = []
        for img in filtered_images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        
        return unique_images
    
    def _validate_images(self, images: List[Dict]) -> List[Dict]:
        """Validate that images are accessible and suitable for Telegram"""
        validated = []
        
        for img in images:
            try:
                # Check if image URL is accessible
                head_response = self.session.head(img['url'], timeout=5)
                
                # Check content type
                content_type = head_response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    continue
                
                # Check file size
                content_length = head_response.headers.get('content-length')
                if content_length and int(content_length) > self.max_image_size:
                    logging.warning(f"Image too large: {img['url']}")
                    continue
                
                # Download and validate image
                response = self.session.get(img['url'], timeout=10)
                response.raise_for_status()
                
                # Try to open with PIL to ensure it's a valid image
                try:
                    image_data = Image.open(io.BytesIO(response.content))
                    width, height = image_data.size
                    
                    # Skip very small images (likely icons or thumbnails)
                    if width < 200 or height < 150:
                        continue
                    
                    # Add image data and metadata
                    img['image_data'] = response.content
                    img['content_type'] = content_type
                    img['file_size'] = len(response.content)
                    img['dimensions'] = (width, height)
                    
                    validated.append(img)
                    
                    logging.info(f"Validated image: {img['url']} ({width}x{height}, {len(response.content)} bytes)")
                    
                except Exception as pil_error:
                    logging.warning(f"Invalid image format: {img['url']} - {str(pil_error)}")
                    continue
                
            except Exception as e:
                logging.warning(f"Failed to validate image {img['url']}: {str(e)}")
                continue
        
        return validated
    
    def create_image_caption(self, image_data: Dict) -> str:
        """Create a caption for the image suitable for Telegram"""
        caption_parts = []
        
        # Add description from alt text or caption
        if image_data.get('caption'):
            caption_parts.append(image_data['caption'])
        elif image_data.get('alt_text') and len(image_data['alt_text']) > 5:
            caption_parts.append(image_data['alt_text'].title())
        
        # Add source attribution
        source_domain = image_data['source_domain']
        caption_parts.append(f"📊 Source: {source_domain}")
        
        return "\n".join(caption_parts)[:1024]  # Telegram caption limit


def extract_article_images(article_urls: List[str]) -> List[Dict]:
    """
    Main function to extract images from article URLs
    
    Args:
        article_urls: List of news article URLs
        
    Returns:
        List of validated image data dictionaries
    """
    extractor = ArticleImageExtractor()
    return extractor.extract_images_from_articles(article_urls)