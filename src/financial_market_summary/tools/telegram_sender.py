import requests
import time
import logging
from typing import List, Dict, Optional, Union
import io
import os
from PIL import Image

class TelegramSender:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = requests.Session()
        
        # Telegram limits
        self.max_message_length = 4096
        self.max_caption_length = 1024
        self.max_file_size = 50 * 1024 * 1024  # 50MB for photos
        self.photo_max_size = 10 * 1024 * 1024  # 10MB recommended for photos
        
    def send_financial_summary_with_images(self, 
                                         summary_text: str, 
                                         images: List[Dict],
                                         language: str = "English") -> Dict:
        """
        Send a complete financial summary with images to Telegram
        
        Args:
            summary_text: The main summary text
            images: List of image dictionaries with metadata
            language: Language of the content
            
        Returns:
            Dictionary with send results and status
        """
        results = {
            'language': language,
            'text_sent': False,
            'images_sent': [],
            'failed_images': [],
            'total_messages': 0,
            'errors': []
        }
        
        try:
            # Send main summary text first
            text_result = self._send_long_message(summary_text, language)
            results['text_sent'] = text_result['success']
            results['total_messages'] += text_result['messages_sent']
            
            if not text_result['success']:
                results['errors'].extend(text_result['errors'])
            
            # Send images with captions
            if images:
                time.sleep(1)  # Brief pause between text and images
                
                for i, image_data in enumerate(images):
                    try:
                        image_result = self._send_image_with_caption(image_data, i + 1, len(images))
                        
                        if image_result['success']:
                            results['images_sent'].append({
                                'url': image_data['url'],
                                'source': image_data['source_domain'],
                                'message_id': image_result.get('message_id')
                            })
                        else:
                            results['failed_images'].append({
                                'url': image_data['url'],
                                'error': image_result['error']
                            })
                        
                        results['total_messages'] += 1
                        
                        # Pause between images to avoid rate limiting
                        if i < len(images) - 1:
                            time.sleep(2)
                            
                    except Exception as e:
                        error_msg = f"Failed to send image {i+1}: {str(e)}"
                        logging.error(error_msg)
                        results['errors'].append(error_msg)
                        results['failed_images'].append({
                            'url': image_data.get('url', 'unknown'),
                            'error': str(e)
                        })
            
            # Calculate success rate
            total_items = 1 + len(images)  # text + images
            successful_items = (1 if results['text_sent'] else 0) + len(results['images_sent'])
            results['success_rate'] = successful_items / total_items if total_items > 0 else 0
            
        except Exception as e:
            error_msg = f"Critical error in send_financial_summary_with_images: {str(e)}"
            logging.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def _send_long_message(self, text: str, language: str) -> Dict:
        """Send long text message, splitting if necessary"""
        result = {'success': False, 'messages_sent': 0, 'errors': []}
        
        try:
            # Add language header
            language_header = f"📊 *Financial Market Summary* ({language})\n" + "="*40 + "\n\n"
            full_text = language_header + text
            
            # Split message if too long
            message_chunks = self._split_message(full_text)
            
            for i, chunk in enumerate(message_chunks):
                try:
                    response = self._send_text_message(chunk)
                    if response['ok']:
                        result['messages_sent'] += 1
                        logging.info(f"Sent message chunk {i+1}/{len(message_chunks)} for {language}")
                    else:
                        error_msg = f"Failed to send chunk {i+1}: {response.get('description', 'Unknown error')}"
                        result['errors'].append(error_msg)
                        logging.error(error_msg)
                    
                    # Pause between chunks
                    if i < len(message_chunks) - 1:
                        time.sleep(1)
                        
                except Exception as e:
                    error_msg = f"Error sending chunk {i+1}: {str(e)}"
                    result['errors'].append(error_msg)
                    logging.error(error_msg)
            
            result['success'] = result['messages_sent'] == len(message_chunks)
            
        except Exception as e:
            error_msg = f"Error in _send_long_message: {str(e)}"
            result['errors'].append(error_msg)
            logging.error(error_msg)
        
        return result
    
    def _send_image_with_caption(self, image_data: Dict, image_num: int, total_images: int) -> Dict:
        """Send a single image with caption"""
        result = {'success': False, 'error': None, 'message_id': None}
        
        try:
            # Prepare image data
            if 'image_data' not in image_data:
                result['error'] = "No image data available"
                return result
            
            image_bytes = image_data['image_data']
            
            # Optimize image if needed
            optimized_image = self._optimize_image_for_telegram(image_bytes)
            
            # Create caption
            caption = self._create_image_caption(image_data, image_num, total_images)
            
            # Prepare files for upload
            files = {
                'photo': ('financial_chart.jpg', optimized_image, 'image/jpeg')
            }
            
            data = {
                'chat_id': self.chat_id,
                'caption': caption,
                'parse_mode': 'Markdown'
            }
            
            # Send photo
            url = f"{self.base_url}/sendPhoto"
            response = self.session.post(url, data=data, files=files, timeout=30)
            response_data = response.json()
            
            if response_data['ok']:
                result['success'] = True
                result['message_id'] = response_data['result']['message_id']
                logging.info(f"Successfully sent image {image_num}/{total_images}")
            else:
                result['error'] = response_data.get('description', 'Unknown error')
                logging.error(f"Failed to send image {image_num}: {result['error']}")
            
        except Exception as e:
            result['error'] = str(e)
            logging.error(f"Exception sending image {image_num}: {str(e)}")
        
        return result
    
    def _optimize_image_for_telegram(self, image_bytes: bytes) -> bytes:
        """Optimize image size and format for Telegram"""
        try:
            # If image is already small enough, return as-is
            if len(image_bytes) <= self.photo_max_size:
                return image_bytes
            
            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # Calculate new dimensions to reduce file size
            width, height = image.size
            max_dimension = 1920  # Telegram's max photo dimension
            
            if width > max_dimension or height > max_dimension:
                ratio = min(max_dimension / width, max_dimension / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save with compression
            output = io.BytesIO()
            quality = 85
            
            # Reduce quality until file size is acceptable
            while quality > 20:
                output.seek(0)
                output.truncate()
                image.save(output, format='JPEG', quality=quality, optimize=True)
                
                if output.tell() <= self.photo_max_size:
                    break
                    
                quality -= 10
            
            return output.getvalue()
            
        except Exception as e:
            logging.warning(f"Image optimization failed: {str(e)}, using original")
            return image_bytes
    
    def _create_image_caption(self, image_data: Dict, image_num: int, total_images: int) -> str:
        """Create a comprehensive caption for the image"""
        caption_parts = []
        
        # Add image number if multiple images
        if total_images > 1:
            caption_parts.append(f"📊 *Chart {image_num}/{total_images}*")
        else:
            caption_parts.append("📊 *Related Chart*")
        
        # Add description from image metadata
        if image_data.get('caption') and len(image_data['caption'].strip()) > 0:
            caption_parts.append(f"_{image_data['caption'].strip()}_")
        elif image_data.get('alt_text') and len(image_data['alt_text'].strip()) > 5:
            alt_text = image_data['alt_text'].strip().title()
            caption_parts.append(f"_{alt_text}_")
        
        # Add source attribution
        source_domain = image_data.get('source_domain', 'Unknown Source')
        caption_parts.append(f"📰 Source: *{source_domain}*")
        
        # Add relevance note if score is high
        if image_data.get('relevance_score', 0) > 5:
            caption_parts.append("🎯 _Highly relevant to current market news_")
        
        # Join all parts and ensure it fits Telegram's limit
        full_caption = "\n".join(caption_parts)
        
        # Truncate if too long
        if len(full_caption) > self.max_caption_length:
            full_caption = full_caption[:self.max_caption_length - 3] + "..."
        
        return full_caption
    
    def _split_message(self, text: str) -> List[str]:
        """Split long message into chunks that fit Telegram's limits"""
        if len(text) <= self.max_message_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If single paragraph is too long, split by sentences
            if len(paragraph) > self.max_message_length:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence = sentence.strip() + '. ' if not sentence.endswith('.') else sentence.strip() + ' '
                    
                    if len(current_chunk) + len(sentence) > self.max_message_length:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            # Single sentence too long, force split
                            chunks.append(sentence[:self.max_message_length])
                    else:
                        current_chunk += sentence
            else:
                if len(current_chunk) + len(paragraph) + 2 > self.max_message_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = paragraph + '\n\n'
                    else:
                        chunks.append(paragraph)
                else:
                    current_chunk += paragraph + '\n\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _send_text_message(self, text: str) -> Dict:
        """Send a single text message"""
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        try:
            response = self.session.post(url, data=data, timeout=10)
            return response.json()
        except Exception as e:
            logging.error(f"Error sending text message: {str(e)}")
            return {'ok': False, 'description': str(e)}
    
    def test_connection(self) -> bool:
        """Test if the bot token and chat ID are valid"""
        try:
            # Test bot token
            url = f"{self.base_url}/getMe"
            response = self.session.get(url, timeout=10)
            bot_info = response.json()
            
            if not bot_info['ok']:
                logging.error(f"Invalid bot token: {bot_info.get('description')}")
                return False
            
            # Test chat ID by sending a simple message
            test_message = "🤖 Telegram connection test successful"
            result = self._send_text_message(test_message)
            
            if result['ok']:
                logging.info("Telegram connection test passed")
                return True
            else:
                logging.error(f"Failed to send test message: {result.get('description')}")
                return False
                
        except Exception as e:
            logging.error(f"Telegram connection test failed: {str(e)}")
            return False
    
    def send_error_notification(self, error_message: str) -> bool:
        """Send error notification to Telegram"""
        try:
            error_text = f"🚨 *Financial Bot Error*\n\n`{error_message}`\n\n_Time: {time.strftime('%Y-%m-%d %H:%M:%S')}_"
            result = self._send_text_message(error_text)
            return result['ok']
        except Exception as e:
            logging.error(f"Failed to send error notification: {str(e)}")
            return False


def send_financial_content_to_telegram(bot_token: str, 
                                     chat_id: str,
                                     content_data: Dict) -> Dict:
    """
    Main function to send financial content with images to Telegram
    
    Args:
        bot_token: Telegram bot token
        chat_id: Telegram chat ID
        content_data: Dictionary containing summary text, images, and language info
        
    Returns:
        Dictionary with sending results
    """
    sender = TelegramSender(bot_token, chat_id)
    
    # Test connection first
    if not sender.test_connection():
        return {
            'success': False,
            'error': 'Failed to connect to Telegram',
            'results': {}
        }
    
    results = {}
    overall_success = True
    
    # Send content for each language
    for language, content in content_data.items():
        if language == 'images':  # Skip images key, it's handled per language
            continue
            
        try:
            summary_text = content.get('summary', '')
            images = content_data.get('images', [])  # Use shared images
            
            result = sender.send_financial_summary_with_images(
                summary_text=summary_text,
                images=images,
                language=language
            )
            
            results[language] = result
            
            if result['success_rate'] < 0.5:  # Less than 50% success
                overall_success = False
            
            # Brief pause between languages
            time.sleep(3)
            
        except Exception as e:
            error_msg = f"Failed to send content for {language}: {str(e)}"
            logging.error(error_msg)
            results[language] = {
                'success': False,
                'error': error_msg
            }
            overall_success = False
    
    return {
        'success': overall_success,
        'results': results,
        'total_languages': len([k for k in content_data.keys() if k != 'images'])
    }