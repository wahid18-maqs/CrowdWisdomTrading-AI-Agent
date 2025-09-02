# config.py - Rate limiting and API management configuration
import os
import time
import logging
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting"""
    requests_per_minute: int
    requests_per_hour: int
    delay_between_requests: float
    max_retries: int
    backoff_multiplier: float

class APIRateLimiter:
    """Rate limiter to manage API calls within quotas"""
    
    def __init__(self):
        self.api_configs = {
            'gemini': RateLimitConfig(
                requests_per_minute=10,  # Conservative for free tier
                requests_per_hour=100,
                delay_between_requests=6.0,  # 6 seconds between calls
                max_retries=5,
                backoff_multiplier=2.0
            ),
            'tavily': RateLimitConfig(
                requests_per_minute=20,
                requests_per_hour=200,
                delay_between_requests=3.0,
                max_retries=3,
                backoff_multiplier=1.5
            ),
            'serper': RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=300,
                delay_between_requests=2.0,
                max_retries=3,
                backoff_multiplier=1.5
            ),
            'telegram': RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=1000,
                delay_between_requests=1.0,
                max_retries=3,
                backoff_multiplier=2.0
            )
        }
        
        # Track API usage
        self.api_usage = {}
        self.last_request_times = {}
    
    def wait_if_needed(self, api_name: str):
        """Wait if necessary to respect rate limits"""
        if api_name not in self.api_configs:
            return
        
        config = self.api_configs[api_name]
        current_time = time.time()
        
        # Check if we need to wait based on last request
        if api_name in self.last_request_times:
            time_since_last = current_time - self.last_request_times[api_name]
            if time_since_last < config.delay_between_requests:
                wait_time = config.delay_between_requests - time_since_last
                logger.info(f"Rate limiting {api_name}: waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Update last request time
        self.last_request_times[api_name] = time.time()
    
    def get_retry_delay(self, api_name: str, attempt: int) -> float:
        """Get delay for retry attempt"""
        if api_name not in self.api_configs:
            return 2.0 ** attempt
        
        config = self.api_configs[api_name]
        return config.delay_between_requests * (config.backoff_multiplier ** attempt)

class WorkflowConfig:
    """Configuration for the entire workflow"""
    
    def __init__(self):
        self.rate_limiter = APIRateLimiter()
        
        # Workflow settings
        self.settings = {
            'max_summary_words': 500,
            'max_search_results': 8,  # Reduced to limit processing
            'search_hours_back': 2,
            'max_images': 2,
            'languages': ['arabic', 'hindi', 'hebrew'],
            'telegram_max_message_length': 4096,
            'enable_fallbacks': True,
            'save_outputs_locally': True
        }
        
        # Performance monitoring
        self.performance_targets = {
            'max_total_execution_time': 600,  # 10 minutes
            'max_translation_time': 120,     # 2 minutes per language
            'max_search_time': 180,          # 3 minutes
            'max_formatting_time': 120       # 2 minutes
        }
    
    def get_optimized_settings(self) -> Dict[str, Any]:
        """Get settings optimized for current API quotas"""
        # Check available API quotas and adjust settings
        optimized = self.settings.copy()
        
        # If using free tier, reduce intensity
        if self._is_free_tier():
            optimized.update({
                'max_search_results': 5,
                'max_images': 1,
                'search_hours_back': 1,
                'languages': ['arabic']  # Start with one language for testing
            })
            logger.info("Using optimized settings for free tier API usage")
        
        return optimized
    
    def _is_free_tier(self) -> bool:
        """Check if we're using free tier APIs"""
        # Simple heuristic: if GOOGLE_API_KEY is short, likely free tier
        google_key = os.getenv('GOOGLE_API_KEY', '')
        return len(google_key) < 50  # Adjust based on your key format

# Global rate limiter instance
rate_limiter = APIRateLimiter()

def get_workflow_config() -> WorkflowConfig:
    """Get the current workflow configuration"""
    return WorkflowConfig()

def apply_rate_limiting(api_name: str):
    """Decorator-like function to apply rate limiting"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            rate_limiter.wait_if_needed(api_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator