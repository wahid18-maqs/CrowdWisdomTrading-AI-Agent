from dotenv import load_dotenv
import os
import time
import logging
from typing import Dict, Optional
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class RateLimitedLLM:
    """Rate-limited wrapper for Google Gemini LLM"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", api_tier: str = "auto"):
        """
        Initialize rate-limited LLM
        
        Args:
            model_name: Gemini model to use
            api_tier: "free", "paid", or "auto" (auto-detect)
        """
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        self.model_name = model_name
        
        # Auto-detect or set API tier
        self.api_tier = self._detect_api_tier() if api_tier == "auto" else api_tier
        
        # Set rate limits based on tier
        self.rate_limits = self._get_rate_limits()
        
        # Initialize the LangChain wrapper
        self.llm = self._create_llm()
        
        # Track usage
        self.last_request_time = 0
        self.request_count = 0
        self.daily_request_count = 0
        self.current_day = time.strftime("%Y-%m-%d")
        
        logging.info(f"LLM initialized: {model_name} ({self.api_tier} tier)")
        logging.info(f"Rate limits: {self.rate_limits['requests_per_minute']} req/min, {self.rate_limits['requests_per_day']} req/day")
    
    def _detect_api_tier(self) -> str:
        """Auto-detect API tier based on usage patterns"""
        try:
            # Try a simple test request to check quota
            test_model = genai.GenerativeModel(self.model_name)
            
            # Test with minimal content
            response = test_model.generate_content("Test")
            
            if response:
                # If we can make requests quickly, likely paid tier
                # This is a simple heuristic - you might want to refine this
                return "paid"
            else:
                return "free"
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for quota exceeded errors (typical of free tier)
            if "quota" in error_msg or "limit" in error_msg:
                logging.info("Detected free tier based on quota limitations")
                return "free"
            else:
                # Default to free tier for safety
                logging.info("Unable to detect tier, defaulting to free tier limits")
                return "free"
    
    def _get_rate_limits(self) -> Dict:
        """Get rate limits based on API tier"""
        if self.api_tier == "free":
            return {
                "requests_per_minute": 15,    # Conservative for free tier
                "requests_per_day": 1500,     # Daily limit for free tier
                "delay_between_requests": 4,   # 4 seconds between requests
                "max_tokens_per_minute": 32000,
                "max_tokens_per_day": 50000
            }
        else:  # paid tier
            return {
                "requests_per_minute": 60,     # Higher limit for paid
                "requests_per_day": 10000,     # Much higher daily limit
                "delay_between_requests": 1,   # 1 second between requests
                "max_tokens_per_minute": 100000,
                "max_tokens_per_day": 1000000
            }
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create the LangChain LLM wrapper with minimal configuration"""
        try:
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.1,
                max_output_tokens=2048
            )
        except Exception as e:
            logging.error(f"Failed to create LLM with model {self.model_name}: {str(e)}")
            # Try with basic model name
            try:
                return ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=self.api_key,
                    temperature=0.1
                )
            except Exception as e2:
                logging.error(f"Failed to create LLM with fallback model: {str(e2)}")
                raise e2
    
    def invoke(self, messages, **kwargs):
        """Rate-limited invoke method"""
        # Check daily limits
        current_day = time.strftime("%Y-%m-%d")
        if current_day != self.current_day:
            # Reset daily counter
            self.daily_request_count = 0
            self.current_day = current_day
        
        if self.daily_request_count >= self.rate_limits["requests_per_day"]:
            raise Exception(f"Daily request limit exceeded ({self.rate_limits['requests_per_day']})")
        
        # Apply rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limits["delay_between_requests"]:
            sleep_time = self.rate_limits["delay_between_requests"] - time_since_last
            logging.info(f"Rate limiting: waiting {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        # Make the request
        try:
            response = self.llm.invoke(messages, **kwargs)
            
            # Update tracking
            self.last_request_time = time.time()
            self.request_count += 1
            self.daily_request_count += 1
            
            logging.debug(f"LLM request completed. Daily count: {self.daily_request_count}")
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle specific quota errors
            if "quota" in error_msg or "limit" in error_msg:
                if self.api_tier == "paid":
                    logging.warning("Quota exceeded on paid tier, switching to free tier limits")
                    self.api_tier = "free"
                    self.rate_limits = self._get_rate_limits()
                
                # Wait longer on quota errors
                logging.warning("Quota exceeded, waiting 60 seconds before retry")
                time.sleep(60)
                raise Exception("Rate limit exceeded, please retry in a few minutes")
            
            raise e
    
    def get_usage_stats(self) -> Dict:
        """Get current usage statistics"""
        return {
            "api_tier": self.api_tier,
            "total_requests": self.request_count,
            "daily_requests": self.daily_request_count,
            "daily_limit": self.rate_limits["requests_per_day"],
            "requests_remaining": self.rate_limits["requests_per_day"] - self.daily_request_count,
            "rate_limit_per_minute": self.rate_limits["requests_per_minute"],
            "delay_between_requests": self.rate_limits["delay_between_requests"]
        }


class WorkflowConfig:
    """Configuration settings for the entire workflow"""
    
    def __init__(self, api_tier: str = "auto"):
        self.api_tier = api_tier
        
        # Workflow settings based on API tier
        if api_tier == "free":
            self.config = {
                # Search settings
                "max_search_results": 6,        # Fewer articles to process
                "search_hours_back": 3,         # Slightly longer time window
                
                # Summary settings
                "max_summary_words": 450,       # Shorter summaries
                "include_sector_analysis": True,
                "include_tomorrow_watch": True,
                
                # Image settings
                "max_images": 2,                # Limit images
                "image_validation_timeout": 5,  # Shorter timeouts
                
                # Translation settings
                "target_languages": ["Arabic", "Hindi", "Hebrew"],
                "translation_batch_size": 1,    # Process one language at a time
                
                # Execution settings
                "max_execution_time": 900,      # 15 minutes max
                "enable_caching": True,
                "verbose_logging": True
            }
        else:  # paid tier
            self.config = {
                # Search settings
                "max_search_results": 10,       # More articles
                "search_hours_back": 2,         # Recent news
                
                # Summary settings
                "max_summary_words": 500,       # Full-length summaries
                "include_sector_analysis": True,
                "include_tomorrow_watch": True,
                
                # Image settings
                "max_images": 3,                # More images
                "image_validation_timeout": 10,
                
                # Translation settings
                "target_languages": ["Arabic", "Hindi", "Hebrew"],
                "translation_batch_size": 2,    # Process multiple languages
                
                # Execution settings
                "max_execution_time": 600,      # 10 minutes max
                "enable_caching": True,
                "verbose_logging": False
            }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def update(self, updates: Dict):
        """Update configuration values"""
        self.config.update(updates)


# Global instances
_llm_instance = None
_config_instance = None

def get_llm(model_name: str = "gemini-1.5-flash", api_tier: str = "auto") -> RateLimitedLLM:
    """
    Get or create the global LLM instance
    
    Args:
        model_name: Gemini model to use
        api_tier: API tier ("free", "paid", or "auto")
        
    Returns:
        RateLimitedLLM instance
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = RateLimitedLLM(model_name, api_tier)
    
    return _llm_instance

def get_rate_limits() -> Dict:
    """Get current rate limit configuration"""
    llm = get_llm()
    return llm.rate_limits

def get_workflow_config(api_tier: str = "auto") -> WorkflowConfig:
    """
    Get or create the global workflow configuration
    
    Args:
        api_tier: API tier to configure for
        
    Returns:
        WorkflowConfig instance
    """
    global _config_instance
    
    if _config_instance is None:
        # Auto-detect tier from LLM if needed
        if api_tier == "auto":
            llm = get_llm()
            api_tier = llm.api_tier
        
        _config_instance = WorkflowConfig(api_tier)
    
    return _config_instance

def get_usage_stats() -> Dict:
    """Get current LLM usage statistics"""
    llm = get_llm()
    return llm.get_usage_stats()

def test_llm_connection() -> bool:
    """Test LLM connection and configuration"""
    try:
        llm = get_llm()
        
        # Test with a simple request
        test_response = llm.invoke("Respond with 'Connection test successful'")
        
        if test_response and "successful" in str(test_response).lower():
            logging.info("LLM connection test passed")
            return True
        else:
            logging.error("LLM connection test failed - unexpected response")
            return False
            
    except Exception as e:
        logging.error(f"LLM connection test failed: {str(e)}")
        return False

def reset_usage_tracking():
    """Reset usage tracking (useful for testing)"""
    global _llm_instance
    if _llm_instance:
        _llm_instance.request_count = 0
        _llm_instance.daily_request_count = 0
        _llm_instance.current_day = time.strftime("%Y-%m-%d")
        logging.info("Usage tracking reset")

# Configuration constants
DEFAULT_MODEL = "gemini-1.5-flash"
FALLBACK_MODEL = "gemini-pro"  # Fallback model

# Model parameters optimized for financial analysis
MODEL_PARAMS = {
    'temperature': 0.1,           # Low for consistent analysis
    'max_output_tokens': 2048,    # Sufficient for summaries
    'top_p': 0.8,                # Focused responses
    'top_k': 40                   # Balanced creativity
}