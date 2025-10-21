import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting parameters for API calls."""
    requests_per_minute: int
    requests_per_hour: int
    delay_between_requests: float
    max_retries: int
    backoff_multiplier: float


class APIRateLimiter:
    """Manages API calls with rate limiting."""

    def __init__(self):
        """Initialize rate limit configurations."""
        self.api_configs = {
            "gemini": RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                delay_between_requests=8.0,
                max_retries=5,
                backoff_multiplier=2.5,
            ),
            "tavily": RateLimitConfig(
                requests_per_minute=20,
                requests_per_hour=200,
                delay_between_requests=3.0,
                max_retries=3,
                backoff_multiplier=1.5,
            ),
            "telegram": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=1000,
                delay_between_requests=1.0,
                max_retries=3,
                backoff_multiplier=2.0,
            ),
        }
        self.last_request_times = {}

    def wait_if_needed(self, api_name: str):
        """Pauses execution to respect configured delay between API requests."""
        if api_name not in self.api_configs:
            return

        config = self.api_configs[api_name]
        current_time = time.time()

        if api_name in self.last_request_times:
            time_since_last = current_time - self.last_request_times[api_name]
            if time_since_last < config.delay_between_requests:
                wait_time = config.delay_between_requests - time_since_last
                logger.info(f"Rate limiting for {api_name}: waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        self.last_request_times[api_name] = time.time()

    def get_retry_delay(self, api_name: str, attempt: int) -> float:
        """Calculates appropriate backoff delay for retry attempts."""
        if api_name not in self.api_configs:
            return 2.0**attempt

        config = self.api_configs[api_name]
        return config.delay_between_requests * (config.backoff_multiplier**attempt)


class WorkflowConfig:
    """Workflow configuration and API usage tracking."""

    def __init__(self):
        """Initialize workflow configuration."""
        self.rate_limiter = APIRateLimiter()

        self.settings = {
            "max_summary_words": 500,
            "max_search_results": 8,
            "search_hours_back": 1,
            "max_images": 2,
            "telegram_max_message_length": 4096,
            "enable_fallbacks": True,
            "save_outputs_locally": True,
        }

        self.performance_targets = {
            "max_total_execution_time": 600,
            "max_translation_time": 120,
            "max_search_time": 180,
            "max_formatting_time": 120,
        }

        self.api_usage_stats = {
            "gemini_calls": 0,
            "tavily_calls": 0,
            "telegram_calls": 0,
        }

    def get_optimized_settings(self) -> Dict[str, Any]:
        """Retrieve settings optimized for current API usage tier."""
        optimized = self.settings.copy()

        if self._is_free_tier():
            optimized.update({
                "max_search_results": 5,
                "max_images": 1,
                "search_hours_back": 1,
            })
            logger.info("Using optimized settings for free tier API usage.")
        else:
            optimized.update({
                "max_search_results": 10,
                "max_images": 3,
            })
            logger.info("Using enhanced settings for paid tier API usage.")

        return optimized

    def track_api_usage(self, api_name: str):
        """Track API usage for monitoring and optimization."""
        usage_key = f"{api_name}_calls"
        if usage_key in self.api_usage_stats:
            self.api_usage_stats[usage_key] += 1

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        return {
            "api_usage": self.api_usage_stats.copy(),
            "current_tier": "paid" if not self._is_free_tier() else "free",
            "settings": self.get_optimized_settings(),
        }

    def _is_free_tier(self) -> bool:
        """Determine if current API key suggests free-tier account."""
        google_key = os.getenv("GOOGLE_API_KEY", "")
        return len(google_key) < 50

    def reset_stats(self):
        """Reset usage statistics."""
        self.api_usage_stats = {key: 0 for key in self.api_usage_stats}
        logger.info("Statistics reset successfully.")


# Global instances
rate_limiter = APIRateLimiter()
workflow_config = WorkflowConfig()


def get_workflow_config() -> WorkflowConfig:
    """Provide centralized access to workflow configuration."""
    return workflow_config


def apply_rate_limiting(api_name: str):
    """Decorator to apply rate limiting to API functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            rate_limiter.wait_if_needed(api_name)
            workflow_config.track_api_usage(api_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator
