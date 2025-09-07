import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """
    A data class to hold rate limiting parameters for a specific API.
    This provides a structured way to define and access different
    rate-limiting constraints.
    """

    requests_per_minute: int
    requests_per_hour: int
    delay_between_requests: float
    max_retries: int
    backoff_multiplier: float

class APIRateLimiter:
    """
    Manages API calls by enforcing defined rate limits and backoff strategies.
    This class provides methods to check and enforce rate limits before
    making an API call, and to calculate appropriate delays for retries.
    """

    def __init__(self):
        """Initializes rate limit configurations for various APIs."""
        self.api_configs = {
            "gemini": RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                delay_between_requests=6.0,
                max_retries=5,
                backoff_multiplier=2.0,
            ),
            "tavily": RateLimitConfig(
                requests_per_minute=20,
                requests_per_hour=200,
                delay_between_requests=3.0,
                max_retries=3,
                backoff_multiplier=1.5,
            ),
            "serper": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=300,
                delay_between_requests=2.0,
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
        """
        Pauses execution to respect the configured delay between API requests.
        Args:
            api_name: The name of the API being called.
        """
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
        """
        Calculates the appropriate backoff delay for a retry attempt.
        This uses an exponential backoff strategy to prevent overwhelming the API.
        Args:
            api_name: The name of the API.
            attempt: The current retry attempt number.

        Returns:
            The recommended delay in seconds.
        """
        if api_name not in self.api_configs:
            return 2.0**attempt

        config = self.api_configs[api_name]
        return config.delay_between_requests * (config.backoff_multiplier**attempt)


class WorkflowConfig:
    """
    Holds general configuration settings for the entire financial workflow.
    This class includes settings for summary length, search parameters,
    and performance targets, with an ability to optimize them based on tier.
    """

    def __init__(self):
        """Initializes the workflow with default settings and a rate limiter."""
        self.rate_limiter = APIRateLimiter()

        self.settings = {
            "max_summary_words": 500,
            "max_search_results": 8,
            "search_hours_back": 2,
            "max_images": 2,
            "languages": ["arabic", "hindi", "hebrew"],
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

    def get_optimized_settings(self) -> Dict[str, Any]:
        """
        Retrieves settings optimized for the current API usage tier.
        This method adjusts the workflow's intensity to respect API limits,
        especially for free-tier users.
        Returns:
            A dictionary of optimized workflow settings.
        """
        optimized = self.settings.copy()

        if self._is_free_tier():
            optimized.update(
                {
                    "max_search_results": 5,
                    "max_images": 1,
                    "search_hours_back": 1,
                    "languages": ["arabic"],
                }
            )
            logger.info("Using optimized settings for free tier API usage.")

        return optimized

    def _is_free_tier(self) -> bool:
        """
        Determines if the current API key suggests a free-tier account.

        Returns:
            True if the API key length is short, suggesting a free tier.
        """
        google_key = os.getenv("GOOGLE_API_KEY", "")
        return len(google_key) < 50


# Creating a global instance of the rate limiter to be shared across modules
rate_limiter = APIRateLimiter()


def get_workflow_config() -> WorkflowConfig:
    """
    Provides a centralized function to retrieve the workflow configuration.
    This ensures all parts of the application use the same settings.
    """
    return WorkflowConfig()


def apply_rate_limiting(api_name: str):
    """
    A decorator-like function to apply rate limiting to a function.
    This wrapper can be used to easily add rate-limiting logic to any
    API-calling function.
    Args:
        api_name: The name of the API to apply the rate limit for.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            rate_limiter.wait_if_needed(api_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator
