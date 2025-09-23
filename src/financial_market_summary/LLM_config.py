import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting parameters for API calls."""
    requests_per_minute: int
    requests_per_hour: int
    delay_between_requests: float
    max_retries: int
    backoff_multiplier: float


@dataclass
class ValidationConfig:
    """Configuration for summary validation thresholds."""
    min_confidence_score: int = 60
    min_source_reliability: bool = True
    require_stock_verification: bool = True
    require_number_accuracy: bool = True
    max_hallucination_threshold: float = 0.7
    enable_validation_logging: bool = True


class APIRateLimiter:
    """Manages API calls with rate limiting and validation tracking."""

    def __init__(self):
        """Initialize rate limit configurations and validation tracking."""
        self.api_configs = {
            "gemini": RateLimitConfig(
                requests_per_minute=10,  # More conservative
                requests_per_hour=100,
                delay_between_requests=8.0,  # Longer delays
                max_retries=5,
                backoff_multiplier=2.5,  # Higher backoff
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
        self.validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "average_confidence": 0.0,
            "validation_history": []
        }

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

    def track_validation_result(self, validation_result: Dict[str, Any]):
        """Track validation statistics for monitoring accuracy."""
        confidence_score = validation_result.get("confidence_score", 0)
        passed = validation_result.get("confidence_score", 0) >= 60
        
        self.validation_stats["total_validations"] += 1
        if passed:
            self.validation_stats["passed_validations"] += 1
        else:
            self.validation_stats["failed_validations"] += 1
            
        # Update rolling average confidence
        total_validations = self.validation_stats["total_validations"]
        current_avg = self.validation_stats["average_confidence"]
        new_avg = (current_avg * (total_validations - 1) + confidence_score) / total_validations
        self.validation_stats["average_confidence"] = new_avg
        
        # Store validation history (keep last 50)
        validation_entry = {
            "timestamp": datetime.now().isoformat(),
            "confidence_score": confidence_score,
            "passed": passed,
            "details": validation_result
        }
        self.validation_stats["validation_history"].append(validation_entry)
        
        # Keep only last 50 entries
        if len(self.validation_stats["validation_history"]) > 50:
            self.validation_stats["validation_history"] = self.validation_stats["validation_history"][-50:]
        
        logger.info(f"Validation tracked: Score {confidence_score}/100, Pass rate: {self.get_pass_rate():.1f}%")

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics."""
        return self.validation_stats.copy()

    def get_pass_rate(self) -> float:
        """Calculate current validation pass rate percentage."""
        if self.validation_stats["total_validations"] == 0:
            return 0.0
        return (self.validation_stats["passed_validations"] / self.validation_stats["total_validations"]) * 100


class WorkflowConfig:
    """Enhanced workflow configuration with validation settings."""

    def __init__(self):
        """Initialize workflow with enhanced validation settings."""
        self.rate_limiter = APIRateLimiter()
        self.validation_config = ValidationConfig()

        self.settings = {
            "max_summary_words": 500,
            "max_search_results": 8,
            "search_hours_back": 1,  # Enforced 1-hour limit
            "max_images": 2,
            "telegram_max_message_length": 4096,
            "enable_fallbacks": True,
            "save_outputs_locally": True,
            
            # Enhanced validation settings
            "enable_validation": True,
            "validation_retry_on_low_confidence": True,
            "log_validation_failures": True,
            "require_source_verification": True,
        }

        self.performance_targets = {
            "max_total_execution_time": 600,
            "max_translation_time": 120,
            "max_search_time": 180,
            "max_formatting_time": 120,
            "max_validation_time": 60,  # New validation time limit
        }

        # Enhanced API usage tracking
        self.api_usage_stats = {
            "gemini_calls": 0,
            "tavily_calls": 0,
            "serper_calls": 0,
            "telegram_calls": 0,
            "validation_calls": 0,  # Track validation-specific calls
        }

    def get_optimized_settings(self) -> Dict[str, Any]:
        """Retrieve settings optimized for current API usage tier and validation."""
        optimized = self.settings.copy()

        if self._is_free_tier():
            # More conservative settings for free tier
            optimized.update({
                "max_search_results": 5,
                "max_images": 1,
                "search_hours_back": 1,
                "validation_retry_on_low_confidence": False,  # Reduce retries
            })
            logger.info("Using optimized settings for free tier API usage.")
        else:
            # Enhanced settings for paid tier
            optimized.update({
                "max_search_results": 10,
                "max_images": 3,
                "validation_retry_on_low_confidence": True,
                })
            logger.info("Using enhanced settings for paid tier API usage.")

        return optimized

    def update_validation_config(self, **kwargs):
        """Update validation configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.validation_config, key):
                setattr(self.validation_config, key, value)
                logger.info(f"Updated validation config: {key} = {value}")

    def track_api_usage(self, api_name: str):
        """Track API usage for monitoring and optimization."""
        usage_key = f"{api_name}_calls"
        if usage_key in self.api_usage_stats:
            self.api_usage_stats[usage_key] += 1
        
        # Special tracking for validation calls
        if api_name == "gemini":
            self.api_usage_stats["validation_calls"] += 1

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage and performance summary."""
        validation_stats = self.rate_limiter.get_validation_stats()
        
        return {
            "api_usage": self.api_usage_stats.copy(),
            "validation_stats": validation_stats,
            "current_tier": "paid" if not self._is_free_tier() else "free",
            "settings": self.get_optimized_settings(),
            "performance_summary": {
                "total_workflows": validation_stats.get("total_validations", 0),
                "success_rate": self.rate_limiter.get_pass_rate(),
                "average_confidence": validation_stats.get("average_confidence", 0),
            }
        }

    def _is_free_tier(self) -> bool:
        """Determine if current API key suggests free-tier account."""
        google_key = os.getenv("GOOGLE_API_KEY", "")
        return len(google_key) < 50

    def reset_stats(self):
        """Reset usage and validation statistics."""
        self.api_usage_stats = {key: 0 for key in self.api_usage_stats}
        self.rate_limiter.validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "average_confidence": 0.0,
            "validation_history": []
        }
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


def track_validation(validation_result: Dict[str, Any]):
    """Track validation results for monitoring and improvement."""
    rate_limiter.track_validation_result(validation_result)


def get_validation_config() -> ValidationConfig:
    """Get current validation configuration."""
    return workflow_config.validation_config


def log_performance_summary():
    """Log comprehensive performance and validation summary."""
    summary = workflow_config.get_usage_summary()
    
    logger.info("=== PERFORMANCE SUMMARY ===")
    logger.info(f"API Calls: {summary['api_usage']}")
    logger.info(f"Validation Pass Rate: {summary['performance_summary']['success_rate']:.1f}%")
    logger.info(f"Average Confidence: {summary['performance_summary']['average_confidence']:.1f}/100")
    logger.info(f"Total Workflows: {summary['performance_summary']['total_workflows']}")
    logger.info(f"Current Tier: {summary['current_tier']}")
    logger.info("=== END SUMMARY ===")


def is_validation_enabled() -> bool:
    """Check if validation is currently enabled."""
    return workflow_config.settings.get("enable_validation", True)


def should_retry_on_low_confidence() -> bool:
    """Check if workflow should retry on low confidence scores."""
    return workflow_config.settings.get("validation_retry_on_low_confidence", True)


# Validation helper functions
def meets_confidence_threshold(confidence_score: int) -> bool:
    """Check if confidence score meets minimum threshold."""
    return confidence_score >= workflow_config.validation_config.min_confidence_score


def log_validation_failure(validation_result: Dict[str, Any], summary: str):
    """Log detailed information about validation failures."""
    if workflow_config.settings.get("log_validation_failures", True):
        logger.warning("=== VALIDATION FAILURE ===")
        logger.warning(f"Confidence Score: {validation_result.get('confidence_score', 0)}/100")
        logger.warning(f"Failed Checks: {[k for k, v in validation_result.items() if not v and k != 'confidence_score']}")
        logger.warning(f"Summary Preview: {summary[:200]}...")
        logger.warning("=== END VALIDATION FAILURE ===")