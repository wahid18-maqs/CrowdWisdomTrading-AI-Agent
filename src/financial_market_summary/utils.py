import os
import logging
import json
import yaml
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

class FinancialLogger:
    """
    Enhanced logging utility for the financial AI agent.
    This class provides a comprehensive and structured logging solution to
    track the workflow, errors, and performance of the agent.
    """

    @staticmethod
    def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
        """
        Sets up a comprehensive logging configuration for the application.

        Args:
            log_level (int): The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
            log_file (Optional[str]): The path to the log file. A timestamped file is created by default.

        Returns:
            logging.Logger: The configured logger instance.
        """
        # Creating the logs directory if it does not exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Set the default log file name with a timestamp
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"financial_agent_{timestamp}.log"

        # Define the logging format for all handlers
        log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

        # Configure the root logger to output to both a file and the console
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        # Create a specific logger for the financial agent to allow for more granular control
        logger = logging.getLogger("FinancialAgent")
        logger.setLevel(log_level)

        # Add a dedicated handler to capture all ERROR level logs in a separate file
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(error_handler)

        return logger
    
    @staticmethod
    def log_flow_step(logger: logging.Logger, step_name: str, status: str, details: Optional[Dict[str, Any]] = None):
        """
        Logs a specific step in the workflow with structured information.

        This method standardizes the logging format for different stages of the
        agent's execution, making it easier to parse and monitor the log file.

        Args:
            logger (logging.Logger): The logger instance to use.
            step_name (str): The name of the current step in the workflow.
            status (str): The status of the step (e.g., "success", "error", "in_progress").
            details (Optional[Dict[str, Any]]): A dictionary of additional
                details to include in the log entry.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step_name,
            "status": status,
            "details": details or {}
        }

        # Log based on the status provided, using a clear message instead of emojis
        if status.lower() == "success":
            logger.info(f"Workflow step completed successfully: {step_name} - {json.dumps(details, default=str)}")
        elif status.lower() == "error":
            logger.error(f"Workflow step encountered an error: {step_name} - {json.dumps(details, default=str)}")
        else:
            logger.info(f"Workflow step is in progress: {step_name} - {json.dumps(details, default=str)}")

class ConfigManager:
    """
    A utility class for managing application configurations.
    This class handles loading and saving configuration files, and retrieving
    API keys from environment variables.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initializes the ConfigManager and creates the configuration directory.

        Args:
            config_dir (str): The name of the directory to store configuration files.
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Loads configuration data from a YAML file.

        Args:
            config_name (str): The name of the configuration file (without extension).

        Returns:
            Dict[str, Any]: The loaded configuration data as a dictionary.
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            return {}
        
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Error loading configuration file '{config_name}': {e}")
            return {}
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]):
        """
        Saves configuration data to a YAML file.

        Args:
            config_name (str): The name of the configuration file.
            config_data (Dict[str, Any]): The configuration data to save.
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving configuration file '{config_name}': {e}")
    
    def get_api_keys(self) -> Dict[str, str]:
        """
        Retrieves API keys from environment variables.
        Returns:
            Dict[str, str]: A dictionary of API keys found in the environment.
        """
        api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'tavily': os.getenv('TAVILY_API_KEY'),
            'serper': os.getenv('SERPER_API_KEY'),
            'groq': os.getenv('GROQ_API_KEY'),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'google_translate': os.getenv('GOOGLE_TRANSLATE_API_KEY')
        }
        
        # Filter out keys that were not found in the environment
        return {k: v for k, v in api_keys.items() if v is not None}
    
    def validate_required_keys(self, required_keys: List[str]) -> Dict[str, bool]:
        """
        Validates that a list of required API keys are present in the environment.
        Args:
            required_keys (List[str]): A list of key names to check.
        Returns:
            Dict[str, bool]: A dictionary mapping each required key to a boolean
                indicating if it is present.
        """
        api_keys = self.get_api_keys()
        validation = {}
        
        for key in required_keys:
            validation[key] = key in api_keys and api_keys[key] is not None
        
        return validation

class MarketTimeUtils:
    """
    Utility functions for handling market timing and schedules.
    This class provides methods to check if the market is open and to
    retrieve relevant time-based information.
    """
    
    @staticmethod
    def is_market_open() -> bool:
        """
        Checks if the US stock market is currently open.
        The US market operates from 9:30 AM to 4:00 PM ET, Monday to Friday.
        Returns:
            bool: True if the market is open, False otherwise.
        """
        from datetime import datetime, time
        import pytz
        # Use US Eastern Time for market hours
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        market_open_time = time(9, 30)
        market_close_time = time(16, 0)
        
        # Check if it is a weekday and within the market's operating hours
        is_weekday = now.weekday() < 5
        is_market_hours = market_open_time <= now.time() <= market_close_time
        return is_weekday and is_market_hours
    
    @staticmethod
    def get_market_status() -> Dict[str, Any]:
        """
        Provides a comprehensive report on the current market status.

        Returns:
            Dict[str, Any]: A dictionary containing market status details,
                including current times in both ET and IST, and the market's day of the week.
        """
        from datetime import datetime
        import pytz
        et = pytz.timezone('US/Eastern')
        ist = pytz.timezone('Asia/Kolkata')
        now_et = datetime.now(et)
        now_ist = datetime.now(ist)
        return {
            "is_open": MarketTimeUtils.is_market_open(),
            "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "current_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "is_weekday": now_et.weekday() < 5,
            "market_day": now_et.strftime("%A"),
            "suggested_run_time": "01:30 AM IST (after US market close)"
        }
    
    @staticmethod
    def get_next_market_close() -> datetime:
        """
        Calculates and returns the time of the next US market close.

        Returns:
            datetime: The datetime object representing the next market close time.
        """
        from datetime import datetime, time, timedelta
        import pytz
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        market_close_time = time(16, 0)
        if now.time() > market_close_time or now.weekday() >= 5:
            days_ahead = 1
            while (now + timedelta(days=days_ahead)).weekday() >= 5:
                days_ahead += 1
            next_close = (now + timedelta(days=days_ahead)).replace(
                hour=16, minute=0, second=0, microsecond=0
            )
        else:
            next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return next_close

class OutputFormatter:
    """
    Utility for formatting and saving the agent's outputs.
    """
    @staticmethod
    def save_summary_output(summary_data: Dict[str, Any], output_dir: str = "outputs"):
        """
        Saves the financial summary and its translations in various formats.
        Args:
            summary_data (Dict[str, Any]): A dictionary containing the summary content
                in different languages.
            output_dir (str): The directory to save the output files.
        Returns:
            Dict[str, str]: A dictionary with information about the saved files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_path / f"financial_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
        if 'english' in summary_data:
            text_file = output_path / f"summary_english_{timestamp}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(summary_data['english'])  
        for lang, content in summary_data.items():
            if lang != 'english' and isinstance(content, str):
                lang_file = output_path / f"summary_{lang}_{timestamp}.txt"
                with open(lang_file, 'w', encoding='utf-8') as f:
                    f.write(content)
        return {
            'json_file': str(json_file),
            'output_dir': str(output_path),
            'timestamp': timestamp
        }
    @staticmethod
    def create_sample_inputs_outputs():
        """
        Creates sample input and output files for demonstration purposes.
        """
        samples_dir = Path("samples")
        samples_dir.mkdir(exist_ok=True)
        
        sample_input = {
            "search_query": "US financial news stock market trading",
            "time_range": "last 1 hour",
            "target_languages": ["arabic", "hindi", "hebrew"],
            "max_summary_words": 500
        }
        sample_output = {
            "timestamp": "2024-01-15T18:30:00Z",
            "market_status": "closed",
            "english_summary": "Sample financial market summary...",
            "translations": {
                "arabic": "ملخص السوق المالي...",
                "hindi": "वित्तीय बाजार सारांश...",
                "hebrew": "סיכום שוק כספי..."
            },
            "images_found": 2,
            "telegram_delivery": "success",
            "execution_time_seconds": 120
        }
        with open(samples_dir / "sample_input.json", 'w', encoding="utf-8") as f:
            json.dump(sample_input, f, indent=2, ensure_ascii=False)
        
        with open(samples_dir / "sample_output.json", 'w', encoding="utf-8") as f:
            json.dump(sample_output, f, indent=2, ensure_ascii=False)
        return str(samples_dir)
    
class ErrorHandler:
    """
    A centralized utility for handling and reporting errors.
    """
    
    @staticmethod
    def handle_api_error(api_name: str, error: Exception) -> str:
        """
        Handles API-specific errors and returns a user-friendly message.
        Args:
            api_name (str): The name of the API that failed.
            error (Exception): The exception object.
        Returns:
            str: A detailed error message including a suggested fallback or action.
        """
        error_messages = {
            'tavily': "Tavily search failed. Trying Serper as fallback.",
            'serper': "Serper search failed. Using cached financial data.",
            'Google Gemini': "Gemini API error. Check API key and quota.",
            'telegram': "Telegram delivery failed. Content saved locally.",
        }
        base_message = error_messages.get(api_name, f"{api_name} API error")
        detailed_error = f"{base_message} Details: {str(error)}"
        
        logging.error(detailed_error)
        return detailed_error
    
    @staticmethod
    def create_error_report(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Creates a comprehensive error report from a list of logged errors.
        Args:
            errors (List[Dict[str, Any]]): A list of error dictionaries.
        Returns:
            Dict[str, Any]: A summary report of all errors.
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_errors": len(errors),
            "errors": errors,
            "severity_breakdown": {
                "critical": len([e for e in errors if e.get('severity') == 'critical']),
                "warning": len([e for e in errors if e.get('severity') == 'warning']),
                "info": len([e for e in errors if e.get('severity') == 'info'])
            }
        }

class PerformanceMonitor:
    """
    A utility class for monitoring and tracking performance metrics of the agent.
    """
    def __init__(self):
        """
        Initializes the PerformanceMonitor with default metric values.
        """
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'step_timings': {},
            'api_calls': {},
            'error_count': 0
        }
    
    def start_monitoring(self):
        """Starts the performance monitoring session."""
        self.metrics['start_time'] = datetime.now()
    
    def end_monitoring(self):
        """Ends the performance monitoring session and returns a final report."""
        self.metrics['end_time'] = datetime.now()
        return self.get_performance_report()
    
    def track_step(self, step_name: str, duration: float):
        """
        Tracks the execution duration of a specific step.
        Args:
            step_name (str): The name of the step.
            duration (float): The duration in seconds.
        """
        self.metrics['step_timings'][step_name] = duration
    
    def track_api_call(self, api_name: str, duration: float, success: bool):
        """
        Tracks the performance of an API call.
        Args:
            api_name (str): The name of the API.
            duration (float): The duration of the call in seconds.
            success (bool): A boolean indicating if the call was successful.
        """
        if api_name not in self.metrics['api_calls']:
            self.metrics['api_calls'][api_name] = []
        
        self.metrics['api_calls'][api_name].append({
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def increment_error(self):
        """Increments the error counter."""
        self.metrics['error_count'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive performance report.

        Returns:
            Dict[str, Any]: A dictionary containing various performance metrics.
        """
        total_duration = 0
        if self.metrics['start_time'] and self.metrics['end_time']:
            total_duration = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
        
        return {
            'total_execution_time': total_duration,
            'step_timings': self.metrics['step_timings'],
            'api_performance': self.metrics['api_calls'],
            'error_count': self.metrics['error_count'],
            'success_rate': self._calculate_success_rate(),
            'bottleneck_analysis': self._identify_bottlenecks()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculates the overall success rate of all API calls."""
        total_calls = sum(len(calls) for calls in self.metrics['api_calls'].values())
        if total_calls == 0:
            return 100.0  
        successful_calls = sum(
            len([call for call in calls if call['success']]) 
            for calls in self.metrics['api_calls'].values()
        )
        
        return (successful_calls / total_calls) * 100
    
    def _identify_bottlenecks(self) -> Dict[str, Any]:
        """Identifies potential performance bottlenecks by finding the slowest steps and APIs."""
        bottlenecks = {}
        
        if self.metrics['step_timings']:
            slowest_step = max(self.metrics['step_timings'].items(), key=lambda x: x[1])
            bottlenecks['slowest_step'] = {
                'step': slowest_step[0],
                'duration': slowest_step[1]
            }
        
        api_averages = {}
        for api, calls in self.metrics['api_calls'].items():
            if calls:
                avg_duration = sum(call['duration'] for call in calls) / len(calls)
                api_averages[api] = avg_duration
        
        if api_averages:
            slowest_api = max(api_averages.items(), key=lambda x: x[1])
            bottlenecks['slowest_api'] = {
                'api': slowest_api[0],
                'average_duration': slowest_api[1]
            }
        
        return bottlenecks
    
    
