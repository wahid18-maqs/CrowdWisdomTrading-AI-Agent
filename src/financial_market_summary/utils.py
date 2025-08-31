import os
import logging
import json
import yaml
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import requests
from pathlib import Path
import sys

class FinancialLogger:
    """Enhanced logging utility for financial AI agent"""
    
    @staticmethod
    def setup_logging(log_level=logging.INFO, log_file: Optional[str] = None):
        """Setup comprehensive logging configuration"""
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Default log file with timestamp
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"financial_agent_{timestamp}.log"
        
        # Configure logging format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create logger for the financial agent
        logger = logging.getLogger("FinancialAgent")
        logger.setLevel(log_level)
        
        # Add custom handlers for different log levels
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(error_handler)
        
        return logger
    
    @staticmethod
    def log_flow_step(logger, step_name: str, status: str, details: Dict[str, Any] = None):
        """Log flow step with structured information"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step_name,
            "status": status,
            "details": details or {}
        }
        
        if status.lower() == "success":
            logger.info(f"✅ {step_name}: {status} - {json.dumps(details, default=str)}")
        elif status.lower() == "error":
            logger.error(f"❌ {step_name}: {status} - {json.dumps(details, default=str)}")
        else:
            logger.info(f"🔄 {step_name}: {status} - {json.dumps(details, default=str)}")

class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            return {}
        
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Error loading config {config_name}: {e}")
            return {}
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]):
        """Save configuration to YAML file"""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving config {config_name}: {e}")
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get all API keys from environment variables"""
        api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'tavily': os.getenv('TAVILY_API_KEY'),
            'serper': os.getenv('SERPER_API_KEY'),
            'groq': os.getenv('GROQ_API_KEY'),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'google_translate': os.getenv('GOOGLE_TRANSLATE_API_KEY')
        }
        
        # Filter out None values
        return {k: v for k, v in api_keys.items() if v is not None}
    
    def validate_required_keys(self, required_keys: List[str]) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        api_keys = self.get_api_keys()
        validation = {}
        
        for key in required_keys:
            validation[key] = key in api_keys and api_keys[key] is not None
        
        return validation

class MarketTimeUtils:
    """Utility functions for market timing and schedules"""
    
    @staticmethod
    def is_market_open() -> bool:
        """Check if US markets are currently open"""
        from datetime import datetime, time
        import pytz
        
        # US Eastern Time
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Market hours: 9:30 AM to 4:00 PM ET, Monday to Friday
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        # Check if it's a weekday and within market hours
        is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
        is_market_hours = market_open <= now.time() <= market_close
        
        return is_weekday and is_market_hours
    
    @staticmethod
    def get_market_status() -> Dict[str, Any]:
        """Get comprehensive market status information"""
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
        """Get the next US market close time"""
        from datetime import datetime, time, timedelta
        import pytz
        
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Market closes at 4:00 PM ET
        market_close_time = time(16, 0)
        
        # If it's already past market close today or weekend, get next weekday close
        if now.time() > market_close_time or now.weekday() >= 5:
            # Find next weekday
            days_ahead = 1
            while (now + timedelta(days=days_ahead)).weekday() >= 5:
                days_ahead += 1
            next_close = (now + timedelta(days=days_ahead)).replace(
                hour=16, minute=0, second=0, microsecond=0
            )
        else:
            # Today's close
            next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return next_close

class OutputFormatter:
    """Utility for formatting and saving outputs"""
    
    @staticmethod
    def save_summary_output(summary_data: Dict[str, Any], output_dir: str = "outputs"):
        """Save summary outputs in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = output_path / f"financial_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Save English summary as text
        if 'english' in summary_data:
            text_file = output_path / f"summary_english_{timestamp}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(summary_data['english'])
        
        # Save translations as separate files
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
        """Create sample input/output examples for demonstration"""
        samples_dir = Path("samples")
        samples_dir.mkdir(exist_ok=True)
        
        # Sample input
        sample_input = {
            "search_query": "US financial news stock market trading",
            "time_range": "last 1 hour",
            "target_languages": ["arabic", "hindi", "hebrew"],
            "max_summary_words": 500
        }
        
        # Sample output structure
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
        
        # Save samples with UTF-8 encoding
        with open(samples_dir / "sample_input.json", 'w', encoding="utf-8") as f:
            json.dump(sample_input, f, indent=2, ensure_ascii=False)
        
        with open(samples_dir / "sample_output.json", 'w', encoding="utf-8") as f:
            json.dump(sample_output, f, indent=2, ensure_ascii=False)
        
        return str(samples_dir)

class ErrorHandler:
    """Centralized error handling utility"""
    
    @staticmethod
    def handle_api_error(api_name: str, error: Exception) -> str:
        """Handle API-specific errors with appropriate fallbacks"""
        error_messages = {
            'tavily': "Tavily search failed. Trying Serper as fallback.",
            'serper': "Serper search failed. Using cached financial data.",
            'openai': "OpenAI API error. Check API key and quota.",
            'telegram': "Telegram delivery failed. Content saved locally.",
            'groq': "Groq API unavailable. Using primary LLM."
        }
        
        base_message = error_messages.get(api_name, f"{api_name} API error")
        detailed_error = f"{base_message} Details: {str(error)}"
        
        logging.error(detailed_error)
        return detailed_error
    
    @staticmethod
    def create_error_report(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive error report"""
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
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'step_timings': {},
            'api_calls': {},
            'error_count': 0
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.metrics['start_time'] = datetime.now()
    
    def end_monitoring(self):
        """End performance monitoring"""
        self.metrics['end_time'] = datetime.now()
        return self.get_performance_report()
    
    def track_step(self, step_name: str, duration: float):
        """Track individual step performance"""
        self.metrics['step_timings'][step_name] = duration
    
    def track_api_call(self, api_name: str, duration: float, success: bool):
        """Track API call performance"""
        if api_name not in self.metrics['api_calls']:
            self.metrics['api_calls'][api_name] = []
        
        self.metrics['api_calls'][api_name].append({
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def increment_error(self):
        """Increment error counter"""
        self.metrics['error_count'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
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
        """Calculate overall success rate"""
        total_calls = sum(len(calls) for calls in self.metrics['api_calls'].values())
        if total_calls == 0:
            return 100.0
        
        successful_calls = sum(
            len([call for call in calls if call['success']]) 
            for calls in self.metrics['api_calls'].values()
        )
        
        return (successful_calls / total_calls) * 100
    
    def _identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify performance bottlenecks"""
        bottlenecks = {}
        
        # Find slowest step
        if self.metrics['step_timings']:
            slowest_step = max(self.metrics['step_timings'].items(), key=lambda x: x[1])
            bottlenecks['slowest_step'] = {
                'step': slowest_step[0],
                'duration': slowest_step[1]
            }
        
        # Find slowest API
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