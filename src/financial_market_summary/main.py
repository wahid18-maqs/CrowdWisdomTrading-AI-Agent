# Fixed main.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_env_keys():
    """Check for required environment variables"""
    required_keys = ["GOOGLE_API_KEY", "TAVILY_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SERPER_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        logger.error(f"Missing required API keys: {missing_keys}")
        logger.error("Please set these in your .env file:")
        for key in missing_keys:
            logger.error(f"  {key}=your_key_here")
        return False
    
    logger.info("All required API keys found")
    return True

def run_financial_summary():
    """Main function to run the financial summary"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for all required API keys
    if not check_env_keys():
        return {"status": "failed", "error": "Missing API keys"}
    
    try:
        # Import the crew class here after env vars are loaded and checked
        # Choose which crew implementation to run. 
        # Using the "Updated" step-by-step version here.
        # To use the traditional one, import from '.crew_traditional' (rename your file)
        from .crew import FinancialMarketCrew
        
        logger.info("=== Starting Financial Market Summary Workflow ===")
        logger.info(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create and run the crew
        crew_manager = FinancialMarketCrew()
        # Use the appropriate run method for the selected crew class
        result = crew_manager.run_complete_workflow()
        
        logger.info(f"Execution completed with status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            logger.info("✅ Financial summary generated and sent successfully!")
        else:
            logger.error(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"A fatal error occurred in the main execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "failed", "error": error_msg}

if __name__ == "__main__":
    final_result = run_financial_summary()
    print("\n--- WORKFLOW FINAL RESULT ---")
    print(final_result)