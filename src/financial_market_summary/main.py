import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_env_keys() -> bool:
    """
    Checks for the presence of all required environment variables.
    Provides detailed feedback on missing keys to help with configuration.
    Returns:
        True if all required keys are found, False otherwise.
    """
    required_keys = {
        "GOOGLE_API_KEY": "Gemini API key for LLM operations",
        "TAVILY_API_KEY": "Tavily API key for news search",
        "TELEGRAM_BOT_TOKEN": "Telegram bot token for sending messages",
        "TELEGRAM_CHAT_ID": "Telegram chat ID for the target channel",
        "SERPER_API_KEY": "Serper API key for backup search"
    }
    
    missing_keys = []
    
    logger.info("Beginning check for required API keys...")
    for key, description in required_keys.items(): 
        value = os.getenv(key)
        if not value:
            missing_keys.append(f"  {key} ({description})")
    
    if missing_keys:
        logger.error("The following required API keys were not found:")
        for key in missing_keys:
            logger.error(key)
        logger.error("Please add these to your .env file to proceed.")
        return False
    
    logger.info("All required API keys were found.")
    return True

def check_api_quotas() -> str:
    """
    Provides a heuristic check for the Gemini API quota tier.
    Based on a simple key length check, this function logs recommendations
    to manage potential rate limiting for free-tier users.
    Returns:
        A string indicating the detected tier ('free_tier' or 'paid_tier').
    """
    logger.info("Checking API quotas and providing recommendations...")
    google_key = os.getenv('GOOGLE_API_KEY', '')
    if len(google_key) < 50:
        logger.warning("Detected a likely free tier for Gemini API. The workflow will be more conservative with API calls to prevent rate limiting.")
        return 'free_tier'
    else:
        logger.info("Detected a paid tier for Gemini API. Using standard rate limiting settings.")
        return 'paid_tier'

def run_financial_summary() -> Dict[str, Any]:
    """
    Orchestrates and runs the complete financial summary workflow.

    This function performs all necessary environment checks before
    initializing and executing the CrewAI workflow. It also handles
    logging of execution status, duration, and any errors.

    Returns:
        A dictionary containing the final status and results of the workflow.
    """
    if not check_env_keys():
        return {"status": "failed", "error": "Missing required API keys"}
    
    quota_tier = check_api_quotas()
    
    try:
        from .crew_bot import FinancialMarketCrew
        
        logger.info("--- Starting Financial Market Summary Workflow ---")
        logger.info(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"API Tier: {quota_tier}")
        
        # Determine expected execution time based on the quota tier
        if quota_tier == 'free_tier':
            logger.info("Using conservative rate limiting for the free tier. Expected execution time: 8-12 minutes.")
        else:
            logger.info("Using standard rate limiting for the paid tier. Expected execution time: 3-5 minutes.")
        
        # Initialize and run the crew
        crew_manager = FinancialMarketCrew()
        start_time = time.time()
        result = crew_manager.run_complete_workflow()
        end_time = time.time()
        execution_duration = end_time - start_time
        logger.info(f"Total execution time: {execution_duration:.1f} seconds")
        logger.info(f"Execution completed with status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            logger.info("Financial summary generated and sent successfully!")
            
            summary = result.get('summary', {})
            logger.info(f"Summary of Accomplishments: {summary.get('sends_completed', 0)} deliveries.")
        else:
            logger.error(f"Execution failed: {result.get('error', 'Unknown error')}")
            
            # Provide troubleshooting tips for common failures
            error_msg = result.get('error', '')
            if '429' in error_msg or 'quota' in error_msg.lower():
                logger.info("Troubleshooting a rate limit error:")
                logger.info(" - Try running the workflow again in 1-2 minutes.")
                logger.info(" - Consider upgrading to a paid Gemini API tier for higher limits.")
                logger.info(" - Check your API quotas at https://ai.google.dev/")
        
        return result
        
    except ImportError as e:
        error_msg = f"Import error - please ensure all dependencies are installed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "failed", "error": error_msg}
        
    except Exception as e:
        error_msg = f"A fatal error occurred during execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "failed", "error": error_msg}

# --- API Connection Test Function ---
def test_api_connections() -> Dict[str, str]:
    """
    Tests connections to all required APIs before starting the main workflow.

    This function provides a pre-flight check to ensure the application
    can communicate with external services.

    Returns:
        A dictionary with the connection status of each API.
    """
    logger.info("Testing API connections...")
    
    test_results = {}
    
    # Test Gemini API connection
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        gemini_llm.invoke([{"role": "user", "content": "Test"}])
        test_results['gemini'] = "Connected successfully"
        logger.info("Gemini API: Connected successfully.")
    except Exception as e:
        test_results['gemini'] = f"Failed: {str(e)}"
        logger.error(f"Gemini API: Connection failed with error: {str(e)}")
    
    # Test Telegram API connection
    try:
        import requests
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if bot_token:
            url = f"https://api.telegram.org/bot{bot_token}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                test_results['telegram'] = "Connected successfully"
                logger.info("Telegram API: Connected successfully.")
            else:
                test_results['telegram'] = f"HTTP {response.status_code}"
                logger.error(f"Telegram API: Connection failed with HTTP status code {response.status_code}.")
        else:
            test_results['telegram'] = "No token"
            logger.error("Telegram API: No token provided.")
    except Exception as e:
        test_results['telegram'] = f"Failed: {str(e)}"
        logger.error(f"Telegram API: Connection failed with error: {str(e)}")
    
    # Test Tavily API connection
    try:
        import requests
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_key,
                "query": "test",
                "max_results": 1
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                test_results['tavily'] = "Connected successfully"
                logger.info("Tavily API: Connected successfully.")
            else:
                test_results['tavily'] = f"HTTP {response.status_code}"
                logger.error(f"Tavily API: Connection failed with HTTP status code {response.status_code}.")
        else:
            test_results['tavily'] = "No key"
            logger.error("Tavily API: No key provided.")
    except Exception as e:
        test_results['tavily'] = f"Failed: {str(e)}"
        logger.error(f"Tavily API: Connection failed with error: {str(e)}")
    
    return test_results

# Main Entry Point 
if __name__ == "__main__":
    logger.info("--- Financial Market Summary Bot ---")
    
    # Perform pre-flight API connection tests
    api_status = test_api_connections()
    
    successful_apis = len([k for k, v in api_status.items() if "Connected" in v])
    total_apis = len(api_status)
    
    logger.info(f"API Status: {successful_apis}/{total_apis} APIs are connected and ready to use.")
    
    # Check if minimum APIs are available for the workflow to run
    if successful_apis >= 2:
        logger.info("Minimum APIs are available. Proceeding with the workflow...")
        final_result = run_financial_summary()
        logger.info("--- WORKFLOW FINAL RESULT ---")
        logger.info(final_result)
    else:
        logger.error("Insufficient API connections. The workflow will be aborted.")
        logger.info("Please check your API keys and the connection status logs above.")
        final_result = {"status": "failed", "error": "Insufficient API connections"}
    
    # Saving the final result to a JSON file in the output folder
    try:
        output_dir = ROOT_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"workflow_result_{timestamp}.json"
        
        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'api_status': api_status,
                'workflow_result': final_result,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        logger.info(f"Final results saved to: {result_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save results to a file due to an error: {e}")