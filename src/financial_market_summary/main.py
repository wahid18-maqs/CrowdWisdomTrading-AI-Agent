# Enhanced main.py with rate limiting and better error handling
import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(ENV_PATH)


# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_env_keys():
    """Check for required environment variables with detailed feedback"""
    required_keys = {
        "GOOGLE_API_KEY": "Gemini API key for LLM operations",
        "TAVILY_API_KEY": "Tavily API key for news search",
        "TELEGRAM_BOT_TOKEN": "Telegram bot token for sending messages",
        "TELEGRAM_CHAT_ID": "Telegram chat ID for the target channel",
        "SERPER_API_KEY": "Serper API key for backup search"
    }
    
    missing_keys = []
    available_keys = []
    
    for key, description in required_keys.items(): 
        value = os.getenv(key)
        if not value:
            missing_keys.append(f"  {key}={description}")
        else:
            available_keys.append(key)
            # Mask the key for security
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            logger.info(f"✅ {key}: {masked_value}")
    
    if missing_keys:
        logger.error("❌ Missing required API keys:")
        for key in missing_keys:
            logger.error(key)
        logger.error("\nPlease add these to your .env file:")
        return False
    
    logger.info(f"✅ All {len(available_keys)} required API keys found")
    return True

def check_api_quotas():
    """Check API quotas and provide recommendations"""
    logger.info("🔍 Checking API quotas and providing recommendations...")
    
    # Check Gemini quota (this is where we're hitting limits)
    google_key = os.getenv('GOOGLE_API_KEY', '')
    if len(google_key) < 50:  # Heuristic for free tier
        logger.warning("⚠️  Detected possible free tier Gemini API - applying conservative rate limiting")
        logger.info("💡 Recommendations:")
        logger.info("  - Workflow will use longer delays between requests")
        logger.info("  - Consider upgrading to paid tier for faster execution")
        logger.info("  - Current free tier: 15 requests/minute, 1500 requests/day")
        return 'free_tier'
    else:
        logger.info("✅ Detected paid tier Gemini API - using standard rate limiting")
        return 'paid_tier'

def run_financial_summary():
    """Enhanced main function with rate limiting and better error handling"""
    
    # Check for all required API keys
    if not check_env_keys():
        return {"status": "failed", "error": "Missing required API keys"}
    
    # Check API quotas and adjust settings
    quota_tier = check_api_quotas()
    
    try:
        # Import after env check
        from .crew import FinancialMarketCrew
        
        logger.info("=== Starting Financial Market Summary Workflow ===")
        logger.info(f"🕐 Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⚙️  API Tier: {quota_tier}")
        
        if quota_tier == 'free_tier':
            logger.info("🐌 Using conservative rate limiting for free tier")
            logger.info("⏱️  Expected execution time: 8-12 minutes")
        else:
            logger.info("🚀 Using standard rate limiting for paid tier")
            logger.info("⏱️  Expected execution time: 3-5 minutes")
        
        # Create and run the crew
        crew_manager = FinancialMarketCrew()
        
        # Set rate limiting based on quota tier
        if quota_tier == 'free_tier':
            crew_manager.rate_limit_delay = 8  # 8 seconds between calls for free tier
        else:
            crew_manager.rate_limit_delay = 3  # 3 seconds for paid tier
        
        start_time = time.time()
        result = crew_manager.run_complete_workflow()
        end_time = time.time()
        
        execution_duration = end_time - start_time
        logger.info(f"⏱️  Total execution time: {execution_duration:.1f} seconds")
        logger.info(f"Execution completed with status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            logger.info("✅ Financial summary generated and sent successfully!")
            
            # Log summary of what was accomplished
            summary = result.get('summary', {})
            logger.info(f"📊 Summary: {summary.get('translations_completed', 0)} translations, {summary.get('sends_completed', 0)} deliveries")
        else:
            logger.error(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
            
            # Provide troubleshooting tips
            error_msg = result.get('error', '')
            if '429' in error_msg or 'quota' in error_msg.lower():
                logger.info("💡 Rate limit troubleshooting:")
                logger.info("  - Try running again in 1-2 minutes")
                logger.info("  - Consider upgrading to paid Gemini API tier")
                logger.info("  - Check your API quotas at https://ai.google.dev/")
        
        return result
        
    except ImportError as e:
        error_msg = f"Import error - check if all dependencies are installed: {str(e)}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}
        
    except Exception as e:
        error_msg = f"Fatal error in main execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "failed", "error": error_msg}

def test_api_connections():
    """Test all API connections before running the full workflow"""
    logger.info("🧪 Testing API connections...")
    
    test_results = {}
    
    # Test Gemini API
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        test_response = gemini_llm.invoke([{"role": "user", "content": "Test"}])
        test_results['gemini'] = "✅ Connected"
        logger.info("✅ Gemini API: Connected successfully")
    except Exception as e:
        test_results['gemini'] = f"❌ Failed: {str(e)}"
        logger.error(f"❌ Gemini API: {str(e)}")
    
    # Test Telegram API
    try:
        import requests
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if bot_token:
            url = f"https://api.telegram.org/bot{bot_token}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                test_results['telegram'] = "✅ Connected"
                logger.info("✅ Telegram API: Connected successfully")
            else:
                test_results['telegram'] = f"❌ HTTP {response.status_code}"
                logger.error(f"❌ Telegram API: HTTP {response.status_code}")
        else:
            test_results['telegram'] = "❌ No token"
            logger.error("❌ Telegram API: No token provided")
    except Exception as e:
        test_results['telegram'] = f"❌ Failed: {str(e)}"
        logger.error(f"❌ Telegram API: {str(e)}")
    
    # Test Tavily API
    try:
        import requests
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            # Simple test query
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_key,
                "query": "test",
                "max_results": 1
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                test_results['tavily'] = "✅ Connected"
                logger.info("✅ Tavily API: Connected successfully")
            else:
                test_results['tavily'] = f"❌ HTTP {response.status_code}"
                logger.error(f"❌ Tavily API: HTTP {response.status_code}")
        else:
            test_results['tavily'] = "❌ No key"
            logger.error("❌ Tavily API: No key provided")
    except Exception as e:
        test_results['tavily'] = f"❌ Failed: {str(e)}"
        logger.error(f"❌ Tavily API: {str(e)}")
    
    return test_results

if __name__ == "__main__":
    print("🚀 Financial Market Summary Bot")
    print("=" * 50)
    
    # Test connections first
    api_status = test_api_connections()
    
    # Count successful connections
    successful_apis = len([k for k, v in api_status.items() if "✅" in v])
    total_apis = len(api_status)
    
    print(f"\n📊 API Status: {successful_apis}/{total_apis} APIs connected successfully")
    
    if successful_apis >= 2:  # Need at least Gemini + Telegram
        print("✅ Minimum APIs available - proceeding with workflow...")
        final_result = run_financial_summary()
        print("\n" + "=" * 50)
        print("--- WORKFLOW FINAL RESULT ---")
        print(final_result)
    else:
        print("❌ Insufficient API connections - workflow aborted")
        print("💡 Please check your API keys and try again")
        final_result = {"status": "failed", "error": "Insufficient API connections"}
    
    # Save result to file for debugging
    try:
        output_dir = Path("logs")
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
        
        logger.info(f"📄 Results saved to: {result_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")