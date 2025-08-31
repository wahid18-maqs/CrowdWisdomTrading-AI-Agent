# main.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from .crew import FinancialMarketFlow

# Load environment variables from custom path
dotenv_path = Path(__file__).resolve().parent.parent / '../.env'
load_dotenv(dotenv_path=dotenv_path)

REQUIRED_KEYS = ["GOOGLE_API_KEY", "TAVILY_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_env_keys():
    missing_keys = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing_keys:
        logger.error(f"Missing required API keys: {missing_keys}")
        return False
    return True

def run():
    if not check_env_keys():
        logger.error("Please set all required API keys in your environment before running.")
        return

    try:
        logger.info("=== Starting CrowdWisdomTrading Financial Market Summary ===")
        flow = FinancialMarketFlow()
        result = flow.run_with_guardrails()

        # Print final flow status
        status = result.get("status", "failed")
        logger.info(f"Flow completed with status: {status}")

        # Print summaries
        formatted_summary = flow.flow_state.get("formatted_summary", "")
        translations = flow.flow_state.get("translations", {})

        logger.info("=== English Summary ===")
        logger.info(formatted_summary)

        for lang, translation in translations.items():
            logger.info(f"=== {lang.capitalize()} Translation ===")
            logger.info(translation)

        # Telegram result
        telegram_result = result.get("flow_state", {}).get("send_result", "No Telegram output")
        logger.info("=== Telegram Sending Result ===")
        logger.info(telegram_result)

    except Exception as e:
        logger.error(f"Error running financial flow: {str(e)}")

if __name__ == "__main__":
    run()
