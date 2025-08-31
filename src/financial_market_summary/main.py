#!/usr/bin/env python3
"""
CrowdWisdomTrading AI Agent - Main Runner
Financial Market Summary Generator with Multi-language Support

Author: [Your Name]
Created for: CrowdWisdomTrading Internship Assessment
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
import argparse
import json
from pydantic import BaseModel, Field
from pathlib import Path
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,   # Show everything (DEBUG and above)
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_market_summary.crew import FinancialMarketFlow, run_financial_flow
from financial_market_summary.utils import (
    FinancialLogger, ConfigManager, MarketTimeUtils, 
    OutputFormatter, ErrorHandler, PerformanceMonitor
)

def setup_environment():
    """Setup environment and validate configuration"""
    print("🚀 CrowdWisdomTrading AI Agent - Financial Market Summary Generator")
    print("=" * 70)
    
    # Setup logging
    logger = FinancialLogger.setup_logging()
    logger.info("Starting Financial Market Summary AI Agent")
    
    # Setup configuration
    config_manager = ConfigManager()
    
    # Validate required API keys
    required_keys = ['groq', 'tavily', 'telegram_bot_token', 'telegram_chat_id'] # Changed from 'openai' to 'groq'
    key_validation = config_manager.validate_required_keys(required_keys)
    
    missing_keys = [k for k, v in key_validation.items() if not v]
    if missing_keys:
        logger.error(f"❌ Missing required API keys: {missing_keys}")
        print(f"\n❌ ERROR: Missing required environment variables:")
        for key in missing_keys:
            print(f"  - {key.upper()}_API_KEY" if not key.startswith('telegram') else f"  - {key.upper()}")
        print(f"\nPlease set these in your .env file or environment variables.")
        return False, None, None
    
    logger.info("✅ All required API keys validated")
    
    # Get market status
    market_status = MarketTimeUtils.get_market_status()
    logger.info(f"Market Status: {market_status}")
    
    return True, logger, config_manager

def run_financial_summary(test_mode: bool = False, save_output: bool = True):
    """Run the complete financial summary generation flow"""
    
    # Setup environment
    setup_success, logger, config_manager = setup_environment()
    if not setup_success:
        return False
    
    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor()
    performance_monitor.start_monitoring()
    
    try:
        logger.info("🔄 Initializing Financial Market Flow...")
        
        # Create and run the flow
        if test_mode:
            logger.info("🧪 Running in TEST MODE")
            result = run_test_flow(logger)
        else:
            logger.info("🚀 Running PRODUCTION FLOW")
            result = run_financial_flow()
        
        # Process results
        if result.get('status') == 'success':
            logger.info("✅ Financial summary generation completed successfully!")
            
            # Save outputs if requested
            if save_output:
                output_info = OutputFormatter.save_summary_output(
                    result.get('flow_state', {})
                )
                logger.info(f"📁 Outputs saved to: {output_info['output_dir']}")
            
            # Print summary
            print_execution_summary(result, performance_monitor.end_monitoring())
            
            return True
            
        else:
            logger.error("❌ Financial summary generation failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
            # Save partial results if available
            if result.get('partial_state') and save_output:
                output_info = OutputFormatter.save_summary_output(result['partial_state'])
                logger.info(f"📁 Partial results saved to: {output_info['output_dir']}")
            
            return False
            
    except Exception as e:
        error_msg = ErrorHandler.handle_api_error("main_flow", e)
        logger.error(f"❌ Unexpected error: {error_msg}")
        performance_monitor.increment_error()
        return False
    
    finally:
        performance_report = performance_monitor.end_monitoring()
        logger.info(f"📊 Performance Report: {json.dumps(performance_report, indent=2, default=str)}")

def run_test_flow(logger):
    """Run a simplified test flow for development"""
    logger.info("Running test flow with sample data...")
    
    # Create sample test data
    test_result = {
        'status': 'success',
        'flow_state': {
            'raw_news_data': 'Sample financial news data for testing...',
            'summary': '''
# Daily Financial Market Summary - TEST MODE

## Market Overview
- Test market data showing strong performance
- Major indices up 2.5% on average
- Volume above normal levels

## Key Movers
- AAPL: +3.2% on strong earnings report
- MSFT: +2.8% on cloud growth
- TSLA: -1.5% on production concerns

## Sector Analysis
- Technology leading gains
- Energy sector mixed performance
- Healthcare showing resilience

*This is test data generated for development purposes.*
            ''',
            'formatted_summary': 'Formatted test summary with placeholder images...',
            'translations': {
                'arabic': 'ملخص السوق المالي - وضع الاختبار',
                'hindi': 'वित्तीय बाजार सारांश - परीक्षण मोड',
                'hebrew': 'סיכום שוק כספי - מצב בדיקה'
            }
        },
        'final_result': 'Test execution completed successfully'
    }
    
    return test_result

def print_execution_summary(result: dict, performance_report: dict):
    """Print a formatted execution summary"""
    print("\n" + "=" * 70)
    print("📊 EXECUTION SUMMARY")
    print("=" * 70)
    
    flow_state = result.get('flow_state', {})
    
    # Basic info
    print(f"✅ Status: {result.get('status', 'Unknown').upper()}")
    print(f"⏱️  Total Time: {performance_report.get('total_execution_time', 0):.2f} seconds")
    print(f"🌐 Languages Generated: English + 3 translations")
    print(f"📱 Telegram Delivery: {'✅ Success' if 'send_completed' in str(result) else '❌ Check logs'}")
    
    # Content summary
    summary = flow_state.get('summary', '')
    if summary:
        word_count = len(summary.split())
        print(f"📝 Summary Length: {word_count} words")
    
    # Performance breakdown
    if performance_report.get('step_timings'):
        print(f"🚀 Success Rate: {performance_report.get('success_rate', 0):.1f}%")
        print(f"🐌 Slowest Step: {performance_report.get('bottleneck_analysis', {}).get('slowest_step', {}).get('step', 'N/A')}")
    
    print("=" * 70)

def create_sample_outputs():
    """Create sample input/output files for assessment submission"""
    samples_dir = OutputFormatter.create_sample_inputs_outputs()
    print(f"📁 Sample files created in: {samples_dir}")
    return samples_dir

def test_api_connections():
    """Test all API connections before running main flow"""
    print("\n🔧 TESTING API CONNECTIONS")
    print("-" * 40)
    
    config_manager = ConfigManager()
    api_keys = config_manager.get_api_keys()
    
    # Test results
    test_results = {}
    
    # Test Telegram connection
    if 'telegram_bot_token' in api_keys:
        try:
            from src.financial_market_summary.tools.telegram_sender import TelegramSender
            telegram_sender = TelegramSender()
            result = telegram_sender.test_connection()
            test_results['telegram'] = result
            print(f"📱 Telegram: {result}")
        except Exception as e:
            test_results['telegram'] = f"❌ Error: {str(e)}"
            print(f"📱 Telegram: ❌ Error: {str(e)}")
    
    # Test Tavily
    if 'tavily' in api_keys:
        try:
            from src.financial_market_summary.tools.tavily_search import TavilyFinancialTool
            tavily_tool = TavilyFinancialTool()
            # Quick test search
            result = tavily_tool._run("test financial news", hours_back=1, max_results=1)
            if "Error:" not in result:
                test_results['tavily'] = "✅ Connected"
                print("🔍 Tavily: ✅ Connected")
            else:
                test_results['tavily'] = "❌ Connection failed"
                print("🔍 Tavily: ❌ Connection failed")
        except Exception as e:
            test_results['tavily'] = f"❌ Error: {str(e)}"
            print(f"🔍 Tavily: ❌ Error: {str(e)}")
    
    # Test Serper (if available)
    if 'serper' in api_keys:
        try:
            from src.financial_market_summary.tools.tavily_search import SerperFinancialTool
            serper_tool = SerperFinancialTool()
            result = serper_tool._run("test financial news", hours_back=1, max_results=1)
            if "Error:" not in result:
                test_results['serper'] = "✅ Connected"
                print("🔍 Serper: ✅ Connected")
            else:
                test_results['serper'] = "❌ Connection failed"
                print("🔍 Serper: ❌ Connection failed")
        except Exception as e:
            test_results['serper'] = f"❌ Error: {str(e)}"
            print(f"🔍 Serper: ❌ Error: {str(e)}")
    
    print("-" * 40)
    return test_results

def main():
    """Main entry point with command line argument support"""
    parser = argparse.ArgumentParser(
        description="CrowdWisdomTrading AI Agent - Financial Market Summary Generator"
    )
    parser.add_argument(
        '--test', 
        action='store_true', 
        help='Run in test mode with sample data'
    )
    parser.add_argument(
        '--no-save', 
        action='store_true', 
        help='Do not save output files'
    )
    parser.add_argument(
        '--test-apis', 
        action='store_true', 
        help='Test API connections only'
    )
    parser.add_argument(
        '--create-samples', 
        action='store_true', 
        help='Create sample input/output files'
    )
    parser.add_argument(
        '--schedule', 
        action='store_true', 
        help='Show optimal scheduling information'
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.test_apis:
        test_api_connections()
        return
    
    if args.create_samples:
        create_sample_outputs()
        return
    
    if args.schedule:
        market_status = MarketTimeUtils.get_market_status()
        next_close = MarketTimeUtils.get_next_market_close()
        
        print("\n📅 OPTIMAL SCHEDULING INFORMATION")
        print("-" * 50)
        print(f"Current Market Status: {'🟢 Open' if market_status['is_open'] else '🔴 Closed'}")
        print(f"Next Market Close: {next_close.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Recommended Run Time: 01:30 AM IST (after US market close)")
        print(f"Current Time (IST): {market_status['current_time_ist']}")
        print("-" * 50)
        return
    
    # Run main flow
    success = run_financial_summary(
        test_mode=args.test,
        save_output=not args.no_save
    )
    
    if success:
        print("\n🎉 Financial Market Summary completed successfully!")
        print("📧 Ready for submission to: gilad@crowdwisdomtrading.com")
    else:
        print("\n❌ Financial Market Summary failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()