import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import our crew implementation
from .crew import FinancialMarketCrew

def setup_environment():
    """Setup environment variables and validate configuration"""
    # Load environment variables
    load_dotenv()
    
    # Required environment variables
    required_vars = [
        'GOOGLE_API_KEY',     # For Gemini LLM
        'TAVILY_API_KEY',     # For news search
        'TELEGRAM_BOT_TOKEN', # For message sending
        'TELEGRAM_CHAT_ID'    # For target channel
    ]
    
    # Optional environment variables
    optional_vars = [
        'SERPER_API_KEY',     # Backup search API
    ]
    
    missing_vars = []
    present_vars = []
    
    # Check required variables
    for var in required_vars:
        if os.getenv(var):
            present_vars.append(var)
            # Mask the actual values in logs
            masked_value = os.getenv(var)[:8] + "..." if len(os.getenv(var)) > 8 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            missing_vars.append(var)
            print(f"❌ {var}: Not found")
    
    # Check optional variables
    for var in optional_vars:
        if os.getenv(var):
            masked_value = os.getenv(var)[:8] + "..." if len(os.getenv(var)) > 8 else "***"
            print(f"🔶 {var}: {masked_value} (optional)")
        else:
            print(f"🔶 {var}: Not found (optional)")
    
    if missing_vars:
        print(f"\n❌ Missing required environment variables: {missing_vars}")
        print("\nPlease ensure all required API keys are set in your .env file:")
        print("GOOGLE_API_KEY=your_gemini_api_key")
        print("TAVILY_API_KEY=your_tavily_api_key")
        print("TELEGRAM_BOT_TOKEN=your_bot_token")
        print("TELEGRAM_CHAT_ID=your_chat_id")
        return False
    
    print(f"\n✅ Environment setup complete. {len(present_vars)} required variables found.")
    return True

def save_execution_results(results, filename=None):
    """Save execution results to JSON file"""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workflow_result_{timestamp}.json"
        
        filepath = logs_dir / filename
        
        # Save results with proper JSON formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 Execution results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"⚠️ Failed to save execution results: {str(e)}")
        return None

def print_execution_summary(results):
    """Print a formatted summary of execution results"""
    print("\n" + "="*60)
    print("📊 EXECUTION SUMMARY")
    print("="*60)
    
    # Basic execution info
    print(f"🕒 Start Time: {results.get('execution_start', 'Unknown')}")
    print(f"🕒 End Time: {results.get('execution_end', 'Unknown')}")
    print(f"⏱️ Duration: {results.get('execution_duration_minutes', 0):.1f} minutes")
    print(f"✅ Success: {results.get('success', False)}")
    
    if not results.get('success'):
        print(f"❌ Error: {results.get('error', 'Unknown error')}")
        return
    
    # API connection status
    api_status = results.get('api_connections', {})
    print(f"\n🔌 API CONNECTIONS:")
    for api, status in api_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {api.title()}: {'Connected' if status else 'Failed'}")
    
    # Detailed results from execution
    detailed = results.get('detailed_results', {})
    
    # Search results
    search_results = detailed.get('search_results', {})
    if search_results:
        articles_count = len(search_results.get('articles', []))
        print(f"\n📰 NEWS SEARCH:")
        print(f"   📄 Articles Found: {articles_count}")
        print(f"   🔗 URLs for Image Extraction: {len(search_results.get('article_urls', []))}")
        
        # Show top articles
        articles = search_results.get('articles', [])[:3]
        for i, article in enumerate(articles, 1):
            print(f"   {i}. {article.get('title', 'Untitled')[:60]}...")
    
    # Image extraction results
    images = detailed.get('extracted_images', [])
    if images:
        print(f"\n🖼️ IMAGE EXTRACTION:")
        print(f"   📊 Charts/Graphs Found: {len(images)}")
        for i, img in enumerate(images, 1):
            source = img.get('source_domain', 'Unknown')
            score = img.get('relevance_score', 0)
            print(f"   {i}. Source: {source} (Relevance: {score})")
    
    # Translation results
    translations = detailed.get('translations', {})
    if translations:
        print(f"\n🌐 TRANSLATIONS:")
        print(f"   📝 Languages: {len(translations)}")
        for lang in translations.keys():
            print(f"   ✅ {lang}")
    
    # Telegram distribution results
    telegram_results = detailed.get('telegram_distribution', {})
    if telegram_results:
        print(f"\n📱 TELEGRAM DISTRIBUTION:")
        print(f"   📤 Overall Success: {telegram_results.get('success', False)}")
        
        # Per-language results
        lang_results = telegram_results.get('results', {})
        for lang, result in lang_results.items():
            if lang != 'images':  # Skip images key
                success_rate = result.get('success_rate', 0) * 100
                print(f"   {lang}: {success_rate:.0f}% success rate")
    
    print("\n" + "="*60)

def main():
    """Main execution function"""
    print("🚀 Financial Market Summary Bot - Starting Execution")
    print(f"📅 Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Environment Setup
    print("\n1️⃣ ENVIRONMENT SETUP")
    if not setup_environment():
        print("❌ Environment setup failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Initialize Crew
    print("\n2️⃣ INITIALIZING WORKFLOW")
    try:
        crew = FinancialMarketCrew()
        print("✅ Crew initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize crew: {str(e)}")
        sys.exit(1)
    
    # Step 3: Execute Workflow
    print("\n3️⃣ EXECUTING WORKFLOW")
    print("This may take several minutes depending on API tier...")
    print("⏳ Expected time: 3-8 minutes")
    
    try:
        results = crew.execute_workflow()
        
        # Step 4: Save and Display Results
        print("\n4️⃣ PROCESSING RESULTS")
        
        # Save results to file
        save_execution_results(results)
        
        # Print summary
        print_execution_summary(results)
        
        # Final status
        if results.get('success'):
            print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print("📱 Check your Telegram channels for the market summary.")
            
            # Show quick stats
            summary = crew.get_execution_summary()
            print(f"📊 Quick Stats:")
            print(f"   • {summary.get('articles_found', 0)} articles analyzed")
            print(f"   • {summary.get('images_extracted', 0)} charts/graphs included")
            print(f"   • {summary.get('languages_translated', 0)} languages delivered")
        else:
            print("\n❌ WORKFLOW FAILED")
            print("📋 Check the logs for detailed error information.")
            print(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error during execution: {str(e)}")
        
        # Try to save partial results
        try:
            partial_results = {
                'success': False,
                'error': str(e),
                'execution_time': datetime.now().isoformat(),
                'partial_data': crew.execution_results if hasattr(crew, 'execution_results') else {}
            }
            save_execution_results(partial_results, f"failed_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        except:
            pass
        
        sys.exit(1)

def test_mode():
    """Run in test mode - just test API connections"""
    print("🧪 TEST MODE - API Connection Testing")
    print("="*50)
    
    if not setup_environment():
        return
    
    try:
        crew = FinancialMarketCrew()
        connections = crew.test_api_connections()
        
        print("\n📊 CONNECTION TEST RESULTS:")
        all_connected = True
        for service, status in connections.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {service.title()}: {'Connected' if status else 'Failed'}")
            if not status:
                all_connected = False
        
        if all_connected:
            print("\n🎉 All APIs connected successfully!")
            print("Ready to run full workflow.")
        else:
            print("\n⚠️ Some API connections failed.")
            print("Please check your API keys and network connection.")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_mode()
    else:
        main()