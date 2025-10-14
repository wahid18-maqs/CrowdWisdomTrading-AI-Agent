"""
Test script for Telegram two-message format flow
Uses existing search results to test the complete workflow
"""

import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.financial_market_summary.crew_bot import FinancialMarketCrew
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_full_workflow():
    """
    Test the complete workflow by running the crew
    This will search -> summarize -> send to Telegram (ENGLISH ONLY)
    """

    print("=" * 80)
    print("🧪 TESTING COMPLETE TELEGRAM WORKFLOW - ENGLISH ONLY")
    print("=" * 80)
    print()

    try:
        # Create crew instance
        print(f"🚀 Initializing FinancialMarketCrew...")
        crew = FinancialMarketCrew()

        print(f"✅ Crew initialized successfully")
        print()

        # Run the complete workflow
        print(f"🎯 Running complete workflow for ENGLISH...")
        print(f"   This will:")
        print(f"   1. Search for latest financial news (24h)")
        print(f"   2. Capture chart screenshots")
        print(f"   3. Extract AI image descriptions from website")
        print(f"   4. Create two-message format summary with 'Crowd Wisdom' style")
        print(f"   5. Send Message 1 (Image + Caption) to English Telegram")
        print(f"   6. Send Message 2 (Full Summary) to English Telegram")
        print()

        result = crew.run_complete_workflow()

        print()
        print("=" * 80)
        print("✅ ENGLISH WORKFLOW COMPLETED")
        print("=" * 80)
        print()
        print("📋 Result Summary:")
        print(json.dumps(result, indent=2, default=str))
        print()

        # Check results
        status = result.get("status")
        if status == "success":
            summary = result.get("summary", {})
            sends_completed = summary.get("sends_completed", 0)
            print(f"✅ Status: SUCCESS")
            print(f"📨 Messages sent: {sends_completed}")
            print()
            print("Expected messages:")
            print("  • English bot: 2 messages (if image available) or 1 message (if no image)")
            print("  • Other language bots: Multiple messages (translations)")
        else:
            print(f"❌ Status: FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")

        print()
        print("-" * 80)
        print()

    except Exception as e:
        print()
        print(f"❌ Error testing workflow: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()

    print("=" * 80)
    print("🏁 TEST COMPLETED")
    print("=" * 80)
    print()
    print("📱 Check your English Telegram channel to verify:")
    print("   1. Message 1: Image with AI-generated description caption")
    print("   2. Message 2: Full market summary in 'The Crowd Wisdom' style")
    print()
    print("Expected format for Message 2:")
    print("   • Title: 'The Crowd Wisdom's summary'")
    print("   • Market overview (Dow, S&P, Nasdaq)")
    print("   • Macro News: 🔍 [news items]")
    print("   • Notable Stocks: 🟢🔵🟡 [stock highlights]")
    print("   • Commodities/FX (if relevant)")
    print("   • Disclaimer")
    print()

def test_image_description_only():
    """
    Test ONLY the image description extraction (faster test)
    """
    print("=" * 80)
    print("🖼️ TESTING IMAGE DESCRIPTION EXTRACTION ONLY")
    print("=" * 80)
    print()

    search_results_file = r"C:\Users\wahid\Desktop\financial_market_summary\output\search_results\search_results_20251013_114832_US-stock-market-financial-news.json"

    if not os.path.exists(search_results_file):
        print(f"❌ Search results file not found: {search_results_file}")
        return

    print(f"✅ Found search results file:")
    print(f"   {search_results_file}")
    print()

    from src.financial_market_summary.tools.image_finder import EnhancedImageFinder

    try:
        # Create image finder
        finder = EnhancedImageFinder()

        # Run image finding
        print("🔍 Searching for charts and extracting descriptions...")
        result = finder._run(
            search_content="US stock market financial news",
            mentioned_stocks=[],
            max_images=1,
            search_results_file=search_results_file
        )

        print()
        print("✅ Image finder completed")
        print()

        # Parse result
        images = json.loads(result) if isinstance(result, str) else result

        if images and len(images) > 0:
            img = images[0]
            print("📊 IMAGE DETAILS:")
            print(f"   Title: {img.get('title', 'N/A')}")
            print(f"   Source: {img.get('source', 'N/A')}")
            print(f"   Source Article: {img.get('source_article', 'N/A')[:80]}...")
            print(f"   File: {img.get('url', 'N/A')}")
            print(f"   Telegram Compatible: {img.get('telegram_compatible', False)}")
            print()
            print("📝 IMAGE DESCRIPTION (from website <p> tag):")
            print(f"   {img.get('image_description', 'No description')}")
            print()

            if img.get('image_description'):
                print("✅ Image description successfully extracted from website!")
            else:
                print("❌ No image description found - check logs above")
        else:
            print("❌ No images found")

        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def check_existing_results():
    """Check if we have recent output files"""
    print("=" * 80)
    print("📂 CHECKING EXISTING OUTPUT FILES")
    print("=" * 80)
    print()

    output_dir = Path(__file__).resolve().parent / "output"

    # Check search results
    search_dir = output_dir / "search_results"
    if search_dir.exists():
        search_files = list(search_dir.glob("search_results_*.json"))
        if search_files:
            latest = max(search_files, key=lambda p: p.stat().st_mtime)
            print(f"✅ Latest search results:")
            print(f"   {latest}")
            print(f"   Modified: {latest.stat().st_mtime}")
        else:
            print("❌ No search results found")
    else:
        print("❌ Search results directory doesn't exist")

    print()

    # Check screenshots
    screenshots_dir = output_dir / "screenshots"
    if screenshots_dir.exists():
        screenshots = list(screenshots_dir.glob("chart_*.png"))
        if screenshots:
            latest = max(screenshots, key=lambda p: p.stat().st_mtime)
            print(f"✅ Latest screenshot:")
            print(f"   {latest}")
            print(f"   Modified: {latest.stat().st_mtime}")
        else:
            print("❌ No screenshots found")
    else:
        print("❌ Screenshots directory doesn't exist")

    print()

    # Check image results
    image_results_dir = output_dir / "image_results"
    if image_results_dir.exists():
        image_files = list(image_results_dir.glob("image_results_*.json"))
        if image_files:
            latest = max(image_files, key=lambda p: p.stat().st_mtime)
            print(f"✅ Latest image results:")
            print(f"   {latest}")
            print(f"   Modified: {latest.stat().st_mtime}")

            # Show content preview
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('extracted_images'):
                    img = data['extracted_images'][0]
                    desc = img.get('image_description', '')
                    print()
                    print(f"📝 Image description preview:")
                    print(f"   {desc[:150]}...")
        else:
            print("❌ No image results found")
    else:
        print("❌ Image results directory doesn't exist")

    print()

if __name__ == "__main__":
    print()
    print("=" * 80)
    print("📋 TELEGRAM TWO-MESSAGE FORMAT TEST")
    print("=" * 80)
    print()
    print("Choose test mode:")
    print()
    print("1. Full workflow - English + All Translations")
    print("   (search → create summary → send to English + translate to other languages)")
    print()
    print("2. Image description only (fast test, no Telegram)")
    print("   (test image extraction and AI description generation)")
    print()
    print("3. Check existing output files")
    print("   (view latest search results, screenshots, and image descriptions)")
    print()
    print("=" * 80)
    print()

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        print()
        print("🚀 Running full workflow...")
        print("⚠️  Note: This will send to English bot AND all translation bots")
        print()
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            test_full_workflow()
        else:
            print("❌ Test cancelled")
    elif choice == "2":
        test_image_description_only()
    elif choice == "3":
        check_existing_results()
    else:
        print("❌ Invalid choice")
        print("Run the script again and choose 1, 2, or 3")
