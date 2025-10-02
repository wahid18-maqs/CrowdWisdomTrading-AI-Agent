import sys
from pathlib import Path
import json
import os
import logging

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.financial_market_summary.tools.image_finder import EnhancedImageFinder

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

print("="*80)
print("IMAGE FINDER TEST SCRIPT - SINGLE URL TEST")
print("="*80)
print()

# Create a test search results file with just the Yahoo Finance URL
test_url = "https://finance.yahoo.com/news/live/dow-sp-500-nasdaq-notch-records-on-ai-buzz-even-as-government-shutdown-drags-on-200207403.html"

print(f"🎯 Testing with single URL:")
print(f"   {test_url}")
print()

# Create temporary search results structure
test_search_results = {
    "metadata": {
        "timestamp": "2025-10-03 02:30:00 UTC",
        "query": "US stock market financial news",
        "total_articles": 1,
        "search_window_hours": 1.0
    },
    "articles": [
        {
            "article_number": 1,
            "title": "Dow, S&P 500, Nasdaq notch records on AI buzz",
            "source": "Yahoo Finance",
            "url": test_url,
            "date": "October 03, 2025",
            "key_points": ["Stock market activity"]
        }
    ]
}

# Save temporary search results file
test_results_dir = project_root / "output" / "search_results"
test_results_dir.mkdir(parents=True, exist_ok=True)
test_results_file = test_results_dir / "test_single_url.json"

with open(test_results_file, 'w', encoding='utf-8') as f:
    json.dump(test_search_results, f, indent=2)

print(f"📁 Created temporary search results file:")
print(f"   {test_results_file.name}")
print()

# Search content
search_content = "S&P 500 Dow Jones Nasdaq stock market records AI technology"

print(f"🔍 Search content:")
print(f"   {search_content}")
print()

# Initialize Image Finder
image_finder = EnhancedImageFinder()

# Run image extraction with verbose output
print("="*80)
print("RUNNING IMAGE FINDER")
print("="*80)
print()

try:
    result = image_finder._run(
        search_content=search_content,
        mentioned_stocks=[],
        max_images=1,
        search_results_file=str(test_results_file)
    )

    print()
    print("="*80)
    print("TEST RESULTS")
    print("="*80)
    print()

    # Parse and display results
    images = json.loads(result) if isinstance(result, str) else result

    if images:
        print(f"✅ SUCCESS: Found {len(images)} image(s)")
        print()
        for idx, img in enumerate(images, 1):
            print(f"📸 Image {idx}:")
            print(f"   Local Path: {img.get('url', 'N/A')}")
            print(f"   Source: {img.get('source', 'N/A')}")
            print(f"   Type: {img.get('type', 'N/A')}")
            print(f"   Method: {img.get('extraction_method', 'N/A')}")
            print(f"   Description: {img.get('image_description', 'N/A')}")
            print(f"   File Size: {img.get('file_size', 'N/A')} bytes")
            print(f"   Telegram Compatible: {img.get('telegram_compatible', False)}")
            print()

            # Check if file exists
            img_path = img.get('url', '')
            if img_path and os.path.exists(img_path):
                print(f"   ✅ File exists on disk")
            else:
                print(f"   ❌ File NOT found on disk")
            print()
    else:
        print("❌ FAILED: No images found")
        print()
        print("Possible reasons:")
        print("   - All URLs timed out during page load")
        print("   - No chart elements found on pages")
        print("   - All screenshots failed to capture")
        print()
        print("Check the logs above for detailed error messages")

except Exception as e:
    print()
    print("="*80)
    print("TEST FAILED WITH EXCEPTION")
    print("="*80)
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("Test complete!")
print("="*80)
