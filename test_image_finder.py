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
print("IMAGE FINDER TEST - AD BLOCKER + CSS INJECTION TEST")
print("="*80)
print()

# Use latest CNBC URL
test_url = "https://www.cnbc.com/2025/10/20/stock-market-today-live-updates.html"

print(f"🎯 Testing with CNBC URL:")
print(f"   {test_url}")
print()

# Create temporary search results structure
test_search_results = {
    "metadata": {
        "timestamp": "2025-10-10 00:00:00 UTC",
        "query": "US stock market financial news",
        "total_articles": 1,
        "search_window_hours": 1.0
    },
    "articles": [
        {
            "article_number": 1,
            "title": "Stock market today: Live updates",
            "source": "CNBC",
            "url": test_url,
            "date": "October 09, 2025",
            "key_points": ["Stock market activity"]
        }
    ]
}

# Save temporary search results file
test_results_dir = project_root / "output" / "search_results"
test_results_dir.mkdir(parents=True, exist_ok=True)
test_results_file = test_results_dir / "test_cnbc_latest.json"

with open(test_results_file, 'w', encoding='utf-8') as f:
    json.dump(test_search_results, f, indent=2)

print(f"📁 Created test search results file")
print()

# Search content
search_content = "Russell 2000 S&P 500 Dow Jones Nasdaq stock market today"

print(f"🔍 Search content:")
print(f"   {search_content}")
print()

# Initialize Image Finder
image_finder = EnhancedImageFinder()

# Run image extraction
print("="*80)
print("RUNNING IMAGE FINDER")
print("="*80)
print()
print("Watch for these log messages:")
print("  - '🚫 Injecting CSS to hide popups...'")
print("  - '✅ CSS injection complete'")
print()

try:
    result = image_finder._run(
        search_content=search_content,
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
        print(f"✅ SUCCESS: Captured {len(images)} screenshot(s)")
        print()
        for idx, img in enumerate(images, 1):
            img_path = img.get('url', '')

            print(f"📸 Screenshot {idx}:")
            print(f"   Source: {img.get('source', 'N/A')}")
            print(f"   Type: {img.get('type', 'N/A')}")
            print(f"   Method: {img.get('extraction_method', 'N/A')}")
            print(f"   File Size: {img.get('file_size', 'N/A')} bytes")
            print(f"   Telegram Compatible: {img.get('telegram_compatible', False)}")
            print()

            if img_path and os.path.exists(img_path):
                print(f"   ✅ File saved successfully")
                print()
                print(f"   📂 SCREENSHOT PATH:")
                print(f"   {img_path}")
                print()
                print(f"   ⚠️  MANUAL VERIFICATION REQUIRED:")
                print(f"   1. Open the image file above")
                print(f"   2. Check if ANY popups/ads are visible")
                print(f"   3. Verify the chart is clean and unobstructed")
                print()
                print(f"   Expected: Clear chart with NO CNBC AI Summit popup")
            else:
                print(f"   ❌ File NOT found at path")
            print()

        # Show AI description if available
        description = images[0].get('image_description', '')
        if description:
            print(f"🤖 AI Description (first 200 chars):")
            print(f"   {description[:200]}...")
            print()

    else:
        print("❌ FAILED: No screenshots captured")
        print()
        print("Common reasons:")
        print("   - Page load timeout (>20 seconds)")
        print("   - No chart elements found")
        print("   - Network issues")
        print()
        print("Check the detailed logs above for error messages")

except Exception as e:
    print()
    print("="*80)
    print("TEST FAILED WITH EXCEPTION")
    print("="*80)
    print(f"❌ Error: {e}")
    print()
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)
print()
print("Next steps:")
print("  1. Locate the screenshot path shown above")
print("  2. Open the image in your file explorer")
print("  3. Verify NO popups are covering the chart")
print("  4. If popups still appear, report back for alternative solutions")
print("="*80)
