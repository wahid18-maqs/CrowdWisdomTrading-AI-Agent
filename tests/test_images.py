#!/usr/bin/env python3
"""
Test script to verify image finder is working with real financial data
Run this to test image finding before running the full workflow
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime

# Add the correct path to find the module
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    # If running from tests folder, go up one level
    project_root = current_dir.parent
else:
    # If running from project root
    project_root = current_dir

# Add src directory to Python path
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

print(f"🔍 Project root: {project_root}")
print(f"🔍 Source directory: {src_dir}")
print(f"🔍 Current working directory: {Path.cwd()}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_finder():
    """Test the image finder with real financial content"""
    
    # Load environment variables from project root
    env_file = project_root / ".env"
    load_dotenv(env_file)
    
    try:
        # Try to import the image finder
        logger.info(f"Attempting to import from: {src_dir}")
        from financial_market_summary.tools.image_finder import ImageFinder
        
        # Create image finder instance
        image_finder = ImageFinder()
        
        # Test with sample financial content that contains stock symbols
        test_content = """
        **Market Overview**
        The U.S. stock market showed mixed performance today. The S&P 500 (SPY) gained 0.8% while 
        tech stocks led by Apple (AAPL) and Microsoft (MSFT) showed strong gains. 
        
        **Key Movers**
        - AAPL: +2.3% after strong earnings report
        - MSFT: +1.8% on cloud revenue growth
        - TSLA: -1.2% on production concerns
        - GOOGL: +0.9% following AI announcements
        
        **Sector Analysis**
        Technology sector outperformed with semiconductor stocks like NVDA showing gains.
        """
        
        logger.info("🧪 Testing Image Finder with real financial content...")
        logger.info("📄 Test content contains: AAPL, MSFT, TSLA, GOOGL, NVDA, SPY")
        
        # Run the image finder
        result = image_finder._run(
            search_context=test_content,
            max_images=3,
            image_types=["chart", "graph", "financial"]
        )
        
        logger.info("🎯 Image Finder Results:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        # Test individual chart finding methods
        logger.info("\n🔍 Testing individual chart sources...")
        
        test_symbols = ['AAPL', 'MSFT', 'SPY']
        
        for symbol in test_symbols:
            logger.info(f"\n📊 Testing charts for {symbol}:")
            
            # Test Finviz
            finviz_result = image_finder._get_finviz_chart(symbol)
            if finviz_result:
                logger.info(f"  ✅ Finviz: {finviz_result['url']}")
            else:
                logger.warning(f"  ❌ Finviz: No chart found")
            
            # Test Yahoo Finance
            yahoo_result = image_finder._get_yahoo_finance_chart(symbol)
            if yahoo_result:
                logger.info(f"  ✅ Yahoo: {yahoo_result['url']}")
            else:
                logger.warning(f"  ❌ Yahoo: No chart found")
            
            # Test TradingView
            tv_result = image_finder._get_tradingview_chart(symbol)
            if tv_result:
                logger.info(f"  ✅ TradingView: {tv_result['url']}")
            else:
                logger.warning(f"  ❌ TradingView: No chart found")
        
        # Test market index charts
        logger.info("\n📈 Testing market index charts...")
        index_charts = image_finder._find_market_index_charts()
        for chart in index_charts:
            logger.info(f"  ✅ {chart['symbol']}: {chart['url']}")
        
        # Test image verification
        logger.info("\n🔍 Testing image URL verification...")
        test_urls = [
            "https://finviz.com/chart.ashx?t=AAPL&ty=c&ta=1&p=d&s=l",
            "https://chart.yahoo.com/z?s=MSFT&t=1d&q=l&l=on&z=s&p=m50,m200",
            "https://invalid-domain.fake/chart.png"
        ]
        
        for url in test_urls:
            is_valid = image_finder._verify_image_url(url, quick_check=False)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            logger.info(f"  {status}: {url}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error(f"Make sure you're running from the correct directory")
        logger.error(f"Expected file structure:")
        logger.error(f"  {project_root}/")
        logger.error(f"  ├── src/financial_market_summary/tools/image_finder.py")
        logger.error(f"  ├── tests/test_images.py (current file)")
        logger.error(f"  └── .env")
        return False
        
    except Exception as e:
        logger.error(f"❌ Image finder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_chart_urls():
    """Test specific financial chart URLs to ensure they work"""
    
    logger.info("\n🎯 Testing Specific Financial Chart URLs...")
    
    # Test URLs that should definitely work
    test_charts = [
        {
            'name': 'AAPL Finviz Chart',
            'url': 'https://finviz.com/chart.ashx?t=AAPL&ty=c&ta=1&p=d&s=l',
            'expected': True
        },
        {
            'name': 'SPY Finviz Chart', 
            'url': 'https://finviz.com/chart.ashx?t=SPY&ty=c&ta=1&p=d&s=l',
            'expected': True
        },
        {
            'name': 'MSFT Yahoo Chart',
            'url': 'https://chart.yahoo.com/z?s=MSFT&t=1d&q=l&l=on&z=s',
            'expected': True
        },
        {
            'name': 'Market Sector Performance',
            'url': 'https://finviz.com/grp_image.ashx?bar_sector_t.png',
            'expected': True
        }
    ]
    
    working_charts = []
    
    for chart in test_charts:
        try:
            import requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.head(chart['url'], timeout=10, headers=headers, allow_redirects=True)
            
            if response.status_code == 200:
                logger.info(f"  ✅ {chart['name']}: Working")
                working_charts.append(chart)
            else:
                logger.warning(f"  ❌ {chart['name']}: HTTP {response.status_code}")
        
        except Exception as e:
            logger.warning(f"  ❌ {chart['name']}: {str(e)}")
    
    logger.info(f"\n📊 Chart URL Test Results: {len(working_charts)}/{len(test_charts)} charts accessible")
    
    if working_charts:
        logger.info("✅ Working chart URLs found:")
        for chart in working_charts:
            logger.info(f"   📈 {chart['name']}: {chart['url']}")
    else:
        logger.error("❌ No working chart URLs found - check network connectivity")
    
    return working_charts

def generate_test_markdown():
    """Generate a test markdown with real financial images"""
    
    logger.info("\n📝 Generating test markdown with real financial images...")
    
    # Get working charts
    working_charts = test_specific_chart_urls()
    
    if not working_charts:
        logger.error("❌ Cannot generate test markdown - no working chart URLs")
        return
    
    # Create test markdown
    markdown_content = f"""# Financial Market Test Report
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Market Overview
This is a test of the financial image finder functionality.

## Charts Found
"""
    
    for chart in working_charts:
        markdown_content += f"\n### {chart['name']}\n"
        markdown_content += f"![{chart['name']}]({chart['url']})\n"
    
    markdown_content += f"""
## Summary
Successfully found {len(working_charts)} working financial chart URLs.
These images can be used in the actual financial summaries.
"""
    
    # Save to file in project root
    try:
        test_file = project_root / "test_financial_images.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"✅ Test markdown saved to: {test_file}")
        logger.info("📄 You can open this file to verify the images display correctly")
        
        print("\n" + "="*50)
        print("TEST MARKDOWN CONTENT:")
        print("="*50)
        print(markdown_content)
        print("="*50)
        
    except Exception as e:
        logger.error(f"❌ Failed to save test markdown: {e}")

def check_file_structure():
    """Check if the expected file structure exists"""
    logger.info("\n🔍 Checking file structure...")
    
    required_files = [
        src_dir / "financial_market_summary" / "__init__.py",
        src_dir / "financial_market_summary" / "tools" / "__init__.py", 
        src_dir / "financial_market_summary" / "tools" / "image_finder.py",
        project_root / ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        if file_path.exists():
            logger.info(f"  ✅ Found: {file_path}")
        else:
            logger.error(f"  ❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing {len(missing_files)} required files")
        return False
    else:
        logger.info("✅ All required files found")
        return True

if __name__ == "__main__":
    print("🧪 Financial Image Finder Test Suite")
    print("="*50)
    
    # First check file structure
    logger.info("Step 0: File Structure Check")
    if not check_file_structure():
        print("❌ File structure check failed - please fix missing files first")
        sys.exit(1)
    
    # Test 1: Basic image finder functionality
    logger.info("Test 1: Basic Image Finder Functionality")
    success = test_image_finder()
    
    if success:
        logger.info("✅ Image finder test completed successfully")
        
        # Test 2: Generate test markdown
        logger.info("\nTest 2: Generate Test Markdown")
        generate_test_markdown()
        
        print("\n🎉 All tests completed!")
        print("💡 If images are displaying correctly, the image finder is working properly")
        print(f"💡 Now you can run the main workflow from: {project_root}")
        
    else:
        logger.error("❌ Image finder test failed")
        print("\n❌ Tests failed - check the logs above for details")