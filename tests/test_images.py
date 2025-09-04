import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
import requests
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

logger.info(f"Project root directory: {project_root}")
logger.info(f"Source directory added to path: {src_dir}")
logger.info(f"Current working directory: {Path.cwd()}")

def test_image_finder():
    """
    Tests the main functionality of the ImageFinder class with sample financial content.
    This test verifies that the ImageFinder can correctly identify and retrieve
    relevant images based on financial text containing stock tickers.
    Returns:
        bool: True if the test passes without an exception, False otherwise.
    """
    # Load environment variables from the project root's .env file
    env_file = project_root / ".env"
    load_dotenv(env_file)
    try:
        # Attempt to import the ImageFinder class
        logger.info(f"Attempting to import ImageFinder from: {src_dir}")
        from financial_market_summary.tools.image_finder import ImageFinder
        # Instantiate the image finder
        image_finder = ImageFinder()
        # Define sample financial content for the test
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
        
        logger.info("Starting ImageFinder functionality test with sample financial content.")
        logger.info("Test content contains the following tickers: AAPL, MSFT, TSLA, GOOGL, NVDA, SPY")
        
        # Run the image finder to find images based on the content
        result = image_finder._run(
            search_context=test_content,
            max_images=3,
            image_types=["chart", "graph", "financial"]
        )
        
        logger.info("ImageFinder test results:")
        logger.info("=" * 60)
        logger.info(result)
        logger.info("=" * 60)
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure the project structure is correct.")
        logger.error(f"Expected file structure:")
        logger.error(f"  {project_root}/")
        logger.error(f"  ├── src/financial_market_summary/tools/image_finder.py")
        logger.error(f"  ├── tests/test_images.py (current file)")
        logger.error(f"  └── .env")
        return False
        
    except Exception as e:
        logger.error(f"Image finder test failed due to an exception: {e}")
        traceback.print_exc()
        return False

def test_specific_chart_urls():
    """
    Tests specific financial chart URLs to ensure they are accessible.

    This test makes a HEAD request to predefined chart URLs to verify they
    return a 200 OK status, confirming their availability.

    Returns:
        list: A list of dictionaries for each working chart.
    """
    logger.info("Starting test of specific financial chart URLs.")
    
    # Predefined URLs for testing
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
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.head(chart['url'], timeout=10, headers=headers, allow_redirects=True)
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to '{chart['name']}': Working as expected.")
                working_charts.append(chart)
            else:
                logger.warning(f"Connection failed for '{chart['name']}': HTTP status code {response.status_code}.")
        
        except Exception as e:
            logger.warning(f"Connection failed for '{chart['name']}': {str(e)}")
    
    logger.info(f"Chart URL Test Results: {len(working_charts)}/{len(test_charts)} charts accessible.")
    
    if working_charts:
        logger.info("The following chart URLs are working:")
        for chart in working_charts:
            logger.info(f"  - {chart['name']}: {chart['url']}")
    else:
        logger.error("No working chart URLs found. Please check your network connection.")
    
    return working_charts

def generate_test_markdown():
    """
    Generates a markdown file with real financial images to visually verify functionality.
    The markdown file is saved to the project root directory.
    """
    logger.info("Starting markdown generation with real financial images.")
    working_charts = test_specific_chart_urls()
    if not working_charts:
        logger.error("Cannot generate test markdown as no working chart URLs were found.")
        return
    
    markdown_content = f"""# Financial Market Test Report
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## Market Overview
This is a test of the financial image finder functionality. The charts below are
live images retrieved directly from financial data providers.

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
    
    try:
        test_file = project_root / "test_financial_images.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Test markdown successfully saved to: {test_file}")
        logger.info("Please open this file to verify the images display correctly.")
        
        logger.info("=" * 50)
        logger.info("TEST MARKDOWN CONTENT:")
        logger.info("=" * 50)
        logger.info(markdown_content)
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Failed to save test markdown file: {e}")

def check_file_structure():
    """
    Verifies that the required file structure exists for the tests to run.

    Returns:
        bool: True if all required files are found, False otherwise.
    """
    logger.info("Checking for required file structure.")
    
    required_files = [
        src_dir / "financial_market_summary" / "__init__.py",
        src_dir / "financial_market_summary" / "tools" / "__init__.py", 
        src_dir / "financial_market_summary" / "tools" / "image_finder.py",
        project_root / ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        if file_path.exists():
            logger.info(f"Found required file: {file_path}")
        else:
            logger.error(f"Missing required file: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing {len(missing_files)} required files.")
        return False
    else:
        logger.info("All required files found. File structure check passed.")
        return True

if __name__ == "__main__":
    logger.info("Financial Image Finder Test Suite")
    logger.info("=" * 50)
    
    # Step 0: Check the file structure first to ensure dependencies can be imported
    logger.info("Starting Step 0: File Structure Check")
    if not check_file_structure():
        logger.error("File structure check failed. Please fix the missing files before proceeding.")
        sys.exit(1)
    
    # Step 1: Test basic image finder functionality
    logger.info("\nStarting Test 1: Image Finder Functionality")
    success = test_image_finder()
    
    if success:
        logger.info("Image finder test completed successfully.")
        # Step 2: Generate test markdown
        logger.info("\nStarting Test 2: Generate Test Markdown")
        generate_test_markdown()
        logger.info("\nAll tests completed.")
        logger.info("The image finder is working properly if the images display correctly in the generated markdown file.")
        
    else:
        logger.error("Image finder test failed. See logs above for details.")

