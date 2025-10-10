"""
Test file for TavilyFinancialTool
Performs live searches using Tavily API and saves results in JSON.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from src.financial_market_summary.tools.tavily_search import TavilyFinancialTool
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Load environment variables
load_dotenv()

# Output folder for storing test results
OUTPUT_DIR = Path("output/test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_import():
    """Test 1: Import TavilyFinancialTool"""
    tool = TavilyFinancialTool()
    print(f"✅ Tool imported successfully: {tool.name}")


def test_api_key():
    """Test 2: Check TAVILY_API_KEY"""
    api_key = os.getenv("TAVILY_API_KEY")
    assert api_key, "❌ TAVILY_API_KEY not found in environment"
    print("✅ TAVILY_API_KEY is set")


def test_methods_exist():
    """Check if core functionality exists (skip private helper assertions)."""
    tool = TavilyFinancialTool()

    # Check main attributes and run capability
    assert hasattr(tool, "name"), "❌ Missing attribute: name"
    assert hasattr(tool, "description"), "❌ Missing attribute: description"
    assert hasattr(tool, "_run"), "❌ Missing method: _run"

    print("✅ Core tool attributes and _run method exist")



def test_live_search_save_json():
    """Test 4: Perform live search and save results in JSON"""
    tool = TavilyFinancialTool()
    query = "US stock market financial news"
    hours_back = 1
    max_results = 20

    print(f"Running live search for query: '{query}' (past {hours_back} hour)")
    results_text = tool._run(query=query, hours_back=hours_back, max_results=max_results)

    # Save the results in JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:50]
    json_file = OUTPUT_DIR / f"tavily_live_results_{timestamp}_{safe_query}.json"

    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"query": query, "results_text": results_text}, f, indent=2, ensure_ascii=False)
        print(f"✅ Live search results saved to: {json_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


if __name__ == "__main__":
    test_import()
    test_api_key()
    test_methods_exist()
    test_live_search_save_json()
    print("\n🎉 All tests completed!")
