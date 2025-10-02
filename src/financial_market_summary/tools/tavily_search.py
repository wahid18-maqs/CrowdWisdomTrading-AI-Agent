from crewai.tools import BaseTool
from typing import Dict, Any, Type, Optional, List
from pydantic import BaseModel, Field
import requests
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import json
import re
from urllib.parse import urlparse
from pathlib import Path

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TavilySearchInput(BaseModel):
    """Input schema for the financial search tool."""

    query: str = Field(..., description="The search query for financial news.")
    hours_back: int = Field(
        default=1, description="Number of hours back to search for news."
    )
    max_results: int = Field(
        default=10, description="The maximum number of search results to return."
    )

class TavilyFinancialTool(BaseTool):
    """
    Enhanced Tavily search tool with proper source link inclusion and better content structuring.
    """

    name: str = "tavily_financial_search"
    description: str = (
        "Search for financial news across ALL domains using Tavily API. "
        "Stores comprehensive search results in output folder and creates SINGLE concise summary under 400 words for Telegram. "
        "No domain restrictions - searches the entire web for maximum coverage and delivers ONE message."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _get_allowed_domains(self) -> list:
        """Search across ALL domains - no domain restrictions for comprehensive coverage."""
        # Return empty list to search all domains
        return []

    def _get_comprehensive_trusted_domains(self) -> list:
        """Get comprehensive list of all trusted financial news domains."""
        return [
            # Core financial news sites
            "finance.yahoo.com",
            "yahoo.com",
            "investing.com",
            "benzinga.com",
            "cnbc.com",

            # Major financial news outlets
            "reuters.com",
            "bloomberg.com",
            "nasdaq.com",
            "seekingalpha.com",
            "fool.com",
            "thestreet.com",
            "wsj.com",
            "ft.com",
            "morningstar.com",
            "zacks.com",
            "financialpost.com",
            "barrons.com",

            # Additional trusted financial sources
            "sec.gov",
            "federalreserve.gov",
            "bls.gov",
            "treasury.gov"
        ]

    def _safe_format_template(self, template: str, **kwargs) -> str:
        """Safely format template strings with error handling."""
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Template formatting error: {e} for template '{template}' with args {kwargs}")
            # Create fallback by removing template placeholders
            fallback = template
            for key, value in kwargs.items():
                fallback = fallback.replace(f'{{{key}}}', str(value))
            return fallback

    def _run(self, query: str, hours_back: int = 1, max_results: int = 10) -> str:
        """
        Search across ALL domains, store results in output folder, then create summary.
        """
        # ENFORCE 1-hour searches for main workflow queries
        if any(term in query.lower() for term in ['breaking', 'live', 'today', 'stock market', 'financial news']):
            if hours_back > 1:
                logger.warning(f"🚨 FORCING 1-hour search for main workflow query (was {hours_back}h): {query}")
                hours_back = 1

        try:
            # Use the requested timeframe for both search and filtering
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                return "Error: TAVILY_API_KEY not found in environment variables."

            # Enhanced query construction with 1-hour precision
            financial_query = self._build_enhanced_query(query)

            # For 1-hour searches, add breaking news terms
            if hours_back <= 1:
                financial_query = f"{financial_query} breaking news today latest live updates now"
                logger.info(f"🚨 1-HOUR PRECISION QUERY: {financial_query}")

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)

            logger.info(f"⏰ USING TIMEFRAME: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

            logger.info(f"🌐 SEARCHING ALL DOMAINS for financial news from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

            # Search all domains - no restrictions
            logger.info("🔍 Searching across ALL domains without restrictions for comprehensive coverage")

            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_api_key,
                "query": financial_query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": True,  # Get full content for storage
                "max_results": max(max_results, 20),  # Increase for comprehensive coverage

                # NO DOMAIN RESTRICTIONS - search everything
                # "include_domains": [],  # Empty means all domains

                "exclude_domains": ["reddit.com", "twitter.com", "facebook.com", "youtube.com", "instagram.com"],
                "start_published_date": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_published_date": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Filter results by time FIRST
            original_count = len(data.get('results', []))
            filtered_results = self._filter_results_by_time(data.get('results', []), start_time, end_time)

            if not filtered_results:
                logger.warning("No results found across all domains within timeframe")
                logger.info(f"📊 Original results count: {original_count}")
                return "No financial news found across all domains within the search timeframe."

            # Update data with filtered results BEFORE storing
            data['results'] = filtered_results

            # Store search results with ONLY filtered (1-hour) articles
            search_results_file = self._store_search_results(data, query, start_time, end_time)
            logger.info(f"📁 Stored {len(filtered_results)} 1-hour articles (filtered from {original_count}) in: {search_results_file}")

            # Analyze domain coverage
            domain_coverage = self._analyze_domain_coverage(filtered_results)
            logger.info(f"📊 Domain coverage analysis: {len(domain_coverage)} unique domains found")
            for domain, count in sorted(domain_coverage.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  - {domain}: {count} articles")

            # Create two-message format summary for Telegram sending
            summary_dict = self._create_telegram_summary(data, query, start_time, domain_coverage, search_results_file)

            # Find a relevant image using content analysis and search result domains
            content_analysis = summary_dict.get("content_analysis", self._analyze_search_results(filtered_results))

            # Extract domains from filtered results for image search
            search_result_domains = list(domain_coverage.keys())

            # Calculate search timeframe based on hours_back
            if hours_back <= 24:
                search_timeframe = "24h"
            elif hours_back <= 168:  # 7 days
                search_timeframe = "7d"
            else:
                search_timeframe = "1m"

            # Use full summary for image search context
            search_context = summary_dict.get("full_summary", "")
            image_data = self._find_relevant_image(search_context, content_analysis, search_result_domains, search_timeframe, search_results_file)

            # Combine two-message format with image data for Telegram
            summary_for_telegram = self._combine_summary_with_image(summary_dict, image_data)

            # Save telegram summary with image source to output/summary folder
            summary_file = self._save_telegram_summary(summary_for_telegram, image_data, query, start_time)

            logger.info(f"✅ Comprehensive search completed: {len(filtered_results)} results across {len(domain_coverage)} domains")
            logger.info(f"📁 Results stored in: {search_results_file}")
            logger.info(f"📱 Telegram-ready summary created (under 400 words)")
            logger.info(f"💾 Summary with image source saved: {summary_file}")

            return summary_for_telegram

        except requests.exceptions.RequestException as e:
            error_msg = f"Tavily API request failed: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Tavily search error: {e}"
            logger.error(error_msg)
            return error_msg

    def _filter_results_by_time(self, results: list, start_time: datetime, end_time: datetime) -> list:
        """Filter results to ensure they fall within the requested time window."""
        filtered_results = []
        excluded_count = 0
        no_date_count = 0

        # Calculate time window duration for adaptive filtering
        time_window_hours = (end_time - start_time).total_seconds() / 3600
        logger.info(f"⏰ Filtering {len(results)} results for {time_window_hours:.1f} hour timeframe: {start_time} to {end_time}")

        for result in results:
            published_date = result.get('published_date')
            title = result.get('title', 'Unknown')[:50]

            if not published_date:
                if time_window_hours <= 1:
                    domain = urlparse(result.get("url", "")).netloc.lower()
                    if domain.startswith("www."):
                        domain = domain[4:]

                    # Use full trusted domains list
                    trusted_domains = set(self._get_comprehensive_trusted_domains())

                    if self._is_likely_breaking_news(result) or domain in trusted_domains:
                        filtered_results.append(result)
                        logger.info(f"✅ INCLUDED (no date, trusted/breaking): {title}... ({domain})")
                    else:
                        no_date_count += 1
                        logger.info(f"❌ EXCLUDED (no date, untrusted): {title}... ({domain})")
                    continue
                # For wider time windows (>4 hours), include articles without dates
                elif time_window_hours > 4:
                    filtered_results.append(result)
                    logger.debug(f"📅 Including article without date (wide window): {title}...")
                else:
                    no_date_count += 1
                    logger.debug(f"📅 Excluding article without date (narrow window): {title}...")
                continue
                
            try:
                # Parse published date
                logger.info(f"🔍 DEBUG: Parsing date '{published_date}' for article: {result.get('title', 'Unknown')[:50]}...")

                # Enhanced date parsing with multiple formats and validation
                pub_datetime = self._parse_published_date(published_date, time_window_hours <= 1)
                if pub_datetime is None:
                    # If can't parse and it's a 1-hour search, only include if it looks like breaking news
                    if time_window_hours <= 1:
                        if self._is_likely_breaking_news(result):
                            logger.info(f"🚨 Unparseable date but breaking news: {title}...")
                            filtered_results.append(result)
                        else:
                            logger.info(f"❌ Cannot parse date '{published_date}', not breaking news")
                    else:
                        logger.debug(f"❌ Cannot parse date '{published_date}', excluding")
                    continue

                logger.info(f"📅 Parsed datetime: {pub_datetime}")
                
                # Check if within time window
                hours_ago = (end_time - pub_datetime).total_seconds() / 3600
                logger.info(f"⏰ Time check: article={pub_datetime}, hours_ago={hours_ago:.2f}h, window={time_window_hours}h")

                # STRICT 1-hour filtering
                if time_window_hours <= 1:
                    # For 1-hour searches, only include articles from past 90 minutes (with small buffer)
                    if hours_ago <= 1.5:
                        filtered_results.append(result)
                        logger.info(f"✅ PRECISE 1H: {title}... ({hours_ago:.1f}h ago)")
                    else:
                        excluded_count += 1
                        logger.info(f"❌ TOO OLD for 1h search: {title}... ({hours_ago:.1f}h ago)")
                # Standard time window filtering for longer searches
                elif start_time <= pub_datetime <= end_time:
                    filtered_results.append(result)
                    logger.debug(f"✅ Included (in timeframe): {title}...")
                else:
                    excluded_count += 1
                    logger.debug(f"❌ Excluded (outside timeframe): {title}... ({hours_ago:.1f}h ago)")

            except Exception as e:
                excluded_count += 1
                logger.debug(f"❌ Excluded (date parse error): {result.get('title', 'Unknown')[:50]}... (Date: {published_date}, Error: {e})")

        logger.info(f"📊 Time filtering results: {len(filtered_results)} included, {excluded_count} excluded, {no_date_count} had no date")
        logger.info(f"✅ Final filtered results: {len(filtered_results)} articles from past {time_window_hours:.1f} hours")
        return filtered_results

    def _is_likely_breaking_news(self, result: dict) -> bool:
        """Detect if an article without a date is likely breaking/live news."""
        title = result.get('title', '').lower()
        url = result.get('url', '').lower()

        # Live/breaking indicators in title
        breaking_indicators = [
            'live updates', 'breaking', 'just in', 'developing',
            'stock market today', 'live', 'real-time', 'now',
            'today:', 'alert:', 'urgent:', 'flash:'
        ]

        # Live news URL patterns
        live_url_patterns = [
            'live-updates', 'stock-market-today', 'breaking',
            'real-time', 'today', '/live/', '/now/'
        ]

        # Check title
        if any(indicator in title for indicator in breaking_indicators):
            return True

        # Check URL
        if any(pattern in url for pattern in live_url_patterns):
            return True

        # Check for today's date in URL (YYYY/MM/DD or YYYY-MM-DD)
        from datetime import datetime
        today = datetime.now()
        date_patterns = [
            today.strftime('%Y/%m/%d'),
            today.strftime('%Y-%m-%d'),
            today.strftime('%Y%m%d')
        ]

        if any(pattern in url for pattern in date_patterns):
            return True

        return False

    def _clean_and_format_real_title(self, title: str) -> str:
        """Clean and format actual article titles for better readability."""
        if not title or title.strip() == "":
            return "Breaking Financial Market Update"

        # Remove common unwanted suffixes/prefixes
        cleaned_title = title.strip()

        # Remove source names from end (e.g., "Title - CNN", "Title | Reuters")
        patterns_to_remove = [
            r'\s*[-|]\s*(CNN|Reuters|Bloomberg|Yahoo Finance|MarketWatch|CNBC|Benzinga|Investing\.com).*$',
            r'\s*\|\s*.*$',  # Remove everything after |
            r'\s*-\s*[A-Z][a-z]*\s*$',  # Remove trailing source names
        ]

        for pattern in patterns_to_remove:
            cleaned_title = re.sub(pattern, '', cleaned_title, flags=re.IGNORECASE)

        # Capitalize first letter
        cleaned_title = cleaned_title.strip()
        if cleaned_title:
            cleaned_title = cleaned_title[0].upper() + cleaned_title[1:] if len(cleaned_title) > 1 else cleaned_title.upper()

        # Add emoji if title doesn't already have one
        if not any(ord(char) > 127 for char in cleaned_title):  # No existing emojis
            # Add appropriate emoji based on content
            if any(word in cleaned_title.lower() for word in ['break', 'alert', 'urgent']):
                cleaned_title = f"🚨 {cleaned_title}"
            elif any(word in cleaned_title.lower() for word in ['up', 'gain', 'surge', 'rally']):
                cleaned_title = f"📈 {cleaned_title}"
            elif any(word in cleaned_title.lower() for word in ['down', 'fall', 'drop', 'decline']):
                cleaned_title = f"📉 {cleaned_title}"
            elif any(word in cleaned_title.lower() for word in ['fed', 'rate', 'policy']):
                cleaned_title = f"🏛️ {cleaned_title}"
            elif any(word in cleaned_title.lower() for word in ['earnings', 'report', 'results']):
                cleaned_title = f"💼 {cleaned_title}"
            else:
                cleaned_title = f"📰 {cleaned_title}"

        # Truncate if too long (keep under 100 characters for readability)
        if len(cleaned_title) > 100:
            cleaned_title = cleaned_title[:97] + "..."

        return cleaned_title

    def _parse_published_date(self, published_date: str, is_one_hour_search: bool) -> Optional[datetime]:
        """Enhanced date parsing with multiple formats and strict validation for 1-hour searches."""
        if not published_date or published_date.lower() in ['none', 'null', 'undefined', '']:
            return None

        # Common date formats to try
        formats = [
            '%Y-%m-%dT%H:%M:%SZ',           # ISO format with Z
            '%Y-%m-%dT%H:%M:%S.%fZ',        # ISO with microseconds
            '%Y-%m-%dT%H:%M:%S%z',          # ISO with timezone
            '%Y-%m-%d %H:%M:%S',            # Standard datetime
            '%Y/%m/%d %H:%M:%S',            # Alternative separators
            '%Y-%m-%d',                     # Date only
            '%Y/%m/%d',                     # Date only alternative
            '%m/%d/%Y',                     # US format
            '%d/%m/%Y',                     # European format
            '%m-%d-%Y',                     # US format with dashes
            '%d-%m-%Y',                     # European format with dashes
        ]

        # First try ISO format handling
        if 'T' in published_date:
            try:
                # Clean up common timezone indicators
                clean_date = published_date.replace('Z', '+00:00')
                if '+' not in clean_date and clean_date.endswith('Z'):
                    clean_date = clean_date[:-1] + '+00:00'

                pub_datetime = datetime.fromisoformat(clean_date)
                if pub_datetime.tzinfo is not None:
                    # Convert to UTC
                    pub_datetime = pub_datetime.astimezone().replace(tzinfo=None)

                logger.info(f"📅 ISO parsed: {pub_datetime}")
                return pub_datetime
            except Exception as e:
                logger.debug(f"ISO parsing failed: {e}")

        # Try other formats
        for fmt in formats:
            try:
                pub_datetime = datetime.strptime(published_date.strip(), fmt)
                logger.info(f"📅 Parsed with {fmt}: {pub_datetime}")

                # For 1-hour searches, validate that the date is reasonable (not too far in future/past)
                if is_one_hour_search:
                    now = datetime.utcnow()
                    hours_diff = abs((now - pub_datetime).total_seconds() / 3600)

                    # Reject dates more than 24 hours old or in the future
                    if hours_diff > 24:
                        logger.warning(f"Date seems invalid for 1h search: {hours_diff:.1f}h difference")
                        continue

                return pub_datetime
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {published_date}")
        return None

    def _analyze_domain_coverage(self, results: list) -> dict:
        """Analyze domain coverage to ensure comprehensive search."""
        domain_coverage = {}

        for result in results:
            url = result.get('url', '')
            if url:
                try:
                    domain = urlparse(url).netloc.lower()
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    domain_coverage[domain] = domain_coverage.get(domain, 0) + 1
                except:
                    continue

        return domain_coverage

    def _build_enhanced_query(self, original_query: str) -> str:
        """Build enhanced search query for better financial results."""
        base_query = f"{original_query} US stock market financial news"
        
        query_lower = original_query.lower()
        
        if any(term in query_lower for term in ["earnings", "results", "revenue"]):
            base_query += " earnings report quarterly"
        elif any(term in query_lower for term in ["fed", "interest", "rate"]):
            base_query += " federal reserve interest rates"
        elif any(term in query_lower for term in ["tech", "technology"]):
            base_query += " technology stocks"
        
        return base_query


    def _clean_content(self, content: str) -> str:
        """Clean content of formatting artifacts"""
        if not content:
            return ""
        
        # Remove excessive formatting
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^*]+)\*', r'\1', content)
        content = re.sub(r'[#*_`~]+', '', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()

    def _extract_clean_domain(self, url: str) -> str:
        """Extract clean, readable domain name"""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Map to readable names for all trusted domains
            domain_map = {
                'finance.yahoo.com': 'Yahoo Finance',
                'yahoo.com': 'Yahoo Finance',
                'investing.com': 'Investing.com',
                'benzinga.com': 'Benzinga',
                'cnbc.com': 'CNBC',
                'reuters.com': 'Reuters',
                'bloomberg.com': 'Bloomberg',
                'nasdaq.com': 'NASDAQ',
                'seekingalpha.com': 'Seeking Alpha',
                'fool.com': 'The Motley Fool',
                'thestreet.com': 'TheStreet',
                'wsj.com': 'Wall Street Journal',
                'ft.com': 'Financial Times',
                'morningstar.com': 'Morningstar',
                'zacks.com': 'Zacks Investment Research',
                'financialpost.com': 'Financial Post',
                'barrons.com': "Barron's",
                'sec.gov': 'SEC',
                'federalreserve.gov': 'Federal Reserve',
                'bls.gov': 'Bureau of Labor Statistics',
                'treasury.gov': 'U.S. Treasury'
            }
            
            return domain_map.get(domain, domain.title())
        except:
            return "Financial News"

    def _format_news_date(self, date_str: str) -> str:
        """Format date with month name and year for better readability"""
        if not date_str or date_str == "Unknown date":
            # If no date provided, use current time
            return datetime.now().strftime("%B %d, %Y")

        try:
            if 'T' in date_str:
                # Parse ISO format date
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                if dt.tzinfo is not None:
                    # Convert to local time for display
                    dt = dt.astimezone().replace(tzinfo=None)
                return dt.strftime("%B %d, %Y at %I:%M %p")
            else:
                # Try to parse other date formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime("%B %d, %Y")
                    except ValueError:
                        continue
                # If can't parse, return as is
                return date_str
        except Exception as e:
            logger.debug(f"Date formatting error for '{date_str}': {e}")
            # Fallback to current date if parsing fails
            return datetime.now().strftime("%B %d, %Y")

    def _analyze_search_results(self, results: list) -> Dict[str, list]:
        """Analyze results for key financial information"""
        analysis = {
            "key_stocks": [],
            "key_themes": [],
            "mentioned_companies": [],
            "sectors": [],
            "market_events": [],
            "key_movers": []
        }

        all_content = ""
        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")
            all_content += f" {title} {content}"

        # Extract stocks
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_stocks = re.findall(stock_pattern, all_content)
        
        major_stocks = {
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "NVDA", "META",
            "NFLX", "AMD", "INTC", "CRM", "ADBE", "UBER", "SPOT", "ADSK", "FDX"
        }
        
        stock_frequency = {}
        for stock in potential_stocks:
            if stock in major_stocks:
                stock_frequency[stock] = stock_frequency.get(stock, 0) + 1
        
        analysis["key_stocks"] = [
            stock for stock, _ in sorted(stock_frequency.items(), key=lambda x: x[1], reverse=True)
        ][:6]

        # Find key movers with performance data
        mover_patterns = [
            r'([A-Z]{2,5})\s+(?:surged?|jumped?|gained?|rose|rallied)\s+([\d.]+%)',
            r'([A-Z]{2,5})\s+(?:dropped?|fell|declined?|lost)\s+([\d.]+%)',
        ]
        
        for pattern in mover_patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            for symbol, performance in matches:
                if symbol in major_stocks:
                    analysis["key_movers"].append({
                        "symbol": symbol,
                        "performance": performance
                    })
        
        # Remove duplicate movers
        seen_symbols = set()
        unique_movers = []
        for mover in analysis["key_movers"]:
            if mover["symbol"] not in seen_symbols:
                unique_movers.append(mover)
                seen_symbols.add(mover["symbol"])
        analysis["key_movers"] = unique_movers[:5]

        # Identify themes
        content_lower = all_content.lower()
        theme_keywords = {
            "earnings": ["earnings", "quarterly results", "revenue", "profit"],
            "fed_policy": ["federal reserve", "fed", "interest rate", "rate cut"],
            "technology": ["ai", "tech", "software", "semiconductor"],
            "healthcare": ["healthcare", "pharma", "biotech"],
            "energy": ["oil", "gas", "energy"]
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                analysis["key_themes"].append(theme)

        return analysis

    def _enhance_article_content(self, content: str, analysis: Dict[str, list]) -> str:
        """Enhance content by highlighting key information"""
        if not content:
            return "No content available"

        enhanced = self._clean_content(content)

        # Highlight key stocks
        for stock in analysis["key_stocks"][:3]:
            if stock in enhanced:
                enhanced = enhanced.replace(stock, f"**{stock}**", 1)

        return enhanced

    def _generate_market_implications(self, analysis: Dict[str, list], market_overview: str) -> list:
        """Generate 3-5 comprehensive market implications from analysis"""
        implications = []

        # 1. Theme-based implications (prioritize most important)
        if "fed_policy" in analysis["key_themes"]:
            implications.append("Federal Reserve policy adjustments could significantly impact interest-sensitive sectors and overall market liquidity")

        if "earnings" in analysis["key_themes"]:
            implications.append("Corporate earnings performance indicates underlying business strength and may drive sector rotation strategies")

        if "technology" in analysis["key_themes"]:
            implications.append("Technology sector developments could influence innovation investments and growth stock valuations")

        # 2. Stock movement implications
        if analysis["key_movers"]:
            top_mover = analysis["key_movers"][0]
            symbol = top_mover["symbol"]
            performance = top_mover["performance"]

            if "+" in performance or "gain" in performance.lower():
                implications.append(f"Strong performance in {symbol} may signal positive sentiment in related sectors and potentially drive portfolio rebalancing")
            else:
                implications.append(f"Weakness in {symbol} could indicate sector-specific challenges and may prompt defensive positioning strategies")

        # 3. Market structure and sentiment implications
        if len(analysis["key_stocks"]) > 3:
            implications.append("Broad-based stock activity suggests heightened market volatility and increased trading opportunities across multiple sectors")
        elif analysis["key_stocks"]:
            implications.append("Concentrated attention on specific stocks indicates selective investor focus and potential alpha generation opportunities")

        # 4. Sector and thematic implications
        sector_themes = [theme for theme in analysis["key_themes"] if theme in ["healthcare", "energy", "technology"]]
        if len(sector_themes) > 1:
            implications.append("Multi-sector developments suggest market-wide themes that could drive broad portfolio allocation decisions")

        # 5. Economic and policy context implications
        if len(implications) < 4:
            if "fed_policy" in analysis["key_themes"] or any("rate" in theme for theme in analysis["key_themes"]):
                implications.append("Monetary policy developments may create ripple effects across credit markets and asset valuations")
            else:
                implications.append("Current market dynamics reflect evolving economic conditions that may influence long-term investment strategies")

        # Ensure we have 3-5 implications
        if len(implications) < 3:
            implications.extend([
                "Market activity indicates shifting investor sentiment and potential changes in risk appetite",
                "Current developments may create new opportunities for tactical asset allocation adjustments",
                "Emerging trends suggest importance of monitoring both individual stock performance and broader market themes"
            ])

        return implications[:5]  # Return exactly 3-5 implications

    def _generate_key_points(self, analysis: Dict[str, list], top_results: list) -> list:
        """Generate 3-5 comprehensive key points from analysis and top articles"""
        points = []

        # 1. Stock-specific movement points (prioritize top 2 movers)
        if analysis["key_movers"]:
            for mover in analysis["key_movers"][:2]:
                points.append(f"{mover['symbol']} shows significant movement with {mover['performance']} change")

        # 2. Major theme-based points
        theme_points = []
        if "earnings" in analysis["key_themes"]:
            theme_points.append("Corporate earnings reports driving substantial market activity")
        if "fed_policy" in analysis["key_themes"]:
            theme_points.append("Federal Reserve policy decisions creating significant market impact")
        if "technology" in analysis["key_themes"]:
            theme_points.append("Technology sector developments affecting market sentiment")
        if "healthcare" in analysis["key_themes"]:
            theme_points.append("Healthcare sector showing important developments")
        if "energy" in analysis["key_themes"]:
            theme_points.append("Energy sector movements influencing broader markets")

        # Add top 2 theme points
        points.extend(theme_points[:2])

        # 3. Extract key points from top articles if we need more
        if len(points) < 3:
            for result in top_results[:3]:
                title = result.get("title", "").lower()
                content = result.get("content", "").lower()
                combined = f"{title} {content}"

                # Extract specific financial events
                if "merger" in combined or "acquisition" in combined:
                    points.append("Major merger and acquisition activity affecting market dynamics")
                elif "ipo" in combined or "public offering" in combined:
                    points.append("New market listings creating investor opportunities")
                elif "dividend" in combined:
                    points.append("Dividend announcements impacting shareholder returns")
                elif "partnership" in combined:
                    points.append("Strategic partnerships reshaping competitive landscape")
                elif "guidance" in combined or "outlook" in combined:
                    points.append("Company guidance updates influencing investor expectations")

                if len(points) >= 5:
                    break

        # 4. Market sentiment and trend points
        if len(points) < 4:
            if analysis["key_stocks"]:
                points.append(f"Key stocks {', '.join(analysis['key_stocks'][:3])} showing notable investor attention")

        # 5. Broader market context point
        if len(points) < 5:
            if len(analysis["key_themes"]) > 1:
                points.append("Multiple market themes converging to create complex trading environment")
            else:
                points.append("Market movements reflecting current economic and policy environment")

        # Ensure we have 3-5 points
        if len(points) < 3:
            points.extend([
                "Market activity reflects evolving economic conditions",
                "Investor sentiment responding to latest corporate and policy developments",
                "Key sectors showing differentiated performance patterns"
            ])

        return points[:5]  # Return exactly 3-5 points

    def _store_search_results(self, data: dict, query: str, start_time: datetime, end_time: datetime) -> str:
        """Store simple search results in output/search_results folder as JSON."""
        try:
            # Create output directory structure
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            search_results_dir = project_root / "output" / "search_results"
            search_results_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp and query
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r'[^\w\s-]', '', query).strip()
            safe_query = re.sub(r'[-\s]+', '-', safe_query)[:50]

            filename = f"search_results_{timestamp}_{safe_query}.json"
            filepath = search_results_dir / filename

            # Process articles from search results
            articles = []
            for i, result in enumerate(data.get('results', []), 1):
                title = result.get("title", "")
                url = result.get("url", "")
                source_name = self._extract_clean_domain(url)
                date = self._format_news_date(result.get("published_date", ""))
                content = result.get("content", "")

                # Extract key points and market implications
                key_points = self._extract_article_key_points(title, content)
                market_implications = self._extract_article_market_implications(title, content)

                article = {
                    "article_number": i,
                    "title": title,
                    "source": source_name,
                    "url": url,
                    "date": date,
                    "key_points": key_points,
                    "market_implications": market_implications
                }
                articles.append(article)

            # Create simple search results JSON
            # Calculate search window info
            time_window_hours = (end_time - start_time).total_seconds() / 3600

            search_results_data = {
                "metadata": {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                    "query": query,
                    "total_articles": len(articles),
                    "search_window_hours": time_window_hours,
                    "search_timeframe": f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                    "filtered_results": True,
                    "precision_note": f"Contains ONLY articles from past {time_window_hours:.1f} hour(s)" if time_window_hours <= 1 else f"Filtered for {time_window_hours:.1f} hour timeframe"
                },
                "articles": articles
            }

            # Write JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(search_results_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Search results JSON created: {filename} ({len(articles)} articles)")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Failed to store search results: {e}")
            return f"Error storing results: {e}"

    def _create_summary_from_stored_results(self, data: dict, query: str, start_time: datetime, domain_coverage: dict, results_file: str) -> str:
        """Create a comprehensive summary from stored search results."""
        results = data.get("results", [])
        if not results:
            return f"No results found for summary creation."

        # Content analysis for summary
        content_analysis = self._analyze_search_results(results)

        # Generate timestamp for summary
        summary_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        formatted_output = ["=== COMPREHENSIVE FINANCIAL NEWS SUMMARY ==="]
        formatted_output.append(f"Generated: {summary_timestamp}")
        formatted_output.append(f"📁 Full results stored in: {Path(results_file).name}")
        formatted_output.append("")

        # Search metadata
        formatted_output.append("**SEARCH PARAMETERS:**")
        formatted_output.append(f"📝 Query: '{query}'")
        formatted_output.append(f"🌐 Domain Coverage: ALL domains searched (no restrictions)")
        formatted_output.append(f"⏰ Timeframe: {start_time.strftime('%Y-%m-%d %H:%M')} UTC to {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        formatted_output.append(f"📊 Results Found: {len(results)} articles from {len(domain_coverage)} unique domains")
        formatted_output.append("")

        # Market overview from Tavily's answer
        if data.get("answer"):
            market_overview = self._clean_content(data["answer"])
            formatted_output.append("**MARKET OVERVIEW:**")
            formatted_output.append(market_overview)
            formatted_output.append("")

        # Key findings section
        formatted_output.append("**KEY FINDINGS:**")
        if content_analysis["key_stocks"]:
            formatted_output.append(f"🎯 Key Stocks Mentioned: {', '.join(content_analysis['key_stocks'][:6])}")
        if content_analysis["key_themes"]:
            formatted_output.append(f"📋 Major Themes: {', '.join(content_analysis['key_themes'])}")
        if content_analysis["key_movers"]:
            movers_str = ", ".join([f"{m['symbol']} ({m['performance']})" for m in content_analysis["key_movers"][:3]])
            formatted_output.append(f"🚀 Notable Movers: {movers_str}")
        formatted_output.append("")

        # Top domains contributing content
        formatted_output.append("**SOURCE DIVERSITY:**")
        top_domains = sorted(domain_coverage.items(), key=lambda x: x[1], reverse=True)[:8]
        for domain, count in top_domains:
            clean_domain = self._extract_clean_domain(f"https://{domain}")
            formatted_output.append(f"• {clean_domain}: {count} articles")
        formatted_output.append("")

        # Structured article results in requested format
        formatted_output.append("**STRUCTURED SEARCH RESULTS:**")

        # Get the processed articles from stored data with extracted key points and implications
        stored_articles = []
        for result in results:
            title = result.get("title", "")
            url = result.get("url", "")
            source_name = self._extract_clean_domain(url)

            # Combine source name with URL for complete source attribution
            source_with_url = f"{source_name} ({url})" if url else source_name

            date = self._format_news_date(result.get("published_date", ""))
            content = result.get("content", "")

            # Extract key points and market implications for each article
            key_points = self._extract_article_key_points(title, content)
            market_implications = self._extract_article_market_implications(title, content)

            stored_articles.append({
                "title": title,
                "source": source_with_url,
                "date": date,
                "key_points": key_points,
                "market_implications": market_implications
            })

        # Display in numbered format as requested
        for i, article in enumerate(stored_articles[:10], 1):  # Show top 10 articles
            formatted_output.append(f"**{i}. Title:** {article['title']}")
            formatted_output.append(f"   **Source:** {article['source']}")
            formatted_output.append(f"   **Date:** {article['date']}")
            formatted_output.append(f"   **Key Points:** {' | '.join(article['key_points'])}")
            formatted_output.append(f"   **Market Implications:** {' | '.join(article['market_implications'])}")
            formatted_output.append("")

        # Market implications (3-5 implications)
        formatted_output.append("**MARKET IMPLICATIONS (3-5 POINTS):**")
        implications = self._generate_market_implications(content_analysis, data.get("answer", ""))
        for i, impl in enumerate(implications, 1):
            formatted_output.append(f"{i}. {impl}")
        formatted_output.append("")

        # Key points for analysis (3-5 points)
        formatted_output.append("**KEY POINTS FOR ANALYSIS (3-5 POINTS):**")
        key_points = self._generate_key_points(content_analysis, results[:5])
        for i, point in enumerate(key_points, 1):
            formatted_output.append(f"{i}. {point}")
        formatted_output.append("")

        # Summary footer
        formatted_output.append("**SUMMARY METADATA:**")
        formatted_output.append(f"📈 Content Analysis: {len(content_analysis['key_stocks'])} stocks, {len(content_analysis['key_themes'])} themes")
        formatted_output.append(f"🗄️ Raw Data: {len(results)} articles stored for detailed analysis")
        formatted_output.append(f"🌐 Global Search: No domain restrictions applied")
        formatted_output.append(f"📅 Generated: {summary_timestamp}")

        return "\n".join(formatted_output)

    def _extract_article_key_points(self, title: str, content: str) -> list:
        """Extract comprehensive key points from individual article content."""
        if not content and not title:
            return ["No content available for analysis"]

        combined_text = f"{title} {content}"
        original_text = combined_text  # Keep original for context extraction
        combined_text_lower = combined_text.lower()
        key_points = []

        # Extract detailed Fed/Central Bank information
        fed_patterns = [
            (r'fed(?:eral reserve)?\s+(?:officials?|members?|policymakers?)\s+are\s+meeting\s+in\s+([^,.]+)(?:,\s*([^,.]+))?\s+for\s+(?:a\s+)?([^,.]+)\s+meeting',
             "Fed officials are meeting in {location} for a {type} meeting under unprecedented circumstances"),
            (r'economic\s+projections?\s+(?:will\s+show|indicate|suggest)\s+(?:how\s+)?(.*?)(?:\.|,|\s+amidst)',
             "Economic projections will show {detail} in coming months amidst evolving economic conditions"),
            (r'(?:consumer\s+price\s+index|cpi)\s+rose\s+([\d.]+)%\s+in\s+(\w+)\s+from\s+a\s+year\s+earlier',
             "Consumer Price Index rose {rate}% in {month} from a year earlier"),
            (r'inflation\s+(?:uptick|rise|increase)\s+due\s+to\s+(.*?)(?:,|\.|but)',
             "Recent inflation uptick due to {cause}, affecting monetary policy decisions"),
            (r'consumer\s+inflation\s+has\s+been\s+(.*?)(?:,|\.|despite)',
             "Consumer inflation has been {status}, providing insights for policy direction")
        ]

        for pattern, template in fed_patterns:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:2]:  # Limit to 2 per pattern
                if isinstance(match, tuple):
                    if len(match) >= 3:
                        location = match[0].strip() if match[0] else "Washington, DC"
                        detail = match[2].strip() if match[2] else "pivotal"
                        if 'location' in template and 'type' in template:
                            formatted_point = template.format(location=location.title(), type=detail)
                        elif 'detail' in template:
                            formatted_point = template.format(detail=match[0])
                        elif 'rate' in template and 'month' in template:
                            formatted_point = template.format(rate=match[0], month=match[1].title())
                        elif 'cause' in template:
                            formatted_point = template.format(cause=match[0])
                        elif 'status' in template:
                            formatted_point = template.format(status=match[0])
                        else:
                            formatted_point = template
                    else:
                        if 'cause' in template:
                            formatted_point = template.format(cause=match)
                        elif 'status' in template:
                            formatted_point = template.format(status=match)
                        else:
                            formatted_point = template
                else:
                    if 'cause' in template:
                        formatted_point = template.format(cause=match)
                    elif 'status' in template:
                        formatted_point = template.format(status=match)
                    else:
                        formatted_point = template

                key_points.append(formatted_point)

        # Enhanced economic data extraction
        economic_patterns = [
            (r'(?:unemployment|jobless)\s+rate\s+(?:is|was|stands\s+at|reached?)\s+([\d.]+)%',
             "Unemployment rate {status} at {rate}%, indicating labor market conditions"),
            (r'(?:gdp|gross\s+domestic\s+product)\s+(?:grew|expanded|increased|rose)\s+([\d.]+)%',
             "GDP expanded {rate}% showing economic growth momentum"),
            (r'(?:job|employment)\s+(?:growth|gains?|creation)\s+(?:of|totaled?|reached?)\s+([\d,]+)',
             "Job creation totaled {number} positions, reflecting employment trends"),
            (r'inflation\s+(?:expectations?|forecasts?)\s+(?:are|remain|stand)\s+(?:at\s+)?([\d.]+)%',
             "Inflation expectations remain at {rate}%, guiding monetary policy decisions")
        ]

        for pattern, template in economic_patterns:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:2]:
                if 'rate' in template and 'status' in template:
                    formatted_point = template.format(status="currently", rate=match)
                elif 'rate' in template:
                    formatted_point = template.format(rate=match)
                elif 'number' in template:
                    formatted_point = template.format(number=match)
                else:
                    formatted_point = template
                key_points.append(formatted_point)

        # Detailed corporate earnings and financial metrics
        earnings_patterns = [
            (r'(?:earnings|profits?)\s+(?:per\s+share|eps)\s+(?:of|reached?|came\s+in\s+at)\s+\$?([\d.]+)',
             "Earnings per share reached ${amount}, {performance} analyst expectations"),
            (r'revenue\s+(?:of|reached?|totaled?|came\s+in\s+at)\s+\$?([\d.]+)\s*(billion|million|trillion)?',
             "Revenue totaled ${amount}{unit}, {comparison} previous period performance"),
            (r'(?:profit|net\s+income)\s+(?:of|reached?|totaled?)\s+\$?([\d.]+)\s*(billion|million|trillion)?',
             "Net profit reached ${amount}{unit}, demonstrating operational efficiency"),
            (r'(?:sales|total\s+sales)\s+(?:of|reached?|totaled?)\s+\$?([\d.]+)\s*(billion|million|trillion)?',
             "Total sales reached ${amount}{unit}, indicating market demand trends")
        ]

        for pattern, template in earnings_patterns:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:2]:
                if isinstance(match, tuple):
                    amount = match[0]
                    unit = f" {match[1]}" if match[1] else ""
                    # Determine performance vs expectations
                    performance = "meeting" if "beat" in combined_text_lower or "exceeded" in combined_text_lower else "aligning with"
                    comparison = "outperforming" if "growth" in combined_text_lower or "increased" in combined_text_lower else "compared to"
                    formatted_point = template.format(amount=amount, unit=unit, performance=performance, comparison=comparison)
                else:
                    performance = "meeting" if "beat" in combined_text_lower else "aligning with"
                    formatted_point = template.format(amount=match, performance=performance)
                key_points.append(formatted_point)

        # Enhanced stock movement and market activity
        stock_patterns = [
            (r'([A-Z]{2,5})\s+(?:stock\s+)?(?:surged?|jumped?|soared?|rallied)\s+(?:by\s+)?([\d.]+)%\s+(?:after|following|on)\s+(.*?)(?:\.|,|;)',
             "{symbol} stock surged {percentage}% following {catalyst}, indicating strong market response"),
            (r'([A-Z]{2,5})\s+(?:stock\s+)?(?:dropped?|fell|declined?|plunged)\s+(?:by\s+)?([\d.]+)%\s+(?:after|following|on)\s+(.*?)(?:\.|,|;)',
             "{symbol} stock declined {percentage}% after {catalyst}, reflecting market concerns"),
            (r'([A-Z]{2,5})\s+(?:shares?|stock)\s+(?:are\s+)?(?:up|gained?|rose)\s+([\d.]+)%\s+in\s+(?:pre-market|after-hours|regular)\s+trading',
             "{symbol} shares gained {percentage}% in extended trading, showing investor confidence"),
            (r'market\s+(?:volatility|uncertainty)\s+(?:increased?|rose|surged)\s+(?:due\s+to|following|amid)\s+(.*?)(?:\.|,|;)',
             "Market volatility increased due to {factor}, creating trading opportunities and risks")
        ]

        for pattern, template in stock_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:2]:
                if isinstance(match, tuple) and len(match) >= 3:
                    formatted_point = template.format(symbol=match[0], percentage=match[1], catalyst=match[2].strip())
                elif isinstance(match, tuple) and len(match) == 2:
                    formatted_point = template.format(symbol=match[0], percentage=match[1])
                else:
                    formatted_point = template.format(factor=match)
                key_points.append(formatted_point)

        # Comprehensive market events and policy developments
        event_patterns = [
            (r'(?:federal\s+reserve|fed)\s+(?:meeting|decision|announcement)\s+(.*?)(?:\.|,|;)',
             "Federal Reserve meeting focuses on {topic}, with significant implications for monetary policy"),
            (r'(?:merger|acquisition)\s+(?:deal|agreement)\s+(?:worth|valued\s+at)\s+\$?([\d.]+)\s*(billion|million|trillion)?',
             "Major merger and acquisition activity worth ${amount}{unit} reshaping industry landscape"),
            (r'(?:ipo|initial\s+public\s+offering)\s+(?:of|by)\s+(.*?)\s+(?:raised|valued\s+at)\s+\$?([\d.]+)',
             "{company} IPO raised ${amount}, creating new investment opportunities in public markets"),
            (r'dividend\s+(?:payment|announcement)\s+of\s+\$?([\d.]+)\s+per\s+share',
             "Dividend announcement of ${amount} per share enhancing shareholder returns"),
            (r'(?:partnership|collaboration)\s+(?:between|with)\s+(.*?)\s+(?:and|&)\s+(.*?)(?:\.|,|;)',
             "Strategic partnership between {partner1} and {partner2} creating new market opportunities")
        ]

        for pattern, template in event_patterns:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:2]:
                if isinstance(match, tuple):
                    if len(match) >= 3:
                        if 'partner1' in template and 'partner2' in template:
                            formatted_point = template.format(partner1=match[0].title(), partner2=match[1].title())
                        elif 'company' in template:
                            formatted_point = template.format(company=match[0].title(), amount=match[1])
                        elif 'amount' in template and 'unit' in template:
                            unit = f" {match[1]}" if match[1] else ""
                            formatted_point = template.format(amount=match[0], unit=unit)
                        else:
                            formatted_point = template.format(topic=match[0])
                    elif len(match) == 2:
                        if 'amount' in template and 'unit' in template:
                            unit = f" {match[1]}" if match[1] else ""
                            formatted_point = template.format(amount=match[0], unit=unit)
                        elif 'company' in template:
                            formatted_point = template.format(company=match[0].title(), amount=match[1])
                        else:
                            formatted_point = template.format(topic=match[0])
                    else:
                        formatted_point = template.format(amount=match)
                else:
                    formatted_point = template.format(topic=match)
                key_points.append(formatted_point)

        # Sector-specific developments with context
        sector_patterns = [
            (r'technology\s+(?:sector|stocks?|companies?)\s+(.*?)(?:\.|,|;)',
             "Technology sector {development}, influencing innovation investment strategies"),
            (r'healthcare\s+(?:sector|stocks?|companies?)\s+(.*?)(?:\.|,|;)',
             "Healthcare sector {development}, affecting regulatory and investment landscapes"),
            (r'energy\s+(?:sector|stocks?|companies?)\s+(.*?)(?:\.|,|;)',
             "Energy sector {development}, impacting broader economic conditions"),
            (r'financial\s+(?:sector|stocks?|companies?)\s+(.*?)(?:\.|,|;)',
             "Financial sector {development}, reflecting banking and credit market conditions")
        ]

        for pattern, template in sector_patterns:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:1]:  # One per sector
                formatted_point = template.format(development=match.strip())
                key_points.append(formatted_point)

        # Ensure we have at least 3-5 key points
        if len(key_points) < 3:
            # Add comprehensive default points based on content
            if "federal reserve" in combined_text_lower or "fed" in combined_text_lower:
                key_points.append("Federal Reserve policy decisions creating significant market implications for monetary conditions")
            if "earnings" in combined_text_lower:
                key_points.append("Corporate earnings reports providing insights into business fundamentals and market valuations")
            if "inflation" in combined_text_lower:
                key_points.append("Inflation data influencing central bank policy decisions and market expectations")
            if any(word in combined_text_lower for word in ["stock", "share", "trading"]):
                key_points.append("Stock market activity reflecting investor sentiment and economic conditions")
            if "market" in combined_text_lower:
                key_points.append("Financial market developments indicating evolving economic and policy environment")

            # Add more default points if still not enough
            if len(key_points) < 3:
                additional_defaults = [
                    "Market participants responding to latest economic developments and policy signals",
                    "Investment strategies adapting to current financial conditions and risk factors",
                    "Corporate performance metrics influencing investor decision-making processes",
                    "Economic indicators providing insights into market direction and trends"
                ]
                key_points.extend(additional_defaults[:3 - len(key_points)])

        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for point in key_points:
            if point not in seen:
                seen.add(point)
                unique_points.append(point)

        return unique_points[:6]  # Increased limit to 6 comprehensive key points

    def _extract_article_market_implications(self, title: str, content: str) -> list:
        """Extract comprehensive market implications from individual article content."""
        if not content and not title:
            return ["Unable to determine market implications from available content"]

        combined_text = f"{title} {content}"
        combined_text_lower = combined_text.lower()
        implications = []

        # Fed policy and monetary implications (most important)
        fed_implications = [
            (r'fed(?:eral reserve)?\s+(?:officials?|meeting|decision)\s+.*?(?:signals?|indicate|suggest)\s+(.*?)(?:\.|,|;)',
             "Markets closely watch for signals on {signal}, affecting monetary policy expectations and asset allocation strategies"),
            (r'(?:interest\s+rate|rate)\s+(?:cut|reduction|lower|decrease)\s+(?:of\s+)?([\d.]+)?\s*(?:basis\s+points|%)?',
             "Rate cuts could stimulate economic growth and market liquidity, benefiting risk assets and growth sectors"),
            (r'(?:interest\s+rate|rate)\s+(?:hike|increase|raise)\s+(?:of\s+)?([\d.]+)?\s*(?:basis\s+points|%)?',
             "Rate increases may cool economic activity and affect valuations, particularly in interest-sensitive sectors"),
            (r'inflation\s+(?:target|expectation|forecast)\s+(?:of\s+)?([\d.]+)%',
             "Inflation expectations at {rate}% guide Fed policy decisions, influencing bond yields and equity valuations"),
            (r'(?:economic\s+projections?|outlook)\s+(?:show|indicate|suggest)\s+(.*?)(?:\.|,|;)',
             "Economic projections indicating {outlook} create uncertainty for market participants and policy direction"),
            (r'(?:labor\s+market|employment|jobs)\s+(?:data|conditions?)\s+(.*?)(?:\.|,|;)',
             "Labor market conditions {status} influence Fed policy timing and market expectations for rate adjustments")
        ]

        for pattern, template in fed_implications:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:2]:
                if isinstance(match, tuple):
                    if 'signal' in template:
                        formatted_implication = template.format(signal=match[0].strip())
                    elif 'rate' in template:
                        rate = match[0] if match[0] else "target level"
                        formatted_implication = template.format(rate=rate)
                    elif 'outlook' in template:
                        formatted_implication = template.format(outlook=match[0].strip())
                    elif 'status' in template:
                        formatted_implication = template.format(status=match[0].strip())
                    else:
                        formatted_implication = template
                else:
                    if 'signal' in template:
                        formatted_implication = template.format(signal=match.strip())
                    elif 'outlook' in template:
                        formatted_implication = template.format(outlook=match.strip())
                    elif 'status' in template:
                        formatted_implication = template.format(status=match.strip())
                    else:
                        formatted_implication = template
                implications.append(formatted_implication)

        # Economic data and market sentiment implications
        economic_implications = [
            (r'inflation\s+(?:rose|increased|climbed)\s+(?:to\s+)?([\d.]+)%\s+.*?(?:expectations?|forecasts?)',
             "Inflation rising to {rate}% versus expectations affects Fed policy credibility and market volatility"),
            (r'unemployment\s+(?:rate|data)\s+(?:fell|dropped|declined)\s+(?:to\s+)?([\d.]+)%',
             "Unemployment declining to {rate}% supports economic growth narratives but may influence Fed hawkishness"),
            (r'gdp\s+(?:growth|expansion)\s+(?:of\s+)?([\d.]+)%\s+(?:indicates?|shows?|reflects?)\s+(.*?)(?:\.|,|;)',
             "GDP growth of {rate}% {indication} affects investor confidence in economic resilience and corporate earnings"),
            (r'(?:consumer\s+spending|retail\s+sales)\s+(.*?)(?:\.|,|;)',
             "Consumer spending trends {trend} impact corporate revenue expectations and discretionary sector performance"),
            (r'(?:market\s+volatility|uncertainty)\s+(?:increased?|rose|surged)\s+(?:due\s+to|amid|following)\s+(.*?)(?:\.|,|;)',
             "Increased market volatility due to {factor} creates both portfolio risks and tactical trading opportunities"),
            (r'mixed\s+signals?\s+on\s+(.*?)\s+(?:increase|create|add)\s+uncertainty',
             "Mixed signals on {factor} increase uncertainty, affecting equities and fixed income markets through risk premium adjustments")
        ]

        for pattern, template in economic_implications:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:2]:
                try:
                    if isinstance(match, tuple):
                        if 'rate' in template and 'indication' in template:
                            formatted_implication = template.format(rate=match[0], indication=match[1].strip())
                        elif 'rate' in template:
                            formatted_implication = template.format(rate=match[0])
                        elif 'trend' in template:
                            formatted_implication = template.format(trend=match[0].strip())
                        elif 'factor' in template:
                            formatted_implication = template.format(factor=match[0].strip())
                        else:
                            formatted_implication = template
                    else:
                        if 'rate' in template:
                            formatted_implication = template.format(rate=match)
                        elif 'trend' in template:
                            formatted_implication = template.format(trend=match.strip())
                        elif 'factor' in template:
                            formatted_implication = template.format(factor=match.strip())
                        else:
                            formatted_implication = template
                except (KeyError, IndexError, AttributeError) as e:
                    logger.warning(f"Template formatting error for pattern '{template}' with match '{match}': {e}")
                    # Use template without formatting as fallback
                    formatted_implication = template.replace('{trend}', 'trending').replace('{factor}', 'market factors').replace('{rate}', 'current rate')
                implications.append(formatted_implication)

        # Corporate earnings and financial performance implications
        earnings_implications = [
            (r'(?:earnings|profits?)\s+(?:beat|exceeded|surpassed)\s+(?:expectations?|estimates?)\s+by\s+([\d.]+)%?',
             "Earnings beating expectations by {margin} may boost investor confidence and support sector valuation multiples"),
            (r'(?:earnings|profits?)\s+(?:missed|fell\s+short|disappointed)\s+(?:expectations?|estimates?)\s+by\s+([\d.]+)%?',
             "Earnings missing expectations by {margin} could pressure stock valuations and trigger portfolio rebalancing"),
            (r'revenue\s+(?:growth|increase)\s+of\s+([\d.]+)%\s+(?:indicates?|shows?|reflects?)\s+(.*?)(?:\.|,|;)',
             "Revenue growth of {rate}% indicating {factor} supports business model sustainability and market share expansion"),
            (r'(?:corporate\s+guidance|outlook)\s+(?:raised|increased|upgraded|improved)',
             "Improved corporate guidance may drive positive earnings revisions and support forward-looking valuations"),
            (r'(?:corporate\s+guidance|outlook)\s+(?:lowered|decreased|downgraded|reduced)',
             "Reduced corporate guidance could prompt analyst downgrades and create headwinds for sector performance"),
            (r'(?:margin\s+expansion|profit\s+margins?\s+improved)',
             "Margin expansion demonstrates operational efficiency and pricing power, supporting sustainable profitability growth")
        ]

        for pattern, template in earnings_implications:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:2]:
                if isinstance(match, tuple):
                    if 'margin' in template:
                        formatted_implication = template.format(margin=match[0])
                    elif 'rate' in template and 'factor' in template:
                        formatted_implication = template.format(rate=match[0], factor=match[1].strip())
                    else:
                        formatted_implication = template
                else:
                    if 'margin' in template:
                        formatted_implication = template.format(margin=match)
                    else:
                        formatted_implication = template
                implications.append(formatted_implication)

        # Sector and industry-specific implications
        sector_implications = [
            (r'technology\s+(?:sector|stocks?|companies?)\s+(.*?)(?:\.|,|;)',
             "Technology sector {development} could influence innovation investments, growth stock valuations, and tech-heavy index performance"),
            (r'healthcare\s+(?:sector|stocks?|companies?)\s+(.*?)(?:\.|,|;)',
             "Healthcare sector {development} affects regulatory landscapes, demographic investment themes, and defensive portfolio positioning"),
            (r'energy\s+(?:sector|stocks?|companies?)\s+(.*?)(?:\.|,|;)',
             "Energy sector {development} impacts inflation expectations, commodity cycles, and ESG investment considerations"),
            (r'financial\s+(?:sector|stocks?|companies?)\s+(.*?)(?:\.|,|;)',
             "Financial sector {development} reflects credit conditions, interest rate sensitivity, and economic cycle positioning"),
            (r'(?:merger|acquisition)\s+activity\s+(.*?)(?:\.|,|;)',
             "M&A activity {status} may drive sector consolidation, premium valuations, and strategic repositioning opportunities")
        ]

        for pattern, template in sector_implications:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:1]:  # One per sector
                if 'development' in template:
                    formatted_implication = template.format(development=match.strip())
                elif 'status' in template:
                    formatted_implication = template.format(status=match.strip())
                else:
                    formatted_implication = template
                implications.append(formatted_implication)

        # Market structure and risk implications
        market_structure_implications = [
            (r'(?:volatility|market\s+swings?|price\s+fluctuations?)\s+(.*?)(?:\.|,|;)',
             "Market volatility patterns {pattern} create both portfolio hedging needs and tactical allocation opportunities"),
            (r'(?:institutional|investor)\s+(?:flows?|positioning)\s+(.*?)(?:\.|,|;)',
             "Institutional investor positioning {trend} may signal broader market sentiment shifts and liquidity dynamics"),
            (r'(?:risk\s+appetite|sentiment)\s+(?:remains?|appears?|shows?)\s+(.*?)(?:\.|,|;)',
             "Market sentiment showing {status} influences asset allocation decisions and risk premium requirements across asset classes")
        ]

        for pattern, template in market_structure_implications:
            matches = re.findall(pattern, combined_text_lower, re.IGNORECASE)
            for match in matches[:1]:
                if 'pattern' in template:
                    formatted_implication = template.format(pattern=match.strip())
                elif 'trend' in template:
                    formatted_implication = template.format(trend=match.strip())
                elif 'status' in template:
                    formatted_implication = template.format(status=match.strip())
                else:
                    formatted_implication = template
                implications.append(formatted_implication)

        # Ensure we have at least 3-4 market implications
        if len(implications) < 3:
            # Add comprehensive default implications based on content
            if any(term in combined_text_lower for term in ["fed", "federal reserve", "rate", "monetary policy"]):
                implications.append("Markets closely watch for signals on the Fed's willingness to adjust policy stance, affecting asset allocation and risk appetite")
            if any(term in combined_text_lower for term in ["inflation", "cpi", "pce", "price index"]):
                implications.append("Inflation developments influence monetary policy expectations and create ripple effects across bond and equity markets")
            if any(term in combined_text_lower for term in ["earnings", "revenue", "profits", "guidance"]):
                implications.append("Corporate earnings performance may drive sector rotation strategies and portfolio allocation decisions based on fundamental strength")
            if any(term in combined_text_lower for term in ["unemployment", "jobs", "labor", "employment"]):
                implications.append("Labor market dynamics affect consumer spending power and Fed policy timing, influencing broad market sentiment")
            if any(term in combined_text_lower for term in ["market", "stock", "trading", "volatility"]):
                implications.append("Current market developments may create new opportunities for investment strategy adjustments and risk management considerations")

            # Add more default implications if still not enough
            if len(implications) < 3:
                additional_defaults = [
                    "Economic data releases may influence investor sentiment and drive tactical portfolio positioning decisions",
                    "Sector-specific developments could create both opportunities and risks for diversified investment strategies",
                    "Market volatility patterns may require enhanced risk management and hedging considerations for portfolio protection"
                ]
                implications.extend(additional_defaults[:3 - len(implications)])

        # Remove duplicates while preserving order
        seen = set()
        unique_implications = []
        for implication in implications:
            if implication not in seen:
                seen.add(implication)
                unique_implications.append(implication)

        return unique_implications[:4]  # Increased limit to 4 comprehensive implications

    def _deduplicate_and_rank_points(self, points_list: list) -> list:
        """Deduplicate and rank points by relevance and uniqueness."""
        if not points_list:
            return []

        # Remove duplicates while preserving order
        seen = set()
        unique_points = []

        for point in points_list:
            # Normalize point for comparison (remove emojis, extra spaces)
            normalized = re.sub(r'[^\w\s]', '', point.lower()).strip()

            if normalized and normalized not in seen and len(normalized) > 10:
                seen.add(normalized)
                unique_points.append(point)

        # Score points by financial keywords and length
        scored_points = []
        for point in unique_points:
            score = 0
            point_lower = point.lower()

            # Score by financial relevance
            financial_keywords = ['earnings', 'revenue', 'profit', 'stock', 'market', 'fed', 'rate', 'inflation', 'gdp', 'economic']
            score += sum(1 for keyword in financial_keywords if keyword in point_lower)

            # Score by informativeness (length and detail)
            if 50 <= len(point) <= 150:  # Optimal length range
                score += 2
            elif len(point) > 150:
                score += 1

            # Bonus for specific numbers or percentages
            if re.search(r'\d+\.?\d*%|\$\d+|\d+\.\d+', point):
                score += 1

            scored_points.append((score, point))

        # Sort by score (descending) and return points
        scored_points.sort(key=lambda x: x[0], reverse=True)
        return [point for score, point in scored_points]

    def _create_telegram_summary(self, data: dict, query: str, start_time: datetime, domain_coverage: dict, results_file: str) -> dict:
        """Create two-message format: image caption + full summary for Telegram sending."""
        results = data.get("results", [])
        if not results:
            return {
                "has_image_caption": False,
                "full_summary": "No financial news found for the specified timeframe."
            }

        # Use ALL results for comprehensive analysis
        logger.info(f"📊 Creating two-message format from {len(results)} total articles")

        # Analyze ALL search results for comprehensive insights
        content_analysis = self._analyze_search_results(results)

        # Get the most relevant article for reference
        top_result = results[0] if results else None

        # Use ACTUAL title from search results, not generated one
        if top_result:
            actual_title = self._clean_content(top_result.get("title", ""))
            # Clean up title for better readability
            catchy_title = self._clean_and_format_real_title(actual_title)
            logger.info(f"📰 Using ACTUAL title from search: '{actual_title}' → '{catchy_title}'")
        else:
            catchy_title = "Breaking Financial Market Update"
            logger.info("📰 No search results, using fallback title")

        # Extract comprehensive key points and implications from ALL articles
        all_key_points = []
        all_implications = []

        # Process multiple top articles for diverse insights
        articles_processed = 0
        for i, result in enumerate(results[:5]):
            article_title = result.get("title", "")
            article_content = result.get("content", "")

            if article_title or article_content:
                points = self._extract_article_key_points(article_title, article_content)
                implications = self._extract_article_market_implications(article_title, article_content)

                all_key_points.extend(points)
                all_implications.extend(implications)
                articles_processed += 1

        logger.info(f"📝 Processed {articles_processed} articles for insights extraction")

        # Deduplicate and select best key points and implications
        key_points = self._deduplicate_and_rank_points(all_key_points)[:6]  # Up to 6 key points
        implications = self._deduplicate_and_rank_points(all_implications)[:5]  # Up to 5 implications

        # Ensure minimum content
        if not key_points:
            key_points = [
                "Financial markets showing significant activity based on recent developments",
                "Market participants responding to latest economic data and corporate earnings",
                "Investment sentiment shifting as traders evaluate new information"
            ]

        if not implications:
            implications = [
                "Markets may experience continued volatility as investors digest recent news",
                "Portfolio positioning strategies likely to adjust based on emerging trends"
            ]

        # Create IMAGE CAPTION (with emojis, ≤150 words, ≤1024 chars)
        image_caption = self._create_image_caption(catchy_title, key_points, content_analysis)

        # Create FULL SUMMARY (no emojis, ≤400 words, ≤4096 chars)
        full_summary = self._create_full_summary(catchy_title, key_points, implications, content_analysis)

        logger.info(f"✅ Two-message format created successfully")
        logger.info(f"📝 Image caption: {len(image_caption.split())} words, {len(image_caption)} chars")
        logger.info(f"📄 Full summary: {len(full_summary.split())} words, {len(full_summary)} chars")

        return {
            "has_image_caption": True,
            "image_caption": image_caption,
            "full_summary": full_summary,
            "content_analysis": content_analysis
        }

    def _create_image_caption(self, catchy_title: str, key_points: list, content_analysis: dict) -> str:
        """Create image caption with emojis, ≤150 words, ≤1024 chars."""
        # Start with emoji title
        caption_parts = [f"📈 {catchy_title}", ""]

        # Add top 3 key highlights with emojis
        highlights = []
        for i, point in enumerate(key_points[:3]):
            # Shorten point for caption
            short_point = point[:80] + "..." if len(point) > 80 else point

            # Add appropriate emoji based on content
            if any(word in point.lower() for word in ["fed", "rate", "interest"]):
                emoji = "🏛️"
            elif any(word in point.lower() for word in ["earnings", "profit", "revenue"]):
                emoji = "💰"
            elif any(word in point.lower() for word in ["stock", "surge", "gain"]):
                emoji = "📈"
            elif any(word in point.lower() for word in ["market", "trading"]):
                emoji = "📊"
            elif any(word in point.lower() for word in ["tech", "technology"]):
                emoji = "💻"
            else:
                emoji = "📉" if any(word in point.lower() for word in ["fall", "drop", "decline"]) else "💡"

            highlights.append(f"{emoji} {short_point}")

        caption_parts.extend(highlights)
        caption_parts.extend(["", "Full summary below ⬇️"])

        caption_text = "\n".join(caption_parts)

        # Ensure under limits (≤150 words, ≤1024 chars)
        while len(caption_text.split()) > 140 or len(caption_text) > 1000:
            if len(highlights) > 2:
                highlights = highlights[:2]
            else:
                # Shorten highlights further
                highlights = [h[:60] + "..." if len(h) > 60 else h for h in highlights]

            caption_parts = [f"📈 {catchy_title}", ""]
            caption_parts.extend(highlights)
            caption_parts.extend(["", "Full summary below ⬇️"])
            caption_text = "\n".join(caption_parts)

        logger.info(f"📝 Image caption created: {len(caption_text.split())} words, {len(caption_text)} chars")
        return caption_text

    def _create_full_summary(self, catchy_title: str, key_points: list, implications: list, content_analysis: dict) -> str:
        """Create full summary without emojis, ≤400 words, ≤4096 chars."""
        summary_parts = []

        # Title (no emoji in full summary)
        summary_parts.append(catchy_title)
        summary_parts.append("")

        # Key Points section
        if key_points:
            summary_parts.append("Key Points:")
            for point in key_points:
                summary_parts.append(f"- {point}")
            summary_parts.append("")

        # Market Implications section
        if implications:
            summary_parts.append("Market Implications:")
            for impl in implications:
                summary_parts.append(f"- {impl}")
            summary_parts.append("")

        # Live Charts section
        summary_parts.append("Live Charts:")
        summary_parts.append('🔗 📊 <a href="https://finance.yahoo.com/quote/%5EGSPC/chart/">S&P 500 Chart</a>')
        summary_parts.append('🔗 📈 <a href="https://finance.yahoo.com/quote/%5EIXIC/chart/">NASDAQ Chart</a>')
        summary_parts.append('🔗 📉 <a href="https://finance.yahoo.com/quote/%5EDJI/chart/">Dow Jones Chart</a>')
        summary_parts.append('🔗 ⚡ <a href="https://finance.yahoo.com/quote/%5EVIX/chart/">VIX Chart</a>')
        summary_parts.append('🔗 🏛️ <a href="https://finance.yahoo.com/quote/%5ETNX/chart/">10-Year Chart</a>')
        summary_parts.append('🔗 💰 <a href="https://finance.yahoo.com/quote/GC%3DF/chart/">Gold Chart</a>')

        summary_text = "\n".join(summary_parts)

        # Ensure under limits (≤400 words, ≤4096 chars)
        while len(summary_text.split()) > 390 or len(summary_text) > 4000:
            if len(implications) > 3:
                implications = implications[:3]
            elif len(key_points) > 4:
                key_points = key_points[:4]
            elif len(implications) > 2:
                implications = implications[:2]
            else:
                # Shorten individual points
                key_points = [point[:150] + "..." if len(point) > 150 else point for point in key_points[:3]]
                implications = [impl[:180] + "..." if len(impl) > 180 else impl for impl in implications[:2]]
                break

            # Rebuild summary
            summary_parts = [catchy_title, ""]

            if key_points:
                summary_parts.append("Key Points:")
                for point in key_points:
                    summary_parts.append(f"- {point}")
                summary_parts.append("")

            if implications:
                summary_parts.append("Market Implications:")
                for impl in implications:
                    summary_parts.append(f"- {impl}")
                summary_parts.append("")

            if content_analysis.get("key_stocks"):
                primary_stock = content_analysis["key_stocks"][0]
                summary_parts.append(f"Live Chart: {primary_stock} Stock Performance")
            else:
                summary_parts.append("Live Chart: S&P 500 Market Index")

            summary_text = "\n".join(summary_parts)

        logger.info(f"📄 Full summary created: {len(summary_text.split())} words, {len(summary_text)} chars")
        return summary_text

    def _generate_catchy_title(self, original_title: str, content_analysis: dict, query: str) -> str:
        """Generate a catchy title based on content analysis and market activity."""

        # Extract key elements for title generation
        key_stocks = content_analysis.get("key_stocks", [])
        key_themes = content_analysis.get("key_themes", [])
        key_movers = content_analysis.get("key_movers", [])

        title_lower = original_title.lower()

        # Fed/Policy focused titles
        if any(theme in ["fed_policy"] for theme in key_themes) or any(word in title_lower for word in ["fed", "federal reserve", "rate"]):
            import random
            if "meeting" in title_lower:
                meeting_titles = [
                    "🏛️ Fed's Pivotal Policy Meeting Shakes Markets",
                    "📊 Federal Reserve Meeting Sparks Market Volatility",
                    "🎯 Critical Fed Decision Day: Markets on Edge"
                ]
                return random.choice(meeting_titles)
            elif "cut" in title_lower or "lower" in title_lower:
                cut_titles = [
                    "📉 Fed Rate Cut Signals: Market Rally Ahead?",
                    "🚀 Rate Cut Hopes Drive Market Optimism",
                    "💰 Fed Dovish Stance Ignites Investor Confidence"
                ]
                return random.choice(cut_titles)
            elif "inflation" in title_lower:
                inflation_titles = [
                    "📊 Inflation Data Sparks Fed Policy Speculation",
                    "🔥 Price Pressures Shape Fed's Next Move",
                    "📈 Inflation Metrics Drive Market Expectations"
                ]
                return random.choice(inflation_titles)
            else:
                fed_titles = [
                    "🎯 Federal Reserve Decision Rocks Financial Markets",
                    "🏛️ Fed Policy Shift Sends Shockwaves Through Trading",
                    "⚡ Central Bank Action Transforms Market Sentiment"
                ]
                return random.choice(fed_titles)

        # Earnings focused titles
        elif "earnings" in key_themes or "earnings" in title_lower:
            import random
            if key_movers and any("+" in str(mover.get("performance", "")) for mover in key_movers):
                top_stock = key_movers[0].get("symbol", key_stocks[0] if key_stocks else "Market")
                beat_templates = [
                    f"🚀 {top_stock} Earnings Beat Ignites Market Surge",
                    f"📈 {top_stock} Smashes Expectations: Rally Continues",
                    f"💰 {top_stock} Strong Earnings Fuel Investor Optimism"
                ]
                return random.choice(beat_templates)
            elif key_stocks:
                stock_templates = [
                    f"💼 {key_stocks[0]} Earnings Report Moves Markets",
                    f"📊 {key_stocks[0]} Results Drive Trading Action",
                    f"🎯 {key_stocks[0]} Earnings Shape Market Sentiment"
                ]
                return random.choice(stock_templates)
            else:
                general_earnings = [
                    "📈 Corporate Earnings Drive Market Momentum",
                    "💼 Q4 Results Season Reshapes Investment Landscape",
                    "🚀 Earnings Surprises Fuel Market Optimism"
                ]
                return random.choice(general_earnings)

        # Technology focused titles
        elif "technology" in key_themes or any(word in title_lower for word in ["tech", "ai", "innovation"]):
            if key_stocks:
                return f"⚡ Tech Rally: {key_stocks[0]} Leads Innovation Surge"
            else:
                return "💻 Technology Sector Breakthrough Captivates Investors"

        # Stock movement focused titles
        elif key_movers:
            top_mover = key_movers[0]
            symbol = top_mover.get("symbol", "")
            performance = top_mover.get("performance", "")

            if "+" in performance or "gain" in performance.lower():
                return f"📈 {symbol} Soars {performance}: Market Optimism Peaks"
            else:
                return f"📉 {symbol} Tumbles {performance}: Investor Concerns Mount"

        # Market volatility titles
        elif any(word in title_lower for word in ["volatility", "uncertainty", "swings"]):
            return "🌊 Market Volatility Surge: Opportunities Amid Chaos"

        # Economic data titles
        elif any(word in title_lower for word in ["unemployment", "jobs", "gdp", "economic"]):
            return "📊 Economic Data Shockwave Ripples Through Markets"

        # Sector specific titles
        elif "healthcare" in key_themes:
            return "🏥 Healthcare Sector Breakthrough Attracts Investors"
        elif "energy" in key_themes:
            return "⚡ Energy Market Dynamics Shift Investment Flows"

        # M&A and corporate action titles
        elif any(word in title_lower for word in ["merger", "acquisition", "deal"]):
            return "🤝 Major Corporate Deal Reshapes Industry Landscape"

        # General market titles based on query
        elif "market" in query.lower():
            if key_stocks:
                return f"🎯 Market Focus: {', '.join(key_stocks[:2])} Drive Trading Action"
            else:
                return "📈 Market Dynamics Shift as Investors React"

        # Default catchy titles
        else:
            import random
            catchy_options = [
                "🔥 Breaking: Financial Markets React to Latest Developments",
                "⚡ Market Alert: Key Developments Shape Trading Sentiment",
                "📈 Investor Focus: Critical Market Moves Unfold",
                "🎯 Market Watch: Significant Financial Developments",
                "💼 Wall Street Update: Major Market Movements",
                "🚀 Market Momentum: Key Financial Shifts Drive Action",
                "💡 Trading Spotlight: Major Market Catalysts Emerge",
                "🌟 Financial Flash: Critical Developments Shake Markets",
                "📊 Market Pulse: Significant Events Reshape Trading",
                "⭐ Wall Street Wire: Major Financial Movements Unfold"
            ]

            # Add randomization based on current time and content
            seed = hash(str(content_analysis) + str(original_title) + str(query)) % len(catchy_options)
            return catchy_options[seed]

    def _extract_article_key_points_formatted(self, title: str, content: str, content_analysis: dict) -> list:
        """Extract and format key points specifically for the structured format."""
        key_points = []
        combined_text = f"{title} {content}".lower()

        # Fed/Policy specific points
        if any(term in combined_text for term in ["fed", "federal reserve", "meeting", "rate"]):
            if "meeting" in combined_text:
                key_points.append("Fed officials are meeting in Washington, DC for a pivotal meeting under unprecedented circumstances.")
            if "rate" in combined_text and ("cut" in combined_text or "lower" in combined_text):
                key_points.append("Economic projections will show how aggressively the central bank might lower rates in coming months amidst a precarious labor market.")
            if "inflation" in combined_text:
                key_points.append("Recent inflation uptick due to policy changes, but Fed officials believe this may be temporary.")

        # Inflation and economic data points
        if "inflation" in combined_text or "cpi" in combined_text:
            # Look for specific numbers
            import re
            inflation_numbers = re.findall(r'(\d+\.?\d*)%.*inflation', combined_text)
            if inflation_numbers:
                rate = inflation_numbers[0]
                key_points.append(f"Consumer Price Index rose {rate}% in August from a year earlier, consistent with expectations.")
            else:
                key_points.append("Consumer inflation has been mostly predictable, despite economic disruptions.")

        # Market and earnings points
        if any(term in combined_text for term in ["earnings", "revenue", "profit"]):
            key_points.append("Corporate earnings reports driving substantial market activity and investor sentiment.")

        if any(term in combined_text for term in ["market", "stocks", "trading"]):
            key_points.append("Market conditions reflect ongoing investor attention to economic policy and corporate performance.")

        # Stock-specific points from content analysis
        if content_analysis.get("key_movers"):
            for mover in content_analysis["key_movers"][:2]:
                key_points.append(f"{mover['symbol']} shows significant movement with {mover['performance']} change in trading.")

        # Technology/sector specific points
        if any(term in combined_text for term in ["tech", "technology", "ai"]):
            key_points.append("Technology sector developments affecting market sentiment and investment flows.")

        # Default financial points if none specific found
        if not key_points:
            key_points = [
                "Market activity reflects current economic conditions and policy developments.",
                "Investor sentiment responding to latest corporate and economic data releases.",
                "Key financial indicators showing varied performance patterns across sectors."
            ]

        return key_points[:5]  # Limit to 5 key points

    def _extract_market_implications_formatted(self, title: str, content: str, content_analysis: dict) -> list:
        """Extract and format market implications specifically for the structured format."""
        implications = []
        combined_text = f"{title} {content}".lower()

        # Fed policy implications
        if any(term in combined_text for term in ["fed", "rate", "policy"]):
            implications.append("Markets closely watch for signals on the Fed's willingness to cut rates further.")

        # Economic uncertainty implications
        if any(term in combined_text for term in ["inflation", "jobs", "employment", "economic"]):
            implications.append("Mixed signals on inflation and jobs data increase uncertainty, affecting equities and fixed income markets.")

        # Earnings implications
        if "earnings" in combined_text:
            implications.append("Corporate earnings performance may drive sector rotation strategies and portfolio allocation decisions.")

        # Market volatility implications
        if any(term in combined_text for term in ["volatility", "uncertainty", "risk"]):
            implications.append("Increased market volatility creates both challenges and opportunities for tactical investment strategies.")

        # Technology sector implications
        if any(term in combined_text for term in ["tech", "technology", "innovation"]):
            implications.append("Technology sector developments could influence innovation investments and growth stock valuations.")

        # Default implications if none specific found
        if not implications:
            implications = [
                "Current market dynamics may create new opportunities for investment strategy adjustments.",
                "Emerging trends suggest importance of monitoring both individual performance and broader market themes."
            ]

        return implications[:3]  # Limit to 3 implications

    def _find_relevant_image(self, summary_content: str, content_analysis: dict,
                            search_result_domains: List[str] = None, search_timeframe: str = "24h", search_results_file: str = "") -> dict:
        """Find a relevant image for the summary content with domain and date filtering"""
        try:
            # Import the image finder
            from .image_finder import EnhancedImageFinder

            image_finder = EnhancedImageFinder()

            # Extract stock symbols for image search
            mentioned_stocks = content_analysis.get("key_stocks", [])

            if search_result_domains is None:
                search_result_domains = []

            logger.info(f"🖼️ Searching for relevant image with stocks: {mentioned_stocks}")
            logger.info(f"📍 Using {len(search_result_domains)} search result domains for image filtering")
            logger.info(f"⏰ Image search timeframe: {search_timeframe}")

            # Search for images with enhanced filtering using search results file
            result = image_finder._run(
                search_content=summary_content,
                mentioned_stocks=mentioned_stocks,
                max_images=1,  # We only need one image
                search_results_file=search_results_file
            )

            # Parse the result
            import json
            images = json.loads(result)

            if images and len(images) > 0:
                best_image = images[0]
                logger.info(f"📊 Image finder returned: {best_image.get('title', 'Unknown')}")
                logger.info(f"🔗 URL: {best_image.get('url', 'No URL')}")
                logger.info(f"📱 Telegram Compatible: {best_image.get('telegram_compatible', False)}")

                if best_image.get("telegram_compatible", False):
                    logger.info(f"✅ Returning Telegram-compatible image: {best_image.get('title', 'Unknown')}")
                    return best_image
                else:
                    logger.warning(f"⚠️ Image not compatible but returning anyway: {best_image.get('verification_status', 'Unknown')}")
                    return best_image

            logger.warning("⚠️ No images returned from image finder")
            return None

        except Exception as e:
            logger.error(f"Error finding relevant image: {e}")
            return None

    def _combine_summary_with_image(self, summary_dict: dict, image_data: dict) -> str:
        """Combine two-message format with image data for Telegram sending"""
        if not summary_dict.get("has_image_caption"):
            # No image caption case - return single message
            return summary_dict.get("full_summary", "No summary available.")

        # Get the two messages
        image_caption = summary_dict.get("image_caption", "")
        full_summary = summary_dict.get("full_summary", "")

        # Update Live Chart references if we have actual image data
        if image_data and image_data.get("url"):
            chart_description = image_data.get("title", "Financial Chart")
            full_summary = full_summary.replace("Live Chart: S&P 500 Market Index", f"Live Chart: {chart_description}")
            full_summary = full_summary.replace("Live Chart: ", "Live Chart: ")

        # Create image metadata with detailed logging
        logger.info(f"🖼️ Creating image metadata from image_data: {image_data}")

        if image_data:
            logger.info(f"✅ Image data available: URL={image_data.get('url', 'NO_URL')}, Compatible={image_data.get('telegram_compatible', 'NO_COMPAT')}")
        else:
            logger.warning("❌ No image_data received from image finder")

        image_info = {
            "primary_image": {
                "url": image_data.get("url", "") if image_data else "",
                "title": image_data.get("title", "Financial Chart") if image_data else "",
                "image_description": image_data.get("image_description", "") if image_data else "",
                "description": image_data.get("image_description", "") if image_data else "",
                "type": image_data.get("file_type", "financial_chart") if image_data else "",
                "source": image_data.get("source", "Unknown") if image_data else "",
                "source_article": image_data.get("source_article", "") if image_data else "",
                "telegram_compatible": image_data.get("telegram_compatible", False) if image_data else False,
                "relevance_score": image_data.get("relevance_score", 0) if image_data else 0,
                "extraction_method": image_data.get("extraction_method", "") if image_data else "",
                "content_type": image_data.get("content_type", "") if image_data else "",
                "file_size": image_data.get("file_size", "") if image_data else ""
            }
        }

        logger.info(f"📊 Final primary_image structure: URL='{image_info['primary_image']['url']}', Compatible={image_info['primary_image']['telegram_compatible']}")

        # Format for Telegram bot parsing
        telegram_format = {
            "message_type": "two_message_format",
            "has_image": bool(image_data and image_data.get("url")),
            "message_1_image_caption": image_caption,
            "message_2_full_summary": full_summary,
            "image_metadata": image_info
        }

        # Return in the format expected by Telegram sender
        combined_content = "=== TELEGRAM_TWO_MESSAGE_FORMAT ===\n"
        combined_content += f"Message 1 (Image Caption):\n{image_caption}\n\n"
        combined_content += f"Message 2 (Full Summary):\n{full_summary}\n\n"
        combined_content += "---TELEGRAM_IMAGE_DATA---\n"
        combined_content += json.dumps(image_info, indent=2)

        logger.info(f"📝 Two-message format combined with image data")
        logger.info(f"📊 Has image: {bool(image_data and image_data.get('url'))}")
        logger.info(f"📏 Combined content length: {len(combined_content)} chars")
        logger.info(f"🔍 Contains IMAGE_DATA marker: {'---TELEGRAM_IMAGE_DATA---' in combined_content}")
        logger.info(f"📄 Last 200 chars: ...{combined_content[-200:]}")

        return combined_content

    def _save_telegram_summary(self, summary_content: str, image_data: dict, query: str, start_time: datetime) -> str:
        """Save telegram summary with image source information to output/summary folder as JSON"""
        try:
            # Create output directory structure
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            output_dir = project_root / "output"
            output_dir.mkdir(exist_ok=True)

            # Create summary subdirectory
            summary_dir = output_dir / "summary"
            summary_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp and query
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r'[^\w\s-]', '', query).strip()
            safe_query = re.sub(r'[-\s]+', '-', safe_query)[:50]

            filename = f"telegram_summary_{timestamp}_{safe_query}.json"
            filepath = summary_dir / filename

            # Extract clean summary content (remove image data section if present)
            clean_summary = summary_content
            if "---TELEGRAM_IMAGE_DATA---" in summary_content:
                clean_summary = summary_content.split("---TELEGRAM_IMAGE_DATA---")[0].strip()

            # Prepare summary data structure
            summary_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "search_timeframe": start_time.isoformat(),
                    "summary_length_chars": len(clean_summary),
                    "summary_length_words": len(clean_summary.split())
                },
                "telegram_summary": {
                    "raw_content": clean_summary,
                    "formatted_content": clean_summary.replace('\n\n', '\n').strip()
                },
                "image_information": {
                    "has_image": image_data is not None,
                    "image_url": image_data.get("url", "") if image_data else "",
                    "image_title": image_data.get("title", "") if image_data else "",
                    "image_description": image_data.get("image_description", "") if image_data else "",
                    "image_source": image_data.get("source", "") if image_data else "",
                    "source_article_link": image_data.get("source_article", "") if image_data else "",
                    "source_domain": image_data.get("source", "") if image_data else "",
                    "telegram_compatible": image_data.get("telegram_compatible", False) if image_data else False,
                    "relevance_score": image_data.get("relevance_score", 0) if image_data else 0,
                    "image_type": image_data.get("file_type", "") if image_data else "",
                    "extraction_method": image_data.get("extraction_method", "") if image_data else ""
                }
            }

            # Write JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"💾 Telegram summary saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save telegram summary: {e}")
            return f"Error saving summary: {e}"

    def _add_image_info_to_summary(self, summary: str, content_analysis: dict, results: list) -> str:
        """Add image information to summary for Telegram delivery."""

        # Determine best image based on content analysis
        image_suggestions = []

        # Stock charts for mentioned stocks
        if content_analysis["key_stocks"]:
            primary_stock = content_analysis["key_stocks"][0]
            image_suggestions.append({
                "type": "stock_chart",
                "symbol": primary_stock,
                "url": f"https://chart.yahoo.com/z?s={primary_stock}&t=1d&q=l&l=on&z=s&p=s",
                "description": f"{primary_stock} Stock Chart"
            })

        # Market index chart (always include as backup)
        image_suggestions.append({
            "type": "market_index",
            "symbol": "^GSPC",
            "url": "https://chart.yahoo.com/z?s=%5EGSPC&t=1d&q=l&l=on&z=s&p=s",
            "description": "S&P 500 Index Chart"
        })

        # Add concise image info for single message
        enhanced_summary = summary + "\n"
        enhanced_summary += f"**📊 Chart:** {image_suggestions[0]['description'] if image_suggestions else 'Market Index'}"

        # Add metadata for the system to use
        enhanced_summary += f"\n---TELEGRAM_IMAGE_DATA---\n"
        enhanced_summary += json.dumps({
            "primary_image": image_suggestions[0] if image_suggestions else None,
            "backup_images": image_suggestions[1:] if len(image_suggestions) > 1 else [],
            "content_analysis": {
                "key_stocks": content_analysis["key_stocks"][:3],
                "key_themes": content_analysis["key_themes"],
                "total_articles": len(results)
            }
        })

        return enhanced_summary