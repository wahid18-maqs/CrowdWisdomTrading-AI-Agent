import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Type
from urllib.parse import urljoin, urlparse
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Input and Tool Definition ---
class ImageFinderInput(BaseModel):
    """Input schema for the financial image finder tool."""

    search_context: str = Field(
        ..., description="The context or content to find relevant financial images for."
    )
    max_images: int = Field(
        default=2, description="The maximum number of images to find."
    )
    image_types: List[str] = Field(
        default=["chart", "graph", "financial"],
        description="The types of images to search for.",
    )


class ImageFinder(BaseTool):
    """
    Finds real, working URLs for financial charts, graphs, and visualizations
    based on news content or a given context.
    """

    name: str = "financial_image_finder"
    description: str = (
        "Find relevant financial charts, graphs, and images based on news content. "
        "Returns real, working URLs for stock charts, market graphs, and financial visualizations."
    )
    args_schema: Type[BaseModel] = ImageFinderInput

    def _run(
        self,
        search_context: str,
        max_images: int = 2,
        image_types: List[str] = None,
    ) -> str:
        """
        Executes the image search process by calling multiple internal methods.
        """
        try:
            if image_types is None:
                image_types = ["chart", "graph", "financial"]

            logger.info(f"Searching for {max_images} financial images.")

            financial_terms = self._extract_financial_terms(search_context)
            logger.info(f"Extracted terms: {financial_terms}")

            images = []

            # Prioritize real stock charts if symbols are found
            if financial_terms["stocks"]:
                stock_images = self._find_real_stock_charts(financial_terms["stocks"])
                images.extend(stock_images)

            # Add market index charts
            if len(images) < max_images:
                index_images = self._find_market_index_charts()
                images.extend(index_images)

            # Search for more images using the Serper API as a fallback
            if len(images) < max_images and os.getenv("SERPER_API_KEY"):
                search_images = self._search_real_financial_images(
                    financial_terms, max_images - len(images)
                )
                images.extend(search_images)

            # Fallback to general, real-time charts if needed
            if len(images) < max_images:
                realtime_images = self._get_realtime_financial_charts(
                    max_images - len(images)
                )
                images.extend(realtime_images)

            # Verify all found images for accessibility and filter out broken links
            verified_images = self._verify_and_filter_images(images)

            return self._format_image_results(verified_images)

        except Exception as e:
            error_msg = f"An error occurred during image search: {e}"
            logger.error(error_msg)
            return error_msg

    # --- Private Helper Methods ---
    def _extract_financial_terms(self, content: str) -> Dict[str, List[str]]:
        """
        Extracts financial terms, such as stock symbols and keywords, from a given string.
        """
        financial_data = {
            "stocks": [],
            "indices": [],
            "sectors": [],
            "keywords": [],
        }

        # Regex to find potential stock symbols (2-5 uppercase letters)
        stock_pattern = r"\b([A-Z]{2,5})\b"
        potential_stocks = re.findall(stock_pattern, content)

        # Common English words to exclude
        exclude_words = {
            "THE",
            "AND",
            "FOR",
            "ARE",
            "BUT",
            "NOT",
            "YOU",
            "ALL",
            "CAN",
            "WAS",
            "ONE",
            "NEW",
            "NOW",
            "OLD",
            "SEE",
            "WHO",
            "ITS",
            "GET",
            "MAY",
            "USE",
            "WAR",
            "FAR",
            "ANY",
            "DAY",
            "END",
            "WAY",
            "OUT",
            "MAN",
            "TOP",
            "PUT",
            "SET",
            "RUN",
            "GOT",
            "LET",
            "NEWS",
            "SAID",
            "ALSO",
            "MADE",
            "OVER",
            "HERE",
            "TIME",
            "YEAR",
            "WEEK",
            "HOUR",
            "PLUS",
            "THAN",
            "ONLY",
            "JUST",
            "LIKE",
            "INTO",
            "MORE",
            "SOME",
            "VERY",
            "WHAT",
            "FROM",
            "THEY",
            "KNOW",
            "WANT",
            "BEEN",
            "GOOD",
            "MUCH",
            "COME",
            "COULD",
            "WOULD",
            "MARKET",
            "STOCK",
            "PRICE",
            "DOWN",
            "CLOSE",
            "OPEN",
            "HIGH",
            "TRADE",
            "SELL",
            "BUY",
            "META",
            "ORCL",
        }

        # Major stock symbols to prioritize, reducing false positives
        major_stocks = {
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "AMZN",
            "TSLA",
            "NVDA",
            "NFLX",
            "AMD",
            "INTC",
            "CRM",
            "ADBE",
            "PYPL",
            "UBER",
            "SPOT",
            "ZOOM",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "C",
            "JNJ",
            "PFE",
            "UNH",
            "CVS",
        }

        # Filter potential symbols
        financial_data["stocks"] = list(
            {
                stock
                for stock in potential_stocks
                if (stock in major_stocks or (stock not in exclude_words))
            }
        )[:5]

        # Major index symbols
        indices = ["SPY", "QQQ", "DIA", "VIX", "IWM", "VTI", "VOO"]
        financial_data["indices"] = [
            idx for idx in indices if idx in content.upper()
        ]

        # General financial keywords
        financial_keywords = [
            "earnings",
            "revenue",
            "profit",
            "growth",
            "trading",
            "market",
            "nasdaq",
            "dow",
        ]
        financial_data["keywords"] = [
            kw for kw in financial_keywords if kw.lower() in content.lower()
        ]

        return financial_data

    def _find_real_stock_charts(
        self, stocks: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Attempts to find real stock charts from multiple reliable sources.
        """
        images = []
        chart_sources = [
            self._get_finviz_chart,
            self._get_yahoo_finance_chart,
            self._get_tradingview_chart,
            self._get_investing_com_chart,
        ]

        for stock in stocks[:3]:
            for chart_source in chart_sources:
                try:
                    chart_data = chart_source(stock)
                    if chart_data:
                        images.append(chart_data)
                        logger.info(
                            f"Found chart for {stock} from {chart_source.__name__}"
                        )
                        break
                except Exception as e:
                    logger.debug(
                        f"Failed to get chart from {chart_source.__name__} for {stock}: {e}"
                    )
        return images

    def _get_finviz_chart(self, symbol: str) -> Dict[str, Any]:
        """Retrieves a daily chart from Finviz."""
        chart_url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
        if self._verify_image_url(chart_url):
            return {
                "url": chart_url,
                "type": "stock_chart",
                "symbol": symbol,
                "source": "finviz",
                "description": f"{symbol} daily stock chart from Finviz",
            }
        return None

    def _get_yahoo_finance_chart(self, symbol: str) -> Dict[str, Any]:
        """Retrieves a daily chart from Yahoo Finance."""
        chart_url = f"https://chart.yahoo.com/z?s={symbol}&t=1d&q=l&l=on&z=s&p=m50,m200&a=v&c="
        if self._verify_image_url(chart_url):
            return {
                "url": chart_url,
                "type": "stock_chart",
                "symbol": symbol,
                "source": "yahoo_finance",
                "description": f"{symbol} stock chart from Yahoo Finance",
            }
        return None

    def _get_tradingview_chart(self, symbol: str) -> Dict[str, Any]:
        """Retrieves a chart snapshot from TradingView."""
        chart_url = f"https://s3.tradingview.com/snapshots/u/{symbol}.png"
        if self._verify_image_url(chart_url):
            return {
                "url": chart_url,
                "type": "stock_chart",
                "symbol": symbol,
                "source": "tradingview",
                "description": f"{symbol} chart snapshot from TradingView",
            }
        return None

    def _get_investing_com_chart(self, symbol: str) -> Dict[str, Any]:
        """Retrieves a daily chart from Investing.com."""
        chart_url = (
            f"https://i-invdn-com.investing.com/charts/us_stocks_{symbol.lower()}_1d.png"
        )
        if self._verify_image_url(chart_url):
            return {
                "url": chart_url,
                "type": "stock_chart",
                "symbol": symbol,
                "source": "investing_com",
                "description": f"{symbol} chart from Investing.com",
            }
        return None

    def _find_market_index_charts(self) -> List[Dict[str, Any]]:
        """
        Finds charts for major market indices.
        """
        images = []
        indices = [
            {"symbol": "SPY", "name": "S&P 500"},
            {"symbol": "QQQ", "name": "NASDAQ"},
            {"symbol": "DIA", "name": "Dow Jones"},
        ]

        for index in indices[:2]:
            chart_url = (
                f"https://finviz.com/chart.ashx?t={index['symbol']}&ty=c&ta=1&p=d&s=l"
            )
            if self._verify_image_url(chart_url):
                images.append(
                    {
                        "url": chart_url,
                        "type": "index_chart",
                        "symbol": index["symbol"],
                        "source": "finviz",
                        "description": f"{index['name']} ({index['symbol']}) index chart",
                    }
                )
        return images

    def _search_real_financial_images(
        self, financial_terms: Dict[str, List[str]], max_images: int
    ) -> List[Dict[str, Any]]:
        """
        Searches for images using the Serper API based on extracted financial terms.
        """
        images = []
        serper_key = os.getenv("SERPER_API_KEY")
        if not serper_key:
            logger.warning("SERPER_API_KEY not found. Skipping Serper search.")
            return images

        query_parts = ["financial chart"]
        if financial_terms["stocks"]:
            query_parts.append(f"{financial_terms['stocks'][0]} stock chart")
        if financial_terms["keywords"]:
            query_parts.extend(financial_terms["keywords"][:2])

        query = " ".join(query_parts) + " market graph today"
        logger.info(f"Serper search query: {query}")

        url = "https://google.serper.dev/images"
        payload = {
            "q": query,
            "num": max_images * 3,
            "gl": "us",
            "safe": "active",
        }
        headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}

        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=20
            )
            response.raise_for_status()
            data = response.json()

            for img_data in data.get("images", []):
                if len(images) >= max_images:
                    break

                img_url = img_data.get("imageUrl") or img_data.get("link")
                title = img_data.get("title", "Financial Chart")

                if self._is_quality_financial_image(title, img_url):
                    if self._verify_image_url(img_url, quick_check=True):
                        images.append(
                            {
                                "url": img_url,
                                "type": "financial_search",
                                "title": title,
                                "source": "serper_search",
                                "description": title,
                            }
                        )
                        logger.info(f"Found quality image: {title}")
        except Exception as e:
            logger.error(f"Serper image search failed: {e}")
        return images

    def _get_realtime_financial_charts(self, max_images: int) -> List[Dict[str, Any]]:
        """
        Provides a fallback list of real-time market overview charts.
        """
        images = []
        market_charts = [
            {
                "url": "https://finviz.com/grp_image.ashx?bar_sector_t.png",
                "type": "sector_performance",
                "source": "finviz",
                "description": "Sector Performance Heatmap",
            },
            {
                "url": "https://finviz.com/grp_image.ashx?bar_industry_d.png",
                "type": "industry_performance",
                "source": "finviz",
                "description": "Industry Performance Overview",
            },
        ]

        for chart in market_charts:
            if len(images) >= max_images:
                break
            if self._verify_image_url(chart["url"], quick_check=True):
                images.append(chart)
        return images

    def _is_quality_financial_image(self, title: str, url: str) -> bool:
        """
        Checks if an image is a high-quality financial chart based on its title and URL.
        """
        if not title or not url:
            return False

        quality_indicators = [
            "chart",
            "graph",
            "stock",
            "market",
            "trading",
            "financial",
            "price",
            "volume",
            "earnings",
            "revenue",
            "index",
            "nasdaq",
            "dow",
            "analysis",
        ]

        low_quality_indicators = [
            "meme",
            "joke",
            "funny",
            "cartoon",
            "logo",
            "icon",
            "avatar",
            "wallpaper",
            "template",
        ]

        text_to_check = (title + " " + url).lower()

        has_quality = any(indicator in text_to_check for indicator in quality_indicators)
        has_low_quality = any(
            indicator in text_to_check for indicator in low_quality_indicators
        )

        return has_quality and not has_low_quality

    def _verify_image_url(self, image_url: str, quick_check: bool = False) -> bool:
        """
        Verifies that an image URL is accessible and valid.
        """
        try:
            if quick_check:
                return image_url.startswith(("http://", "https://"))

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.head(
                image_url, timeout=5, headers=headers, allow_redirects=True
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            return content_type.startswith("image/")
        except Exception as e:
            logger.debug(f"URL verification failed for {image_url}: {e}")
            return False

    def _verify_and_filter_images(
        self, images: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Iterates through a list of images, verifies each URL, and returns a list of
        only the accessible ones.
        """
        verified_images = []
        for img in images:
            url = img.get("url")
            if url and self._verify_image_url(url):
                verified_images.append(img)
                logger.info(f"Verified image: {img.get('description', 'Unknown')}")
            else:
                logger.warning(
                    f"Failed verification for URL: {img.get('url', 'Unknown')}"
                )
        return verified_images

    def _format_image_results(self, images: List[Dict[str, Any]]) -> str:
        """
        Formats the list of verified images into a human-readable string.
        """
        if not images:
            return "No accessible financial images found. Please proceed without images."

        result_parts = ["=== VERIFIED FINANCIAL IMAGES FOUND ===\n"]

        for i, img in enumerate(images, 1):
            image_info = f"""
Image {i}:
- URL: {img['url']}
- Type: {img.get('type', 'unknown')}
- Description: {img.get('description', 'Financial visualization')}
- Source: {img.get('source', 'unknown')}
- Symbol: {img.get('symbol', 'N/A')}
---
"""
            result_parts.append(image_info)

        result_parts.append(f"Total verified images: {len(images)}")
        return "\n".join(result_parts)
    
