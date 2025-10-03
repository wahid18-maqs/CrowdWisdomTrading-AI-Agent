import os
import logging
from typing import Optional
from crewai import Agent, LLM
from crewai.tools import BaseTool
from .tools.telegram_sender import EnhancedTelegramSender
from .tools.image_finder import EnhancedImageFinder as ImageFinder
from .tools.tavily_search import TavilyFinancialTool
from .tools.translator import TranslatorTool
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FinancialAgents:
    """Factory class to create specialized financial agents for the workflow with web source and image verification."""
    
    def __init__(self):
        """Initialize tools used by agents and the LLM."""
        self.telegram_sender = EnhancedTelegramSender()
        self.image_finder = ImageFinder()
        self.tavily_tool = TavilyFinancialTool()
        self.translator = TranslatorTool()
        
        # Set up the LLM with Google Gemini 2.5 Flash
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        self.llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=google_api_key,
            temperature=0.3,
            max_tokens=2048,
            request_timeout=60,
        )

        # Vision-enabled LLM for image analysis
        self.vision_llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=google_api_key,
            temperature=0.1,  # Lower temperature for consistent analysis
            max_tokens=1024,
            request_timeout=60,
        )
            
    def search_agent(self) -> Agent:
        """Creates an agent for comprehensive financial news searching across all trusted domains."""
        return Agent(
            role="Comprehensive Financial News Searcher",
            goal="Search comprehensively across ALL trusted financial domains for the latest US financial news, ensuring complete market coverage from major outlets, specialized sites, premium sources, and government agencies.",
            backstory="You are an expert financial researcher specialized in comprehensive multi-domain news aggregation. You systematically search across all major financial news outlets (Yahoo Finance, CNBC, Reuters, Bloomberg), specialized sites (Seeking Alpha, Benzinga), premium sources (WSJ, FT, Barron's), market data providers (NASDAQ, Morningstar), and government sources (SEC, Fed, Treasury) to ensure no critical market news is missed. You validate source diversity and URL accessibility.",
            tools=[self.tavily_tool],
            llm=self.llm,
            verbose=True
        )

    def summary_agent(self) -> Agent:
        """Creates an agent for summarizing financial news."""
        return Agent(
            role="Financial News Summarizer",
            goal="Analyze financial news and create a concise, structured market summary under 300 words for Telegram with accurate data and stock symbols.",
            backstory="You are an expert financial analyst who excels at distilling complex market data into clear, concise summaries with key points and implications, ensuring all numbers and stock symbols are accurate.",
            tools=[],
            llm=self.llm,
            verbose=True
        )

    def web_searching_source_agent(self) -> Agent:
        """Creates an agent that can search the web to find the best source article for a summary."""
        return Agent(
            role="Web-Searching Financial Source Finder",
            goal="Search the web to find the most relevant and authoritative source article that matches the financial summary title and content, then verify the URL accessibility.",
            backstory="You are an expert researcher who specializes in finding the perfect source articles by searching financial news websites to match summary content with original reporting. You always verify that URLs work and are accessible before recommending them.",
            tools=[self.tavily_tool], 
            llm=self.llm,
            verbose=True
        )

    def enhanced_image_finder_agent(self) -> Agent:
        """Creates an agent that works with pre-extracted images from search results."""
        return Agent(
            role="Financial Image Content Selector",
            goal="Select and verify the most relevant financial images from pre-extracted image results saved during the search phase.",
            backstory="You are an expert visual content specialist who analyzes pre-extracted financial charts and images from image_results JSON files. You select the most relevant images that match the financial summary content and verify their accessibility for Telegram delivery.",
            tools=[],  
            llm=self.llm,
            verbose=True
        )

    def formatting_agent(self) -> Agent:
        """Creates an agent for formatting summaries with verified sources and visuals."""
        return Agent(
            role="Financial Content Formatter with Source and Image Integration",
            goal="Format financial summaries with markdown, integrate verified source links, and add pre-extracted images from image_results, ensuring all content is properly structured for Telegram delivery.",
            backstory="You are a content specialist skilled in creating visually appealing, well-structured financial updates for Telegram. You work with pre-extracted images from image_results files and integrate them with verified source links using professional formatting and proper attribution.",
            tools=[],  
            llm=self.llm,
            verbose=True
        )


    def content_extractor_agent(self) -> Agent:
        """Creates an agent that extracts formatted content from agent final answers."""
        return Agent(
            role="Agent Content Extractor and Formatter",
            goal="Extract properly formatted financial content with verified source links from agent final answers and prepare it for Telegram delivery, preserving all source information, links, and verification status.",
            backstory="You are a content extraction specialist who can parse agent outputs and extract the final formatted content with proper source attribution, markdown formatting, and verification indicators. You ensure that source links from agent answers are preserved exactly as they appear.",
            tools=[],
            llm=self.llm,
            verbose=True
        )

    def header_ordering_agent(self) -> Agent:
        """Creates an agent for ordering message headers by priority for all languages."""
        return Agent(
            role="Financial Content Header Organizer",
            goal="Reorganize financial message headers and sections by priority order to optimize readability and information hierarchy for Telegram delivery across all languages.",
            backstory="You are a content organization specialist who understands financial information hierarchy and user reading patterns. You optimize content structure by prioritizing the most important information first, ensuring that headers and sections are arranged for maximum impact and readability across English, Arabic, Hindi, and Hebrew languages.",
            tools=[],
            llm=self.llm,
            verbose=True
        )

    def image_analysis_agent(self) -> Agent:
        """Creates an agent for analyzing images to determine financial relevance."""
        return Agent(
            role="Financial Image Content Analyzer",
            goal="Analyze extracted images to determine financial relevance and filter out generic content like logos, ads, and author photos.",
            backstory="You are a vision AI specialist who can identify financial charts, graphs, market data, and earnings visualizations while filtering out irrelevant images. You score images based on their financial content relevance.",
            tools=[], 
            llm=self.vision_llm,
            verbose=True
        )

    def send_agent(self) -> Agent:
        """Creates an agent for sending summaries and translations to Telegram."""
        return Agent(
            role="Telegram Content Distributor and Translator",
            goal="Send formatted financial summaries to Telegram in multiple languages (English, Arabic, Hindi, Hebrew), translating content while preserving financial data and routing to language-specific bots.",
            backstory="You are a multilingual communication specialist skilled in distributing financial updates to Telegram channels. You translate content accurately while preserving stock symbols and numbers, and route messages to the appropriate language-specific bots.",
            tools=[self.telegram_sender, self.translator],
            llm=self.llm,
            verbose=True
        )

