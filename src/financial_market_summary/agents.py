import os
import logging
from typing import Optional
from crewai import Agent, LLM
from crewai.tools import BaseTool
from .tools.telegram_sender import EnhancedTelegramSender
from .tools.image_finder import ImageFinder
from .tools.tavily_search import TavilyFinancialTool
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FinancialAgents:
    """Factory class to create specialized financial agents for the workflow."""
    
    def __init__(self):
        """Initialize tools used by agents and the LLM."""
        self.telegram_sender = EnhancedTelegramSender()
        self.image_finder = ImageFinder()
        self.tavily_tool = TavilyFinancialTool()
        
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
            
    def search_agent(self) -> Agent:
        """Creates an agent for searching financial news."""
        return Agent(
            role="Financial News Searcher",
            goal="Search for the latest US financial news from the past 2 hours, focusing on major stock movements, earnings, and economic data.",
            backstory="You are a skilled financial researcher with expertise in identifying the most relevant and recent market news using advanced search tools.",
            tools=[self.tavily_tool],
            llm=self.llm,
            verbose=True
        )

    def summary_agent(self) -> Agent:
        """Creates an agent for summarizing financial news."""
        return Agent(
            role="Financial News Summarizer",
            goal="Analyze financial news and create a concise, structured market summary under 500 words for Telegram.",
            backstory="You are an expert financial analyst who excels at distilling complex market data into clear, concise summaries with key points and implications.",
            tools=[],
            llm=self.llm,
            verbose=True
        )

    def formatting_agent(self) -> Agent:
        """Creates an agent for formatting summaries with visuals."""
        return Agent(
            role="Financial Content Formatter",
            goal="Format financial summaries with markdown and integrate 1-2 relevant charts or images.",
            backstory="You are a content specialist skilled in creating visually appealing, well-structured financial updates for Telegram, incorporating charts and professional formatting.",
            tools=[self.image_finder],
            llm=self.llm,
            verbose=True
        )

    def translation_agent(self) -> Agent:
        """Creates an agent for translating financial summaries."""
        return Agent(
            role="Financial Content Translator",
            goal="Translate financial summaries into multiple languages while preserving stock symbols, numbers, and markdown formatting.",
            backstory="You are a multilingual financial translator with expertise in maintaining accuracy and professional terminology across languages.",
            tools=[],
            llm=self.llm,
            verbose=True
        )

    def send_agent(self) -> Agent:
        """Creates an agent for sending summaries to Telegram."""
        return Agent(
            role="Telegram Content Distributor",
            goal="Send formatted financial summaries and their translations to a Telegram channel using the telegram_sender tool.",
            backstory="You are a communication specialist skilled in distributing financial updates to Telegram channels with proper formatting and reliability.",
            tools=[self.telegram_sender],
            llm=self.llm,
            verbose=True
        )