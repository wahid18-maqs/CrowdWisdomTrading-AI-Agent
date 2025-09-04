import logging
import os
import time
from typing import Dict, Type
from crewai import Agent, LLM
from .tools.tavily_search import SerperFinancialTool
from .tools.tavily_search import TavilyFinancialTool, SerperFinancialTool
from .tools.telegram_sender import TelegramSender
from .tools.image_finder import ImageFinder
from .tools.translator import MultiLanguageTranslator


logger = logging.getLogger(__name__)


class FinancialAgents:
    """
    A factory class for creating specialized financial AI agents.

    This class manages the initialization of tools and a shared LLM instance,
    providing a consistent way to create and configure agents for a financial
    news workflow.
    """

    def __init__(self):
        """Initializes all the necessary tools and the Gemini LLM client."""
        self.tavily_tool = TavilyFinancialTool()
        self.serper_tool = SerperFinancialTool()
        self.telegram_sender = TelegramSender()
        self.image_finder = ImageFinder()
        self.translator = MultiLanguageTranslator()

        # Initialize the shared Gemini LLM client
        self.gemini_llm = self._create_gemini_llm()

    def _create_gemini_llm(self) -> LLM:
        """
        Creates and configures a Gemini LLM instance.

        This method includes error handling for the API key and sets
        conservative parameters to manage token usage and request timeouts.
        """
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")

            # Configure LLM with conservative settings to manage API usage
            llm = LLM(
                model="gemini/gemini-2.5-flash",
                api_key=google_api_key,
                temperature=0.3,
                max_tokens=2048,
                request_timeout=60,
            )

            logger.info("Gemini LLM initialized successfully.")
            return llm

        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            # Fallback to a basic LLM configuration if initialization fails
            return LLM(
                model="gemini/gemini-2.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=0.3,
            )

    # --- Agent Creation Methods ---

    def search_agent(self) -> Agent:
        """
        Creates an agent specialized in searching for financial news.

        This agent is designed to efficiently find relevant market data and
        news from reliable sources.
        """
        return Agent(
            role="Financial News Search Specialist",
            goal="Search and gather the latest US financial news and market data efficiently.",
            backstory="""You are an expert financial news researcher with deep knowledge of
            US markets. You excel at finding the most relevant and recent financial news from
            reliable sources like Bloomberg, Reuters, CNBC, and MarketWatch. You work efficiently
            to minimize API calls while maximizing information quality.""",
            tools=[self.tavily_tool, self.serper_tool],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm,
        )

    def summary_agent(self) -> Agent:
        """
        Creates an agent specialized in creating concise financial summaries.

        This agent distills complex financial information into clear, actionable
        insights while adhering to strict word count limits.
        """
        return Agent(
            role="Senior Financial Market Analyst",
            goal="Create concise, actionable financial market summaries under 500 words efficiently.",
            backstory="""You are a seasoned financial analyst with 15+ years of experience
            in equity research and market analysis. You excel at distilling complex financial
            information into clear, actionable insights quickly and efficiently. You understand
            the importance of working within API constraints while maintaining quality.""",
            tools=[],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=1,
            llm=self.gemini_llm,
        )

    def formatting_agent(self) -> Agent:
        """
        Creates an agent specialized in formatting content with visual elements.

        This agent combines analytical expertise with design knowledge to create
        well-formatted content with appropriate visual elements.
        """
        return Agent(
            role="Financial Content Formatter and Visualization Expert",
            goal="Format financial summaries with relevant visual elements efficiently.",
            backstory="""You are a skilled financial content specialist who combines
            analytical expertise with visual design knowledge. You work efficiently to
            create well-formatted content with appropriate visual elements while being
            mindful of resource constraints.""",
            tools=[self.image_finder],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=1,
            llm=self.gemini_llm,
        )

    def translation_agent(self) -> Agent:
        """
        Creates an agent specialized in high-fidelity financial translation.

        This agent accurately translates financial content to Arabic, Hindi, or
        Hebrew, preserving technical terminology, numbers, and formatting.
        """
        return Agent(
            role="Multilingual Financial Translation Specialist",
            goal="Translate financial content accurately while preserving technical terminology.",
            backstory="""You are a professional financial translator with expertise in
            Arabic, Hindi, and Hebrew. You work efficiently to provide accurate translations
            while being mindful of API rate limits. You preserve all financial terms,
            numbers, and formatting in your translations.""",
            tools=[self.translator],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=1,
            llm=self.gemini_llm,
        )

    def send_agent(self) -> Agent:
        """
        Creates an agent specialized in content distribution via Telegram.

        This agent ensures the timely and accurate dissemination of financial
        information through Telegram channels.
        """
        return Agent(
            role="Financial Content Distribution Manager",
            goal="Efficiently distribute financial summaries to Telegram channels.",
            backstory="""You are a digital content distribution expert specializing in
            financial communications. You ensure timely, accurate dissemination of information
            through Telegram while working efficiently within system constraints.""",
            tools=[self.telegram_sender],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=1,
            llm=self.gemini_llm,
        )

