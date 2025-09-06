from crewai import Agent, LLM
from textwrap import dedent
import os
import logging

logger = logging.getLogger(__name__)


class FinancialMarketAgents:
    def __init__(self, llm=None):
        """
        Initialize agents with a Gemini LLM.
        If llm is not provided, it creates one using the GOOGLE_API_KEY.
        """
        self.llm = llm or self._create_gemini_llm()

    def _create_gemini_llm(self) -> LLM:
        """
        Creates and configures a Gemini LLM instance using GOOGLE_API_KEY.
        """
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.warning("GOOGLE_API_KEY not found. Agents may fail without a valid API key.")
        
        try:
            llm = LLM(
                model="gemini/gemini-2.5-flash",
                api_key=google_api_key or "",
                provider="api",  # ensures we do NOT use Vertex
                temperature=0.3,
                max_tokens=2048,
                request_timeout=60,
            )
            logger.info("Gemini LLM initialized successfully.")
            print("Using LLM provider:", llm.provider)
            return llm

        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            return LLM(
                model="gemini/gemini-2.5-flash",
                api_key=google_api_key or "",
                provider="api"
            )

    # --------------------- Agents ---------------------

    def search_agent(self):
        return Agent(
            role='Financial News Search Specialist',
            goal='Find the most recent and relevant US financial market news with complete article information',
            backstory=dedent("""\
                You are an expert financial news researcher with deep knowledge of US markets.
                You focus on NYSE, NASDAQ, S&P 500, Dow Jones, and major American companies.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_execution_time=300,
        )

    def summary_agent(self):
        return Agent(
            role='Senior Financial Market Analyst',
            goal='Create professional market summaries with visual content awareness',
            backstory=dedent("""\
                Experienced Wall Street analyst summarizing US market news clearly and professionally.
                You consider charts and graphs while writing summaries.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_execution_time=300,
        )

    def image_extraction_agent(self):
        return Agent(
            role='Financial Content Visual Specialist',
            goal='Extract and curate financial charts, graphs, and visual content',
            backstory=dedent("""\
                Expert in financial visual content. Focus on charts, graphs, and infographics that support market news.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_execution_time=600,
            tools=[],
        )

    def formatting_agent(self):
        return Agent(
            role='Financial Content Integration Specialist',
            goal='Combine written market analysis with visual content into professional reports',
            backstory=dedent("""\
                Expert in financial content presentation and formatting for professional multi-channel distribution.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_execution_time=180,
        )

    def translation_agent(self):
        return Agent(
            role='Financial Content Multilingual Specialist',
            goal='Provide accurate translations of financial content while preserving technical accuracy',
            backstory=dedent("""\
                Professional financial translator fluent in English, Arabic, Hindi, and Hebrew. Maintains technical accuracy.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_execution_time=240,
            tools=[],
        )

    def distribution_agent(self):
        return Agent(
            role='Financial Content Distribution Specialist',
            goal='Deliver financial content with images to audiences via Telegram',
            backstory=dedent("""\
                Specialist in multi-channel financial content distribution, ensuring formatting and readability.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_execution_time=300,
            tools=[],
        )

    def quality_assurance_agent(self):
        return Agent(
            role='Financial Content Quality Assurance Specialist',
            goal='Ensure all financial content meets professional standards',
            backstory=dedent("""\
                Meticulous QA specialist verifying accuracy, formatting, and translations for professional financial content.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_execution_time=180,
        )
