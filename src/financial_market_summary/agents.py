# agents.py
from crewai import Agent, LLM
from crewai_tools import SerperDevTool
import os

# It's good practice to import tools from a centralized location
from .tools.tavily_search import TavilyFinancialTool, SerperFinancialTool
from .tools.telegram_sender import TelegramSender
from .tools.image_finder import ImageFinder
from .tools.translator import MultiLanguageTranslator

class FinancialAgents:
    """
    A factory class for creating specialized financial AI agents.
    
    This class initializes all the necessary tools and provides methods
    to create each agent with a pre-defined configuration (role, goal,
    backstory, tools, etc.), powered by a Gemini LLM.
    """
    
    def __init__(self):
        # Initialize all the tools the agents might need
        self.tavily_tool = TavilyFinancialTool()
        self.serper_tool = SerperFinancialTool()
        self.telegram_sender = TelegramSender()
        self.image_finder = ImageFinder()
        self.translator = MultiLanguageTranslator()
        
        # Initialize the Gemini LLM client with a specific model and temperature
        # This LLM will be shared across all agents
        self.gemini_llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )

    def search_agent(self) -> Agent:
        """🔎 Agent specialized in searching for financial news."""
        return Agent(
            role='Financial News Search Specialist',
            goal='Search and gather the latest US financial news and market data from the past hour',
            backstory="""You are an expert financial news researcher with deep knowledge of 
            US markets. You excel at finding the most relevant and recent financial news from 
            reliable sources like Bloomberg, Reuters, CNBC, and MarketWatch. You understand 
            the importance of timing in financial markets and focus on finding news that could 
            impact trading decisions.""",
            tools=[self.tavily_tool, self.serper_tool],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=3,
            llm=self.gemini_llm
        )
    
    def summary_agent(self) -> Agent:
        """✍️ Agent specialized in creating concise financial summaries."""
        return Agent(
            role='Senior Financial Market Analyst',
            goal='Create concise, actionable financial market summaries under 500 words',
            backstory="""You are a seasoned financial analyst with 15+ years of experience 
            in equity research and market analysis. You have worked at top-tier investment 
            banks and have a talent for distilling complex financial information into clear, 
            actionable insights. You excel at identifying market trends, key price movements, 
            earnings impacts, and economic indicators that drive market sentiment.""",
            tools=[],  # This agent processes text, so it doesn't need external tools
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm
        )
    
    def formatting_agent(self) -> Agent:
        """🎨 Agent specialized in formatting content with visual elements."""
        return Agent(
            role='Financial Content Formatter and Visualization Expert',
            goal='Format financial summaries with relevant charts, graphs, and visual elements',
            backstory="""You are a skilled financial content specialist who combines 
            analytical expertise with visual design knowledge. You understand that financial 
            information is best conveyed through a combination of text and visual elements,
            and you are an expert at finding the perfect chart to illustrate a point.""",
            tools=[self.image_finder],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm
        )
    
    def translation_agent(self) -> Agent:
        """🌐 Agent specialized in high-fidelity financial translation."""
        return Agent(
            role='Multilingual Financial Translation Specialist',
            goal='Translate financial content accurately while preserving technical terminology and formatting',
            backstory="""You are a professional financial translator with expertise in 
            Arabic, Hindi, and Hebrew. You have a deep understanding of financial markets and 
            terminology in multiple languages, ensuring that all translations are not just
            linguistically correct but also contextually accurate.""",
            tools=[self.translator],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm
        )
    
    def send_agent(self) -> Agent:
        """🚀 Agent specialized in content distribution via Telegram."""
        return Agent(
            role='Financial Content Distribution Manager',
            goal='Efficiently distribute financial summaries to multiple channels and languages',
            backstory="""You are a digital content distribution expert specializing in 
            financial communications. You have extensive experience managing multi-channel 
            content delivery for financial services companies and ensuring timely, accurate
            dissemination of information.""",
            tools=[self.telegram_sender],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm
        )