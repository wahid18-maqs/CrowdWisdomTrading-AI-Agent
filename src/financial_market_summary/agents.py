from crewai import Agent,LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
import os
from .tools.tavily_search import TavilyFinancialTool, SerperFinancialTool
from .tools.telegram_sender import TelegramSender
from .tools.image_finder import ImageFinder
from .tools.translator import MultiLanguageTranslator
from langchain_google_genai import ChatGoogleGenerativeAI

class FinancialAgents:
    """Factory class for creating specialized financial AI agents"""
    
    def __init__(self):
        # Initialize tools
        self.tavily_tool = TavilyFinancialTool()
        self.serper_tool = SerperFinancialTool()
        self.telegram_sender = TelegramSender()
        self.image_finder = ImageFinder()
        self.translator = MultiLanguageTranslator()
        
        # Initialize the Gemini LLM client
        self.llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )

    def search_agent(self):
        """Agent specialized in searching financial news"""
        return Agent(
            role='Financial News Search Specialist',
            goal='Search and gather the latest US financial news and market data from the past hour',
            backstory="""You are an expert financial news researcher with deep knowledge of 
            US markets. You excel at finding the most relevant and recent financial news from 
            reliable sources like Bloomberg, Reuters, CNBC, and MarketWatch. You understand 
            the importance of timing in financial markets and focus on finding news that could 
            impact trading decisions. You are skilled at identifying key market movers, 
            earnings reports, economic data releases, and breaking financial news.""",
            tools=[self.tavily_tool, self.serper_tool],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=3,
            llm=self.gemini_llm # Use the new Gemini LLM object
        )
    
    def summary_agent(self):
        """Agent specialized in creating financial summaries"""
        return Agent(
            role='Senior Financial Market Analyst',
            goal='Create concise, actionable financial market summaries under 500 words',
            backstory="""You are a seasoned financial analyst with 15+ years of experience 
            in equity research and market analysis. You have worked at top-tier investment 
            banks and have a talent for distilling complex financial information into clear, 
            actionable insights. You understand what information is most valuable to traders, 
            investors, and financial professionals. You excel at identifying market trends, 
            key price movements, earnings impacts, and economic indicators that drive market 
            sentiment. Your summaries are known for being precise, well-structured, and 
            highly valuable for decision-making.""",
            tools=[],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm # Use the new Gemini LLM object
        )
    
    def formatting_agent(self):
        """Agent specialized in formatting content with visual elements"""
        return Agent(
            role='Financial Content Formatter and Visualization Expert',
            goal='Format financial summaries with relevant charts, graphs, and visual elements',
            backstory="""You are a skilled financial content specialist who combines 
            analytical expertise with visual design knowledge. You understand that financial 
            information is best conveyed through a combination of text and visual elements. 
            You are expert at finding relevant stock charts, market graphs, and financial 
            visualizations that support and enhance written analysis. You know how to structure 
            content for maximum readability and impact, and you understand which types of 
            charts and graphs are most valuable for different types of financial news. You 
            excel at integrating visual elements seamlessly into financial reports.""",
            tools=[self.image_finder],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm # Use the new Gemini LLM object
        )
    
    def translation_agent(self):
        """Agent specialized in financial translation"""
        return Agent(
            role='Multilingual Financial Translation Specialist',
            goal='Translate financial content accurately while preserving technical terminology and formatting',
            backstory="""You are a professional financial translator with expertise in 
            Arabic, Hindi, and Hebrew. You have deep understanding of financial markets and 
            terminology in multiple languages. You worked for international investment banks 
            and financial news organizations, specializing in making complex financial content 
            accessible to diverse global audiences. You understand the nuances of financial 
            terminology across different cultures and markets. You are meticulous about 
            preserving the accuracy of numbers, stock symbols, and technical terms while 
            ensuring the translated content flows naturally in the target language. You know 
            that precision in financial translation can impact investment decisions.""",
            tools=[self.translator],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm # Use the new Gemini LLM object
        )
    
    def send_agent(self):
        """Agent specialized in content distribution"""
        return Agent(
            role='Financial Content Distribution Manager',
            goal='Efficiently distribute financial summaries to multiple channels and languages',
            backstory="""You are a digital content distribution expert specializing in 
            financial communications. You have extensive experience managing multi-channel 
            content delivery for financial services companies. You understand the importance 
            of timely, accurate financial communication and you excel at formatting content 
            appropriately for different platforms and audiences. You are skilled at managing 
            Telegram channels, ensuring content is properly formatted with appropriate 
            headers, timestamps, and visual elements. You understand the global nature of 
            financial markets and the need to serve diverse linguistic audiences. You are 
            meticulous about maintaining consistent quality across all distribution channels.""",
            tools=[self.telegram_sender],
            verbose=True,
            memory=True,
            allow_delegation=False,
            max_iter=2,
            llm=self.gemini_llm # Use the new Gemini LLM object
        )
    
    def create_crew_agents(self):
        """Create all agents for the crew"""
        return [
            self.search_agent(),
            self.summary_agent(), 
            self.formatting_agent(),
            self.translation_agent(),
            self.send_agent()
        ]
    
    def get_agent_by_role(self, role: str):
        """Get specific agent by role"""
        agent_mapping = {
            'search': self.search_agent,
            'summary': self.summary_agent,
            'formatting': self.formatting_agent,
            'translation': self.translation_agent,
            'send': self.send_agent
        }
        
        agent_func = agent_mapping.get(role.lower())
        if agent_func:
            return agent_func()
        else:
            raise ValueError(f"Unknown agent role: {role}. Available: {list(agent_mapping.keys())}")
    
    def configure_all_agents(self, custom_config: dict = None):
        """Configure all agents with custom settings"""
        if custom_config is None:
            custom_config = {}
        
        # Default configuration
        default_config = {
            "verbose": True,
            "memory": True,
            "allow_delegation": False,
            "llm": self.gemini_llm # Change 4: Use the new Gemini LLM object
        }
        
        # Merge configurations
        config = {**default_config, **custom_config}
        
        
        agents = self.create_crew_agents()
        for agent in agents:
            for key, value in config.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
        
        return agents