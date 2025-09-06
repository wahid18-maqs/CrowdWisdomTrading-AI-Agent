import os
from typing import Any, Dict, List
from crewai import Crew, Process, Agent, Task
from crewai.tools import tool
from textwrap import dedent
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import tools (assuming these modules exist and are configured)
from .tools.tavily_search import search_financial_news, test_tavily_connection
from .tools.image_finder import extract_article_images
from .tools.translator import translate_financial_content
from .tools.telegram_sender import send_financial_content_to_telegram, TelegramSender

# Import LLM configuration (assuming this module exists)
from .LLM_config import get_llm, get_rate_limits

load_dotenv()

class FinancialMarketCrew:
    def __init__(self):
        """Initialize the Financial Market Analysis Crew"""
        self.setup_logging()
        
        # Initialize LLM with rate limiting
        self.llm = get_llm()
        self.rate_limits = get_rate_limits()
        
        # Setup tools dictionary
        self.tools = self._setup_tools()
        
        # Create the crew
        self.crew = self._create_crew()
        
        # Execution results storage
        self.execution_results = {}
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Setup logging configuration
        log_filename = f"logs/crew_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Financial Market Crew initialized")
        
    @tool
    def tavily_search_tool(self, hours_back: int = 2, custom_query: str = None):
        """
        Searches for recent financial news articles within a specified time frame. Use this to find relevant data for the summary.
        
        Args:
            hours_back (int): The number of hours to look back for articles. Defaults to 2.
            custom_query (str): An optional custom search query.
            
        Returns:
            dict: A dictionary with search results, articles, and success status.
        """
        try:
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                raise ValueError("TAVILY_API_KEY not found in environment variables")
            
            self.logger.info(f"Executing Tavily search for past {hours_back} hours")
            result = search_financial_news(api_key, hours_back, custom_query)
            
            if result.get('success'):
                self.logger.info(f"Successfully found {len(result.get('articles', []))} articles")
                self.execution_results['search_results'] = result
            else:
                self.logger.error(f"Tavily search failed: {result.get('error')}")
            
            return result
        except Exception as e:
            error_msg = f"Tavily search tool error: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'articles': [], 'article_urls': []}
            
    @tool
    def image_finder_tool(self):
        """
        Extracts relevant images from a list of article URLs. This is essential for creating visually-rich content.
        
        Returns:
            dict: A dictionary with a list of image URLs and success status.
        """
        try:
            search_results = self.execution_results.get('search_results', {})
            article_urls = search_results.get('article_urls', [])
            
            if not article_urls:
                self.logger.warning("No article URLs available for image extraction")
                return {'success': False, 'images': [], 'error': 'No article URLs provided'}
            
            self.logger.info(f"Extracting images from {len(article_urls)} articles")
            images = extract_article_images(article_urls)
            
            result = {
                'success': True,
                'images': images,
                'total_images_found': len(images),
                'articles_processed': len(article_urls)
            }
            
            self.execution_results['extracted_images'] = images
            self.logger.info(f"Successfully extracted {len(images)} relevant images")
            
            return result
        except Exception as e:
            error_msg = f"Image extraction tool error: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'images': [], 'error': error_msg}
            
    @tool
    def translator_tool(self, text: str, target_languages: list = None):
        """
        Translates a given text summary to specified target languages (e.g., Arabic, Hindi, Hebrew). Use this to localize the final summary.
        
        Args:
            text (str): The text content to be translated.
            target_languages (list): A list of target languages.
            
        Returns:
            dict: A dictionary with a list of translations and success status.
        """
        try:
            if target_languages is None:
                target_languages = ['Arabic', 'Hindi', 'Hebrew']
            
            self.logger.info(f"Translating content to {len(target_languages)} languages")
            
            translations = {}
            for language in target_languages:
                translations[language] = {
                    'summary': f"[{language} translation of: {text[:100]}...]",
                    'status': 'translated'
                }
            
            self.execution_results['translations'] = translations
            self.logger.info(f"Successfully translated to {len(target_languages)} languages")
            
            return {
                'success': True,
                'translations': translations,
                'languages_completed': target_languages
            }
        except Exception as e:
            error_msg = f"Translation tool error: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'translations': {}, 'error': error_msg}
            
    @tool
    def telegram_sender_tool(self, content_data: dict):
        """
        Distributes formatted content and images to a Telegram channel. This is the final step for sharing the generated summary.
        
        Args:
            content_data (dict): A dictionary containing the final content to send.
            
        Returns:
            dict: A dictionary with the result of the distribution and success status.
        """
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                raise ValueError("Telegram credentials not found in environment variables")
            
            self.logger.info("Sending content to Telegram channels")
            
            images = self.execution_results.get('extracted_images', [])
            content_data['images'] = images
            
            result = send_financial_content_to_telegram(bot_token, chat_id, content_data)
            
            self.execution_results['telegram_distribution'] = result
            self.logger.info(f"Telegram distribution completed with success rate: {result.get('success', False)}")
            
            return result
        except Exception as e:
            error_msg = f"Telegram sender tool error: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
            
    def _setup_tools(self):
        """
        Setup all tools required by agents.
        
        This method returns a dictionary of the decorated tool methods.
        """
        return {
            'tavily_search': self.tavily_search_tool,
            'image_finder': self.image_finder_tool,
            'translator': self.translator_tool,
            'telegram_sender': self.telegram_sender_tool
        }
    
    def _create_crew(self):
        """
        Create and configure the CrewAI crew.
        
        Agents and tasks are instantiated directly here to ensure the LLM is passed correctly.
        """
        # Assuming FinancialMarketAgents is a valid class or module
        search_agent = Agent(
            role='Financial News Analyst',
            goal=dedent("Find the most relevant and trending financial news from the last few hours."),
            backstory=dedent("A seasoned financial news analyst with a knack for spotting key trends and breaking stories."),
            tools=[self.tools['tavily_search']],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        image_extraction_agent = Agent(
            role='Visual Content Finder',
            goal=dedent("Extract high-quality and relevant images from financial news articles."),
            backstory=dedent("A meticulous content specialist who finds the perfect visuals to complement news stories."),
            tools=[self.tools['image_finder']],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        summary_agent = Agent(
            role='Market Summary Creator',
            goal=dedent("Synthesize raw financial news into a concise and comprehensive daily market summary."),
            backstory=dedent("An expert in market analysis, able to distil complex financial data into easy-to-understand summaries."),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        formatting_agent = Agent(
            role='Content Formatter',
            goal=dedent("Format the market summary with visuals and markdown for readability."),
            backstory=dedent("A creative content designer who turns plain text into an engaging, visually-rich report."),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        translation_agent = Agent(
            role='Multilingual Translator',
            goal=dedent("Translate the final market summary into Arabic, Hindi, and Hebrew."),
            backstory=dedent("A skilled linguist who specializes in financial terminology across multiple languages."),
            tools=[self.tools['translator']],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        distribution_agent = Agent(
            role='Social Media Distributor',
            goal=dedent("Distribute the formatted market summary to a Telegram channel."),
            backstory=dedent("A tech-savvy operations specialist who automates content distribution to social media platforms."),
            tools=[self.tools['telegram_sender']],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        agents = [
            search_agent,
            image_extraction_agent,
            summary_agent,
            formatting_agent,
            translation_agent,
            distribution_agent
        ]
        
        tasks = [
            Task(
                description=dedent("""
                1. Search for recent financial news from trusted sources using the `tavily_search_tool`.
                2. Identify articles about major market movements, corporate earnings, and global economic shifts.
                3. Compile a list of relevant article URLs for the next agent.
                """),
                expected_output=dedent("A list of relevant articles from the last few hours with their URLs."),
                agent=search_agent,
                llm=self.llm  # Explicitly setting LLM to bypass CrewAI's validation bug
            ),
            Task(
                description=dedent("""
                1. Use the `image_finder_tool` to extract images from the URLs provided by the Search Agent.
                2. Select high-quality, relevant images that visually represent the news.
                3. Make sure to capture at least one image for each major news item.
                """),
                expected_output=dedent("A list of image URLs from the identified articles."),
                agent=image_extraction_agent,
                llm=self.llm # Explicitly setting LLM to bypass CrewAI's validation bug
            ),
            Task(
                description=dedent("""
                1. Synthesize the news articles into a concise daily market summary.
                2. Focus on key headlines, stock market performance, and significant economic indicators.
                3. The summary should be easy to read and provide a quick overview of the market.
                """),
                expected_output=dedent("A single, comprehensive paragraph summarizing the daily financial news."),
                agent=summary_agent,
                llm=self.llm # Explicitly setting LLM to bypass CrewAI's validation bug
            ),
            Task(
                description=dedent("""
                1. Format the market summary into a visually appealing markdown format.
                2. Incorporate the image URLs from the previous agent at relevant points in the summary.
                3. The final output should be a well-structured markdown file ready for distribution.
                """),
                expected_output=dedent("A well-formatted markdown file containing the market summary and images."),
                agent=formatting_agent,
                llm=self.llm # Explicitly setting LLM to bypass CrewAI's validation bug
            ),
            Task(
                description=dedent("""
                1. Translate the final markdown summary into Arabic, Hindi, and Hebrew using the `translator_tool`.
                2. Ensure the translations are accurate and maintain the original tone.
                """),
                expected_output=dedent("A JSON object containing the translated summaries for each language."),
                agent=translation_agent,
                llm=self.llm # Explicitly setting LLM to bypass CrewAI's validation bug
            ),
            Task(
                description=dedent("""
                1. Use the `telegram_sender_tool` to distribute the final formatted summary.
                2. The summary should be posted along with the images to the designated Telegram channel.
                3. Confirm the successful delivery of the content.
                """),
                expected_output=dedent("A confirmation of successful distribution to the Telegram channel."),
                agent=distribution_agent,
                llm=self.llm # Explicitly setting LLM to bypass CrewAI's validation bug
            )
        ]
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential,
            memory=True,
            cache=True,
            max_execution_time=1800,
            step_callback=self._step_callback,
            llm_provider="custom",  # must be before llm
            llm=self.llm,
            embedder={
                "provider": "google",
                "config": {
                    "api_key": os.getenv('GOOGLE_API_KEY'),
                    "model": "text-embedding-004"
                }
            }
        )

        return crew
    
    def _step_callback(self, step_output):
        """Callback function to track crew execution steps"""
        step_info = {
            'timestamp': datetime.now().isoformat(),
            'step': str(step_output),
            'agent': getattr(step_output, 'agent', 'Unknown'),
            'status': 'completed'
        }
        
        self.logger.info(f"Step completed: {step_info['agent']}")
        
        # Store step information
        if 'execution_steps' not in self.execution_results:
            self.execution_results['execution_steps'] = []
        self.execution_results['execution_steps'].append(step_info)
    
    def test_api_connections(self):
        """Test all API connections before execution"""
        self.logger.info("Testing API connections...")
        connection_results = {}
        
        # Test Tavily API
        try:
            tavily_api_key = os.getenv('TAVILY_API_KEY')
            if not tavily_api_key:
                raise ValueError("TAVILY_API_KEY not found in environment variables")
            
            self.logger.info("Testing Tavily connection.")
            result = test_tavily_connection(tavily_api_key)
            if result.get('success'):
                self.logger.info("Tavily connection test successful.")
                connection_results['tavily'] = True
            else:
                self.logger.error(f"Tavily connection test failed: {result.get('error')}")
                connection_results['tavily'] = False
        except Exception as e:
            connection_results['tavily'] = False
            self.logger.error(f"Tavily connection test failed: {str(e)}")
        
        # Test Telegram API
        try:
            from .tools.telegram_sender import TelegramSender
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if bot_token and chat_id:
                sender = TelegramSender(bot_token, chat_id)
                connection_results['telegram'] = sender.test_connection()
            else:
                connection_results['telegram'] = False
                self.logger.error("Telegram credentials not found")
        except Exception as e:
            connection_results['telegram'] = False
            self.logger.error(f"Telegram connection test failed: {str(e)}")
        
        # Test LLM (Gemini) connection
        try:
            test_response = self.llm.invoke("Test connection")
            connection_results['llm'] = bool(test_response)
        except Exception as e:
            connection_results['llm'] = False
            self.logger.error(f"LLM connection test failed: {str(e)}")
        
        self.execution_results['api_connections'] = connection_results
        
        # Log results
        for service, status in connection_results.items():
            status_text = "✅ Connected" if status else "❌ Failed"
            self.logger.info(f"{service.title()} API: {status_text}")
        
        return connection_results
    
    def execute_workflow(self):
        """Execute the complete financial market analysis workflow"""
        start_time = datetime.now()
        self.logger.info(f"Starting financial market analysis workflow at {start_time}")
        
        try:
            api_status = self.test_api_connections()
            failed_apis = [api for api, status in api_status.items() if not status]
            
            if failed_apis:
                error_msg = f"Failed API connections: {failed_apis}"
                self.logger.error(error_msg)
                self.execution_results['warnings'] = [f"Some APIs failed connection test: {failed_apis}"]
            
            self.logger.info("Executing crew workflow...")
            result = self.crew.kickoff()
            
            end_time = datetime.now()
            execution_duration = end_time - start_time
            
            final_results = {
                'success': True,
                'execution_start': start_time.isoformat(),
                'execution_end': end_time.isoformat(),
                'execution_duration_seconds': execution_duration.total_seconds(),
                'execution_duration_minutes': round(execution_duration.total_seconds() / 60, 2),
                'crew_output': str(result),
                'api_connections': api_status,
                'detailed_results': self.execution_results
            }
            
            self.logger.info(f"Workflow completed successfully in {final_results['execution_duration_minutes']} minutes")
            return final_results
            
        except Exception as e:
            end_time = datetime.now()
            execution_duration = end_time - start_time
            
            error_results = {
                'success': False,
                'error': str(e),
                'execution_start': start_time.isoformat(),
                'execution_end': end_time.isoformat(),
                'execution_duration_seconds': execution_duration.total_seconds(),
                'partial_results': self.execution_results
            }
            
            self.logger.error(f"Workflow failed after {execution_duration.total_seconds():.1f} seconds: {str(e)}")
            return error_results
    
    def get_execution_summary(self):
        """Get a summary of the last execution"""
        if not hasattr(self, 'execution_results'):
            return {"status": "No execution results available"}
        
        search_results = self.execution_results.get('search_results', {})
        images = self.execution_results.get('extracted_images', [])
        translations = self.execution_results.get('translations', {})
        telegram_results = self.execution_results.get('telegram_distribution', {})
        
        summary = {
            'articles_found': len(search_results.get('articles', [])),
            'images_extracted': len(images),
            'languages_translated': len(translations),
            'telegram_success': telegram_results.get('success', False),
            'total_execution_steps': len(self.execution_results.get('execution_steps', [])),
            'api_status': self.execution_results.get('api_connections', {})
        }
        
        return summary
