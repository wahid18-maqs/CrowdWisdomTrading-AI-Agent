from crewai import Agent, Crew, Task, Process
from crewai.flow import Flow, listen, start
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai_tools import TavilySearchTool
from crewai_tools import SerperDevTool
from litellm import completion
import logging
from typing import Dict, List, Any
from datetime import datetime
from .agents import FinancialAgents
from .tools.tavily_search import TavilyFinancialTool
from .tools.telegram_sender import TelegramSender
from .tools.image_finder import ImageFinder
from .tools.translator import MultiLanguageTranslator
from .tasks import FinancialTasks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialMarketFlow(Flow):
    """
    CrewAI Flow for Financial Market Summary Generation
    """
    
    def __init__(self):
        super().__init__()
        self.agents = FinancialAgents()
        self.tasks = FinancialTasks(self.agents)  # Pass agents to FinancialTasks
        
        # Initialize tools
        self.tavily_tool = TavilyFinancialTool()
        self.telegram_sender = TelegramSender()
        self.image_finder = ImageFinder()
        self.translator = MultiLanguageTranslator()
        
        # Flow state
        self.flow_state = {
            'raw_news_data': None,
            'summary': None,
            'formatted_summary': None,
            'translations': {},
            'images': []
        }
    
    @start()
    def start_flow(self):
        """Start the financial summary flow"""
        try:
            logger.info("Starting Financial Market Summary Flow")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Flow initiated at: {current_time}")
            self.flow_state['start_time'] = current_time
            return {"status": "flow_started", "timestamp": current_time}
        except Exception as e:
            logger.error(f"Error in start_flow: {str(e)}")
            self.flow_state['start_time'] = None
            return {"status": "start_failed", "error": str(e)}
    
    @listen(start_flow)
    def search_financial_news(self, context: Dict[str, Any]):
        """Search for latest financial news"""
        try:
            logger.info("Step 1: Searching for financial news...")
            
            # Create search agent and task
            search_agent = self.agents.search_agent()
            search_task = self.tasks.search_financial_news()
            
            # Execute search
            crew = Crew(
                agents=[search_agent],
                tasks=[search_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            self.flow_state['raw_news_data'] = result
            logger.info(f"Search completed. Found {len(str(result))} characters of news data")
            return {"status": "search_completed", "data": result}
        except Exception as e:
            logger.error(f"Error in search step: {str(e)}")
            self.flow_state['raw_news_data'] = None
            return {"status": "search_failed", "error": str(e)}
    
    @listen(search_financial_news)
    def create_summary(self, context: Dict[str, Any]):
        """Create summary from search results with guardrails"""
        try:
            logger.info("Step 2: Creating financial summary...")
            
            # Guardrail: Check if we have valid data
            if not self.flow_state['raw_news_data'] or len(str(self.flow_state['raw_news_data'])) < 100:
                logger.error("Insufficient data for summary generation")
                self.flow_state['summary'] = None
                return {"status": "summary_failed", "error": "Insufficient search data"}
            
            summary_agent = self.agents.summary_agent()
            summary_task = self.tasks.create_summary()
            summary_task.context = [self.flow_state['raw_news_data']]
            
            crew = Crew(
                agents=[summary_agent],
                tasks=[summary_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Guardrail: Check summary length
            if len(str(result).split()) > 500:
                logger.warning("Summary exceeds 500 words, trimming...")
                words = str(result).split()[:500]
                result = ' '.join(words) + "..."
            
            self.flow_state['summary'] = result
            logger.info("Summary created successfully")
            return {"status": "summary_completed", "summary": result}
        except Exception as e:
            logger.error(f"Error in summary step: {str(e)}")
            self.flow_state['summary'] = None
            return {"status": "summary_failed", "error": str(e)}
    
    @listen(create_summary)
    def format_with_images(self, context: Dict[str, Any]):
        """Format summary with relevant images"""
        try:
            logger.info("Step 3: Finding and formatting with images...")
            
            # Guardrail: Check if summary exists
            if not self.flow_state['summary']:
                logger.error("No summary available for formatting")
                self.flow_state['formatted_summary'] = None
                return {"status": "formatting_failed", "error": "No summary to format"}
            
            formatting_agent = self.agents.formatting_agent()
            formatting_task = self.tasks.format_with_images()
            formatting_task.context = [self.flow_state['summary']]
            
            crew = Crew(
                agents=[formatting_agent],
                tasks=[formatting_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            self.flow_state['formatted_summary'] = result
            logger.info("Formatting completed")
            return {"status": "formatting_completed", "formatted": result}
        except Exception as e:
            logger.error(f"Error in formatting step: {str(e)}")
            # Fallback: use original summary
            self.flow_state['formatted_summary'] = self.flow_state['summary']
            return {"status": "formatting_partial", "error": str(e)}
    
    @listen(format_with_images)
    def translate_summary(self, context: Dict[str, Any]):
        """Translate summary to multiple languages"""
        try:
            logger.info("Step 4: Translating summary...")
            
            # Guardrail: Check if formatted summary exists
            if not self.flow_state['formatted_summary'] or len(str(self.flow_state['formatted_summary'])) < 50:
                logger.error("No valid formatted summary available for translation")
                self.flow_state['translations'] = {}
                return {"status": "translation_failed", "error": "No valid summary to translate"}
            
            target_languages = ['arabic', 'hindi', 'hebrew']
            translations = {}
            
            for language in target_languages:
                try:
                    translation_agent = self.agents.translation_agent()
                    translation_task = self.tasks.translate_content(language)
                    translation_task.context = [self.flow_state['formatted_summary']]
                    
                    crew = Crew(
                        agents=[translation_agent],
                        tasks=[translation_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    result = crew.kickoff()
                    translations[language] = result.output if hasattr(result, 'output') else str(result)
                    logger.info(f"Translation to {language} completed")
                except Exception as e:
                    logger.error(f"Failed to translate to {language}: {str(e)}")
                    translations[language] = f"Translation failed: {str(e)}"
            
            self.flow_state['translations'] = translations
            return {"status": "translation_completed", "translations": translations}
        except Exception as e:
            logger.error(f"Error in translation step: {str(e)}")
            self.flow_state['translations'] = {}
            return {"status": "translation_failed", "error": str(e)}
    
    @listen(translate_summary)
    def send_to_telegram(self, context: Dict[str, Any]):
        """Send all content to Telegram"""
        try:
            logger.info("Step 5: Sending to Telegram...")
            
            # Prepare content for all languages
            all_content = {
                'english': str(self.flow_state.get('formatted_summary', '')),
                **self.flow_state.get('translations', {})
            }

            # Guardrail: Check if English summary exists
            if not all_content['english'] or len(all_content['english']) < 50:
                logger.error("No valid English summary to send")
                return {"status": "send_failed", "error": "No valid English summary"}

            # Directly use TelegramSender
            send_result = self.telegram_sender.send_multiple_languages(all_content)
            logger.info(f"Telegram send result: {send_result}")

            return {"status": "send_completed", "result": send_result}
        except Exception as e:
            logger.error(f"Error in send step: {str(e)}")
            return {"status": "send_failed", "error": str(e)}
    
    def run_with_guardrails(self):
        """Run the complete flow with error handling and guardrails"""
        try:
            logger.info("=== Starting Financial Market Summary Flow ===")
            
            # Run the flow
            result = self.kickoff()
            
            # Check for failed steps
            error_count = sum(1 for step in ['search_financial_news', 'create_summary', 'format_with_images', 'translate_summary', 'send_to_telegram']
                             if self.flow_state.get(step, {}).get('status', '').startswith('failed'))
            success_rate = (5 - error_count) / 5 * 100 if error_count <= 5 else 0
            status = "success" if error_count == 0 else "partial_success" if error_count < 5 else "failed"
            
            logger.info(f"=== Flow completed with status: {status} ===")
            return {
                "status": status,
                "flow_state": self.flow_state,
                "final_result": result,
                "error_count": error_count,
                "success_rate": success_rate
            }
        except Exception as e:
            logger.error(f"Flow failed with error: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "partial_state": self.flow_state,
                "error_count": 1,
                "success_rate": 0
            }

def run_financial_flow():
    """Main function to run the financial flow"""
    flow = FinancialMarketFlow()
    return flow.run_with_guardrails()