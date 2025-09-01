# Updated crew.py with simplified execution
from crewai import Agent, Crew, Task, Process
import logging
from datetime import datetime
from .agents import FinancialAgents

logger = logging.getLogger(__name__)

class FinancialMarketCrew:
    """Simplified CrewAI implementation for Financial Market Summary"""
    
    def __init__(self):
        self.agents_factory = FinancialAgents()
        self.execution_results = {}

    def _create_and_run_crew(self, agents, tasks):
        """Helper to create and run a temporary crew for a specific phase"""
        crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)
        return crew.kickoff()

    def run_complete_workflow(self):
        """Run the complete financial summary workflow step-by-step"""
        try:
            logger.info("=== Starting Complete Financial Workflow ===")
            
            # --- AGENT INITIALIZATION ---
            search_agent = self.agents_factory.search_agent()
            summary_agent = self.agents_factory.summary_agent()
            formatting_agent = self.agents_factory.formatting_agent()
            translation_agent = self.agents_factory.translation_agent()
            send_agent = self.agents_factory.send_agent()

            # --- STEP 1: SEARCH ---
            logger.info("--- Phase 1: Searching for News ---")
            search_task = Task(
                description="Search for the latest US financial news from the past hour. Focus on major stock movements, earnings, and economic data.",
                expected_output="A structured list of recent US financial news articles with titles, summaries, and sources.",
                agent=search_agent
            )
            search_result = self._create_and_run_crew([search_agent], [search_task])
            if not search_result or "Error:" in str(search_result):
                return {"status": "failed", "error": f"Search phase failed: {search_result}"}
            self.execution_results['search'] = search_result
            logger.info("--- Search Phase Completed ---")

            # --- STEP 2: SUMMARY ---
            logger.info("--- Phase 2: Creating Summary ---")
            summary_task = Task(
                description=f"""Analyze the following financial news and create a concise market summary under 500 words.
                News Data: {search_result}""",
                expected_output="A professional, well-structured financial market summary.",
                agent=summary_agent
            )
            summary_result = self._create_and_run_crew([summary_agent], [summary_task])
            if not summary_result or "Error:" in str(summary_result):
                return {"status": "failed", "error": f"Summary phase failed: {summary_result}"}
            self.execution_results['summary'] = summary_result
            logger.info("--- Summary Phase Completed ---")

            # --- STEP 3: FORMATTING ---
            logger.info("--- Phase 3: Formatting with Visuals ---")
            formatting_task = Task(
                description=f"""Format the following financial summary with 1-2 relevant charts or images using the financial_image_finder tool.
                Summary to Format: {summary_result}""",
                expected_output="A well-formatted summary with markdown and image URLs integrated.",
                agent=formatting_agent,
                tools=[self.agents_factory.image_finder]
            )
            formatted_result = self._create_and_run_crew([formatting_agent], [formatting_task])
            if not formatted_result or "Error:" in str(formatted_result):
                return {"status": "failed", "error": f"Formatting phase failed: {formatted_result}"}
            self.execution_results['formatted_summary'] = formatted_result
            logger.info("--- Formatting Phase Completed ---")
            
            # --- STEP 4: TRANSLATION ---
            logger.info("--- Phase 4: Translating Content ---")
            translations = {}
            for lang in ['arabic', 'hindi', 'hebrew']:
                logger.info(f"Translating to {lang}...")
                translation_task = Task(
                    description=f"Translate the following formatted financial summary to {lang} using the financial_translator tool.",
                    expected_output=f"An accurate and well-formatted translation in {lang}.",
                    agent=translation_agent,
                    tools=[self.agents_factory.translator],
                    inputs={'text': formatted_result, 'target_language': lang} # Pass inputs explicitly
                )
                translation_output = self._create_and_run_crew([translation_agent], [translation_task])
                translations[lang] = translation_output
            self.execution_results['translations'] = translations
            logger.info("--- Translation Phase Completed ---")

            # --- STEP 5: SENDING ---
            logger.info("--- Phase 5: Sending to Telegram ---")
            send_results = {}
            
            # Send English
            send_english_task = Task(
                description=f"Send the English financial summary using the telegram_sender tool.",
                expected_output="Confirmation of successful sending.",
                agent=send_agent,
                tools=[self.agents_factory.telegram_sender],
                inputs={'content': formatted_result, 'language': 'english'}
            )
            send_results['english'] = self._create_and_run_crew([send_agent], [send_english_task])

            # Send Translations
            for lang, content in translations.items():
                send_lang_task = Task(
                    description=f"Send the {lang} financial summary using the telegram_sender tool.",
                    expected_output="Confirmation of successful sending.",
                    agent=send_agent,
                    tools=[self.agents_factory.telegram_sender],
                    inputs={'content': content, 'language': lang}
                )
                send_results[lang] = self._create_and_run_crew([send_agent], [send_lang_task])
            self.execution_results['send_results'] = send_results
            logger.info("--- Sending Phase Completed ---")

            return {
                "status": "success",
                "results": self.execution_results,
                "execution_time": datetime.now().isoformat()
            }
        
        except Exception as e:
            error_msg = f"Complete workflow failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "failed", "error": error_msg}