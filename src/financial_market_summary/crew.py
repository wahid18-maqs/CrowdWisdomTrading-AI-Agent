# Updated crew.py with rate limiting and proper context handling
from crewai import Agent, Crew, Task, Process
import logging
from datetime import datetime
import time
import json
from .agents import FinancialAgents

logger = logging.getLogger(__name__)

class FinancialMarketCrew:
    """Enhanced CrewAI implementation with rate limiting and proper error handling"""
    
    def __init__(self):
        self.agents_factory = FinancialAgents()
        self.execution_results = {}
        self.rate_limit_delay = 5  # 5 seconds between API calls

    def _wait_for_rate_limit(self):
        """Add delay to avoid rate limiting"""
        logger.info(f"Waiting {self.rate_limit_delay} seconds to avoid rate limiting...")
        time.sleep(self.rate_limit_delay)

    def _create_and_run_crew_with_retry(self, agents, tasks, max_retries=3):
        """Helper to create and run a crew with retry logic"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = self.rate_limit_delay * (attempt + 1)
                    logger.info(f"Retry attempt {attempt + 1}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                
                crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)
                result = crew.kickoff()
                return result
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}: {error_msg}")
                    if attempt < max_retries - 1:
                        continue
                else:
                    logger.error(f"Non-rate-limit error on attempt {attempt + 1}: {error_msg}")
                    if attempt < max_retries - 1:
                        continue
                
                # Final attempt failed
                return f"Error after {max_retries} attempts: {error_msg}"
        
        return "Error: Max retries exceeded"

    def run_complete_workflow(self):
        """Run the complete financial summary workflow with proper error handling"""
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
                description="Search for the latest US financial news from the past 2 hours. Focus on major stock movements, earnings, and economic data.",
                expected_output="A structured list of recent US financial news articles with titles, summaries, and sources.",
                agent=search_agent
            )
            
            search_result = self._create_and_run_crew_with_retry([search_agent], [search_task])
            if isinstance(search_result, str) and "Error" in search_result:
                return {"status": "failed", "error": f"Search phase failed: {search_result}"}
            
            self.execution_results['search'] = search_result
            logger.info("--- Search Phase Completed ---")
            self._wait_for_rate_limit()

            # --- STEP 2: SUMMARY ---
            logger.info("--- Phase 2: Creating Summary ---")
            
            # Extract raw content from search result
            search_content = str(search_result.raw) if hasattr(search_result, 'raw') else str(search_result)
            
            summary_task = Task(
                description=f"""Analyze the following financial news and create a concise market summary under 500 words.

Financial News Data:
{search_content}

Structure your summary with:
1. Market Overview (2-3 sentences)
2. Key Movers (top stocks)
3. Sector Analysis
4. Economic Highlights
5. Tomorrow's Watch

Use professional language and include specific figures.""",
                expected_output="A professional, well-structured financial market summary under 500 words.",
                agent=summary_agent
            )
            
            summary_result = self._create_and_run_crew_with_retry([summary_agent], [summary_task])
            if isinstance(summary_result, str) and "Error" in summary_result:
                return {"status": "failed", "error": f"Summary phase failed: {summary_result}"}
            
            self.execution_results['summary'] = summary_result
            logger.info("--- Summary Phase Completed ---")
            self._wait_for_rate_limit()

            # --- STEP 3: FORMATTING ---
            logger.info("--- Phase 3: Formatting with Visuals ---")
            
            # Extract summary content
            summary_content = str(summary_result.raw) if hasattr(summary_result, 'raw') else str(summary_result)
            
            formatting_task = Task(
                description=f"""Format the following financial summary with 1-2 relevant charts or images.

Summary to Format:
{summary_content}

Use the financial_image_finder tool to find relevant financial charts or graphs.
Integrate images using markdown format and create a well-formatted final summary.""",
                expected_output="A well-formatted summary with markdown and image URLs integrated.",
                agent=formatting_agent
            )
            
            formatted_result = self._create_and_run_crew_with_retry([formatting_agent], [formatting_task])
            if isinstance(formatted_result, str) and "Error" in formatted_result:
                return {"status": "failed", "error": f"Formatting phase failed: {formatted_result}"}
            
            self.execution_results['formatted_summary'] = formatted_result
            logger.info("--- Formatting Phase Completed ---")
            self._wait_for_rate_limit()
            
            # --- STEP 4: TRANSLATION ---
            logger.info("--- Phase 4: Translating Content ---")
            
            # Extract formatted content
            formatted_content = str(formatted_result.raw) if hasattr(formatted_result, 'raw') else str(formatted_result)
            
            translations = {}
            languages = ['arabic', 'hindi', 'hebrew']
            
            for lang in languages:
                logger.info(f"Translating to {lang}...")
                
                translation_task = Task(
                    description=f"""Translate the following formatted financial summary to {lang}.

Content to Translate:
{formatted_content}

CRITICAL REQUIREMENTS:
- Keep stock symbols (AAPL, MSFT, etc.) unchanged
- Preserve all numbers, percentages, currency values exactly
- Maintain markdown formatting
- Use professional financial terminology in {lang}
- If unsure about terms, keep English in parentheses

Use the financial_translator tool for accurate translation.""",
                    expected_output=f"An accurate and well-formatted translation in {lang}.",
                    agent=translation_agent
                )
                
                translation_output = self._create_and_run_crew_with_retry([translation_agent], [translation_task])
                
                if isinstance(translation_output, str) and "Error" in translation_output:
                    logger.warning(f"Translation to {lang} failed: {translation_output}")
                    translations[lang] = f"Translation to {lang} failed: {translation_output}"
                else:
                    translation_content = str(translation_output.raw) if hasattr(translation_output, 'raw') else str(translation_output)
                    translations[lang] = translation_content
                
                self._wait_for_rate_limit()
            
            self.execution_results['translations'] = translations
            logger.info("--- Translation Phase Completed ---")

            # --- STEP 5: SENDING ---
            logger.info("--- Phase 5: Sending to Telegram ---")
            send_results = {}
            
            # Send English
            logger.info("Sending English summary...")
            send_english_task = Task(
                description=f"""Send the English financial summary to Telegram.

Content to Send:
{formatted_content}

Use the telegram_sender tool with language='english'.""",
                expected_output="Confirmation of successful sending.",
                agent=send_agent
            )
            
            english_send_result = self._create_and_run_crew_with_retry([send_agent], [send_english_task])
            send_results['english'] = str(english_send_result.raw) if hasattr(english_send_result, 'raw') else str(english_send_result)
            self._wait_for_rate_limit()

            # Send Translations
            for lang, content in translations.items():
                if not content.startswith("Translation to") or "failed" not in content:  # Only send successful translations
                    logger.info(f"Sending {lang} summary...")
                    
                    send_lang_task = Task(
                        description=f"""Send the {lang} financial summary to Telegram.

Content to Send:
{content}

Use the telegram_sender tool with language='{lang}'.""",
                        expected_output="Confirmation of successful sending.",
                        agent=send_agent
                    )
                    
                    lang_send_result = self._create_and_run_crew_with_retry([send_agent], [send_lang_task])
                    send_results[lang] = str(lang_send_result.raw) if hasattr(lang_send_result, 'raw') else str(lang_send_result)
                    self._wait_for_rate_limit()
                else:
                    send_results[lang] = f"Skipped sending {lang} due to translation failure"
            
            self.execution_results['send_results'] = send_results
            logger.info("--- Sending Phase Completed ---")

            return {
                "status": "success",
                "results": self.execution_results,
                "execution_time": datetime.now().isoformat(),
                "summary": {
                    "english_summary": formatted_content[:200] + "..." if len(formatted_content) > 200 else formatted_content,
                    "translations_completed": len([k for k, v in translations.items() if not v.startswith("Translation to")]),
                    "sends_completed": len([k for k, v in send_results.items() if "successfully" in v.lower() or "success" in v.lower()])
                }
            }
        
        except Exception as e:
            error_msg = f"Complete workflow failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "failed", "error": error_msg}