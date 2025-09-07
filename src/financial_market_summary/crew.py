import logging
import time
from typing import Any, Dict, List
from crewai import Agent, Crew, Process, Task
from datetime import datetime
import json
from .agents import FinancialAgents
from .LLM_config import apply_rate_limiting

# Set up logging for the crew module
logger = logging.getLogger(__name__)


class FinancialMarketCrew:
    """
    An enhanced CrewAI implementation for a financial news workflow.

    This class orchestrates a series of specialized agents to search, summarize,
    format, translate, and distribute financial news. It incorporates rate
    limiting and robust error handling with a retry mechanism.
    """

    def __init__(self):
        """Initializes the agent factory and a dictionary to store execution results."""
        self.agents_factory = FinancialAgents()
        self.execution_results: Dict[str, Any] = {}

    def _run_task_with_retry(self, agents: List[Agent], task: Task, max_retries: int = 3) -> str:
        """
        Runs a single task with retry logic to handle potential failures.

        This helper method creates a Crew for the given task and agent, then
        attempts to execute it multiple times with a backoff delay in case of
        errors, particularly rate limits.

        Args:
            agents: A list of agents for the crew.
            task: The task to be executed.
            max_retries: The maximum number of retry attempts.

        Returns:
            The output of the task or an error message if all retries fail.
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"Retry attempt {attempt + 1}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)

                crew = Crew(
                    agents=agents,
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )
                result = crew.kickoff()
                return str(result)

            except Exception as e:
                error_msg = str(e)
                if any(limit_keyword in error_msg.lower() for limit_keyword in ["429", "quota", "rate limit"]):
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}: {error_msg}")
                    if attempt < max_retries - 1:
                        continue
                else:
                    logger.error(f"Non-rate-limit error on attempt {attempt + 1}: {error_msg}")
                    if attempt < max_retries - 1:
                        continue

                return f"Error after {max_retries} attempts: {error_msg}"
        
        return "Error: Max retries exceeded."

    def run_complete_workflow(self) -> Dict[str, Any]:
        """
        Executes the entire financial summary workflow in a sequential manner.

        The workflow consists of five phases: search, summary, formatting,
        translation, and distribution. Each phase is handled by a dedicated
        agent and its output is used as input for the next phase.

        Returns:
            A dictionary containing the workflow's status, results, and a
            summary of the execution.
        """
        try:
            logger.info("--- Starting Complete Financial Workflow ---")

            # --- Phase 1: Search ---
            search_result = self._run_search_phase()
            if "Error" in search_result:
                return {"status": "failed", "error": f"Search phase failed: {search_result}"}
            self.execution_results["search"] = search_result

            # --- Phase 2: Summary ---
            summary_result = self._run_summary_phase(search_result)
            if "Error" in summary_result:
                return {"status": "failed", "error": f"Summary phase failed: {summary_result}"}
            self.execution_results["summary"] = summary_result

            # --- Phase 3: Formatting ---
            formatted_result = self._run_formatting_phase(summary_result)
            if "Error" in formatted_result:
                return {"status": "failed", "error": f"Formatting phase failed: {formatted_result}"}
            self.execution_results["formatted_summary"] = formatted_result

            # --- Phase 4: Translation ---
            translations = self._run_translation_phase(formatted_result)
            self.execution_results["translations"] = translations

            # --- Phase 5: Sending ---
            send_results = self._run_sending_phase(formatted_result, translations)
            self.execution_results["send_results"] = send_results

            logger.info("--- Workflow Completed Successfully ---")
            return {
                "status": "success",
                "results": self.execution_results,
                "execution_time": datetime.now().isoformat(),
                "summary": {
                    "english_summary_excerpt": formatted_result[:200] + "..." if len(formatted_result) > 200 else formatted_result,
                    "translations_completed": len([k for k, v in translations.items() if not isinstance(v, str) or "failed" not in v.lower()]),
                    "sends_completed": len([k for k, v in send_results.items() if "successfully" in v.lower() or "success" in v.lower()]),
                },
            }

        except Exception as e:
            error_msg = f"Complete workflow failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "failed", "error": error_msg}

    # --- Private Phase-specific Methods ---

    def _run_search_phase(self) -> str:
        """Executes the search phase of the workflow."""
        logger.info("--- Phase 1: Searching for News ---")
        search_agent = self.agents_factory.search_agent()
        search_task = Task(
            description="Search for the latest US financial news from the past 2 hours. Focus on major stock movements, earnings, and economic data.",
            expected_output="A structured list of recent US financial news articles with titles, summaries, and sources.",
            agent=search_agent
        )
        return self._run_task_with_retry([search_agent], search_task)

    @apply_rate_limiting("gemini")
    def _run_summary_phase(self, search_result: str) -> str:
        """Executes the summary phase of the workflow."""
        logger.info("--- Phase 2: Creating Summary ---")
        summary_agent = self.agents_factory.summary_agent()
        summary_task = Task(
            description=f"""Analyze the following financial news and create a concise market summary under 500 words.

Financial News Data:
{search_result}

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
        return self._run_task_with_retry([summary_agent], summary_task)

    @apply_rate_limiting("gemini")
    def _run_formatting_phase(self, summary_result: str) -> str:
        """Executes the formatting phase of the workflow."""
        logger.info("--- Phase 3: Formatting with Visuals ---")
        formatting_agent = self.agents_factory.formatting_agent()
        formatting_task = Task(
            description=f"""Format the following financial summary with 1-2 relevant charts or images.

Summary to Format:
{summary_result}

Use the financial_image_finder tool to find relevant financial charts or graphs.
Integrate images using markdown format and create a well-formatted final summary.""",
            expected_output="A well-formatted summary with markdown and image URLs integrated.",
            agent=formatting_agent
        )
        return self._run_task_with_retry([formatting_agent], formatting_task)

    def _run_translation_phase(self, formatted_content: str) -> Dict[str, str]:
        """Executes the translation phase for multiple languages."""
        logger.info("--- Phase 4: Translating Content ---")
        translation_agent = self.agents_factory.translation_agent()
        translations = {}
        languages = ["arabic", "hindi", "hebrew"]

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

            translation_output = self._run_task_with_retry([translation_agent], translation_task)
            translations[lang] = translation_output

        logger.info("--- Translation Phase Completed ---")
        return translations

    def _run_sending_phase(self, formatted_content: str, translations: Dict[str, str]) -> Dict[str, str]:
        """Executes the content distribution phase via Telegram."""
        logger.info("--- Phase 5: Sending to Telegram ---")
        send_agent = self.agents_factory.send_agent()
        send_results = {}

        # Send English summary
        logger.info("Sending English summary...")
        english_send_task = Task(
            description=f"""Send the English financial summary to Telegram.

Content to Send:
{formatted_content}

Use the telegram_sender tool with language='english'.""",
            expected_output="Confirmation of successful sending.",
            agent=send_agent
        )
        english_send_result = self._run_task_with_retry([send_agent], english_send_task)
        send_results["english"] = english_send_result

        # Send translations
        for lang, content in translations.items():
            if "Error" not in content:
                logger.info(f"Sending {lang} summary...")
                lang_send_task = Task(
                    description=f"""Send the {lang} financial summary to Telegram.

Content to Send:
{content}

Use the telegram_sender tool with language='{lang}'.""",
                    expected_output="Confirmation of successful sending.",
                    agent=send_agent
                )
                lang_send_result = self._run_task_with_retry([send_agent], lang_send_task)
                send_results[lang] = lang_send_result
            else:
                send_results[lang] = f"Skipped sending {lang} due to translation failure."

        logger.info("--- Sending Phase Completed ---")
        return send_results
    

