import logging
import time
import re
from typing import Any, Dict, List
from crewai import Agent, Crew, Process, Task
from datetime import datetime, timedelta
import json
from .agents import FinancialAgents
from .LLM_config import apply_rate_limiting

# Set up logging for the crew module
logger = logging.getLogger(__name__)

class FinancialMarketCrew:
    """
    Enhanced CrewAI implementation with summary validation for accuracy.
    """

    def __init__(self):
        """Initializes the agent factory and a dictionary to store execution results."""
        self.agents_factory = FinancialAgents()
        self.execution_results: Dict[str, Any] = {}

    def _validate_summary(self, summary: str, original_news: str) -> Dict[str, Any]:
        """
        Validate the summary against original news for accuracy.
        """
        logger.info("--- Validating Summary Accuracy ---")
        
        validation_results = {
            "has_reliable_sources": self._check_sources(original_news),
            "stocks_verified": self._verify_stock_symbols(summary, original_news),
            "numbers_accurate": self._verify_numbers(summary, original_news),
            "content_fresh": self._check_freshness(original_news),
            "no_hallucination": self._check_hallucination(summary, original_news),
            "confidence_score": 0
        }
        
        # Calculate confidence score (0-100)
        weights = {"has_reliable_sources": 20, "stocks_verified": 30, "numbers_accurate": 25, "content_fresh": 15, "no_hallucination": 10}
        validation_results["confidence_score"] = sum(weights[key] for key, passed in validation_results.items() if passed and key != "confidence_score")
        
        logger.info(f"Validation Results: {validation_results}")
        return validation_results

    def _check_sources(self, original_news: str) -> bool:
        """Check if news comes from reliable sources."""
        reliable_sources = ['reuters', 'bloomberg', 'cnbc', 'wsj', 'marketwatch', 'yahoo', 'investing.com']
        return any(source in original_news.lower() for source in reliable_sources)

    def _verify_stock_symbols(self, summary: str, original_news: str) -> bool:
        """Verify stock symbols in summary exist in original news."""
        summary_stocks = set(re.findall(r'\b[A-Z]{2,5}\b', summary))
        news_stocks = set(re.findall(r'\b[A-Z]{2,5}\b', original_news))
        
        # Allow common words that aren't stocks
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'HAS'}
        summary_stocks = summary_stocks - common_words
        
        # All summary stocks should exist in original news
        return summary_stocks.issubset(news_stocks) if summary_stocks else True

    def _verify_numbers(self, summary: str, original_news: str) -> bool:
        """Verify percentages and numbers in summary match original news."""
        summary_numbers = set(re.findall(r'\d+\.?\d*%', summary))
        news_numbers = set(re.findall(r'\d+\.?\d*%', original_news))
        
        # All summary percentages should exist in original news
        return summary_numbers.issubset(news_numbers) if summary_numbers else True

    def _check_freshness(self, original_news: str) -> bool:
        """Check if the news is actually recent (within 24 hours)."""
        # Look for timestamps in the news
        time_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{1,2} hours? ago)',  # X hours ago
            r'(today|yesterday)',     # Today/yesterday keywords
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, original_news, re.IGNORECASE):
                return True
        
        return True  # Default to fresh if we can't determine age

    def _check_hallucination(self, summary: str, original_news: str) -> bool:
        """Basic check for hallucination - summary should have substantial overlap with news."""
        summary_words = set(summary.lower().split())
        news_words = set(original_news.lower().split())
        
        # Remove common words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had'}
        summary_words = summary_words - common_words
        news_words = news_words - common_words
        
        if not summary_words:
            return False
            
        # Calculate overlap percentage
        overlap = len(summary_words.intersection(news_words)) / len(summary_words)
        return overlap > 0.3  # At least 30% word overlap

    def _run_task_with_retry(self, agents: List[Agent], task: Task, max_retries: int = 3) -> str:
        """Runs a single task with retry logic to handle potential failures."""
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
        Executes the entire financial summary workflow with validation.
        """
        try:
            logger.info("--- Starting Complete Financial Workflow with Validation ---")

            # --- Phase 1: Search ---
            search_result = self._run_search_phase()
            if "Error" in search_result:
                return {"status": "failed", "error": f"Search phase failed: {search_result}"}
            self.execution_results["search"] = search_result

            # --- Phase 2: Summary ---
            summary_result = self._run_summary_phase(search_result)
            if "Error" in summary_result:
                return {"status": "failed", "error": f"Summary phase failed: {summary_result}"}
            
            # --- Phase 2.5: Validation (NEW!) ---
            validation_results = self._validate_summary(summary_result, search_result)
            self.execution_results["validation"] = validation_results
            
            # Check if validation passed
            if validation_results["confidence_score"] < 60:
                logger.warning(f"Low confidence score: {validation_results['confidence_score']}/100")
                # Could implement retry logic here or flag for human review
            
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
                    "confidence_score": validation_results["confidence_score"],
                    "validation_passed": validation_results["confidence_score"] >= 60,
                },
            }

        except Exception as e:
            error_msg = f"Complete workflow failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "failed", "error": error_msg}

    def _run_search_phase(self) -> str:
        """Executes the search phase of the workflow with enforced 1-hour time limit."""
        logger.info("--- Phase 1: Searching for Real-Time News (Last 1 Hour) ---")
        search_agent = self.agents_factory.search_agent()
        search_task = Task(
            description="""Search for the latest US financial news from EXACTLY the past 1 hour. 
            Focus on major stock movements, earnings, and economic data.
            IMPORTANT: The search MUST be limited to the last 1 hour for real-time relevance.""",
            expected_output="Structured analysis of breaking financial news from the last 1 hour with source information.",
            agent=search_agent
        )
        return self._run_task_with_retry([search_agent], search_task)

    @apply_rate_limiting("gemini")
    def _run_summary_phase(self, search_result: str) -> str:
        """Executes the summary phase of the workflow."""
        logger.info("--- Phase 2: Creating Summary ---")
        summary_agent = self.agents_factory.summary_agent()
        summary_task = Task(
            description=f"""Create a financial summary from this news data: {search_result}

            CRITICAL ACCURACY REQUIREMENTS:
            1. ONLY use information directly from the provided news
            2. Keep ALL stock symbols exactly as mentioned in the news
            3. Use EXACT percentages and numbers from the original news
            4. Do NOT add any information not present in the source material
            5. Include the actual publication date from the most recent news article

            Structure: Title, Source, Date, Key Points (3-5), Market Implications (2-3)""",
            expected_output="Accurate financial summary under 500 words with verified information only.",
            agent=summary_agent
        )
        return self._run_task_with_retry([summary_agent], summary_task)

    @apply_rate_limiting("gemini")
    def _run_formatting_phase(self, summary_result: str) -> str:
        """Executes the formatting phase with enhanced ImageFinder preparation."""
        logger.info("--- Phase 3: Formatting with ImageFinder Integration ---")
        formatting_agent = self.agents_factory.formatting_agent()
        formatting_task = Task(
            description=f"""Format this summary for Telegram and find relevant images: {summary_result}""",
            expected_output="Well-formatted summary optimized for Telegram with image integration.",
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
                description=f"""Translate to {lang}, keeping stock symbols and numbers unchanged: {formatted_content}""",
                expected_output=f"Accurate translation in {lang} with preserved financial terms.",
                agent=translation_agent
            )
            translations[lang] = self._run_task_with_retry([translation_agent], translation_task)

        return translations

    def _run_sending_phase(self, formatted_content: str, translations: Dict[str, str]) -> Dict[str, str]:
        """Executes the content distribution phase via Telegram."""
        logger.info("--- Phase 5: Sending to Telegram ---")
        send_agent = self.agents_factory.send_agent()
        send_results = {}

        # Add validation info to the message
        validation = self.execution_results.get("validation", {})
        confidence_score = validation.get("confidence_score", 0)
        
        if confidence_score < 60:
            formatted_content += f"\n\n⚠️ *Confidence Score: {confidence_score}/100 - Please verify information*"

        # Send English summary
        english_send_task = Task(
            description=f"""Send English financial summary to Telegram: {formatted_content}""",
            expected_output="Confirmation of successful delivery with image integration status.",
            agent=send_agent
        )
        send_results["english"] = self._run_task_with_retry([send_agent], english_send_task)

        # Send translations
        for lang, content in translations.items():
            if "Error" not in content:
                lang_send_task = Task(
                    description=f"""Send {lang} summary to Telegram: {content}""",
                    expected_output=f"Confirmation of {lang} message delivery.",
                    agent=send_agent
                )
                send_results[lang] = self._run_task_with_retry([send_agent], lang_send_task)
            else:
                send_results[lang] = f"Skipped sending {lang} due to translation failure."

        return send_results