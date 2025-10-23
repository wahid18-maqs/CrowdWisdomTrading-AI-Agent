import logging
import time
import re
from typing import Any, Dict, List
from crewai import Agent, Crew, Process, Task
from datetime import datetime, timedelta
import json
from .agents import FinancialAgents
from .LLM_config import apply_rate_limiting

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class FinancialMarketCrew:
    """
      CrewAI implementation for financial market summary workflow.
    """

    def __init__(self):
        self.agents_factory = FinancialAgents()
        self.execution_results: Dict[str, Any] = {}
        self.flow = self._create_flow_graph()



    def _run_task_with_retry(self, agents: List[Agent], task: Task, max_retries: int = 3) -> str:
        """Run task with enhanced error handling for empty LLM responses."""
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
                
                # Check if result is empty or None
                if not result or str(result).strip() == "" or str(result).lower() == "none":
                    logger.warning(f"Empty response on attempt {attempt + 1}, retrying...")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return "Error: LLM returned empty response after all retries"
                
                return str(result)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Task execution error on attempt {attempt + 1}: {error_msg}")
                
                if any(limit_keyword in error_msg.lower() for limit_keyword in ["429", "quota", "rate limit", "quota exceeded"]):
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(10 * (attempt + 1)) 
                        continue
                elif "empty" in error_msg.lower() or "none" in error_msg.lower():
                    logger.warning(f"Empty response error on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                else:
                    logger.error(f"Non-recoverable error: {error_msg}")
                    if attempt < max_retries - 1:
                        continue

                return f"Error after {max_retries} attempts: {error_msg}"
        
        return "Error: Max retries exceeded with empty responses"

    def run_complete_workflow(self) -> Dict[str, Any]:
        """Execute simplified workflow: Search → Direct Telegram Send."""
        try:
            logger.info(" Starting Workflow: Search → Direct Telegram Send ")

            # Phase 1: Search and get summary under 400 words
            logger.info(" Phase 1: Search with summary creation")
            search_result = self._run_search_phase()
            if "Error" in search_result:
                return {"status": "failed", "error": f"Search phase failed: {search_result}"}
            self.execution_results["search"] = search_result

            # Phase 2: Send the raw summary directly to Telegram without any formatting
            logger.info("Phase 2: Sending raw summary content directly to Telegram")
            send_results = self._run_raw_telegram_sending(search_result)
            self.execution_results["send_results"] = send_results

            logger.info(" Workflow completed successfully")

            # Count successful sends (English + translations)
            sends_completed = 0
            if isinstance(send_results.get("raw_telegram"), str) and "successfully" in send_results["raw_telegram"].lower():
                sends_completed += 1
            if isinstance(send_results.get("translations"), dict):
                for lang, result in send_results["translations"].items():
                    if isinstance(result, str) and "successfully" in result.lower():
                        sends_completed += 1

            return {
                "status": "success",
                "results": self.execution_results,
                "execution_time": datetime.now().isoformat(),
                "summary": {
                    "workflow_type": "message sended to telegram",
                }
            }

        except Exception as e:
            error_msg = f"Ultra-simplified workflow failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "failed", "error": error_msg}

    def _create_flow_graph(self):
        """Create workflow graph for visualization"""
        if not VISUALIZATION_AVAILABLE:
            return None

        G = nx.DiGraph()

        # Define the workflow steps
        workflow_steps = [
            ("Start", "Tavily Search"),
            ("Tavily Search", "Image Extraction"),
            ("Image Extraction", "Vision AI Analysis"),
            ("Vision AI Analysis", "Image Filtering"),
            ("Image Filtering", "Summary Creation"),
            ("Summary Creation", "Content Formatting"),
            ("Content Formatting", "Telegram Delivery"),
            ("Telegram Delivery", "End")
        ]

        G.add_edges_from(workflow_steps)
        return G

    def plot(self):
        """Plot the financial market workflow"""
        if not VISUALIZATION_AVAILABLE:
            print(" Visualization not available. Install dependencies with:")
            print("pip install matplotlib networkx")
            return

        if self.flow is None:
            print(" Flow graph not created")
            return

        plt.figure(figsize=(14, 10))

        # Create hierarchical layout
        pos = {}
        levels = {
            "Start": (0, 4),
            "Tavily Search": (2, 4),
            "Image Extraction": (4, 4),
            "Vision AI Analysis": (6, 4),
            "Image Filtering": (8, 4),
            "Summary Creation": (10, 4),
            "Content Formatting": (12, 4),
            "Telegram Delivery": (14, 4),
            "End": (16, 4)
        }
        pos.update(levels)

        # Draw the graph
        nx.draw(self.flow, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=3000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                linewidths=2)

        plt.title("Financial Market Summary Workflow", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        print(" Workflow visualization displayed!")

    def _run_search_phase(self) -> str:
        """Search phase that creates two-message format summary."""
        logger.info("--- Phase 1: Searching for Financial News (Last 24 Hours) ---")
        search_agent = self.agents_factory.search_agent()
        search_task = Task(
            description="""Search comprehensively for the latest US financial news, extract chart images, and create a two-message format summary.

            PHASE 1 - SEARCH:
            - Search across ALL domains for comprehensive financial news coverage (24-hour timeframe)
            - Store search results in output folder for archiving
            - Focus on key market movements, earnings, Fed policy, notable stocks

            SEARCH STRATEGY:
            1. Use tavily_financial_search tool with these exact parameters:
               - query: "US stock market financial news"
               - hours_back: 1
               - max_results: 20
            2. Search timeframe: Last 1 hour (Tavily will search 24h, then filter to 1h)
            3. Gather comprehensive news from all available sources
            4. Store complete results in output folder
            5. Note the search_results_file path (will be in output/search_results/ folder)

            PHASE 2 - EXTRACT IMAGES:
            After completing the search, use the enhanced_financial_image_finder tool:
            1. Pass the search_content (summary of what you found)
            2. Pass the search_results_file path from Phase 1
            3. Set max_images to 1 (we only need one chart image)
            4. The tool will:
               - Extract article URLs from the search results file
               - Capture chart screenshots from ALL article URLs
               - Use AI to generate matching descriptions from article text
               - Return the best chart image with description

            PHASE 3 - CREATE TWO-MESSAGE FORMAT SUMMARY:
            - Create a TWO-MESSAGE FORMAT summary for Telegram delivery
            - Use the image description from Phase 2 in Message 1
            - Generate both: short image caption (with emojis) + full comprehensive summary

            TWO-MESSAGE FORMAT REQUIREMENTS:
            1. Message 1 (ENGAGING HOOK): Create an attention-grabbing hook from image description (≤20 words, ≤1024 chars)

               CRITICAL - TRANSFORM IMAGE DESCRIPTION INTO ENGAGING HOOK:
               - Use the AI-extracted image description from Phase 2
               - Transform it into a compelling hook that makes users STOP scrolling
               - Create curiosity and urgency: "Markets just shifted dramatically...", "Breaking: Major move in..."
               - Highlight the most dramatic/interesting aspect
               - Use power words: "surged", "plunged", "breaking", "historic", "unexpected", "dramatic"
               - Ask a compelling question OR state a surprising fact
               - Add relevant emojis (⚠️ 🚀 ⚡ 📈 📉 💰)
               - End with "Full breakdown 👇" or "What's happening? 👇"

               HOOK EXAMPLES (≤20 WORDS):
               Original: "Wednesday's slide in major averages was led by the Dow Jones..."
               Hook: "⚠️ Dow plunges 234 points! What's triggering the sell-off? 👇" (9 words)

               Original: "The S&P 500 rose 1.2% to close at 5,850.23..."
               Hook: "🚀 S&P 500 surges to 5,850 - best session in weeks! What's fueling this? 👇" (14 words)

               Original: "Tesla shares tumbled 8% following weak delivery figures..."
               Hook: "⚡ Tesla crashes 8% on weak deliveries! The shocking reason 👇" (10 words)

            2. Message 2 (Full Summary): Write a comprehensive market summary (≤250 words)
               - **TITLE**: Create a dynamic, engaging title based on the day's main market theme
                 Examples: "Tech Rally Drives Markets Higher" or "Fed Signals Pause as Inflation Cools" or "Markets Hit Record Highs on Strong Earnings"
               - **Market Overview:** Section with Dow Jones, S&P 500, and Nasdaq performance
               - **Macro News:** 1–2 short items about key background events (start each with 🔍)
               - **Notable Stocks:** 2–3 stocks that moved significantly with short explanations (use 🟢🔵🟡 to distinguish them)
               - **Commodities & FX:** Brief description if relevant
               - **Live Charts:** Section with market index chart links
               - **Disclaimer:** "*The above does not constitute investment advice…*"
               - Keep it factual, engaging, and under 250 words

            CRITICAL OUTPUT FORMAT - You MUST return EXACTLY this structure:

            === TELEGRAM_TWO_MESSAGE_FORMAT ===

            Message 1 (Image Caption):
            [Your short image caption here with emojis]

            Message 2 (Full Summary):
            **[Your Dynamic Title Here]**

            **Market Overview:**
            [Paragraph summarizing Dow, S&P, Nasdaq performance with specific numbers/percentages]

            **Macro News:**
            🔍 [First macro news item]
            🔍 [Second macro news item]

            **Notable Stocks:**
            🟢 **[Stock Symbol]** [Description]
            🔵 **[Stock Symbol]** [Description]
            🟡 **[Stock Symbol]** [Description]

            **Commodities & FX:**
            [Brief description if relevant]

            **Live Charts:**
            🔗 📊 <a href="https://finance.yahoo.com/quote/%5EGSPC/chart/">S&P 500</a>
            🔗 📈 <a href="https://finance.yahoo.com/quote/%5EIXIC/chart/">NASDAQ</a>
            🔗 📉 <a href="https://finance.yahoo.com/quote/%5EDJI/chart/">Dow Jones</a>
            🔗 ⚡ <a href="https://finance.yahoo.com/quote/%5EVIX/chart/">VIX</a>
            🔗 🏛️ <a href="https://finance.yahoo.com/quote/%5ETNX/chart/">10-Year</a>
            🔗 💰 <a href="https://finance.yahoo.com/quote/GC%3DF/chart/">Gold</a>

            *The above does not constitute investment advice and is for informational purposes only. Always conduct your own due diligence.*

            ---TELEGRAM_IMAGE_DATA---

            Do NOT add any extra text outside this format. The telegram_sender tool will parse this exact format.""",
            expected_output="Two-message format financial summary ready for Telegram delivery with search results stored in output folder.",
            agent=search_agent
        )
        return self._run_task_with_retry([search_agent], search_task)


    def _run_raw_telegram_sending(self, summary_content: str) -> Dict[str, str]:
        """Send summary content using the new two-message format to Telegram."""
        logger.info(" Sending summary content to Telegram (supports two-message format)")
        send_agent = self.agents_factory.send_agent()

        # Check if content has the new two-message format or old image data format
        if "=== TELEGRAM_TWO_MESSAGE_FORMAT ===" in summary_content:
            logger.info(" Detected new two-message format, sending directly")
            # New two-message format - send directly
            raw_send_task = Task(
                description=f"""Send this Telegram-ready content using the new two-message format to the English bot.

                CONTENT TO SEND (WITH FORMAT MARKERS):
                {summary_content}

                YOUR JOB:
                Use the telegram_sender tool with these parameters:
                - content: Pass the ENTIRE content shown above (including === TELEGRAM_TWO_MESSAGE_FORMAT === and ---TELEGRAM_IMAGE_DATA--- markers)
                - language: "english"

                CRITICAL - DO NOT:
                - Do NOT extract only "Message 1" and "Message 2" text
                - Do NOT remove the === TELEGRAM_TWO_MESSAGE_FORMAT === marker
                - Do NOT remove the ---TELEGRAM_IMAGE_DATA--- marker
                - Do NOT parse or modify the content in ANY way

                CRITICAL - YOU MUST:
                - Pass the COMPLETE content from above to telegram_sender (with ALL format markers intact)
                - The content MUST start with === TELEGRAM_TWO_MESSAGE_FORMAT ===
                - The content MUST end with ---TELEGRAM_IMAGE_DATA---
                - Set language parameter to "english"

                The telegram_sender tool will parse the format markers and send two separate messages.""",
                expected_output="Confirmation that telegram_sender was called with complete formatted content including === TELEGRAM_TWO_MESSAGE_FORMAT === and ---TELEGRAM_IMAGE_DATA--- markers.",
                agent=send_agent
            )

        elif "---TELEGRAM_IMAGE_DATA---" in summary_content:
            logger.info(" Detected old format with embedded image data")
            # Old format with embedded image data - send directly
            raw_send_task = Task(
                description=f"""Send this Telegram-ready content with embedded image data:

                CONTENT TO SEND:
                {summary_content}

                INSTRUCTIONS:
                1. This content has embedded Telegram image data
                2. Use telegram_sender tool with language='english' to handle the embedded data
                3. The tool will extract and process the image data automatically
                4. Send as provided without modifications

                The telegram_sender will handle image processing automatically.""",
                expected_output="Confirmation of successful delivery with embedded image data to Telegram.",
                agent=send_agent
            )

        else:
            logger.info(" No special format detected, sending as plain text")
            # No special format, send as plain text
            raw_send_task = self._create_text_only_task(summary_content, send_agent)

        # Send English version
        result = self._run_task_with_retry([send_agent], raw_send_task)

        # Now translate and send to other languages
        logger.info("Starting translation and sending to other language bots...")
        translation_results = self._translate_and_send(summary_content, send_agent)

        return {
            "raw_telegram": result,
            "translations": translation_results
        }

    def _translate_and_send(self, summary_content: str, send_agent) -> Dict[str, str]:
        """Translate and send summary to Arabic, Hindi, Hebrew, and German bots"""
        results = {}
        languages = ['arabic', 'hindi', 'hebrew', 'german']

        for language in languages:
            try:
                logger.info(f" Translating and sending to {language}...")

                # Create task to translate and send
                translate_task = Task(
                    description=f"""Translate and send content to {language} Telegram bot:

                    ORIGINAL CONTENT:
                    {summary_content}

                    STEP 1 - TRANSLATE WITH ENGAGING HOOK:
                    Call the financial_translator tool with BOTH required parameters:
                    {{
                        "content": "{summary_content[:100]}...[FULL CONTENT HERE]",
                        "target_language": "{language}"
                    }}

                    IMPORTANT: You MUST provide BOTH parameters:
                    - content: The full original content above (including === TELEGRAM_TWO_MESSAGE_FORMAT === and ---TELEGRAM_IMAGE_DATA--- markers)
                    - target_language: Must be exactly '{language}' (one of: 'arabic', 'hindi', 'hebrew', 'german')

                    ENGAGING HOOK STRATEGY FOR {language.upper()}:
                    - The Message 1 hook should be culturally relevant and engaging in {language}
                    - Use {language} power words that create urgency and curiosity
                    - Adapt the hook style to {language} cultural context
                    - Keep emojis (⚠️ 🚀 ⚡ 📈 📉 💰) - they work across all languages
                    - Maintain the compelling question or surprising fact structure
                    - Keep under 20 words in {language} -  punchy!

                    STEP 2 - SEND:
                    After translation, the financial_translator will return translated content in this format:
                    === TELEGRAM_TWO_MESSAGE_FORMAT ===
                    Message 1 (Image Caption):
                    [translated engaging hook in {language}]

                    Message 2 (Full Summary):
                    [translated summary in {language}]

                    ---TELEGRAM_IMAGE_DATA---

                    Pass this COMPLETE output to telegram_sender tool:
                    {{
                        "content": "[COMPLETE translated output from Step 1]",
                        "language": "{language}"
                    }}

                    CRITICAL REQUIREMENTS:
                    - Both financial_translator parameters (content AND target_language) are REQUIRED
                    - Do NOT modify the translator output before passing to telegram_sender
                    - Keep ALL format markers (=== and ---) intact
                    - The language parameter routes to the {language.upper()} bot
                    - The translated hook MUST be engaging and culturally appropriate for {language} speakers""",
                    expected_output=f"Confirmation that {language} translation was sent to {language} Telegram bot with both messages (caption and summary).",
                    agent=send_agent
                )

                result = self._run_task_with_retry([send_agent], translate_task)
                results[language] = result
                logger.info(f"{language} translation sent successfully")

                # Add delay between translations to avoid quota limits
                import time
                if language != languages[-1]:  # Don't delay after last language
                    logger.info(f"Waiting 10 seconds before next translation (quota management)...")
                    time.sleep(10)

            except Exception as e:
                logger.error(f" Failed to translate/send {language}: {e}")
                results[language] = f"Error: {str(e)}"

        return results

    def _create_text_only_task(self, text_content: str, send_agent) -> Task:
        """Create a text-only Telegram sending task"""
        return Task(
            description=f"""Send this financial summary content directly to Telegram exactly as provided:

            CONTENT TO SEND:
            {text_content}

            INSTRUCTIONS:
            1. Send the content exactly as provided above
            2. Do NOT add any formatting, structure, or message templates
            3. Do NOT follow any specific Telegram formatting rules
            4. Just pass the raw content to the Telegram channel
            5. The content is already under 400 words and ready to send
            6. Use telegram_sender tool with language='english' for simple delivery

            CRITICAL: Send the content AS-IS without any modifications, formatting, or structure.""",
            expected_output="Confirmation of successful raw content delivery to Telegram.",
            agent=send_agent
        )

