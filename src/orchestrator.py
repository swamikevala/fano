"""
Orchestrator - The Brain of Fano Explorer

The orchestrator is responsible for:
- Deciding which thread to work on next
- Formulating prompts for exploration and critique
- Judging when a thread is mature enough to synthesize
- Creating chunks from mature threads
- Managing rate limits and model selection
"""

import asyncio
import logging
import random
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from browser import (
    ChatGPTInterface,
    GeminiInterface,
    rate_tracker,
    deep_mode_tracker,
    select_model,
    should_use_deep_mode,
    get_deep_mode_status,
)
from models import (
    ExplorationThread, ThreadStatus, ExchangeRole,
    Chunk, ChunkStatus, ChunkFeedback,
    AxiomStore, BlessedInsight,
)
from storage.db import Database
from chunking import (
    AtomicExtractor, AtomicInsight, InsightStatus,
    get_dedup_checker, DeduplicationChecker,
    PanelExtractor, get_panel_extractor,
)
from review_panel import AutomatedReviewer
from augmentation import get_augmenter, Augmenter

# Setup logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/exploration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


class Orchestrator:
    """
    The brain of the exploration system.
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.db = Database(self.data_dir / "fano_explorer.db")
        self.axioms = AxiomStore(self.data_dir)

        self.chatgpt: Optional[ChatGPTInterface] = None
        self.gemini: Optional[GeminiInterface] = None

        self.running = False
        self.config = CONFIG["orchestration"]

        # Atomic chunking extractors
        self.extractor = AtomicExtractor(data_dir=self.data_dir, config=CONFIG)
        self.panel_extractor = PanelExtractor(data_dir=self.data_dir, config=CONFIG)

        # Use panel extraction by default (can be overridden in config)
        chunking_config = CONFIG.get("chunking", {})
        self.use_panel_extraction = chunking_config.get("use_panel_extraction", True)

        # Automated reviewer (initialized after browsers connect)
        self.reviewer: Optional[AutomatedReviewer] = None

        # Augmenter for blessed insights (initialized after reviewer)
        self.augmenter: Optional[Augmenter] = None

        # Deduplication checker (initialized after reviewer, uses same Claude instance)
        self.dedup_checker: Optional[DeduplicationChecker] = None
        
    async def run(self, process_backlog_first: bool = True):
        """Main exploration loop.

        Args:
            process_backlog_first: If True, process any unextracted threads before
                                   starting the main exploration loop.
        """
        self.running = True
        logger.info("Starting exploration loop")

        # Connect to browsers
        await self._connect_models()

        try:
            # Process backlog of unextracted threads first
            if process_backlog_first:
                await self.process_backlog()

            while self.running:
                await self._exploration_cycle()
                await asyncio.sleep(self.config["poll_interval"])
        finally:
            await self._disconnect_models()
    
    def stop(self):
        """Signal the orchestrator to stop."""
        self.running = False
        logger.info("Stop signal received")

    async def cleanup(self):
        """Clean up resources (called after interrupt)."""
        logger.info("Cleaning up...")
        await self._disconnect_models()

    async def process_backlog(self):
        """
        Process all threads that haven't had atomic extraction run yet.

        Scans for threads where:
        - Status is CHUNK_READY or ARCHIVED (synthesis completed)
        - chunks_extracted is False or missing

        For each such thread, runs atomic extraction and review panel.
        """
        logger.info("[backlog] Scanning for unprocessed threads...")

        threads_dir = self.data_dir / "explorations"
        if not threads_dir.exists():
            logger.info("[backlog] No explorations directory found")
            return

        # Find all threads needing extraction
        unprocessed = []
        for filepath in threads_dir.glob("*.json"):
            try:
                thread = ExplorationThread.load(filepath)

                # Check if thread is ready for extraction but hasn't been processed
                if thread.status in (ThreadStatus.CHUNK_READY, ThreadStatus.ARCHIVED):
                    if not getattr(thread, 'chunks_extracted', False):
                        unprocessed.append(thread)

            except Exception as e:
                logger.warning(f"[backlog] Could not load {filepath}: {e}")

        if not unprocessed:
            logger.info("[backlog] No unprocessed threads found")
            return

        logger.info(f"[backlog] Found {len(unprocessed)} threads to process")

        # Get an available model for extraction
        model_name, model = self._get_backlog_model()
        if not model:
            logger.warning("[backlog] No model available for extraction")
            return

        # Process each thread
        for i, thread in enumerate(unprocessed, 1):
            logger.info(f"[backlog] Processing {i}/{len(unprocessed)}: {thread.id}")

            try:
                await self._extract_and_review(thread, model_name, model)
            except Exception as e:
                logger.error(f"[backlog] Failed to process {thread.id}: {e}")

            # Small delay between threads to avoid rate limits
            if i < len(unprocessed):
                await asyncio.sleep(5)

        logger.info(f"[backlog] Completed processing {len(unprocessed)} threads")

    def _get_backlog_model(self) -> tuple[Optional[str], Optional[object]]:
        """Get an available model for backlog processing."""
        if self.gemini and rate_tracker.is_available("gemini"):
            return ("gemini", self.gemini)
        if self.chatgpt and rate_tracker.is_available("chatgpt"):
            return ("chatgpt", self.chatgpt)
        return (None, None)

    async def _connect_models(self):
        """Connect to LLM browser interfaces."""
        try:
            self.chatgpt = ChatGPTInterface()
            await self.chatgpt.connect()
            logger.info("Connected to ChatGPT")
        except Exception as e:
            logger.warning(f"Could not connect to ChatGPT: {e}")
            self.chatgpt = None

        try:
            self.gemini = GeminiInterface()
            await self.gemini.connect()
            logger.info("Connected to Gemini")
        except Exception as e:
            logger.warning(f"Could not connect to Gemini: {e}")
            self.gemini = None

        # Initialize the automated reviewer with connected browsers
        if CONFIG.get("review_panel", {}).get("enabled", False):
            try:
                self.reviewer = AutomatedReviewer(
                    gemini_browser=self.gemini,
                    chatgpt_browser=self.chatgpt,
                    config=CONFIG,
                    data_dir=self.data_dir,
                )
                logger.info("Initialized automated review panel")

                # Initialize augmenter (uses reviewer's Claude instance)
                if CONFIG.get("augmentation", {}).get("enabled", False):
                    self.augmenter = get_augmenter(
                        claude_reviewer=self.reviewer.claude_reviewer,
                        config=CONFIG,
                        data_dir=self.data_dir,
                    )
                    if self.augmenter:
                        logger.info("Initialized augmenter for blessed insights")

                # Initialize deduplication checker (uses reviewer's Claude instance)
                if CONFIG.get("deduplication", {}).get("enabled", True):
                    self.dedup_checker = get_dedup_checker(
                        claude_reviewer=self.reviewer.claude_reviewer,
                        config=CONFIG,
                    )
                    # Load existing blessed insights into the checker
                    blessed_insights = self._get_blessed_insights()
                    self.dedup_checker.load_known_insights(blessed_insights)
                    logger.info(f"Initialized deduplication checker with {len(blessed_insights)} known insights")

            except Exception as e:
                logger.warning(f"Could not initialize review panel: {e}")
                self.reviewer = None
    
    async def _disconnect_models(self):
        """Disconnect from browser interfaces."""
        if self.chatgpt:
            await self.chatgpt.disconnect()
        if self.gemini:
            await self.gemini.disconnect()
    
    def _get_available_model(self) -> Optional[tuple[str, object]]:
        """Get an available model for use."""
        models = []
        if self.chatgpt and rate_tracker.is_available("chatgpt"):
            models.append(("chatgpt", self.chatgpt))
        if self.gemini and rate_tracker.is_available("gemini"):
            models.append(("gemini", self.gemini))
        
        if not models:
            return None
        
        # Randomize to distribute load
        return random.choice(models)
    
    async def _exploration_cycle(self):
        """Single cycle of the exploration loop."""

        # Build available models dict
        available_models = {}
        if self.chatgpt and rate_tracker.is_available("chatgpt"):
            available_models["chatgpt"] = self.chatgpt
        if self.gemini and rate_tracker.is_available("gemini"):
            available_models["gemini"] = self.gemini

        if not available_models:
            logger.info("All models rate-limited, waiting...")
            await asyncio.sleep(self.config["backoff_base"])
            return

        # Get or create a thread to work on
        thread = self._select_thread()

        if thread is None:
            # Spawn a new thread
            thread = self._spawn_new_thread()
            logger.info(f"Spawned new thread: {thread.topic[:60]}...")

        # Perform work on the thread using weighted model selection
        if thread.needs_exploration:
            model_name = select_model("exploration", available_models)
            if model_name:
                model = available_models[model_name]
                await self._do_exploration(thread, model_name, model)
        elif thread.needs_critique:
            # _do_critique handles its own model selection with critique weights
            model_name = list(available_models.keys())[0]
            model = available_models[model_name]
            await self._do_critique(thread, model_name, model)
        
        # Check if thread is ready for synthesis
        if self._is_chunk_ready(thread):
            await self._synthesize_chunk(thread)
        
        # Save thread state
        thread.save(self.data_dir)
    
    def _select_thread(self) -> Optional[ExplorationThread]:
        """Select an active thread to work on."""
        threads = self._load_active_threads()
        
        if not threads:
            return None
        
        # Prioritize threads that need work
        for thread in threads:
            if thread.needs_exploration or thread.needs_critique:
                return thread
        
        return threads[0] if threads else None
    
    def _load_active_threads(self) -> list[ExplorationThread]:
        """Load all active exploration threads."""
        threads_dir = self.data_dir / "explorations"
        threads = []
        
        if threads_dir.exists():
            for filepath in threads_dir.glob("*.json"):
                thread = ExplorationThread.load(filepath)
                if thread.status == ThreadStatus.ACTIVE:
                    threads.append(thread)
        
        return threads[:self.config["max_active_threads"]]
    
    def _spawn_new_thread(self) -> ExplorationThread:
        """Create a new exploration thread."""
        
        # Get axiom context
        excerpts = self.axioms.get_excerpts()
        numbers = self.axioms.get_target_numbers()
        unexplained = self.axioms.get_unexplained_numbers()
        
        # Prioritize unexplained numbers
        if unexplained:
            target = unexplained[0]
            target_nums = [target]
        elif numbers:
            target_nums = [numbers[0].id]
        else:
            target_nums = []
        
        # Pick seed axioms
        seed_ids = [ex.id for ex in excerpts[:2]] if excerpts else []
        
        # Generate topic based on what we're exploring
        topic = self._generate_topic(target_nums, seed_ids)
        
        thread = ExplorationThread.create_new(
            topic=topic,
            seed_axioms=seed_ids,
            target_numbers=target_nums,
        )
        
        thread.save(self.data_dir)
        return thread
    
    def _generate_topic(self, target_nums: list[str], seed_ids: list[str]) -> str:
        """Generate a topic description for a new thread."""
        # Seed questions for specific numbers
        number_topics = {
            "7": "How do the 7 lines of the Fano plane relate to the 7 lokas in Vedic cosmology?",
            "14": "The 14 Maheshvara Sutras and the 14 lokas — explore geometric necessity",
            "22": "The 22 shrutis as projective structure — why exactly 22?",
            "24": "Samkhya's 24 tattvas — how does this enumeration arise from geometry?",
            "25": "The 25th tattva (Purusha) as geometric completion — structural analysis",
            "36": "Kashmir Shaivism's 36 tattvas — geometric extension of Samkhya's 24",
            "72": "72 melakartas and 72% water — structural coincidence or deep pattern?",
            "84": "84 asanas and 84 siddhas — why this specific number appears",
            "108": "108 sacred repetitions — geometric basis in projective structures",
            "112": "Vijñāna Bhairava's 112 dharanas and the 112 chakras — explore structural reasons",
        }

        if target_nums:
            # Check if we have a specific seed question for this number
            for num in target_nums:
                if num in number_topics:
                    return number_topics[num]
            # Default for other numbers
            return f"Exploring mathematical structure behind {', '.join(target_nums)} across Fano geometry, Sanskrit grammar, music theory, and yogic cosmology"

        # Open exploration topics (rotate through these)
        open_topics = [
            "Open exploration of Fano plane, Sanskrit grammar, music theory, and tantric cosmology connections",
            "How do the incidence structures of projective geometry manifest in yogic/tantric enumeration systems?",
            "Explore the relationship between Panini's grammatical structures and the tattva hierarchies",
            "The geometry of consciousness: connecting Fano structures to dharana techniques",
        ]
        return random.choice(open_topics)
    
    async def _do_exploration(self, thread: ExplorationThread, model_name: str, model):
        """Perform an exploration step."""
        prompt = self._build_exploration_prompt(thread)

        # Determine if we should use deep mode
        use_deep = should_use_deep_mode(model_name, thread, "exploration")

        mode_str = " [DEEP]" if use_deep else ""
        logger.info(f"Exploring [{thread.id}] with {model_name}{mode_str}")

        try:
            await model.start_new_chat()

            # Pass deep mode flag to send_message
            # ChatGPT: use Pro mode for deep, Thinking mode for standard
            if model_name == "chatgpt":
                response = await model.send_message(
                    prompt,
                    use_pro_mode=use_deep,
                    use_thinking_mode=not use_deep
                )
            else:
                response = await model.send_message(prompt, use_deep_think=use_deep)

            # Check if deep mode was actually used and record it
            deep_mode_used = getattr(model, 'last_deep_mode_used', False)
            if deep_mode_used:
                mode_key = "gemini_deep_think" if model_name == "gemini" else "chatgpt_pro"
                deep_mode_tracker.record_usage(mode_key)

            # Extract only structured sections, stripping preamble and recaps
            clean_response = self._extract_structured_response(
                response, ['NEW_INSIGHTS', 'CONNECTIONS', 'QUESTIONS']
            )

            thread.add_exchange(
                role=ExchangeRole.EXPLORER,
                model=model_name,
                prompt=prompt,
                response=clean_response,
                deep_mode_used=deep_mode_used,
            )

            logger.info(f"Exploration complete, {len(response)} -> {len(clean_response)} chars" + (" [DEEP]" if deep_mode_used else ""))

        except Exception as e:
            logger.error(f"Exploration failed: {e}")
    
    async def _do_critique(self, thread: ExplorationThread, model_name: str, model):
        """Perform a critique step."""
        prompt = self._build_critique_prompt(thread)

        # Use weighted selection for critique (prefers ChatGPT)
        available_models = {}
        if self.chatgpt:
            available_models["chatgpt"] = self.chatgpt
        if self.gemini:
            available_models["gemini"] = self.gemini

        selected = select_model("critique", available_models)
        if selected:
            critique_model_name = selected
            critique_model = available_models[selected]
        else:
            # Fallback to original model
            critique_model_name, critique_model = model_name, model

        # Determine if we should use deep mode
        use_deep = should_use_deep_mode(critique_model_name, thread, "critique")

        mode_str = " [DEEP]" if use_deep else ""
        logger.info(f"Critiquing [{thread.id}] with {critique_model_name}{mode_str}")

        try:
            await critique_model.start_new_chat()

            # Pass deep mode flag to send_message
            # ChatGPT: use Pro mode for deep, Thinking mode for standard
            if critique_model_name == "chatgpt":
                response = await critique_model.send_message(
                    prompt,
                    use_pro_mode=use_deep,
                    use_thinking_mode=not use_deep
                )
            else:
                response = await critique_model.send_message(prompt, use_deep_think=use_deep)

            # Check if deep mode was actually used and record it
            deep_mode_used = getattr(critique_model, 'last_deep_mode_used', False)
            if deep_mode_used:
                mode_key = "gemini_deep_think" if critique_model_name == "gemini" else "chatgpt_pro"
                deep_mode_tracker.record_usage(mode_key)

            # Extract only structured sections, stripping preamble and recaps
            clean_response = self._extract_structured_response(
                response, ['CRITICAL_ISSUES', 'PROMISING_DIRECTIONS', 'PROBING_QUESTIONS']
            )

            thread.add_exchange(
                role=ExchangeRole.CRITIC,
                model=critique_model_name,
                prompt=prompt,
                response=clean_response,
                deep_mode_used=deep_mode_used,
            )

            logger.info(f"Critique complete, {len(response)} -> {len(clean_response)} chars" + (" [DEEP]" if deep_mode_used else ""))

        except Exception as e:
            logger.error(f"Critique failed: {e}")
    
    def _get_other_model(self, current: str) -> Optional[tuple[str, object]]:
        """Get a different model than the current one."""
        if current == "chatgpt" and self.gemini and rate_tracker.is_available("gemini"):
            return ("gemini", self.gemini)
        if current == "gemini" and self.chatgpt and rate_tracker.is_available("chatgpt"):
            return ("chatgpt", self.chatgpt)
        return None
    
    def _build_exploration_prompt(self, thread: ExplorationThread) -> str:
        """Build the prompt for exploration."""
        context = self.axioms.get_context_for_exploration()
        date_prefix = datetime.now().strftime("[FANO %m-%d]")

        # Load seed prompt from config
        exploration_config = CONFIG.get("exploration", {})
        intro = exploration_config.get("intro", "You are exploring mathematical connections.")
        key_numbers = exploration_config.get("key_numbers", "")
        goals = exploration_config.get("goals", "")
        guidance = exploration_config.get("guidance", "")

        prompt_parts = [
            f"{date_prefix} {intro.strip()}",
            "",
            "KEY NUMBERS TO DECODE:",
            key_numbers,
            "",
            goals.strip(),
            "",
            guidance.strip(),
            "",
            context,
            "",
        ]
        
        if thread.exchanges:
            prompt_parts.append("=== PREVIOUS EXPLORATION ===")
            prompt_parts.append(thread.get_context_for_prompt())
            prompt_parts.append("")
            prompt_parts.append("Build on this exploration. Go deeper. Find the structure.")
        else:
            prompt_parts.append("Begin your exploration. What mathematical structures might connect these domains?")
            prompt_parts.append("Focus especially on explaining why the numbers appear as they do.")

        # Add anti-bloat instructions with structured format
        prompt_parts.extend([
            "",
            "=== RESPONSE FORMAT ===",
            "IMPORTANT: Do not summarize or recap previous discussion.",
            "State only NEW insights. Assume full context is available.",
            "Skip any preamble - go straight to the sections.",
            "",
            "You MUST structure your response EXACTLY as:",
            "",
            "[NEW_INSIGHTS]",
            "- (bullet points of genuinely new ideas or mathematical structures)",
            "",
            "[CONNECTIONS]",
            "- (new connections discovered between domains)",
            "",
            "[QUESTIONS]",
            "- (open questions worth exploring next)",
        ])

        return "\n".join(prompt_parts)
    
    def _build_critique_prompt(self, thread: ExplorationThread) -> str:
        """Build the prompt for critique."""
        context = thread.get_context_for_prompt()
        date_prefix = datetime.now().strftime("[FANO %m-%d]")

        prompt = f"""{date_prefix} You are a rigorous mathematical critic. Review the following exploration:

{context}

Your task:
1. Identify any FORCING or AD HOC assumptions
2. Point out where the structure feels ARBITRARY rather than INEVITABLE
3. Ask probing questions that would deepen the exploration
4. Note what's working well and feels NATURAL
5. Suggest alternative angles if something feels stuck

The core question: Does this feel DISCOVERED or INVENTED?

Signs of discovery (good):
- "It couldn't be any other way"
- "This explains WHY, not just THAT"
- "Multiple independent paths led here"
- "The numbers fall out naturally"

Signs of invention (bad):
- "We need to assume this for it to work"
- "It matches, but we don't know why"
- "This was chosen to make the numbers fit"
- "It feels clever but arbitrary"

Be constructive but rigorous. The goal is truth, not validation.
Push toward depth, not toward any particular application.

=== RESPONSE FORMAT ===
IMPORTANT: Do not summarize or recap previous discussion.
State only NEW critiques and questions. Assume full context is available.

Structure your response as:
[CRITICAL_ISSUES]
(Genuine problems with the reasoning - be specific)

[PROMISING_DIRECTIONS]
(What's working and worth pursuing deeper)

[PROBING_QUESTIONS]
(Questions that would test or deepen the ideas)"""

        return prompt
    
    def _is_chunk_ready(self, thread: ExplorationThread) -> bool:
        """Determine if a thread is ready to be synthesized into a chunk."""
        
        # Minimum exchanges
        if thread.exchange_count < self.config["min_exchanges_for_chunk"]:
            return False
        
        # Maximum exchanges (force synthesis)
        if thread.exchange_count >= self.config["max_exchanges_per_thread"]:
            return True
        
        # Count critique rounds
        critique_count = sum(1 for e in thread.exchanges if e.role == ExchangeRole.CRITIC)
        if critique_count < CONFIG["synthesis"]["min_critiques"]:
            return False
        
        # Check for profundity signals in recent exchanges
        recent_text = " ".join(e.response for e in thread.exchanges[-4:])
        profundity_score = self._calculate_profundity(recent_text)
        
        # Ready if high profundity
        if profundity_score > 0.6:
            return True
        
        return False
    
    def _calculate_profundity(self, text: str) -> float:
        """Calculate a profundity score based on signal words."""
        text_lower = text.lower()
        
        profundity_signals = CONFIG["synthesis"]["profundity_signals"]
        doubt_signals = CONFIG["synthesis"]["doubt_signals"]
        
        prof_count = sum(1 for sig in profundity_signals if sig in text_lower)
        doubt_count = sum(1 for sig in doubt_signals if sig in text_lower)
        
        # Normalize
        score = (prof_count - doubt_count) / max(len(profundity_signals), 1)
        return max(0, min(1, (score + 1) / 2))  # Scale to 0-1
    
    async def _synthesize_chunk(self, thread: ExplorationThread):
        """Synthesize a thread into a reviewable chunk."""
        # Use weighted selection for synthesis (balanced)
        available_models = {}
        if self.chatgpt:
            available_models["chatgpt"] = self.chatgpt
        if self.gemini:
            available_models["gemini"] = self.gemini

        model_name = select_model("synthesis", available_models)
        if not model_name:
            logger.warning("No model available for synthesis")
            return

        model = available_models[model_name]

        # Always use deep mode for synthesis
        use_deep = should_use_deep_mode(model_name, thread, "synthesis")

        mode_str = " [DEEP]" if use_deep else ""
        logger.info(f"Synthesizing chunk from thread [{thread.id}] with {model_name}{mode_str}")

        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(thread)

        try:
            await model.start_new_chat()

            # Pass deep mode flag to send_message
            # ChatGPT: use Pro mode for deep, Thinking mode for standard
            if model_name == "chatgpt":
                response = await model.send_message(
                    synthesis_prompt,
                    use_pro_mode=use_deep,
                    use_thinking_mode=not use_deep
                )
            else:
                response = await model.send_message(synthesis_prompt, use_deep_think=use_deep)

            # Check if deep mode was actually used and record it
            deep_mode_used = getattr(model, 'last_deep_mode_used', False)
            if deep_mode_used:
                mode_key = "gemini_deep_think" if model_name == "gemini" else "chatgpt_pro"
                deep_mode_tracker.record_usage(mode_key)
            elif use_deep:
                logger.warning(f"Deep mode requested but not available")
            
            # Parse the synthesis response
            title, summary, content = self._parse_synthesis(response, thread)
            
            # Calculate profundity
            profundity = self._calculate_profundity(content)
            critique_count = sum(1 for e in thread.exchanges if e.role == ExchangeRole.CRITIC)
            
            # Create chunk
            chunk = Chunk.create_from_thread(
                thread_id=thread.id,
                title=title,
                content=content,
                summary=summary,
                target_numbers=thread.target_numbers,
                profundity_score=profundity,
                critique_rounds=critique_count,
            )
            
            # Save chunk
            chunk.save(self.data_dir)

            # Mark thread as chunk-ready
            thread.status = ThreadStatus.CHUNK_READY
            thread.save(self.data_dir)

            logger.info(f"Created chunk [{chunk.id}]: {chunk.title}")

            # Atomic extraction and review (if enabled)
            if CONFIG.get("chunking", {}).get("mode") == "atomic":
                await self._extract_and_review(thread, model_name, model)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
    
    def _build_synthesis_prompt(self, thread: ExplorationThread) -> str:
        """Build prompt for chunk synthesis."""
        context = thread.get_context_for_prompt(max_exchanges=10)
        date_prefix = datetime.now().strftime("[FANO %m-%d]")

        return f"""{date_prefix} You are a mathematical writer. Synthesize the following exploration into a clear, well-structured write-up.

{context}

Write a synthesis that:
1. Opens with a clear TITLE (single line)
2. Follows with a one-sentence SUMMARY
3. Then provides the full CONTENT explaining the mathematical structure discovered

The write-up should:
- Be clear and readable
- Preserve the mathematical rigor
- Highlight what feels INEVITABLE vs what remains speculative
- Explicitly show how the structure explains the target numbers

Format:
TITLE: [Your title here]
SUMMARY: [One sentence summary]
CONTENT:
[Full write-up in markdown]"""
    
    def _parse_synthesis(self, response: str, thread: ExplorationThread) -> tuple[str, str, str]:
        """Parse synthesis response into title, summary, content."""

        # Try to extract structured parts
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|SUMMARY:)', response, re.DOTALL)
        summary_match = re.search(r'SUMMARY:\s*(.+?)(?:\n|CONTENT:)', response, re.DOTALL)
        content_match = re.search(r'CONTENT:\s*(.+)', response, re.DOTALL)

        title = title_match.group(1).strip() if title_match else f"Exploration {thread.id}"
        summary = summary_match.group(1).strip() if summary_match else "Mathematical exploration synthesis"
        content = content_match.group(1).strip() if content_match else response

        return title, summary, content

    def _extract_structured_response(self, response: str, sections: list[str]) -> str:
        """
        Extract only the structured sections from a response, stripping preamble.

        Args:
            response: Full LLM response text
            sections: List of section names to extract (e.g. ['NEW_INSIGHTS', 'CONNECTIONS'])

        Returns:
            Cleaned response with only the structured sections
        """
        # Build pattern to find all sections
        section_pattern = r'\[(' + '|'.join(sections) + r')\]'

        # Find the first section marker - everything before it is preamble
        first_match = re.search(section_pattern, response)
        if not first_match:
            # No structured sections found, return original (but log warning)
            logger.warning("No structured sections found in response, keeping original")
            return response

        # Extract from first section onwards
        structured_part = response[first_match.start():]

        # Clean up: remove any trailing content after the last section's content
        # by keeping only content up to any obvious sign-off phrases
        signoff_patterns = [
            r'\n\nLet me know',
            r'\n\nWould you like',
            r'\n\nShall I',
            r'\n\nI hope this',
            r'\n\nIn summary,',  # Catch summary attempts
            r'\n\nTo summarize,',
            r'\n\nOverall,',
        ]
        for pattern in signoff_patterns:
            match = re.search(pattern, structured_part, re.IGNORECASE)
            if match:
                structured_part = structured_part[:match.start()]

        return structured_part.strip()

    async def _extract_and_review(
        self,
        thread: ExplorationThread,
        model_name: str,
        model,
    ):
        """
        Extract atomic insights from a thread and run automated review.

        Claude Opus is the primary extraction author (best at precise articulation).
        Falls back to browser LLMs if Claude is unavailable.

        Args:
            thread: The exploration thread to extract from
            model_name: Name of the fallback model
            model: The fallback model instance (browser LLM)
        """
        if thread.chunks_extracted:
            logger.info(f"[{thread.id}] Already extracted, skipping")
            return

        # Get Claude reviewer for extraction
        claude_reviewer = None
        if self.reviewer:
            claude_reviewer = self.reviewer.claude_reviewer

        try:
            # Get blessed insights for dependency context
            blessed_insights = self._get_blessed_insights()

            # Choose extraction method based on config
            if self.use_panel_extraction:
                logger.info(f"[{thread.id}] Starting PANEL extraction (all 3 LLMs)")
                insights = await self.panel_extractor.extract_from_thread(
                    thread=thread,
                    gemini=self.gemini,
                    chatgpt=self.chatgpt,
                    claude_reviewer=claude_reviewer,
                    blessed_insights=blessed_insights,
                )
            else:
                author = "Claude Opus" if claude_reviewer and claude_reviewer.is_available() else model_name
                logger.info(f"[{thread.id}] Starting single-LLM extraction with {author}")
                insights = await self.extractor.extract_from_thread(
                    thread=thread,
                    claude_reviewer=claude_reviewer,
                    fallback_model=model,
                    fallback_model_name=model_name,
                    blessed_insights=blessed_insights,
                )

            if not insights:
                logger.info(f"[{thread.id}] No insights extracted")
                thread.chunks_extracted = True
                thread.extraction_note = "No insights met extraction criteria"
                thread.save(self.data_dir)
                return

            logger.info(f"[{thread.id}] Extracted {len(insights)} atomic insights")

            # Deduplicate insights against known blessed insights
            if self.dedup_checker:
                unique_insights = []
                duplicate_count = 0

                for insight in insights:
                    is_dup, dup_of_id = await self.dedup_checker.is_duplicate(
                        new_text=insight.insight,
                        new_id=insight.id,
                    )
                    if is_dup:
                        logger.info(f"[{thread.id}] Skipping duplicate insight [{insight.id}] (similar to {dup_of_id})")
                        duplicate_count += 1
                    else:
                        unique_insights.append(insight)
                        # Add to known set to catch duplicates within same batch
                        self.dedup_checker.add_known_insight(insight.id, insight.insight)

                if duplicate_count > 0:
                    logger.info(f"[{thread.id}] Filtered out {duplicate_count} duplicate insights")

                insights = unique_insights

            if not insights:
                logger.info(f"[{thread.id}] All insights were duplicates")
                thread.chunks_extracted = True
                thread.extraction_note = "All insights were duplicates of existing ones"
                thread.save(self.data_dir)
                return

            # Save insights to data/chunks/
            chunks_dir = self.data_dir / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)

            for insight in insights:
                await asyncio.to_thread(insight.save, chunks_dir)

            # Run automated review if enabled
            if self.reviewer and CONFIG.get("review_panel", {}).get("enabled", False):
                blessed_summary = self._get_blessed_summary()
                await self._review_insights(insights, blessed_summary)

            # Mark thread as extracted
            thread.chunks_extracted = True
            await asyncio.to_thread(thread.save, self.data_dir)

            logger.info(f"[{thread.id}] Extraction and review complete")

        except Exception as e:
            logger.error(f"[{thread.id}] Extraction failed: {e}")
            thread.extraction_note = f"Extraction error: {str(e)}"
            await asyncio.to_thread(thread.save, self.data_dir)

    async def _review_insights(
        self,
        insights: list[AtomicInsight],
        blessed_summary: str,
    ):
        """
        Run automated review on extracted insights.

        Args:
            insights: List of AtomicInsight to review
            blessed_summary: Summary of blessed axioms for context
        """
        logger.info(f"Starting automated review of {len(insights)} insights")

        chunks_dir = self.data_dir / "chunks"

        for insight in insights:
            try:
                logger.info(f"Reviewing insight [{insight.id}]")

                review = await self.reviewer.review_insight(
                    chunk_id=insight.id,
                    insight_text=insight.insight,
                    confidence=insight.confidence,
                    tags=insight.tags,
                    dependencies=insight.depends_on,
                    blessed_axioms_summary=blessed_summary,
                )

                # Get outcome action
                action = self.reviewer.get_outcome_action(review)
                logger.info(f"[{insight.id}] Review outcome: {review.final_rating} ({action})")

                # Apply rating to insight
                insight.rating = review.final_rating
                insight.is_disputed = review.is_disputed
                insight.reviewed_at = review.reviewed_at

                # Update status based on rating
                if review.final_rating == "⚡":
                    insight.status = InsightStatus.BLESSED
                    # Build review summary for augmentation context
                    review_summary = self._build_review_summary(review)
                    # Add to blessed insights and augment
                    await self._bless_insight(insight, review_summary)
                elif review.final_rating == "✗":
                    insight.status = InsightStatus.REJECTED
                else:
                    insight.status = InsightStatus.INTERESTING

                # Save updated insight
                await asyncio.to_thread(insight.save, chunks_dir)

            except Exception as e:
                logger.error(f"[{insight.id}] Review failed: {e}")
                insight.status = InsightStatus.PENDING
                await asyncio.to_thread(insight.save, chunks_dir)

    def _get_blessed_summary(self) -> str:
        """Get a summary of blessed axioms/insights for prompts."""
        # Use the axiom store's context method
        return self.axioms.get_context_for_exploration()

    def _build_review_summary(self, review) -> str:
        """
        Build a summary of reviewer findings for augmentation context.

        Extracts key points that reviewers found compelling, which helps
        the augmenter understand what makes this insight worth illustrating.
        """
        parts = []

        if not review.rounds:
            return ""

        # Get final round (most relevant reasoning)
        final_round = review.rounds[-1]

        for llm, resp in final_round.responses.items():
            if resp.rating == "⚡":  # Only include endorsements
                parts.append(f"{llm.title()} found compelling:")
                if resp.naturalness_assessment:
                    parts.append(f"  - Naturalness: {resp.naturalness_assessment}")
                if resp.structural_analysis:
                    parts.append(f"  - Structure: {resp.structural_analysis}")
                if resp.reasoning:
                    parts.append(f"  - Reasoning: {resp.reasoning}")

        # Note if there were mind changes
        if review.mind_changes:
            for mc in review.mind_changes:
                if mc.to_rating == "⚡":
                    parts.append(f"{mc.llm.title()} changed to bless because: {mc.reason}")

        return "\n".join(parts)

    def _get_blessed_insights(self) -> list[dict]:
        """Get list of blessed insights for dependency matching."""
        blessed = []

        # Load seed aphorisms first (user-provided starting points)
        seeds = self.axioms.get_seed_aphorisms()
        for seed in seeds:
            blessed.append({
                "id": seed.id,
                "text": seed.text,
                "tags": seed.tags,
                "is_seed": True,
            })

        # Load from blessed_insights.json if exists
        blessed_file = self.data_dir / "blessed_insights.json"
        if blessed_file.exists():
            import json
            with open(blessed_file, encoding="utf-8") as f:
                data = json.load(f)
                blessed.extend(data.get("insights", []))

        # Also include insights with BLESSED status from insights directory
        blessed_dir = self.data_dir / "chunks" / "insights" / "blessed"
        if blessed_dir.exists():
            for filepath in blessed_dir.glob("*.json"):
                try:
                    insight = AtomicInsight.load(filepath)
                    blessed.append({
                        "id": insight.id,
                        "text": insight.insight,
                        "tags": insight.tags,
                    })
                except Exception:
                    pass

        return blessed

    async def _bless_insight(self, insight: AtomicInsight, review_summary: str = ""):
        """Add an insight to the blessed insights store and augment it."""
        import json

        blessed_file = self.data_dir / "blessed_insights.json"

        # Load existing
        if blessed_file.exists():
            with open(blessed_file, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"insights": []}

        # Add new insight
        data["insights"].append({
            "id": insight.id,
            "text": insight.insight,
            "confidence": insight.confidence,
            "tags": insight.tags,
            "depends_on": insight.depends_on,
            "source_thread_id": insight.source_thread_id,
            "blessed_at": datetime.now().isoformat(),
        })

        # Save
        with open(blessed_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Blessed insight [{insight.id}] added to axiom store")

        # Add to deduplication checker for future duplicate detection
        if self.dedup_checker:
            self.dedup_checker.add_known_insight(insight.id, insight.insight)

        # Augment the blessed insight (generate diagrams, tables, proofs, code)
        if self.augmenter:
            try:
                augmented = await self.augmenter.augment_insight(
                    insight_id=insight.id,
                    insight_text=insight.insight,
                    tags=insight.tags,
                    dependencies=insight.depends_on,
                    review_summary=review_summary,
                )
                aug_count = len(augmented.augmentations)
                if aug_count > 0:
                    logger.info(f"[{insight.id}] Generated {aug_count} augmentations")
            except Exception as e:
                logger.warning(f"[{insight.id}] Augmentation failed: {e}")
