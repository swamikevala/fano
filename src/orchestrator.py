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

from browser import ChatGPTInterface, GeminiInterface, rate_tracker
from models import (
    ExplorationThread, ThreadStatus, ExchangeRole,
    Chunk, ChunkStatus, ChunkFeedback,
    AxiomStore, BlessedInsight,
)
from storage.db import Database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/exploration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
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
        
    async def run(self):
        """Main exploration loop."""
        self.running = True
        logger.info("Starting exploration loop")
        
        # Connect to browsers
        await self._connect_models()
        
        try:
            while self.running:
                await self._exploration_cycle()
                await asyncio.sleep(self.config["poll_interval"])
        finally:
            await self._disconnect_models()
    
    def stop(self):
        """Signal the orchestrator to stop."""
        self.running = False
        logger.info("Stop signal received")
    
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
        
        # Check if any model is available
        model_info = self._get_available_model()
        if not model_info:
            logger.info("All models rate-limited, waiting...")
            await asyncio.sleep(self.config["backoff_base"])
            return
        
        model_name, model = model_info
        
        # Get or create a thread to work on
        thread = self._select_thread()
        
        if thread is None:
            # Spawn a new thread
            thread = self._spawn_new_thread()
            logger.info(f"Spawned new thread: {thread.topic[:60]}...")
        
        # Perform work on the thread
        if thread.needs_exploration:
            await self._do_exploration(thread, model_name, model)
        elif thread.needs_critique:
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
        if target_nums:
            return f"Exploring mathematical structure behind {', '.join(target_nums)} using Fano plane geometry and Sanskrit grammar connections"
        return "Open exploration of Fano plane, Sanskrit grammar, and Indian music theory connections"
    
    async def _do_exploration(self, thread: ExplorationThread, model_name: str, model):
        """Perform an exploration step."""
        prompt = self._build_exploration_prompt(thread)
        
        logger.info(f"Exploring [{thread.id}] with {model_name}")
        
        try:
            if model_name == "chatgpt":
                await model.start_new_chat()
            else:
                await model.start_new_chat()
            
            response = await model.send_message(prompt)
            
            thread.add_exchange(
                role=ExchangeRole.EXPLORER,
                model=model_name,
                prompt=prompt,
                response=response,
            )
            
            logger.info(f"Exploration complete, {len(response)} chars")
            
        except Exception as e:
            logger.error(f"Exploration failed: {e}")
    
    async def _do_critique(self, thread: ExplorationThread, model_name: str, model):
        """Perform a critique step."""
        prompt = self._build_critique_prompt(thread)
        
        # Try to use a different model for critique
        other_model = self._get_other_model(model_name)
        if other_model:
            critique_model_name, critique_model = other_model
        else:
            critique_model_name, critique_model = model_name, model
        
        logger.info(f"Critiquing [{thread.id}] with {critique_model_name}")
        
        try:
            await critique_model.start_new_chat()
            response = await critique_model.send_message(prompt)
            
            thread.add_exchange(
                role=ExchangeRole.CRITIC,
                model=critique_model_name,
                prompt=prompt,
                response=response,
            )
            
            logger.info(f"Critique complete, {len(response)} chars")
            
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

        prompt_parts = [
            f"{date_prefix} You are exploring deep mathematical connections between:",
            "- Fano plane incidence geometry (7 points, 7 lines, 3 points per line)",
            "- Sanskrit grammar (particularly Panini's system)",
            "- Indian classical music theory (swaras, shrutis, ragas)",
            "",
            "Your goal is to find mathematical structures that:",
            "1. Feel NATURAL and INEVITABLE (not forced)",
            "2. Are ELEGANT and SYMMETRIC",
            "3. DECODE the specific numbers in the teachings below",
            "",
            "Follow your mathematical curiosity. Let the structure reveal itself.",
            "If something feels forced, abandon it. If something feels inevitable,",
            "pursue it—even if it seems unrelated to anything practical.",
            "",
            "The criterion for a good direction is: does this feel DISCOVERED",
            "rather than INVENTED? Does it explain WHY the numbers are what they are,",
            "not just THAT they match?",
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
Push toward depth, not toward any particular application."""
        
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
        logger.info(f"Synthesizing chunk from thread [{thread.id}]")
        
        # Get a model for synthesis
        model_info = self._get_available_model()
        if not model_info:
            logger.warning("No model available for synthesis")
            return
        
        model_name, model = model_info
        
        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(thread)
        
        try:
            await model.start_new_chat()
            response = await model.send_message(synthesis_prompt)
            
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
