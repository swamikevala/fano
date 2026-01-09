#!/usr/bin/env python3
"""
Fano Explorer - Autonomous Multi-Agent Research System

Usage:
    python fano_explorer.py auth           # Authenticate with ChatGPT/Gemini
    python fano_explorer.py start          # Start exploration loop
    python fano_explorer.py backlog        # Process unextracted threads (atomic chunking + review)
    python fano_explorer.py review         # Open review interface
    python fano_explorer.py status         # Show current status
    python fano_explorer.py retry-disputed # Re-run Round 3 for disputed insights with Deep Think
    python fano_explorer.py stop           # Graceful shutdown (or use Ctrl+C)
"""

import sys
import os
import asyncio
import io
from pathlib import Path

# Load environment variables from .env file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    # Manual loading to avoid any dotenv issues
    with open(_env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Reconfigure stdout/stderr to use UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # Some environments don't support reconfigure

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel

console = Console(force_terminal=True, legacy_windows=False)


def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ███████╗ █████╗ ███╗   ██╗ ██████╗                     ║
    ║   ██╔════╝██╔══██╗████╗  ██║██╔═══██╗                    ║
    ║   █████╗  ███████║██╔██╗ ██║██║   ██║                    ║
    ║   ██╔══╝  ██╔══██║██║╚██╗██║██║   ██║                    ║
    ║   ██║     ██║  ██║██║ ╚████║╚██████╔╝                    ║
    ║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝                     ║
    ║                                                           ║
    ║   E X P L O R E R                                        ║
    ║                                                           ║
    ║   Fano Plane · Sanskrit Grammar · Indian Music           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def cmd_auth():
    """Authenticate with LLM providers."""
    from browser.base import authenticate_all
    console.print("\n[bold]Starting authentication...[/bold]\n")
    console.print("This will open Chrome windows for you to log in to:")
    console.print("  • ChatGPT (chat.openai.com)")
    console.print("  • Gemini (gemini.google.com)")
    console.print("\nLog in manually, then close the browser when done.\n")
    asyncio.run(authenticate_all())
    console.print("\n[green]✓ Sessions saved![/green]\n")


def cmd_start():
    """Start the exploration loop."""
    from orchestrator import Orchestrator

    console.print("\n[bold]Starting exploration loop...[/bold]")
    console.print("Press Ctrl+C to stop gracefully.\n")

    orchestrator = Orchestrator()

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        # Run cleanup in a new event loop since the previous one was interrupted
        try:
            asyncio.run(orchestrator.cleanup())
        except Exception:
            pass
        console.print("[yellow]Stopped.[/yellow]")


def cmd_backlog():
    """Process backlog of unextracted threads."""
    from orchestrator import Orchestrator
    from models.thread import ExplorationThread

    console.print("\n[bold]Processing backlog...[/bold]")

    # First check if there's anything to process
    data_dir = Path(__file__).parent / "data"
    threads_dir = data_dir / "explorations"

    if not threads_dir.exists():
        console.print("[yellow]No explorations directory found.[/yellow]")
        return

    # Find unprocessed threads
    unprocessed = []
    for filepath in threads_dir.glob("*.json"):
        try:
            thread = ExplorationThread.load(filepath)
            if thread.status in ["CHUNK_READY", "ARCHIVED"]:
                if not getattr(thread, "chunks_extracted", False):
                    unprocessed.append(thread)
        except Exception:
            pass

    if not unprocessed:
        console.print("[yellow]No unprocessed threads found in backlog.[/yellow]")
        console.print("All threads have already been extracted.")
        return

    console.print(f"Found [bold]{len(unprocessed)}[/bold] threads to process:")
    for t in unprocessed[:5]:
        console.print(f"  • {t.id}: {t.topic[:50]}...")
    if len(unprocessed) > 5:
        console.print(f"  ... and {len(unprocessed) - 5} more")

    console.print("\nThis will extract atomic insights from these threads.")
    console.print("Press Ctrl+C to stop gracefully.\n")

    orchestrator = Orchestrator()

    async def run_backlog():
        await orchestrator._connect_models()
        try:
            await orchestrator.process_backlog()
        finally:
            await orchestrator._disconnect_models()

    try:
        asyncio.run(run_backlog())
        console.print("\n[green]✓ Backlog processing complete![/green]\n")
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        try:
            asyncio.run(orchestrator.cleanup())
        except Exception:
            pass
        console.print("[yellow]Stopped.[/yellow]")


def cmd_review():
    """Start the review web interface."""
    from ui.review_server import start_server
    console.print("\n[bold]Starting review server...[/bold]")
    console.print("Open http://localhost:8765 in your browser.\n")
    console.print("Press Ctrl+C to stop.\n")
    start_server()


def cmd_status():
    """Show current exploration status."""
    from storage.db import Database
    from models.thread import ExplorationThread
    from models.chunk import Chunk

    data_dir = Path(__file__).parent / "data"
    db = Database(data_dir / "fano_explorer.db")
    
    # Active threads
    threads = db.get_active_threads()
    console.print(Panel(f"[bold]Active Exploration Threads:[/bold] {len(threads)}"))
    for t in threads:
        console.print(f"  • {t.topic[:60]}... ({t.exchange_count} exchanges)")
    
    # Pending chunks
    pending = list(Path("data/chunks/pending").glob("*.md"))
    console.print(Panel(f"[bold]Chunks Pending Review:[/bold] {len(pending)}"))
    
    # Rate limit status
    from browser.base import get_rate_limit_status
    status = get_rate_limit_status()
    console.print(Panel("[bold]Rate Limit Status:[/bold]"))
    for model, info in status.items():
        if info["limited"]:
            console.print(f"  • {model}: [red]LIMITED[/red] (retry at {info['retry_at']})")
        else:
            console.print(f"  • {model}: [green]OK[/green]")
    
    # Stats
    profound = len(list(Path("data/chunks/profound").glob("*.md")))
    interesting = len(list(Path("data/chunks/interesting").glob("*.md")))
    rejected = len(list(Path("data/chunks/rejected").glob("*.md")))
    console.print(Panel(
        f"[bold]Cumulative Stats:[/bold]\n"
        f"  ⚡ Profound: {profound}\n"
        f"  ?  Interesting: {interesting}\n"
        f"  ✗  Rejected: {rejected}"
    ))


def cmd_retry_disputed():
    """Re-run Round 3 for disputed insights with Deep Think enabled."""
    from pathlib import Path
    from review_panel.models import ChunkReview
    from review_panel.round3 import run_round3
    from review_panel.claude_api import ClaudeReviewer
    from browser.gemini import GeminiInterface
    from browser.chatgpt import ChatGPTInterface

    data_dir = Path(__file__).parent / "data"
    disputed_dir = data_dir / "reviews" / "disputed"
    chunks_dir = data_dir / "chunks" / "insights"  # Chunks are in insights subdirectory

    if not disputed_dir.exists():
        console.print("[yellow]No disputed reviews found.[/yellow]")
        return

    disputed_reviews = list(disputed_dir.glob("*.json"))
    if not disputed_reviews:
        console.print("[yellow]No disputed reviews found.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(disputed_reviews)} disputed reviews to retry[/bold]")
    console.print("This will re-run Round 3 with Deep Think mode for all LLMs.\n")

    async def retry_all():
        # Initialize LLM interfaces
        console.print("[bold]Connecting to LLMs...[/bold]")

        gemini = GeminiInterface()
        chatgpt = ChatGPTInterface()
        claude = ClaudeReviewer()

        try:
            await gemini.connect()
            # Check if actually logged in
            logged_in = await gemini._check_login_status()
            if not logged_in:
                console.print("  [yellow]![/yellow] Gemini not logged in - run 'python fano_explorer.py auth' first")
                gemini = None
            else:
                console.print("  [green]✓[/green] Gemini connected")
        except Exception as e:
            console.print(f"  [red]✗[/red] Gemini failed: {e}")
            gemini = None

        try:
            await chatgpt.connect()
            # Check if actually logged in by looking for login prompts
            page_text = await chatgpt.page.inner_text("body")
            if "log in" in page_text.lower() or "sign up" in page_text.lower():
                console.print("  [yellow]![/yellow] ChatGPT not logged in - run 'python fano_explorer.py auth' first")
                chatgpt = None
            else:
                console.print("  [green]✓[/green] ChatGPT connected")
        except Exception as e:
            console.print(f"  [red]✗[/red] ChatGPT failed: {e}")
            chatgpt = None

        if claude.api_key:
            console.print("  [green]✓[/green] Claude API ready")
        else:
            console.print("  [yellow]![/yellow] Claude API key not found")
            claude = None

        if not gemini and not chatgpt and not claude:
            console.print("[red]No LLMs available. Cannot retry.[/red]")
            return

        console.print("")

        for i, review_path in enumerate(disputed_reviews, 1):
            try:
                review = ChunkReview.load(review_path)
                chunk_id = review.chunk_id

                # Find the chunk file to get the insight text
                chunk_path = None
                for subdir in ["interesting", "blessed", "rejected", "pending"]:
                    potential = chunks_dir / subdir / f"{chunk_id}.md"
                    if potential.exists():
                        chunk_path = potential
                        break

                if not chunk_path:
                    console.print(f"[{i}/{len(disputed_reviews)}] [yellow]Chunk {chunk_id} not found, skipping[/yellow]")
                    continue

                # Read insight from markdown file (insight is in blockquote after header)
                try:
                    content = chunk_path.read_text(encoding="utf-8")
                    # Extract insight from blockquote (line starting with ">")
                    insight_text = ""
                    for line in content.split("\n"):
                        if line.startswith("> "):
                            insight_text = line[2:].strip()
                            break
                    if not insight_text:
                        console.print(f"[{i}/{len(disputed_reviews)}] [yellow]Could not extract insight text, skipping[/yellow]")
                        continue
                except Exception as e:
                    console.print(f"[{i}/{len(disputed_reviews)}] [yellow]Error reading chunk: {e}[/yellow]")
                    continue

                console.print(f"\n[{i}/{len(disputed_reviews)}] [bold]Retrying: {chunk_id}[/bold]")
                console.print(f"  Insight: {insight_text[:80]}...")

                # Get Round 2 from the review (take the last one if duplicates exist)
                round2 = None
                for r in review.rounds:
                    if r.round_number == 2:
                        round2 = r  # Keep overwriting to get the last Round 2

                if not round2:
                    # If no Round 2, try to use Round 1
                    for r in review.rounds:
                        if r.round_number == 1:
                            round2 = r
                            console.print(f"  [yellow]No Round 2 found, using Round 1[/yellow]")
                            break

                if not round2:
                    console.print(f"  [yellow]No suitable round found, skipping[/yellow]")
                    continue

                # Re-run Round 3 with Deep Think
                console.print(f"  [cyan]Running Round 3 with Deep Think...[/cyan]")

                new_round3, mind_changes, is_disputed = await run_round3(
                    chunk_insight=insight_text,
                    round2=round2,
                    gemini_browser=gemini,
                    chatgpt_browser=chatgpt,
                    claude_reviewer=claude,
                    config={},
                )

                # Update the review
                # Remove old Round 3 if present
                review.rounds = [r for r in review.rounds if r.round_number != 3]
                review.rounds.append(new_round3)
                review.mind_changes.extend(mind_changes)

                # Determine final outcome
                final_ratings = list(new_round3.get_ratings().values())
                unique_ratings = set(final_ratings)

                if len(unique_ratings) == 1:
                    review.final_rating = final_ratings[0]
                    review.is_unanimous = True
                    review.is_disputed = False
                    console.print(f"  [green]✓ Resolved to unanimous: {review.final_rating}[/green]")
                else:
                    # Majority wins
                    review.final_rating = new_round3.get_majority_rating()
                    review.is_unanimous = False
                    review.is_disputed = is_disputed
                    if is_disputed:
                        console.print(f"  [yellow]Still disputed: {final_ratings}[/yellow]")
                    else:
                        console.print(f"  [green]Majority decision: {review.final_rating}[/green]")

                # Save review (will move to completed if no longer disputed)
                review.save(data_dir)

                # If no longer disputed, remove from disputed folder
                if not review.is_disputed and review_path.exists():
                    review_path.unlink()
                    console.print(f"  [green]Moved to completed[/green]")

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()

        # Cleanup
        console.print("\n[bold]Cleaning up...[/bold]")
        if gemini:
            await gemini.disconnect()
        if chatgpt:
            await chatgpt.disconnect()

        console.print("\n[green]✓ Done![/green]")

    try:
        asyncio.run(retry_all())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")


def cmd_help():
    """Show help."""
    console.print(__doc__)


COMMANDS = {
    "auth": cmd_auth,
    "start": cmd_start,
    "backlog": cmd_backlog,
    "review": cmd_review,
    "status": cmd_status,
    "retry-disputed": cmd_retry_disputed,
    "help": cmd_help,
    "--help": cmd_help,
    "-h": cmd_help,
}


def main():
    print_banner()
    
    if len(sys.argv) < 2:
        cmd_help()
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        cmd_help()


if __name__ == "__main__":
    main()
