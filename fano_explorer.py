#!/usr/bin/env python3
"""
Fano Explorer - Autonomous Multi-Agent Research System

Usage:
    python fano_explorer.py auth      # Authenticate with ChatGPT/Gemini
    python fano_explorer.py start     # Start exploration loop
    python fano_explorer.py backlog   # Process unextracted threads (atomic chunking + review)
    python fano_explorer.py review    # Open review interface
    python fano_explorer.py status    # Show current status
    python fano_explorer.py stop      # Graceful shutdown (or use Ctrl+C)
"""

import sys
import os
import asyncio
import io
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars
except Exception as e:
    print(f"Warning: Could not load .env: {e}")

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

    console.print("\n[bold]Processing backlog...[/bold]")
    console.print("This will extract atomic insights from threads that haven't been processed yet.")
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


def cmd_help():
    """Show help."""
    console.print(__doc__)


COMMANDS = {
    "auth": cmd_auth,
    "start": cmd_start,
    "backlog": cmd_backlog,
    "review": cmd_review,
    "status": cmd_status,
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
