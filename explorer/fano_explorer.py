#!/usr/bin/env python3
"""
Fano Explorer - Autonomous Multi-Agent Research System

Usage:
    python fano_explorer.py auth           # Authenticate with ChatGPT/Gemini
    python fano_explorer.py start          # Start exploration loop
    python fano_explorer.py backlog        # Process unextracted threads (atomic chunking + review)
    python fano_explorer.py review            # Open review interface
    python fano_explorer.py status            # Show current status
    python fano_explorer.py retry-disputed    # Run Round 4 for disputed insights
    python fano_explorer.py retry-interesting # Run Round 4 for interesting insights (stuck with '?')
    python fano_explorer.py cleanup-dupes     # Find and archive duplicate blessed insights
    python fano_explorer.py stop              # Graceful shutdown (or use Ctrl+C)
"""

import sys
import os
import asyncio
from pathlib import Path

# Load environment variables from .env file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from rich.console import Console
from rich.panel import Panel

console = Console(force_terminal=True, legacy_windows=False)

DATA_DIR = Path(__file__).parent / "data"


def print_banner():
    """Display the Fano Explorer ASCII banner."""
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
    """Show API key configuration info."""
    import os
    console.print("\n[bold]API Configuration[/bold]\n")
    console.print("Fano now uses API-based LLM access via OpenRouter.\n")
    console.print("Set the following environment variable:")
    console.print("  [cyan]OPENROUTER_API_KEY[/cyan] - Your OpenRouter API key\n")

    # Check if key is set
    if os.environ.get("OPENROUTER_API_KEY"):
        console.print("[green]✓ OPENROUTER_API_KEY is set[/green]\n")
    else:
        console.print("[yellow]⚠ OPENROUTER_API_KEY is not set[/yellow]")
        console.print("  Get your key at: https://openrouter.ai/keys\n")


def cmd_start():
    """Start the exploration loop with review UI."""
    import threading
    import yaml
    from explorer.src.orchestrator import Orchestrator
    from explorer.src.ui.review_server import start_server

    # Load review server config
    config_path = Path(__file__).parent / "config.yaml"
    host = "127.0.0.1"
    port = 8765
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        host = config.get("review_server", {}).get("host", host)
        port = config.get("review_server", {}).get("port", port)

    # Start review server in background thread (disable reloader for thread safety)
    def run_review_server():
        from explorer.src.ui.review_server import app, CONFIG
        review_host = CONFIG["review_server"]["host"]
        review_port = CONFIG["review_server"]["port"]
        # use_reloader=False required when running in thread
        app.run(host=review_host, port=review_port, debug=False, use_reloader=False)

    review_thread = threading.Thread(target=run_review_server, daemon=True)
    review_thread.start()

    console.print("\n[bold]Starting exploration loop...[/bold]")
    console.print("Press Ctrl+C to stop gracefully.\n")

    orchestrator = Orchestrator()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        try:
            asyncio.run(orchestrator.cleanup())
        except Exception:
            pass
        console.print("[yellow]Stopped.[/yellow]")


def cmd_backlog():
    """Process backlog of unextracted threads."""
    from explorer.src.commands import find_unprocessed_threads, process_backlog

    console.print("\n[bold]Processing backlog...[/bold]")

    unprocessed = find_unprocessed_threads(DATA_DIR)

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

    try:
        asyncio.run(process_backlog(DATA_DIR))
        console.print("\n[green]✓ Backlog processing complete![/green]\n")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")


def cmd_review():
    """Start the review web interface."""
    import yaml
    from explorer.src.ui.review_server import start_server

    # Load review server config
    config_path = Path(__file__).parent / "config.yaml"
    host = "127.0.0.1"
    port = 8765
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        host = config.get("review_server", {}).get("host", host)
        port = config.get("review_server", {}).get("port", port)

    console.print("\n[bold]Starting review server...[/bold]")
    console.print(f"Open http://{host}:{port} in your browser.\n")
    console.print("Press Ctrl+C to stop.\n")
    start_server()


def cmd_status():
    """Show current exploration status."""
    from explorer.src.commands import get_status

    status = get_status(DATA_DIR)

    console.print(Panel(f"[bold]Active Exploration Threads:[/bold] {len(status.active_threads)}"))
    for t in status.active_threads:
        console.print(f"  • {t.topic[:60]}... ({t.exchange_count} exchanges)")

    console.print(Panel(f"[bold]Chunks Pending Review:[/bold] {status.pending_chunks}"))

    console.print(Panel("[bold]Rate Limit Status:[/bold]"))
    for rl in status.rate_limits:
        if rl.limited:
            console.print(f"  • {rl.model}: [red]LIMITED[/red] (retry at {rl.retry_at})")
        else:
            console.print(f"  • {rl.model}: [green]OK[/green]")

    console.print(Panel(
        f"[bold]Cumulative Stats:[/bold]\n"
        f"  ⚡ Profound: {status.profound_count}\n"
        f"  ?  Interesting: {status.interesting_count}\n"
        f"  ✗  Rejected: {status.rejected_count}"
    ))


def cmd_retry_disputed():
    """Run Round 4 (Modification Focus) for disputed insights."""
    from explorer.src.commands import LLMConnections, RetryProcessor
    from explorer.src.commands.llm_connections import connect_llms, disconnect_llms

    disputed_dir = DATA_DIR / "reviews" / "disputed"
    if not disputed_dir.exists() or not list(disputed_dir.glob("*.json")):
        console.print("[yellow]No disputed reviews found.[/yellow]")
        return

    disputed_count = len(list(disputed_dir.glob("*.json")))
    console.print(f"\n[bold]Found {disputed_count} disputed reviews to retry[/bold]")
    console.print("This will run Round 4 (Modification Focus) with full deliberation history.\n")

    async def run_retry():
        console.print("[bold]Connecting to LLMs...[/bold]")

        def on_status(status):
            if status.connected:
                console.print(f"  [green]✓[/green] {status.name} {status.message.lower()}")
            else:
                console.print(f"  [yellow]![/yellow] {status.name}: {status.message}")

        connections = await connect_llms(on_status)

        if not connections.has_any():
            console.print("[red]No LLMs available. Cannot retry.[/red]")
            return

        console.print("")

        processor = RetryProcessor(DATA_DIR, connections)

        def on_progress(current, total, result):
            console.print(f"\n[{current}/{total}] [bold]{result.chunk_id}[/bold]")
            if result.insight_text:
                console.print(f"  Insight: {result.insight_text[:80]}...")

            if result.error:
                console.print(f"  [red]Error: {result.error}[/red]")
            elif result.outcome == "intractable":
                console.print(f"  [red]✗ Insight marked as INTRACTABLE[/red]")
            elif result.was_modified:
                console.print(f"  [cyan]Insight was MODIFIED in Round 4[/cyan]")
                if result.modified_text:
                    console.print(f"  New insight: {result.modified_text[:100]}...")

            if result.outcome == "resolved":
                console.print(f"  [green]✓ Resolved to unanimous: {result.final_rating}[/green]")
            elif result.outcome == "majority":
                console.print(f"  [yellow]Majority decision (still disputed): {result.final_rating}[/yellow]")
            elif result.outcome == "disputed":
                console.print(f"  [yellow]Still disputed after Round 4[/yellow]")

        await processor.process_disputed(on_progress)

        console.print("\n[bold]Cleaning up...[/bold]")
        await disconnect_llms(connections)
        console.print("\n[green]✓ Done![/green]")

    try:
        asyncio.run(run_retry())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")


def cmd_retry_interesting():
    """Run Round 4 for insights stuck in 'interesting' status."""
    from explorer.src.commands import LLMConnections, RetryProcessor
    from explorer.src.commands.llm_connections import connect_llms, disconnect_llms

    interesting_dir = DATA_DIR / "chunks" / "insights" / "interesting"
    completed_dir = DATA_DIR / "reviews" / "completed"

    if not interesting_dir.exists():
        console.print("[yellow]No interesting insights found.[/yellow]")
        return

    # Count insights with reviews
    count = sum(1 for f in interesting_dir.glob("*.json")
                if (completed_dir / f"{f.stem}.json").exists())

    if count == 0:
        console.print("[yellow]No interesting insights with reviews found.[/yellow]")
        return

    console.print(f"\n[bold]Found {count} interesting insights to retry[/bold]")
    console.print("These insights got '?' rating but may benefit from another Round 4 attempt.\n")

    async def run_retry():
        console.print("[bold]Connecting to LLMs...[/bold]")

        def on_status(status):
            if status.connected:
                console.print(f"  [green]✓[/green] {status.name} {status.message.lower()}")
            else:
                console.print(f"  [yellow]![/yellow] {status.name}: {status.message}")

        connections = await connect_llms(on_status)

        if not connections.has_any():
            console.print("[red]No LLMs available. Cannot retry.[/red]")
            return

        console.print("")

        processor = RetryProcessor(DATA_DIR, connections)

        def on_progress(current, total, result):
            console.print(f"\n[{current}/{total}] [bold]{result.chunk_id}[/bold]")
            if result.insight_text:
                console.print(f"  Insight: {result.insight_text[:80]}...")

            if result.error:
                console.print(f"  [red]Error: {result.error}[/red]")
            elif result.outcome == "intractable":
                console.print(f"  [red]✗ Insight marked as INTRACTABLE[/red]")
            elif result.was_modified:
                console.print(f"  [cyan]Modified insight (v{result.final_rating})[/cyan]")

            if result.outcome == "resolved":
                console.print(f"  [green]✓ Resolved to unanimous: {result.final_rating}[/green]")
                if result.final_rating == "⚡":
                    console.print(f"  [green]✓ Moved to BLESSED[/green]")
                elif result.final_rating == "✗":
                    console.print(f"  [red]Moved to rejected[/red]")
            elif result.outcome == "majority":
                console.print(f"  [yellow]Majority decision (disputed): {result.final_rating}[/yellow]")
            elif result.outcome == "disputed":
                console.print(f"  [yellow]Still no consensus[/yellow]")

        await processor.process_interesting(on_progress)

        console.print("\n[bold]Cleaning up...[/bold]")
        await disconnect_llms(connections)
        console.print("\n[green]✓ Done![/green]")

    try:
        asyncio.run(run_retry())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")


def cmd_cleanup_dupes():
    """Find and archive duplicate blessed insights."""
    import subprocess

    console.print("\n[bold]Cleaning up duplicate blessed insights...[/bold]")

    apply_flag = "--apply" in sys.argv
    cmd = [sys.executable, "cleanup_duplicates.py"]
    if apply_flag:
        cmd.append("--apply")
        console.print("Running in APPLY mode - duplicates will be moved to archive.\n")
    else:
        console.print("Running in DRY RUN mode - no changes will be made.")
        console.print("Use 'cleanup-dupes --apply' to actually archive duplicates.\n")

    try:
        subprocess.run(cmd, cwd=Path(__file__).parent)
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
    "retry-interesting": cmd_retry_interesting,
    "cleanup-dupes": cmd_cleanup_dupes,
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
