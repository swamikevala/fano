#!/usr/bin/env python3
"""
Browser Pool Service - Entry Point

A long-running HTTP service that manages browser instances for LLM access.

Usage:
    python browser_pool.py start          # Start the service
    python browser_pool.py auth gemini    # Authenticate with Gemini
    python browser_pool.py auth chatgpt   # Authenticate with ChatGPT
    python browser_pool.py status         # Check status (requires running service)
"""

import asyncio
import logging
import sys
from pathlib import Path

import yaml
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_banner():
    """Print startup banner."""
    print("""
    +-----------------------------------------------------------+
    |                                                           |
    |   BROWSER POOL SERVICE                                    |
    |                                                           |
    |   Shared LLM Access Layer                                 |
    |                                                           |
    +-----------------------------------------------------------+
    """)


def cmd_start():
    """Start the pool service."""
    print_banner()
    config = load_config()

    host = config.get("server", {}).get("host", "127.0.0.1")
    port = config.get("server", {}).get("port", 9000)

    logger.info(f"Starting Browser Pool on {host}:{port}")

    # Import here to avoid circular imports
    from src.api import create_app

    app = create_app(config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


async def cmd_auth(backend: str):
    """Trigger authentication for a backend."""
    print(f"Authenticating with {backend}...")

    # Add explorer's src to path
    explorer_src = Path(__file__).parent.parent / "explorer" / "src"
    sys.path.insert(0, str(explorer_src))

    if backend == "gemini":
        from browser.gemini import GeminiInterface
        browser = GeminiInterface()
        await browser.connect()
        print(f"[{backend}] Browser opened. Please log in manually.")
        print(f"[{backend}] Close the browser window when done.")
        # Wait for browser to close
        try:
            await browser.context.wait_for_event("close", timeout=0)
        except Exception:
            pass
        print(f"[{backend}] Authentication complete!")

    elif backend == "chatgpt":
        from browser.chatgpt import ChatGPTInterface
        browser = ChatGPTInterface()
        await browser.connect()
        print(f"[{backend}] Browser opened. Please log in manually.")
        print(f"[{backend}] Close the browser window when done.")
        try:
            await browser.context.wait_for_event("close", timeout=0)
        except Exception:
            pass
        print(f"[{backend}] Authentication complete!")

    elif backend == "claude":
        import os
        if os.environ.get("ANTHROPIC_API_KEY"):
            print(f"[{backend}] API key found in environment. Ready to use.")
        else:
            print(f"[{backend}] No API key found. Set ANTHROPIC_API_KEY in .env")

    else:
        print(f"Unknown backend: {backend}")
        print("Available backends: gemini, chatgpt, claude")


def cmd_status():
    """Check status of running service."""
    import requests

    config = load_config()
    host = config.get("server", {}).get("host", "127.0.0.1")
    port = config.get("server", {}).get("port", 9000)

    try:
        response = requests.get(f"http://{host}:{port}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("\nBrowser Pool Status")
            print("=" * 40)
            for backend, info in status.items():
                if info:
                    print(f"\n{backend.upper()}:")
                    print(f"  Available: {info.get('available', False)}")
                    print(f"  Authenticated: {info.get('authenticated', False)}")
                    print(f"  Rate Limited: {info.get('rate_limited', False)}")
                    print(f"  Queue Depth: {info.get('queue_depth', 0)}")
                    if info.get('deep_mode_uses_today') is not None:
                        print(f"  Deep Mode: {info.get('deep_mode_uses_today')}/{info.get('deep_mode_limit')} today")
                    if info.get('pro_mode_uses_today') is not None:
                        print(f"  Pro Mode: {info.get('pro_mode_uses_today')}/{info.get('pro_mode_limit')} today")
        else:
            print(f"Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to pool service at {host}:{port}")
        print("Is the service running? Start with: python browser_pool.py start")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "start":
        cmd_start()

    elif command == "auth":
        if len(sys.argv) < 3:
            print("Usage: python browser_pool.py auth <backend>")
            print("Backends: gemini, chatgpt, claude")
            sys.exit(1)
        backend = sys.argv[2].lower()
        asyncio.run(cmd_auth(backend))

    elif command == "status":
        cmd_status()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
