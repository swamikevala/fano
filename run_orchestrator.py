#!/usr/bin/env python3
"""
Unified Orchestrator Entry Point.

Runs the unified orchestrator with Explorer and Documenter modules.
This replaces the separate explorer/src/orchestrator.py and documenter/main.py
entry points with a single coordinated system.

Usage:
    python run_orchestrator.py                    # Run with both modules
    python run_orchestrator.py --no-documenter    # Explorer only
    python run_orchestrator.py --no-explorer      # Documenter only

Prerequisites:
    - Pool service must be running (python pool/src/api.py)
    - Browsers must be logged in to LLM services
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


async def main():
    """Run the unified orchestrator with HTTP API."""
    import argparse
    from orchestrator.main import Orchestrator
    from orchestrator.api import run_with_api

    parser = argparse.ArgumentParser(
        description="Fano Unified Orchestrator - Coordinates Explorer and Documenter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_orchestrator.py                         # Run both modules
    python run_orchestrator.py --no-documenter         # Explorer only
    python run_orchestrator.py --backends gemini       # Single backend
    python run_orchestrator.py --pool-url http://localhost:9000
    python run_orchestrator.py --api-port 9001         # Custom API port
        """
    )

    parser.add_argument(
        "--data-dir",
        default="data/orchestrator",
        help="Directory for state persistence (default: data/orchestrator)"
    )
    parser.add_argument(
        "--pool-url",
        default="http://localhost:8765",
        help="Pool service URL (default: http://localhost:8765)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["gemini", "chatgpt", "claude"],
        help="LLM backends to use (default: gemini chatgpt claude)"
    )
    parser.add_argument(
        "--no-explorer",
        action="store_true",
        help="Disable Explorer module"
    )
    parser.add_argument(
        "--no-documenter",
        action="store_true",
        help="Disable Documenter module"
    )
    parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="HTTP API host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=9001,
        help="HTTP API port (default: 9001)"
    )

    args = parser.parse_args()

    # Validate at least one module is enabled
    if args.no_explorer and args.no_documenter:
        print("Error: At least one module must be enabled")
        sys.exit(1)

    print(f"Starting Unified Orchestrator...")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Pool URL: {args.pool_url}")
    print(f"  Backends: {args.backends}")
    print(f"  Explorer: {'disabled' if args.no_explorer else 'enabled'}")
    print(f"  Documenter: {'disabled' if args.no_documenter else 'enabled'}")
    print(f"  API: http://{args.api_host}:{args.api_port}")
    print()

    # Create orchestrator
    orchestrator = Orchestrator(
        data_dir=args.data_dir,
        pool_base_url=args.pool_url,
        backends=args.backends,
    )

    # Register modules
    if not args.no_explorer:
        from explorer.src.adapter import ExplorerAdapter
        orchestrator.register_module(ExplorerAdapter())
        print("  Registered: Explorer module")

    if not args.no_documenter:
        from documenter.adapter import DocumenterAdapter
        orchestrator.register_module(DocumenterAdapter())
        print("  Registered: Documenter module")

    print()

    # Run with HTTP API
    await run_with_api(
        orchestrator=orchestrator,
        host=args.api_host,
        port=args.api_port,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
