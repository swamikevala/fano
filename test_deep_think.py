"""
Test script to debug Gemini Deep Think mode enabling.
Run this standalone to see what's happening with the UI.
"""

import asyncio
import sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from browser.gemini import GeminiInterface


async def test_deep_think():
    print("\n" + "="*60)
    print("GEMINI DEEP THINK TEST")
    print("="*60)

    gemini = GeminiInterface()

    print("\n[1] Connecting to Gemini...")
    try:
        await gemini.connect()
        print("    OK - Connected")
    except Exception as e:
        print(f"    FAIL - Connection failed: {e}")
        return

    print("\n[2] Starting new chat...")
    await gemini.start_new_chat()
    print("    OK - New chat started")

    print("\n[3] Attempting to enable Deep Think...")
    print("    (Watch the browser window)")

    # Add extra logging
    success = await gemini.enable_deep_think()

    if success:
        print("\n    OK - Deep Think enabled successfully!")
        print(f"    gemini.deep_think_enabled = {gemini.deep_think_enabled}")
    else:
        print("\n    FAIL - Deep Think FAILED to enable")
        print("    Check the debug screenshot at:")
        print("    data/chat_logs/gemini/deep_think_debug.png")

    print("\n[4] Testing with a simple message...")
    if gemini.deep_think_enabled:
        print("    Sending test message in Deep Think mode...")
        print("    (This may take several minutes)")
    else:
        print("    Sending test message in standard mode...")

    try:
        response = await gemini.send_message(
            "What is 2+2? Reply in one word.",
            use_deep_think=True
        )
        print(f"\n    Response ({len(response)} chars): {response[:200]}...")
    except Exception as e:
        print(f"\n    FAIL - Message failed: {e}")

    print("\n[5] Keeping browser open for inspection...")
    print("    Press Ctrl+C to close")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\nClosing...")
        await gemini.close()


if __name__ == "__main__":
    asyncio.run(test_deep_think())
