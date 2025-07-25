#!/usr/bin/env python3
"""
VoiceShield - Real-Time Multimodal Emotion Detection System
Main entry point for the application.
"""

import sys
import os
import argparse
import signal
import subprocess
from pathlib import Path


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down VoiceShield...")
    sys.exit(0)


def main():
    """Main function to run VoiceShield system."""
    parser = argparse.ArgumentParser(description="VoiceShield - Real-Time Multimodal Emotion Detection (Flask Interface)")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode (not implemented - use Flask interface)")

    args = parser.parse_args()

    # Set up signal handler for graceful shutdown
    try:
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    try:
        if args.headless:
            print("❌ Headless mode not available in this version.")
            print("🛡️ Please use the Flask interface instead.")
            print("🌐 Starting Flask interface...")

        print("🛡️ VoiceShield - Flask Interface")
        print("=" * 50)
        print("📹 Real face detection with OpenCV")
        print("🎭 Actual emotion analysis with DeepFace")
        print("🎤 Voice emotion detection")
        print("✅ Modern web interface")
        print("🌐 Access at: http://localhost:5001")
        print("=" * 50)

        # Run Flask app directly
        subprocess.run([sys.executable, "working_flask_app.py"], check=True)

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        print("VoiceShield shutdown complete")


if __name__ == "__main__":
    main()
