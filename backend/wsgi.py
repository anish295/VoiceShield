#!/usr/bin/env python3
"""
WSGI entry point for VoiceShield Flask Application
Used for production deployment with Gunicorn
"""

from app import app, socketio

if __name__ == "__main__":
    socketio.run(app, allow_unsafe_werkzeug=True)
