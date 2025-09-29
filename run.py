#!/usr/bin/env python3
"""
Main application runner for Code Quality Assessment system
"""

import os
import sys
import argparse

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.app import app


def main():
    parser = argparse.ArgumentParser(description='Code Quality Assessment Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    print("=" * 60)
    print("🔍 Code Quality Assessment System")
    print("=" * 60)
    print(f"🌐 Server starting on http://{args.host}:{args.port}")
    print(f"🐛 Debug mode: {'ON' if args.debug else 'OFF'}")
    print("=" * 60)
    print("\nFeatures:")
    print("📸 Upload code screenshots for OCR analysis")
    print("✏️  Direct code text analysis")
    print("🎯 Real-time quality assessment")
    print("📚 Educational feedback and suggestions")
    print("🚀 RESTful API endpoints")
    print("=" * 60)
    print("\nAPI Endpoints:")
    print("GET  /                    - Web interface")
    print("GET  /api/health          - Health check")
    print("POST /api/analyze         - Analyze code image")
    print("POST /api/analyze-text    - Analyze code text")
    print("GET  /api/feedback/<score> - Get contextual feedback")
    print("=" * 60)

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()