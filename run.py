#!/usr/bin/env python3
"""
Main application runner for Code Quality Assessment system.
"""

from pathlib import Path
import sys
import argparse
import os


def _looks_like_project(path: Path) -> bool:
    """Return True if the expected repository files exist in the path."""
    required = ['src', 'requirements.txt', 'README.md']
    return all((path / item).exists() for item in required)


def find_project_root(start_dir: Path) -> Path:
    """Locate the repository root even when the script runs elsewhere."""
    start_dir = start_dir.resolve()
    candidates = []
    visited = set()

    for candidate in [start_dir, *start_dir.parents]:
        candidates.append(candidate)
        try:
            candidates.extend(child for child in candidate.iterdir() if child.is_dir())
        except (OSError, PermissionError):
            continue

    for candidate in candidates:
        if candidate in visited:
            continue
        visited.add(candidate)
        if _looks_like_project(candidate):
            return candidate

    raise RuntimeError(
        'Could not locate the project root. Start the script inside the repo or set PYTHONPATH manually.'
    )


def running_inside_notebook(argv=None) -> bool:
    """Detect if the code runs inside an IPython/Jupyter kernel."""
    if 'ipykernel' in sys.modules:
        return True

    argv = argv if argv is not None else sys.argv
    if not argv:
        return False

    launcher = Path(argv[0]).name
    has_connection_file = '-f' in argv or any(arg.startswith('--f') for arg in argv)
    return (
        launcher.startswith('ipykernel')
        or 'JPY_PARENT_PID' in os.environ
        or has_connection_file
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from api.app import app  # noqa: E402  (import after sys.path tweak)


def main():
    parser = argparse.ArgumentParser(description='Code Quality Assessment Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    parse_args_from = [] if running_inside_notebook() else None
    args = parser.parse_args(args=parse_args_from)

    print('=' * 60)
    print('🔍 Code Quality Assessment System')
    print('=' * 60)
    print(f'🌐 Server starting on http://{args.host}:{args.port}')
    print(f'🐛 Debug mode: {"ON" if args.debug else "OFF"}')
    print('=' * 60)
    print()
    print('Features:')
    print('📸 Upload code screenshots for OCR analysis')
    print('✏️  Direct code text analysis')
    print('🎯 Real-time quality assessment')
    print('📚 Educational feedback and suggestions')
    print('🚀 RESTful API endpoints')
    print('=' * 60)
    print()
    print('API Endpoints:')
    print('GET  /                    - Web interface')
    print('GET  /api/health          - Health check')
    print('POST /api/analyze         - Analyze code image')
    print('POST /api/analyze-text    - Analyze code text')
    print('GET  /api/feedback/<score> - Get contextual feedback')
    print('=' * 60)

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print()
        print('👋 Server stopped by user')
    except Exception as e:  # pragma: no cover - only triggered on failure
        print()
        print(f'❌ Server error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()

