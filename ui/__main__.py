"""
halo-forge UI Entry Point

Run with: python -m ui
"""

import argparse
from ui.app import run


def main():
    """Main entry point for the UI."""
    parser = argparse.ArgumentParser(
        description='halo-forge Web UI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to listen on (default: 8080)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable hot reload for development'
    )
    
    args = parser.parse_args()
    
    print(f"""
╭──────────────────────────────────────────────────────────────╮
│                                                              │
│   HALO-FORGE Web UI                                          │
│                                                              │
│   Starting server at http://{args.host}:{args.port}              │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
""")
    
    run(host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
