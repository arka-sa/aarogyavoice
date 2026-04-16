#!/bin/bash
# VoiceDoc Quick Start Script

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   VoiceDoc - Voice Health Navigator  ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Run: cp .env.example .env"
    echo "   Then fill in your API keys."
    exit 1
fi

# Install dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q
echo "      Done."

# Run ingestion if Qdrant collection not mentioned in state
echo ""
echo "[2/3] Ingesting medical knowledge into Qdrant..."
python ingest.py

# Start the server
echo ""
echo "[3/3] Starting FastAPI server on port 8000..."
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  Server: http://localhost:8000                      │"
echo "│  Health: http://localhost:8000/health               │"
echo "│                                                     │"
echo "│  Next: run ngrok in another terminal:               │"
echo "│    ngrok http 8000                                  │"
echo "│  Then run:                                          │"
echo "│    python setup_vapi.py --url https://xyz.ngrok.io  │"
echo "└─────────────────────────────────────────────────────┘"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
