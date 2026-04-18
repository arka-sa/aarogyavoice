#!/bin/bash
# VoiceDoc Quick Start Script — starts server + ngrok together

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

# Check ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found! Install it from https://ngrok.com/download"
    exit 1
fi

# Install dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q
echo "      Done."

# Run ingestion
echo ""
echo "[2/3] Ingesting medical knowledge into Qdrant..."
python ingest.py

# Start FastAPI server in background
echo ""
echo "[3/3] Starting FastAPI server on port 8000..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "      Waiting for server to be ready..."
for i in $(seq 1 15); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "      Server is up!"
        break
    fi
    sleep 1
done

# Start ngrok and grab the public URL
echo ""
echo "      Starting ngrok tunnel..."
ngrok http 8000 --log=stdout --log-format=json > /tmp/ngrok_voicedoc.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to establish tunnel
sleep 3
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnels = data.get('tunnels', [])
    for t in tunnels:
        if t.get('proto') == 'https':
            print(t['public_url'])
            break
except:
    pass
")

echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Server : http://localhost:8000                             │"
echo "│  Health : http://localhost:8000/health                      │"
if [ -n "$NGROK_URL" ]; then
echo "│  Ngrok  : $NGROK_URL          │"
echo "│                                                             │"
echo "│  Update Vapi Custom LLM URL to:                            │"
echo "│  $NGROK_URL/chat/completions  │"
else
echo "│  Ngrok  : check http://localhost:4040 for your URL          │"
fi
echo "│                                                             │"
echo "│  Press Ctrl+C to stop everything                           │"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

# Cleanup both processes on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SERVER_PID 2>/dev/null
    kill $NGROK_PID 2>/dev/null
    exit 0
}
trap cleanup INT TERM

# Keep script alive
wait
