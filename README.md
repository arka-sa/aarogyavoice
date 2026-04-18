<div align="center">

# 🩺 AarogyaVoice

### *Voice AI Health Navigator for Rural India*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Vapi](https://img.shields.io/badge/Vapi-Voice_AI-6C47FF?style=for-the-badge)](https://vapi.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF4081?style=for-the-badge)](https://qdrant.tech)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**A real-time voice health assistant that helps rural communities access medical guidance — in plain spoken language, over a simple phone call.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Configuration](#-configuration) • [API](#-api-reference)

---

</div>

## 🌟 What is AarogyaVoice?

AarogyaVoice is a voice-first AI health navigator built for people in rural India who may not have internet access, health literacy, or easy access to doctors. A caller simply phones a number, describes their symptoms in natural language, and receives warm, practical, conversational health guidance — like talking to a knowledgeable friend.

It combines **Vapi** (voice AI infrastructure), **Qdrant** (medical knowledge retrieval), and **Claude / Gemini** (language understanding) into a seamless voice pipeline.

```
Caller speaks  →  Vapi transcribes  →  AarogyaVoice RAG  →  LLM responds  →  Vapi speaks back
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ **Voice-First** | Works over a phone call — no app, no internet needed for the user |
| 🧠 **Medical RAG** | Retrieves relevant knowledge from a curated medical knowledge base before responding |
| 🚨 **Emergency Detection** | Instantly directs callers to 108 (free ambulance) for life-threatening symptoms |
| 💬 **Conversational** | Asks follow-up questions, explores causes, gives home remedies — not just "go to PHC" |
| 🔄 **Dual LLM Support** | Switch between Claude (Anthropic) and Gemini (Google) via a single env variable |
| 📡 **Streaming** | Full OpenAI-compatible streaming for low-latency voice responses |
| 🏥 **Government Schemes** | Proactively mentions 108, PHC, Ayushman Bharat, JSY, ASHA workers |
| 🌐 **Free for callers** | Designed around free government health infrastructure |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         VAPI PLATFORM                           │
│   Voice Call  →  Deepgram STT  →  Custom LLM Hook  →  11labs   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ POST /chat/completions
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AAROGYAVOICE SERVER                         │
│                                                                 │
│   FastAPI  →  SentenceTransformer  →  Qdrant (Vector Search)   │
│                                           │                     │
│                                    Medical Context              │
│                                           │                     │
│                              ┌────────────▼───────────┐        │
│                              │  Claude / Gemini LLM   │        │
│                              └────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

**Stack:**
- **Voice Layer** — [Vapi](https://vapi.ai) + Deepgram (STT) + ElevenLabs (TTS)
- **API Server** — FastAPI (Python), OpenAI-compatible `/chat/completions`
- **Embeddings** — `all-MiniLM-L6-v2` (runs locally, no API key needed)
- **Vector DB** — Qdrant Cloud (free tier works)
- **LLM** — Anthropic Claude or Google Gemini (switchable)
- **Tunnel** — ngrok (for local development)

---

## 🚀 Quick Start

### Prerequisites

- Python **3.12** or **3.13** (not 3.14 — pydantic-core doesn't support it yet)
- [ngrok](https://ngrok.com/download) installed
- A [Vapi](https://vapi.ai) account
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier)
- An [Anthropic](https://console.anthropic.com) or [Google AI Studio](https://aistudio.google.com) API key

### 1. Clone & set up environment

```bash
git clone https://github.com/your-username/aarogyavoice.git
cd aarogyavoice

python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# LLM — choose "claude" or "gemini"
LLM_PROVIDER=gemini

# Anthropic (if LLM_PROVIDER=claude)
ANTHROPIC_API_KEY=sk-ant-...

# Google (if LLM_PROVIDER=gemini)
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-pro
```

### 3. Start everything with one command

```bash
./start.sh
```

This will:
1. Install dependencies
2. Ingest the medical knowledge base into Qdrant
3. Start the FastAPI server on port 8000
4. Start an ngrok tunnel and **print the public URL**

### 4. Connect to Vapi

1. Open your [Vapi Dashboard](https://dashboard.vapi.ai)
2. Go to your assistant → **Model** tab
3. Set **Provider** to `Custom LLM`
4. Set **Custom LLM URL** to the ngrok URL printed by the script, e.g.:
   ```
   https://xxxx-xxxx.ngrok-free.app/chat/completions
   ```
5. Click **Publish** and hit **Talk** to test

---

## ⚙️ Configuration

### Switching LLM Providers

Change `LLM_PROVIDER` in `.env`:

```env
LLM_PROVIDER=claude   # Uses Anthropic Claude Sonnet
LLM_PROVIDER=gemini   # Uses Google Gemini 2.5 Pro (default)
```

No code changes needed — the server picks it up on startup.

### Adding Medical Knowledge

Edit `data/medical_kb.json` to add new entries:

```json
[
  {
    "id": "fever_001",
    "category": "fever",
    "text": "Fever above 38°C lasting more than 3 days in adults may indicate..."
  }
]
```

Then re-run ingestion:

```bash
python ingest.py
```

---

## 📡 API Reference

### `GET /health`
Returns server status and active LLM provider.

```json
{ "status": "ok", "service": "VoiceDoc", "llm": "gemini" }
```

### `POST /chat/completions`
OpenAI-compatible chat completion endpoint. This is what Vapi calls.

**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "I have had fever for 3 days" }
  ],
  "stream": true
}
```

**Response:** OpenAI-compatible JSON or SSE stream.

### `POST /vapi/webhook`
Receives Vapi server-side events (call started, call ended, etc.) for logging.

---

## 🩺 Example Conversation

> **User:** I've been having fever since the last 3 days.
>
> **AarogyaVoice:** I understand you've had fever for 3 days — that must be worrying. Can I ask, do you also have any body aches, headache, or are you feeling cold and shivering? Knowing this will help me understand what might be going on.
>
> **User:** Yes, body aches and feeling cold.
>
> **AarogyaVoice:** That combination — fever, body aches, and chills lasting 3 days — could be a sign of malaria, viral fever, or dengue, especially if there are mosquitoes in your area. Right now, please drink plenty of boiled water, rest as much as possible, and avoid cold food. But since it has been 3 days, please visit your nearest PHC today — they can do a quick blood test which is free, and get you the right medicine.

---

## 🤝 Government Health Resources Referenced

| Service | What it does |
|---|---|
| **108** | Free emergency ambulance, available 24 hours |
| **PHC** (Primary Health Centre) | Free outpatient consultation near you |
| **Ayushman Bharat / PM-JAY** | Free hospital treatment up to ₹5 lakh |
| **JSY** (Janani Suraksha Yojana) | Free support for pregnant women |
| **ASHA Worker** | Free home health visits in your village |

---

## 📁 Project Structure

```
aarogyavoice/
├── main.py              # FastAPI server — RAG pipeline + LLM routing
├── ingest.py            # One-time Qdrant knowledge base ingestion
├── start.sh             # Single script to start server + ngrok
├── setup_vapi.py        # Helper to configure Vapi assistant via API
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── data/
    └── medical_kb.json  # Curated medical knowledge base
```

---

## 📄 License

MIT © 2025 — Built with care for rural India 🇮🇳

---

<div align="center">

*"Healthcare is a right, not a privilege. AarogyaVoice is a small step toward making that real."*

</div>
