"""
VoiceDoc - Voice AI Health Navigator for Rural India
FastAPI server acting as a Custom LLM endpoint for Vapi.

Flow:
  User speaks → Vapi transcribes → POST /chat (OpenAI-compatible)
  → Qdrant RAG → Claude or Gemini → JSON response → Vapi speaks it back

Set LLM_PROVIDER=claude or LLM_PROVIDER=gemini in your .env
"""

import os
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import httpx

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicedoc")

app = FastAPI(title="VoiceDoc API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (initialized at startup) ─────────────────────────────────────────

qdrant: QdrantClient = None
embedder: SentenceTransformer = None
llm_provider: str = None          # "claude" or "gemini"

# Provider-specific clients
claude_client = None
gemini_model = None

COLLECTION_NAME = "voicedoc_medical"


@app.on_event("startup")
async def startup():
    global qdrant, embedder, llm_provider, claude_client, gemini_model

    logger.info("Starting VoiceDoc server...")

    # Qdrant
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    logger.info("Qdrant connected.")

    # Embeddings (free, runs locally — no API key needed)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded.")

    # LLM Provider
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if llm_provider == "claude":
        import anthropic
        claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info("LLM: Claude (Anthropic) ready.")

    elif llm_provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        gemini_model = genai.GenerativeModel(model_name)
        logger.info(f"LLM: Gemini ({model_name}) ready.")

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{llm_provider}'. Use 'claude' or 'gemini'.")

    logger.info("VoiceDoc is ready!")


# ── Helpers ───────────────────────────────────────────────────────────────────


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Embed query and fetch top-k relevant medical passages from Qdrant."""
    vector = embedder.encode(query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    if not results:
        return ""
    passages = [hit.payload["text"] for hit in results]
    return "\n\n---\n\n".join(passages)


def build_system_prompt(context: str) -> str:
    return f"""You are AarogyaVoice, a warm and trusted AI health guide helping people in rural India over a phone call. You speak like a knowledgeable, caring friend — someone who genuinely wants to help, not just give a quick answer and hang up. Be conversational, human, and thorough.

MEDICAL KNOWLEDGE (always use this first to answer):
{context}

═══ EMERGENCY PROTOCOL — HIGHEST PRIORITY ═══
If the caller mentions ANY of these: chest pain, difficulty breathing, unconsciousness, face drooping or arm weakness or slurred speech (stroke), severe bleeding, snakebite, poisoning, or pregnancy emergency — IMMEDIATELY respond:
"This sounds very serious. Please call 108 right now — it is the free government ambulance, available 24 hours. Do not wait, call immediately."
Then give ONE simple first-aid tip if helpful. Do not say anything else until they confirm they will call.

═══ HOW TO HAVE A GOOD CONVERSATION ═══
1. Acknowledge what they said warmly — show you are listening and you care.
2. Ask follow-up questions to understand better — ask about other symptoms, how long, severity, age of the patient, any medicines already taken. Ask ONE follow-up question at a time.
3. Once you have enough context, explain:
   - What this could be (possible causes, explained simply — "this might be because...")
   - What they can do at home right now (practical steps — hydration, rest, cold compress, light food, etc.)
   - When they must go to a doctor or PHC
4. Keep the tone like a helpful elder sibling or a trusted friend — warm, clear, never scary or dismissive.
5. Speak in natural sentences only — no bullet points, no lists, no formatting. This is a voice call.
6. Keep each response focused — 4 to 6 sentences is ideal. Long enough to be helpful, short enough to be easy to follow on a call.

═══ MEDICAL GUIDANCE ═══
- Use the medical knowledge above as your primary source.
- Never diagnose definitively. Say "this could be..." or "this might be because of..."
- Always give at least one practical home remedy or action they can take right now.
- For any symptom lasting more than 3 days, recommend visiting the nearest PHC.
- For children under 5 with fever — recommend PHC visit the same day, do not wait.
- Mention free government services when relevant:
  108 is the free emergency ambulance available 24 hours. PHC or Primary Health Centre gives free outpatient care nearby. Ayushman Bharat or PM-JAY covers free hospital treatment up to 5 lakh rupees. ASHA workers do free home health visits in the village.

═══ NEVER DO THESE ═══
- Never say "As an AI" or "I am just a bot"
- Never give specific medicine dosages
- Never use complex medical words without immediately explaining them in simple language
- Never dismiss any symptom as unimportant — always take it seriously
- Never give a one-line answer and stop — always be helpful and complete
"""


def openai_response(reply_text: str) -> dict:
    """Wrap reply text in OpenAI-compatible response shape."""
    return {
        "id": "voicedoc-chat",
        "object": "chat.completion",
        "model": "voicedoc-rag",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── LLM Calls ─────────────────────────────────────────────────────────────────


def call_claude(system_prompt: str, messages: list) -> str:
    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text


def call_gemini(system_prompt: str, messages: list) -> str:
    # Build a single prompt string from conversation history
    history_text = ""
    for m in messages:
        role = "User" if m["role"] == "user" else "VoiceDoc"
        history_text += f"{role}: {m.get('content', '')}\n"

    full_prompt = f"{system_prompt}\n\nConversation:\n{history_text}VoiceDoc:"

    response = gemini_model.generate_content(
        full_prompt,
        generation_config={
            "max_output_tokens": 400,
            "temperature": 0.4,
        },
    )
    # Gemini 2.5 Pro may return thinking parts — extract only text parts
    try:
        return response.text.strip()
    except Exception:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                return part.text.strip()
        return "I'm sorry, I couldn't process that. Please try again."


async def stream_claude(system_prompt: str, messages: list):
    """Yield Claude response in OpenAI SSE format."""
    with claude_client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text_chunk in stream.text_stream:
            chunk = {
                "id": "voicedoc-stream",
                "object": "chat.completion.chunk",
                "model": "voicedoc-rag",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text_chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    done_chunk = {
        "id": "voicedoc-stream",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_gemini(system_prompt: str, messages: list):
    """Yield Gemini response in OpenAI SSE format."""
    history_text = ""
    for m in messages:
        role = "User" if m["role"] == "user" else "VoiceDoc"
        history_text += f"{role}: {m.get('content', '')}\n"

    full_prompt = f"{system_prompt}\n\nConversation:\n{history_text}VoiceDoc:"

    response = gemini_model.generate_content(
        full_prompt,
        generation_config={"max_output_tokens": 400, "temperature": 0.4},
        stream=True,
    )

    for chunk in response:
        try:
            text = chunk.text if chunk.text else ""
        except Exception:
            continue  # skip thinking-only chunks (Gemini 2.5 Pro)
        if text:
            data = {
                "id": "voicedoc-stream",
                "object": "chat.completion.chunk",
                "model": "voicedoc-rag",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

    done_chunk = {
        "id": "voicedoc-stream",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "VoiceDoc", "llm": llm_provider}


@app.post("/chat/completions")
@app.post("/chat")
async def custom_llm_endpoint(request: Request):
    """
    OpenAI-compatible chat completion endpoint for Vapi Custom LLM.
    Vapi sends conversation here; we do Qdrant RAG and return LLM response.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages: list = body.get("messages", [])
    stream: bool = body.get("stream", False)

    # Debug: write raw Vapi request to file so we can inspect it
    try:
        with open("/tmp/vapi_last_request.json", "w") as f:
            json.dump(body, f, indent=2)
    except Exception:
        pass
    logger.info(f"Raw body keys: {list(body.keys())}")

    def get_content(m: dict) -> str:
        """Vapi uses 'message' key; OpenAI uses 'content'. Handle both."""
        val = m.get("content") or m.get("message") or ""
        if isinstance(val, list):
            # OpenAI content array: [{"type": "text", "text": "..."}]
            return " ".join(p.get("text", "") for p in val if isinstance(p, dict))
        return str(val)

    def normalize_role(role: str) -> str:
        """Vapi uses 'bot'; OpenAI uses 'assistant'."""
        return "assistant" if role == "bot" else role

    # Extract latest user message for RAG
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = get_content(msg)
            break

    logger.info(f"[{llm_provider.upper()}] User: {user_query[:100]}")

    # Retrieve relevant medical context from Qdrant
    context = retrieve_context(user_query) if user_query else ""
    logger.info(f"Qdrant returned {len(context)} chars of context")

    system_prompt = build_system_prompt(context)

    filtered_messages = [
        {"role": normalize_role(m["role"]), "content": get_content(m)}
        for m in messages
        if normalize_role(m.get("role", "")) in ("user", "assistant")
    ]

    # ── Streaming ──────────────────────────────────────────────────────────
    if stream:
        if llm_provider == "claude":
            return StreamingResponse(
                stream_claude(system_prompt, filtered_messages),
                media_type="text/event-stream",
            )
        else:
            return StreamingResponse(
                stream_gemini(system_prompt, filtered_messages),
                media_type="text/event-stream",
            )

    # ── Non-streaming ──────────────────────────────────────────────────────
    if llm_provider == "claude":
        reply_text = call_claude(system_prompt, filtered_messages)
    else:
        reply_text = call_gemini(system_prompt, filtered_messages)

    logger.info(f"Response: {reply_text[:100]}")
    return JSONResponse(content=openai_response(reply_text))


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    """Handles Vapi server-side events for logging and analytics."""
    body = await request.json()
    event_type = body.get("type", "unknown")
    logger.info(f"Vapi event: {event_type}")

    if event_type == "call-ended":
        call_id = body.get("call", {}).get("id")
        duration = body.get("call", {}).get("duration", 0)
        logger.info(f"Call {call_id} ended after {duration}s")

    return {"received": True}
