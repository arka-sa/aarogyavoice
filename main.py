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
    return f"""You are VoiceDoc, a compassionate AI health navigator helping people in rural India get reliable health guidance over a voice call.

MEDICAL KNOWLEDGE (use this to answer):
{context}

RULES:
1. Keep responses SHORT and CLEAR — this is a voice call, not a text message. Max 3-4 sentences.
2. Use simple, everyday language. No complex medical terms.
3. ALWAYS recommend seeing a doctor or calling 108 for serious or unclear symptoms.
4. For ANY life-threatening emergency (chest pain, stroke, difficulty breathing, unconscious person, snake bite), immediately say "Call 108 right now — this is an emergency."
5. Be warm, calm, and reassuring.
6. Do not make definitive diagnoses. Guide people toward care.
7. Free government services are available — mention them when relevant (PHC, Ayushman Bharat, JSY).
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
            "max_output_tokens": 200,
            "temperature": 0.4,
        },
    )
    return response.text.strip()


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
    """Yield Gemini response in OpenAI SSE format (simulated streaming)."""
    # Gemini streaming via generate_content(stream=True)
    history_text = ""
    for m in messages:
        role = "User" if m["role"] == "user" else "VoiceDoc"
        history_text += f"{role}: {m.get('content', '')}\n"

    full_prompt = f"{system_prompt}\n\nConversation:\n{history_text}VoiceDoc:"

    response = gemini_model.generate_content(
        full_prompt,
        generation_config={"max_output_tokens": 200, "temperature": 0.4},
        stream=True,
    )

    for chunk in response:
        text = chunk.text if chunk.text else ""
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

    # Extract latest user message for RAG
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            user_query = content if isinstance(content, str) else str(content)
            break

    logger.info(f"[{llm_provider.upper()}] User: {user_query[:100]}")

    # Retrieve relevant medical context from Qdrant
    context = retrieve_context(user_query) if user_query else ""
    logger.info(f"Qdrant returned {len(context)} chars of context")

    system_prompt = build_system_prompt(context)

    filtered_messages = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in messages
        if m.get("role") in ("user", "assistant")
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
