"""
VoiceDoc - Vapi Assistant Setup
Run this after your server is live (ngrok URL ready) to create the Vapi assistant.
Usage: python setup_vapi.py --url https://your-ngrok-url.ngrok.io
"""

import os
import sys
import argparse
import httpx
from dotenv import load_dotenv

load_dotenv()

VAPI_API_URL = "https://api.vapi.ai"


def create_voicedoc_assistant(server_url: str) -> dict:
    """Create the VoiceDoc assistant on Vapi."""
    vapi_key = os.getenv("VAPI_API_KEY")
    if not vapi_key:
        print("ERROR: VAPI_API_KEY not set in .env")
        sys.exit(1)

    assistant_config = {
        "name": "VoiceDoc Health Navigator",
        "model": {
            "provider": "custom-llm",
            "url": f"{server_url}/chat",
            # Tells Vapi to send conversation to our RAG endpoint
            "model": "voicedoc-rag",
        },
        "voice": {
            "provider": "playht",
            "voiceId": "jennifer",  # Clear, warm female voice
        },
        "firstMessage": (
            "Hello! I'm VoiceDoc, your health guide. "
            "Please describe what you or your family member is feeling, "
            "and I'll help you understand what to do. How can I help you today?"
        ),
        "systemPrompt": (
            "You are VoiceDoc, a compassionate AI health navigator for rural India. "
            "Keep all responses under 3 sentences and very simple. "
            "Always recommend calling 108 for emergencies."
        ),
        "endCallPhrases": ["goodbye", "thank you goodbye", "ok bye", "thank you doctor"],
        "serverUrl": f"{server_url}/vapi/webhook",
        "serverUrlSecret": "voicedoc-secret-2026",
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-2",
            "language": "en-IN",  # Indian English
        },
        "silenceTimeoutSeconds": 10,
        "maxDurationSeconds": 600,  # 10 minute max call
        "backgroundSound": "off",
        "backchannelingEnabled": False,
        "analysisPlan": {
            "summaryPrompt": "Summarize what health issue was discussed and what guidance was given.",
        },
    }

    headers = {
        "Authorization": f"Bearer {vapi_key}",
        "Content-Type": "application/json",
    }

    print(f"Creating VoiceDoc assistant pointing to: {server_url}/chat")

    response = httpx.post(
        f"{VAPI_API_URL}/assistant",
        json=assistant_config,
        headers=headers,
        timeout=30,
    )

    if response.status_code not in (200, 201):
        print(f"ERROR creating assistant: {response.status_code}")
        print(response.text)
        sys.exit(1)

    return response.json()


def create_phone_number(assistant_id: str) -> dict:
    """Buy a phone number and link it to the assistant (optional)."""
    vapi_key = os.getenv("VAPI_API_KEY")
    headers = {
        "Authorization": f"Bearer {vapi_key}",
        "Content-Type": "application/json",
    }

    phone_config = {
        "provider": "twilio",
        "assistantId": assistant_id,
        "name": "VoiceDoc Helpline",
    }

    response = httpx.post(
        f"{VAPI_API_URL}/phone-number",
        json=phone_config,
        headers=headers,
        timeout=30,
    )

    if response.status_code not in (200, 201):
        print(f"  (Phone number creation skipped: {response.status_code})")
        return {}

    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Set up VoiceDoc Vapi assistant")
    parser.add_argument(
        "--url",
        required=True,
        help="Your public server URL (e.g. https://abc123.ngrok.io)",
    )
    args = parser.parse_args()

    server_url = args.url.rstrip("/")

    print("\nVoiceDoc - Vapi Setup")
    print("=" * 40)

    assistant = create_voicedoc_assistant(server_url)
    assistant_id = assistant.get("id")

    print(f"\n✅ Assistant created!")
    print(f"   ID:   {assistant_id}")
    print(f"   Name: {assistant.get('name')}")
    print(f"   LLM:  {server_url}/chat")

    print("\n📋 Next steps:")
    print("   1. Go to https://dashboard.vapi.ai")
    print(f"   2. Find assistant: '{assistant.get('name')}'")
    print("   3. Click 'Test' to try a voice call in your browser")
    print("   4. Or buy a phone number in Vapi dashboard and assign this assistant")
    print(f"\n   Assistant ID (save this): {assistant_id}")

    # Save assistant ID to .env style file for reference
    with open(".voicedoc_state", "w") as f:
        f.write(f"ASSISTANT_ID={assistant_id}\n")
        f.write(f"SERVER_URL={server_url}\n")
    print("\n   Saved to .voicedoc_state")


if __name__ == "__main__":
    main()
