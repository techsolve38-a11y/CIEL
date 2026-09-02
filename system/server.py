"""
CIEL Server — access CIEL from your phone or other devices on your home network
====================================================================================

Wraps the Orchestrator in a small web server so any device on the SAME
WiFi network as this computer can talk to CIEL — no cloud, no hosting
fees, no exposing anything to the public internet.

Run this, then find this computer's local IP address (see instructions
below) and open http://THAT-IP:8000 in your phone's browser, as long as
your phone is on the same WiFi network as this computer.

For access AWAY from home (not on the same WiFi) without paying a cloud
provider or opening risky router ports, the recommended next step is
Tailscale (https://tailscale.com) — free for personal use, creates a
private network between only YOUR OWN devices, nothing exposed publicly.
That's a separate, later step — this file only handles local network access.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ciel.orchestrator.core import Orchestrator

app = FastAPI(title="CIEL")
orchestrator = Orchestrator()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/", response_class=HTMLResponse)
def home():
    """A minimal, dependency-free chat page — no separate frontend build
    needed, just plain HTML+JS served directly, so a phone browser can
    talk to CIEL immediately."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>CIEL</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 20px auto; padding: 0 15px; }
        #chat { border: 1px solid #ccc; border-radius: 8px; padding: 10px; height: 60vh; overflow-y: auto; margin-bottom: 10px; }
        .msg { margin: 8px 0; padding: 8px 12px; border-radius: 8px; }
        .user { background: #e3f2fd; text-align: right; }
        .ciel { background: #f0f0f0; }
        input { width: 75%; padding: 10px; }
        button { width: 20%; padding: 10px; }
    </style>
    </head>
    <body>
        <h2>CIEL</h2>
        <div id="chat"></div>
        <input id="input" placeholder="Type a message..." onkeydown="if(event.key==='Enter')send()">
        <button onclick="send()">Send</button>
        <script>
            async function send() {
                const input = document.getElementById('input');
                const chat = document.getElementById('chat');
                const text = input.value.trim();
                if (!text) return;
                chat.innerHTML += `<div class="msg user">${text}</div>`;
                input.value = '';
                chat.scrollTop = chat.scrollHeight;
                const res = await fetch('/chat', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                const data = await res.json();
                chat.innerHTML += `<div class="msg ciel">${data.response}</div>`;
                chat.scrollTop = chat.scrollHeight;
            }
        </script>
    </body>
    </html>
    """


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response_text = orchestrator.process(req.message)
    return ChatResponse(response=response_text)


@app.get("/status")
def status():
    return {
        "constitution_version": orchestrator.constitution.version,
        "reasoning_engine_active": orchestrator._client is not None,
        "web_search_enabled": orchestrator.enable_web_search,
    }


if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" is what makes this reachable from OTHER devices on
    # your network, not just this computer — "127.0.0.1" (the default)
    # would only ever be reachable from this machine itself.
    uvicorn.run(app, host="0.0.0.0", port=8000)