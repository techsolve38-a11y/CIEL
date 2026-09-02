"""
Reasoning Engines — swappable backends behind one interface
==================================================================

CIEL's orchestrator needs SOME reasoning engine to think with. This
module defines that as a clean interface with two implementations:

  OllamaEngine  — free, local, runs on your own machine, zero cost ever.
                  No tool-use support here (kept deliberately simple and
                  robust rather than fragile) — instead, useful skill
                  results get computed proactively and included directly
                  in the context, since that's free too and doesn't
                  depend on a model's tool-calling reliability.

  ClaudeEngine  — paid, via the Anthropic API, with full native tool use
                  (the skill-calling loop, web search). Available for
                  when there's budget for it — nothing else in CIEL
                  needs to change to switch back.

Orchestrator picks one at startup. Swapping later is a one-line change,
which is the entire point of building this as an interface rather than
hardcoding either choice.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod


class ReasoningEngine(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """Given a system prompt (context) and a user message, return
        CIEL's response as plain text."""
        raise NotImplementedError


class OllamaEngine(ReasoningEngine):
    """Free, local reasoning via Ollama (https://ollama.com). Requires
    Ollama installed and running on this machine (`ollama serve`, usually
    started automatically), with at least one model pulled (e.g.
    `ollama pull llama3.2`)."""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.host = host
        self.timeout = timeout

    def generate(self, system_prompt: str, user_message: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result["message"]["content"]
        except urllib.error.URLError as e:
            return (
                f"[CIEL's local reasoning engine (Ollama) couldn't be reached: {e}. "
                f"Is Ollama installed and running? Try 'ollama serve' in a terminal, "
                f"and confirm a model is pulled with 'ollama pull {self.model}'.]"
            )
        except (KeyError, json.JSONDecodeError) as e:
            return f"[CIEL got an unexpected response from Ollama: {e}]"


class CIEL0Engine(ReasoningEngine):
    """CIEL-0 itself — the from-scratch model — as the reasoning engine.
    This is the ONLY genuinely independent option: no external model, no
    API, no other company's weights. Runs entirely from a checkpoint you
    trained yourself.

    HONEST LIMITATION, stated here rather than hidden: CIEL-0's context
    window (block_size) is far smaller than the rich context the
    orchestrator builds (constitution + memory + skills + tools), and its
    current training (Shakespeare, small scale) gives it no ability to
    use injected facts to answer questions — this was proven directly in
    testing. This engine is honest about both: it truncates context to
    fit what CIEL-0 can actually see, and returns exactly what CIEL-0
    generates, good or bad, rather than dressing it up."""

    def __init__(self, checkpoint_path: str, temperature: float = 0.7, max_new_tokens: int = 200):
        import torch
        self.torch = torch
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._load_model()

    def _load_model(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.n_heads = n_heads; self.d_head = d_model // n_heads
                self.W_q = nn.Linear(d_model, d_model); self.W_k = nn.Linear(d_model, d_model)
                self.W_v = nn.Linear(d_model, d_model); self.W_out = nn.Linear(d_model, d_model)
            def forward(self, x):
                b, t, d = x.shape
                Q = self.W_q(x).view(b,t,self.n_heads,self.d_head).transpose(1,2)
                K = self.W_k(x).view(b,t,self.n_heads,self.d_head).transpose(1,2)
                V = self.W_v(x).view(b,t,self.n_heads,self.d_head).transpose(1,2)
                out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                out = out.transpose(1,2).contiguous().view(b,t,d)
                return self.W_out(out)

        class FeedForward(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d_model,d_ff), nn.GELU(), nn.Linear(d_ff,d_model))
            def forward(self, x): return self.net(x)

        class TransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff):
                super().__init__()
                self.attention = MultiHeadAttention(d_model, n_heads)
                self.feedforward = FeedForward(d_model, d_ff)
                self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
            def forward(self, x):
                x = x + self.attention(self.norm1(x))
                x = x + self.feedforward(self.norm2(x))
                return x

        class TinyGPT(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, block_size):
                super().__init__()
                self.block_size = block_size
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.position_embedding = nn.Embedding(block_size, d_model)
                self.blocks = nn.ModuleList([TransformerBlock(d_model,n_heads,d_ff) for _ in range(n_layers)])
                self.final_norm = nn.LayerNorm(d_model)
                self.output_head = nn.Linear(d_model, vocab_size)
            def forward(self, idx):
                b, t = idx.shape
                x = self.token_embedding(idx) + self.position_embedding(torch.arange(t, device=idx.device))
                for block in self.blocks: x = block(x)
                x = self.final_norm(x)
                return self.output_head(x)
            @torch.no_grad()
            def generate(self, idx, max_new_tokens, temperature=0.7):
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -self.block_size:]
                    logits = self(idx_cond)[:, -1, :] / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, next_id], dim=1)
                return idx

        ckpt = torch.load(self.checkpoint_path, weights_only=True, map_location="cpu")

        if "vocab" in ckpt:
            self.chars = ckpt["vocab"]
            d_model, n_heads, n_layers, d_ff, block_size = (
                ckpt["d_model"], ckpt["n_heads"], ckpt["n_layers"], ckpt["d_ff"], ckpt["block_size"]
            )
        else:
            # BACKWARD COMPAT: checkpoints from 10_tiny_gpt.py (before this
            # session added vocab/architecture info to saved checkpoints)
            # only contain model_state/optimizer_state/total_steps. This is
            # the EXACT 65-character tinyshakespeare.txt vocabulary and the
            # architecture that script hardcoded — confirmed directly
            # earlier in this project, not guessed.
            self.chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            d_model, n_heads, n_layers, d_ff, block_size = 128, 4, 4, 512, 64

        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.block_size = block_size

        self.model = TinyGPT(
            vocab_size=len(self.chars), d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, block_size=block_size,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _encode(self, s: str):
        # Any character CIEL-0 never saw during training (e.g. from a
        # constitution/memory format it wasn't trained on) gets skipped
        # rather than crashing — an honest, visible limitation of a
        # character-level model with a fixed, small vocabulary.
        return [self.char_to_id[c] for c in s if c in self.char_to_id]

    def _decode(self, ids) -> str:
        return "".join(self.id_to_char[i] for i in ids)

    def generate(self, system_prompt: str, user_message: str) -> str:
        full_prompt = f"{system_prompt}\nUser: {user_message}\nCIEL:"
        # HONEST TRUNCATION: CIEL-0 can only see its last `block_size`
        # characters. Rather than silently failing on long context, take
        # exactly what it CAN see — this was proven necessary directly
        # in testing (see the earlier live capability test).
        truncated = full_prompt[-self.block_size:]

        ids = self._encode(truncated)
        idx = self.torch.tensor([ids], dtype=self.torch.long)
        out = self.model.generate(idx, self.max_new_tokens, temperature=self.temperature)
        full_output = self._decode(out[0].tolist())
        return full_output[len(truncated):]


class ClaudeEngine(ReasoningEngine):
    """Paid, via the Anthropic API — full native tool use (executable
    skills, web search). This is the ORIGINAL reasoning path, kept intact
    and available whenever there's budget for it again."""

    def __init__(self, client, model: str, tools: list, memory_store, max_tool_rounds: int = 3):
        self.client = client
        self.model = model
        self.tools = tools
        self.memory_store = memory_store
        self.max_tool_rounds = max_tool_rounds

    def generate(self, system_prompt: str, user_message: str) -> str:
        # Local import to avoid a hard dependency on `anthropic` and the
        # skill-handler module for people running Ollama-only.
        import anthropic
        from ciel.skills.handlers import execute_skill_by_tool_name

        messages = [{"role": "user", "content": user_message}]
        text = "[CIEL used the maximum number of tool calls for this turn without reaching a final answer.]"

        try:
            for _ in range(self.max_tool_rounds):
                response = self.client.messages.create(
                    model=self.model, max_tokens=1500,
                    system=system_prompt, messages=messages, tools=self.tools,
                )
                custom_tool_calls = [
                    b for b in response.content if b.type == "tool_use" and b.name != "web_search"
                ]
                if not custom_tool_calls:
                    text = "".join(b.text for b in response.content if b.type == "text")
                    break
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for call in custom_tool_calls:
                    try:
                        result_text = execute_skill_by_tool_name(call.name, call.input, self.memory_store)
                    except Exception as e:
                        result_text = f"Skill execution failed: {e}"
                    tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": result_text})
                messages.append({"role": "user", "content": tool_results})
        except anthropic.BadRequestError as e:
            try:
                detail = e.response.json().get("error", {}).get("message", "")
            except Exception:
                detail = str(e)
            if "credit balance" in detail.lower():
                return ("[CIEL can't reach its reasoning engine: your Anthropic account has "
                        "insufficient credits. Add credits at console.anthropic.com -> "
                        "Plans & Billing, then try again.]")
            return f"[CIEL's reasoning engine rejected this request: {detail}]"
        except anthropic.RateLimitError:
            return "[CIEL's reasoning engine is rate-limited right now. Wait a moment and try again.]"
        except anthropic.APIConnectionError:
            return "[CIEL couldn't reach the Anthropic API — check your internet connection.]"
        except anthropic.APIError as e:
            return f"[CIEL's reasoning engine returned an unexpected error: {e}]"

        return text