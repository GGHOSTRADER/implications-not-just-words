"""LLM Registry (local models only: this release has no cloud backend) — reads role→model assignments from .env.

To swap a role's model, edit .env:
    LLM_BRIEFING_NLG=groq   # or qwen
    LLM_ALERT_NLG=qwen
    ...

Available backends: qwen (local, fast), qwen_think (same model, thinking mode, token-capped), groq (optional cloud)
Available roles:    briefing_nlg, alert_nlg, tactical, strategic, classifier, gym, recipe_judge, weekly_review
A role with no setting is local `qwen`.

recipe_judge (Ship 3, recipe_discovery.py): short/fast coherence-judging
calls, same shape as the existing forced-tool-call classifier pattern
(agent.py's _classify_confirmation/_classify_dont_know/
_classify_inventory_intent). Defaults to qwen, matching classifier's own
assignment — not exposed for casual backend-swapping since it's a
cheap/fast judgment call, not a role a non-technical operator needs to
retune. recipe_discovery.py uses the direct `llm_registry.qwen` instance
reference (bypassing get()) rather than get("recipe_judge"), matching
the exact pattern the 3 existing agent.py examples already use — this
role entry exists for documentation/consistency and as a one-line escape
hatch if backend flexibility is ever needed later.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import PrivateAttr

load_dotenv(Path(__file__).parent / ".env")   # optional: MODEL_NAME, MODEL_BASE_URL

log = logging.getLogger(__name__)

# ── Base model instances (infrastructure — don't change for role swaps) ───────

# Local Qwen via Ollama's NATIVE API (not the OpenAI-compat /v1 endpoint).
# ChatOllama lets us disable Qwen3 "thinking" with reasoning=False — without it,
# qwen3:8b spends ~8s/call generating a reasoning trace. With it: ~0.7s, clean output,
# tool calling intact. base_url must be the native host (no /v1 suffix).
_ollama_base = os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1").replace("/v1", "").rstrip("/")
qwen = ChatOllama(
    base_url=_ollama_base,
    model=os.getenv("MODEL_NAME", "qwen3:8b"),
    temperature=0.1,
    reasoning=False,
    # Keep the model resident so requests after an idle gap don't pay a cold reload
    # (Ollama's default unloads after 5 min). "-1s" = stay loaded; override via env.
    # Must include a unit suffix — Ollama rejects a bare number like "-1".
    keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "1h"),
)

# Same local model with thinking ON, for the agents that must choose and call tools correctly (2026-10-04).
# Measured on the tactical agent's real prompt: reasoning off picked a sensible tool 5 times in 8 (it invented the
# time instead of calling get_current_time); thinking on picked sensibly 8 in 8, at 2-10 s per model step instead of
# under 1 s. Classifiers and extraction stay on `qwen` (fast).
#
# SAFETY NET: thinking is unbounded by default, and one call in a test run thought for ~20 minutes (1214 s). So the
# thinking model is capped (THINK_MAX_TOKENS, about 70 s at this GPU's ~36 tokens/s). A capped call returns an EMPTY
# answer with done_reason="length" (measured), which an agent would show as a blank reply, so in that case the same
# request is answered by the fast model instead.
THINK_MAX_TOKENS = int(os.getenv("THINK_MAX_TOKENS", "2500"))


def _cut_off_with_nothing(msg) -> bool:
    """True for a thinking call stopped by the token cap before it produced any answer or tool call."""
    meta = getattr(msg, "response_metadata", None) or {}
    return (meta.get("done_reason") == "length"
            and not (getattr(msg, "content", "") or "").strip()
            and not getattr(msg, "tool_calls", None))


class ThinkingOllama(ChatOllama):
    """ChatOllama in thinking mode that falls back to a fast sibling when the token cap cuts it off empty."""

    _fast: "ChatOllama | None" = PrivateAttr(default=None)

    def invoke(self, input, config=None, **kwargs):          # bind_tools(...) forwards its tools in kwargs
        msg = super().invoke(input, config, **kwargs)
        if self._fast is not None and _cut_off_with_nothing(msg):
            log.warning(f"[llm] thinking call hit the {THINK_MAX_TOKENS}-token cap with no answer; "
                        f"answering with the fast model instead")
            return self._fast.invoke(input, config, **kwargs)
        return msg


qwen_think = ThinkingOllama(
    base_url=_ollama_base,
    model=os.getenv("MODEL_NAME", "qwen3:8b"),
    temperature=0.1,
    reasoning=True,
    num_predict=THINK_MAX_TOKENS,
    keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "1h"),
)
qwen_think._fast = qwen

_BACKENDS: dict = {"qwen": qwen, "qwen_think": qwen_think}

# ── Role → env var mapping ────────────────────────────────────────────────────

_ROLE_ENV: dict[str, str] = {
    "briefing_nlg": "LLM_BRIEFING_NLG",
    "alert_nlg":    "LLM_ALERT_NLG",
    "tactical":     "LLM_TACTICAL",
    "strategic":    "LLM_STRATEGIC",
    "classifier":   "LLM_CLASSIFIER",
    "gym":          "LLM_GYM",
    "recipe_judge": "LLM_RECIPE_JUDGE",
    # The weekly review is two small focused questions. Measured 2026-10-04 on 8 runs: fast mode made no invented
    # numbers and broke no format rule, in ~10 s; thinking mode was no better and took 30-1214 s. So it defaults to fast.
    "weekly_review": "LLM_WEEKLY_REVIEW",
}


def get(role: str):
    """Return the model assigned to *role* (reads from .env at startup)."""
    if role not in _ROLE_ENV:
        raise KeyError(f"Unknown LLM role: {role!r}. Available: {list(_ROLE_ENV)}")
    # Default is the LOCAL model (2026-10-04): the app runs on local models, so a role whose setting is missing
    # must not quietly start sending messages to a cloud API. Groq is used only where .env says so explicitly.
    backend_name = os.getenv(_ROLE_ENV[role], "qwen").strip().lower()
    if backend_name not in _BACKENDS:
        raise ValueError(f"Role {role!r} has unknown backend {backend_name!r}. Use: qwen, qwen_think (this release has no cloud backend)")
    return _BACKENDS[backend_name]
