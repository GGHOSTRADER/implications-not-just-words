"""Stand-in for the application's agent module: only the answer cleaner the evaluation uses, copied verbatim (commit
19077f5), and a fixed identity line (the evaluation replaces it anyway)."""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


def _identity_header(*_a, **_k) -> str:
    return "You are Calvis, a friendly fitness coach. The user is G."


def _load_goals(*_a, **_k) -> dict:
    """The application reads the user's goals from its database. The evaluation replaces this with each case's goals."""
    raise RuntimeError("no database in this release: evals.run_message_eval supplies each case's goals")


def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks (and any unclosed leading block)."""
    if "<think>" not in text and "</think>" not in text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed/leading think block (thinking got cut or only opening tag present)
    text = re.sub(r"^\s*<think>.*?$", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def _clean_response(text: str) -> str:
    """Strip Qwen3 reasoning blocks, markdown and LaTeX from model output."""
    text = _strip_think(text)
    # LaTeX display math \[ ... \]
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    # LaTeX inline math \( ... \)
    text = re.sub(r"\\\(.*?\\\)", "", text, flags=re.DOTALL)
    # LaTeX \text{...}
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    # Markdown bold **...**
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Markdown italic *...*  (single asterisk)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Markdown headers ## Title
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Collapse extra blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
