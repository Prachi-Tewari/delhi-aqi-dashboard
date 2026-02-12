"""Conversation summarization memory.

Keeps the most recent N messages verbatim and summarizes older ones
into a compact context block, preventing the conversation from exceeding
the LLM context window while preserving important information.
"""

from __future__ import annotations

import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def _messages_tokens(messages: list[dict]) -> int:
    return sum(_approx_tokens(m.get("content", "")) for m in messages)


class ConversationMemory:
    """Manages conversation history with automatic summarization.

    Keeps `keep_recent` messages verbatim and summarizes the rest
    when the total conversation exceeds `max_tokens`.
    """

    def __init__(
        self,
        max_tokens: int = 6000,
        keep_recent: int = 6,
        groq_api_key: str = "",
    ):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self._api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._summary: str | None = None
        self._summarized_count: int = 0  # how many messages have been summarized

    @property
    def summary(self) -> str | None:
        return self._summary

    def prepare_messages(self, full_history: list[dict]) -> list[dict]:
        """Return an optimized message list for the LLM.

        If the conversation is short enough, returns it unchanged.
        Otherwise, prepends a summary of older messages and keeps
        only the most recent ones verbatim.
        """
        if not full_history:
            return []

        # Separate system message if present
        system_msg = None
        chat_msgs = full_history
        if full_history[0].get("role") == "system":
            system_msg = full_history[0]
            chat_msgs = full_history[1:]

        total_tok = _messages_tokens(chat_msgs)

        # If under limit, return as-is
        if total_tok <= self.max_tokens and len(chat_msgs) <= self.keep_recent * 2:
            return full_history

        # Split into old (to summarize) and recent (to keep)
        n_keep = min(self.keep_recent, len(chat_msgs))
        old_msgs = chat_msgs[:-n_keep] if n_keep < len(chat_msgs) else []
        recent_msgs = chat_msgs[-n_keep:]

        result = []
        if system_msg:
            result.append(system_msg)

        # Summarize old messages
        if old_msgs:
            summary = self._summarize(old_msgs)
            if summary:
                result.append({
                    "role": "system",
                    "content": (
                        f"[Conversation Summary — {len(old_msgs)} earlier messages]\n"
                        f"{summary}"
                    ),
                })

        result.extend(recent_msgs)
        return result

    def _summarize(self, messages: list[dict]) -> str:
        """Summarize a list of messages using the LLM."""
        # Check if we already have a summary that covers these messages
        if self._summary and self._summarized_count >= len(messages):
            return self._summary

        # Build summarization prompt
        conversation_text = ""
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")[:500]  # truncate long messages
            conversation_text += f"{role}: {content}\n\n"

        # If we have a previous summary, include it
        prior = ""
        if self._summary:
            prior = f"Previous summary:\n{self._summary}\n\nNew messages to include:\n"

        prompt = (
            f"{prior}{conversation_text}\n"
            "Create a concise summary of this conversation in 3-5 sentences. "
            "Include: key topics discussed, any specific data/facts mentioned, "
            "user preferences or context established, and unanswered questions."
        )

        summary = self._call_llm(prompt)
        if summary and not summary.startswith("["):
            self._summary = summary
            self._summarized_count = len(messages)
            return summary

        # Fallback: simple extractive summary
        return self._extractive_fallback(messages)

    def _extractive_fallback(self, messages: list[dict]) -> str:
        """Simple extractive summary when LLM is unavailable."""
        topics = []
        for m in messages:
            content = m.get("content", "")
            if m.get("role") == "user":
                topics.append(f"User asked: {content[:100]}")
            else:
                topics.append(f"Assistant discussed: {content[:80]}")
        # Keep first and last few
        if len(topics) > 6:
            topics = topics[:3] + ["..."] + topics[-2:]
        self._summary = " | ".join(topics)
        self._summarized_count = len(messages)
        return self._summary

    def _call_llm(self, prompt: str) -> str:
        if not self._api_key:
            return ""
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a concise summarizer."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.2,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        return ""

    def reset(self):
        self._summary = None
        self._summarized_count = 0
