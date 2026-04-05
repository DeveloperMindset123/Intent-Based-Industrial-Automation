"""Context management for MCP tool responses — prevents context bloating.

Implements response summarization and truncation strategies to keep
agent context windows manageable when processing long tool outputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_RESPONSE_CHARS = 4000
DEFAULT_MAX_CONTEXT_CHARS = 32000
DEFAULT_MAX_HISTORY_STEPS = 20


@dataclass
class ContextConfig:
    """Configuration for context management."""

    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_history_steps: int = DEFAULT_MAX_HISTORY_STEPS
    truncation_strategy: str = "tail"  # "tail", "head", or "middle"
    summarize_json: bool = True


class ContextManager:
    """Manages tool response context to prevent context window bloating.

    Strategies:
    1. Response truncation — cap individual tool outputs
    2. JSON summarization — collapse large JSON arrays/objects
    3. History windowing — keep only the N most recent step results
    4. Total context budgeting — enforce overall character limit
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        self._config = config or ContextConfig()
        self._history: list[dict] = []

    @property
    def config(self) -> ContextConfig:
        return self._config

    def process_response(self, tool_name: str, raw_response: str) -> str:
        """Process a tool response: summarize JSON, then truncate if needed."""
        response = raw_response

        # Step 1: Summarize JSON structures if applicable
        if self._config.summarize_json:
            response = self._summarize_json(response)

        # Step 2: Truncate to max response size
        if len(response) > self._config.max_response_chars:
            response = self._truncate(response, self._config.max_response_chars)

        # Step 3: Record in history
        self._history.append({
            "tool": tool_name,
            "response": response,
            "original_length": len(raw_response),
            "processed_length": len(response),
        })

        # Step 4: Trim history if too long
        if len(self._history) > self._config.max_history_steps:
            self._history = self._history[-self._config.max_history_steps:]

        return response

    def get_context_window(self) -> str:
        """Build a context string from history, respecting total budget."""
        parts = []
        total = 0
        # Build from most recent backward
        for entry in reversed(self._history):
            text = f"[{entry['tool']}]: {entry['response']}"
            if total + len(text) > self._config.max_context_chars:
                break
            parts.append(text)
            total += len(text)
        parts.reverse()
        return "\n\n".join(parts)

    def clear(self) -> None:
        """Clear the history."""
        self._history.clear()

    def _truncate(self, text: str, max_chars: int) -> str:
        """Truncate text according to configured strategy."""
        if len(text) <= max_chars:
            return text

        marker = f"\n... [truncated {len(text) - max_chars} chars] ..."

        if self._config.truncation_strategy == "head":
            return text[:max_chars] + marker
        elif self._config.truncation_strategy == "tail":
            return marker + text[-(max_chars):]
        else:  # middle
            half = max_chars // 2
            return text[:half] + marker + text[-half:]

    def _summarize_json(self, text: str) -> str:
        """Summarize large JSON structures in the response."""
        try:
            data = json.loads(text)
            return self._summarize_value(data)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON within the text
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            prefix = text[:start]
            suffix = text[end:]
            try:
                data = json.loads(text[start:end])
                summarized = self._summarize_value(data)
                return prefix + summarized + suffix
            except (json.JSONDecodeError, TypeError):
                pass

        return text

    def _summarize_value(self, value: object, depth: int = 0) -> str:
        """Recursively summarize a JSON value."""
        if isinstance(value, list):
            if len(value) <= 3:
                return json.dumps(value)
            # Show first 2 and last 1, with count
            items = [json.dumps(value[0]), json.dumps(value[1])]
            return f"[{', '.join(items)}, ... ({len(value)} items total) ..., {json.dumps(value[-1])}]"
        elif isinstance(value, dict):
            if len(value) <= 5 or depth > 1:
                return json.dumps(value)
            # Show key names and summarize large nested values
            parts = []
            for k, v in list(value.items())[:5]:
                if isinstance(v, (list, dict)) and len(str(v)) > 200:
                    parts.append(f'"{k}": {self._summarize_value(v, depth + 1)}')
                else:
                    parts.append(f'"{k}": {json.dumps(v)}')
            if len(value) > 5:
                parts.append(f"... +{len(value) - 5} more keys")
            return "{" + ", ".join(parts) + "}"
        else:
            return json.dumps(value)
