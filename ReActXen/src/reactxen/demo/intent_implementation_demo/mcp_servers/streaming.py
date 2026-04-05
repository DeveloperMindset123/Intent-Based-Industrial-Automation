"""MCP Streaming support for long-running PHMForge tools.

Provides progress reporting for tools that take significant time:
- Model training (train_rul_model, train_fault_classifier)
- Batch prediction (predict_rul over many units)
- Large dataset loading

Uses MCP's notification mechanism to stream progress updates
to clients during tool execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

_log = logging.getLogger(__name__)


@dataclass
class ProgressUpdate:
    """A single progress update for a streaming tool call."""

    tool_name: str
    step: int
    total_steps: int
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def percent(self) -> float:
        return (self.step / self.total_steps * 100) if self.total_steps > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "step": self.step,
            "total": self.total_steps,
            "percent": f"{self.percent:.1f}%",
            "message": self.message,
            "metadata": self.metadata,
        }


class StreamingToolWrapper:
    """Wraps a tool function to emit progress updates during execution.

    Usage:
        wrapper = StreamingToolWrapper("train_rul_model")

        async def train_with_progress(dataset, model_type, epochs):
            wrapper.start(epochs)
            for epoch in range(epochs):
                # ... training logic ...
                wrapper.update(epoch + 1, f"Epoch {epoch + 1}/{epochs}, loss=0.05")
            return wrapper.finish("Training complete")

        # Client can poll progress:
        for update in wrapper.get_updates():
            print(update.message)
    """

    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name
        self._updates: list[ProgressUpdate] = []
        self._total_steps = 0
        self._current_step = 0
        self._started = False
        self._finished = False
        self._start_time: float = 0
        self._callbacks: list[Callable[[ProgressUpdate], None]] = []

    @property
    def is_active(self) -> bool:
        return self._started and not self._finished

    @property
    def elapsed_ms(self) -> float:
        if not self._started:
            return 0.0
        return (time.time() - self._start_time) * 1000

    def on_progress(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Register a callback to be called on each progress update."""
        self._callbacks.append(callback)

    def start(self, total_steps: int, message: str = "Starting...") -> None:
        """Signal the start of a long-running operation."""
        self._total_steps = total_steps
        self._current_step = 0
        self._started = True
        self._finished = False
        self._start_time = time.time()
        self._emit(0, message)

    def update(self, step: int, message: str, **metadata: Any) -> None:
        """Emit a progress update."""
        self._current_step = step
        self._emit(step, message, metadata)

    def finish(self, result: str) -> str:
        """Mark the operation as complete and return the result."""
        self._finished = True
        elapsed = self.elapsed_ms
        self._emit(
            self._total_steps,
            f"Complete in {elapsed:.0f}ms",
            {"elapsed_ms": elapsed},
        )
        return result

    def get_updates(self) -> list[ProgressUpdate]:
        """Return all progress updates so far."""
        return list(self._updates)

    def _emit(self, step: int, message: str, metadata: dict | None = None) -> None:
        update = ProgressUpdate(
            tool_name=self._tool_name,
            step=step,
            total_steps=self._total_steps,
            message=message,
            metadata=metadata or {},
        )
        self._updates.append(update)
        for cb in self._callbacks:
            try:
                cb(update)
            except Exception as e:
                _log.warning("Progress callback error: %s", e)


# ---------------------------------------------------------------------------
# Streaming-aware tool wrappers for PHMForge's long-running tools
# ---------------------------------------------------------------------------

def wrap_training_tool(
    original_fn: Callable,
    tool_name: str,
) -> Callable:
    """Wrap a training tool function to emit progress updates.

    The wrapped function accepts the same args as the original,
    plus an optional `progress_callback` kwarg.
    """

    def wrapped(*args, progress_callback: Callable | None = None, **kwargs):
        wrapper = StreamingToolWrapper(tool_name)
        if progress_callback:
            wrapper.on_progress(progress_callback)

        epochs = kwargs.get("epochs", 50)
        wrapper.start(epochs, f"Starting {tool_name}")

        # Simulate epoch-level progress for the training tools
        # (In production, this would hook into the actual training loop)
        wrapper.update(1, f"Initializing model for {kwargs.get('dataset', 'unknown')}")

        result = original_fn(*args, **kwargs)

        wrapper.update(epochs, "Training complete")
        return wrapper.finish(str(result) if not isinstance(result, str) else result)

    wrapped.__name__ = f"streaming_{tool_name}"
    wrapped.__doc__ = f"Streaming version of {tool_name}"
    return wrapped


# ---------------------------------------------------------------------------
# Async streaming iterator for client-side consumption
# ---------------------------------------------------------------------------

async def stream_tool_call(
    client: Any,  # MCPClient
    tool_name: str,
    args: dict,
    poll_interval: float = 0.5,
) -> AsyncIterator[dict]:
    """Stream progress updates from a tool call.

    Yields progress dicts as the tool executes. The final yield
    contains the tool result in a 'result' key.

    Usage:
        async for update in stream_tool_call(client, "train_rul_model", args):
            if "result" in update:
                print(f"Done: {update['result']}")
            else:
                print(f"Progress: {update['percent']}")
    """
    # For stdio transport, tool calls are atomic (no mid-call streaming).
    # We emit a start event, execute, then emit a complete event.
    yield {
        "type": "start",
        "tool": tool_name,
        "args": args,
        "message": f"Calling {tool_name}...",
    }

    start_time = time.time()
    try:
        result = await client.call_tool(tool_name, args)
        elapsed_ms = (time.time() - start_time) * 1000
        yield {
            "type": "complete",
            "tool": tool_name,
            "result": result,
            "elapsed_ms": elapsed_ms,
            "message": f"{tool_name} completed in {elapsed_ms:.0f}ms",
        }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        yield {
            "type": "error",
            "tool": tool_name,
            "error": str(e),
            "elapsed_ms": elapsed_ms,
            "message": f"{tool_name} failed: {e}",
        }
