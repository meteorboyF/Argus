"""Gemma 4 E2B reasoning agent via native llama.cpp (OpenAI-compatible server).

The agent receives one privacy-gated camera frame and the user's transcribed
question. It either answers directly (one or two short spoken sentences, hazards
first) or emits a single find_object(name) tool call which the runtime executes
with YOLO-World, then feeds the result back for a final spoken answer.

llama.cpp is launched separately (see scripts/run_llama_server.sh). This client
just talks to its /v1/chat/completions endpoint with an image attached.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import cv2
import numpy as np
import requests

from .config import AgentConfig

ARGUS_SYSTEM = """You are ARGUS, an assistant embedded in smart glasses for a blind user.
You see one camera frame and hear one question.
Rules:
1. Answer in one or two short spoken sentences.
2. State any safety hazard first.
3. To locate a specific named object, call the tool find_object(name).
4. Never describe faces or private documents.
Return either a spoken answer, or a single tool call."""

FIND_OBJECT_TOOL = {
    "type": "function",
    "function": {
        "name": "find_object",
        "description": "Locate a named object in the current camera view.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "open-vocabulary object name, e.g. 'keys'"}
            },
            "required": ["name"],
        },
    },
}


@dataclass
class AgentReply:
    text: str                       # spoken answer ("" if a tool call was made)
    tool_call: str | None = None    # tool name, e.g. "find_object"
    tool_args: dict | None = None


def _encode_image(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class GemmaAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.endpoint = cfg.server_url.rstrip("/") + "/v1/chat/completions"

    def _post(self, messages: list[dict], tools: list[dict] | None) -> dict:
        payload = {
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if tools:
            payload["tools"] = tools
        r = requests.post(self.endpoint, json=payload, timeout=self.cfg.request_timeout_s)
        r.raise_for_status()
        return r.json()

    def ask(self, frame_gated_bgr: np.ndarray, question: str) -> AgentReply:
        """First turn: frame + question, with the find_object tool available."""
        messages = [
            {"role": "system", "content": ARGUS_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": _encode_image(frame_gated_bgr)}},
            ]},
        ]
        return self._parse(self._post(messages, [FIND_OBJECT_TOOL]))

    def with_tool_result(self, question: str, tool_result: dict) -> AgentReply:
        """Second turn: feed the find_object result back for a final answer."""
        messages = [
            {"role": "system", "content": ARGUS_SYSTEM},
            {"role": "user", "content": question},
            {"role": "tool", "name": "find_object", "content": json.dumps(tool_result)},
        ]
        return self._parse(self._post(messages, None))

    @staticmethod
    def _parse(resp: dict) -> AgentReply:
        choice = resp["choices"][0]["message"]
        calls = choice.get("tool_calls")
        if calls:
            fn = calls[0]["function"]
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"name": args}
            return AgentReply(text="", tool_call=fn["name"], tool_args=args or {})
        return AgentReply(text=(choice.get("content") or "").strip())
