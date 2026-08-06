"""Gemma reasoning agent via native llama.cpp (OpenAI-compatible server).

The agent receives one privacy-gated camera frame and the user's transcribed
question. It either answers directly (one or two short spoken sentences, hazards
first) or requests the find_object tool, which the runtime executes with
YOLO-World, then feeds the result back for a final spoken answer.

llama.cpp is launched separately (see scripts/run_llama_server.sh). This client
just talks to its /v1/chat/completions endpoint with an image attached.

TOOL PROTOCOL. Two modes (config agent.tool_protocol):
  - "prompt" (default): the system prompt instructs the model to answer with a
    single JSON object {"tool": "find_object", "name": "..."} when it needs to
    locate something, and we parse that out of the plain text reply. This works
    with EVERY llama.cpp chat template, including Gemma's.
  - "native": the OpenAI `tools` parameter is sent. Requires llama-server to be
    started with --jinja AND a chat template with tool support — many Gemma
    templates lack it, which is why this is not the default.
Native tool_calls in the response are honoured in both modes.
"""
from __future__ import annotations

import base64
import json
import re
import time
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
3. Never describe faces or private documents.
4. If the user asks to locate/find a specific object, do NOT guess. Instead reply with ONLY this JSON on a single line, nothing else:
{"tool": "find_object", "name": "<object name>"}
Otherwise reply with the spoken answer only (no JSON)."""

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

# A JSON object containing a "tool" key anywhere in the reply (models often wrap
# it in prose or markdown fences despite instructions).
_TOOL_JSON_RE = re.compile(r"\{[^{}]*\"tool\"[^{}]*\}")


class AgentError(RuntimeError):
    """The llama.cpp server could not be reached or returned garbage."""


@dataclass
class AgentReply:
    text: str                       # spoken answer ("" if a tool call was made)
    tool_call: str | None = None    # tool name, e.g. "find_object"
    tool_args: dict | None = None


def _encode_image(frame_bgr: np.ndarray, max_side: int) -> str:
    h, w = frame_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise AgentError("Failed to JPEG-encode the camera frame")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class GemmaAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.endpoint = cfg.server_url.rstrip("/") + "/v1/chat/completions"

    def healthy(self) -> bool:
        try:
            r = requests.get(self.cfg.server_url.rstrip("/") + "/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _post(self, messages: list[dict], tools: list[dict] | None) -> dict:
        payload = {
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if tools and self.cfg.tool_protocol == "native":
            payload["tools"] = tools
        last_err: Exception | None = None
        for attempt in range(self.cfg.retries + 1):
            try:
                r = requests.post(self.endpoint, json=payload,
                                  timeout=self.cfg.request_timeout_s)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_err = e
                if attempt < self.cfg.retries:
                    time.sleep(1.0)
        raise AgentError(f"llama.cpp server request failed: {last_err}") from last_err

    def ask(self, frame_gated_bgr: np.ndarray, question: str) -> AgentReply:
        """First turn: frame + question, with the find_object tool available."""
        messages = [
            {"role": "system", "content": ARGUS_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {
                    "url": _encode_image(frame_gated_bgr, self.cfg.image_max_side)}},
            ]},
        ]
        return self._parse(self._post(messages, [FIND_OBJECT_TOOL]))

    def with_tool_result(self, question: str, tool_result: dict) -> AgentReply:
        """Second turn: feed the find_object result back for a final answer.

        Sent as plain text (not the OpenAI `tool` role) so it works with chat
        templates that have no tool-role support, which includes Gemma's.
        """
        messages = [
            {"role": "system", "content": ARGUS_SYSTEM},
            {"role": "user", "content": (
                f"The user asked: {question}\n"
                f"The find_object tool returned: {json.dumps(tool_result)}\n"
                "Using only this result, give the final spoken answer now "
                "(one or two short sentences, direction and distance if known). "
                "Do not output JSON."
            )},
        ]
        reply = self._parse(self._post(messages, None))
        # Never loop: a tool call in the second turn is treated as a plain miss.
        if reply.tool_call:
            found = tool_result.get("found")
            name = tool_result.get("name", "it")
            reply = AgentReply(text=(f"I could not find the {name}." if not found
                                     else f"The {name} is {tool_result.get('direction', 'ahead')}."))
        return reply

    @classmethod
    def _parse(cls, resp: dict) -> AgentReply:
        try:
            choice = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise AgentError(f"Unexpected server response shape: {resp!r:.300}") from e

        # Native tool_calls (present when --jinja + a tool-capable template).
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

        text = (choice.get("content") or "").strip()

        # Prompt-protocol tool request embedded in the text.
        m = _TOOL_JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                tool = obj.get("tool")
                if tool:
                    args = {k: v for k, v in obj.items() if k != "tool"}
                    return AgentReply(text="", tool_call=str(tool), tool_args=args)
            except json.JSONDecodeError:
                pass  # looked like JSON but wasn't — treat as speech

        return AgentReply(text=text)
