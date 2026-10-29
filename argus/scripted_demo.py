"""Explicit opt-in scripted presentation; never detection evidence."""
import re


def seat_reply(question: str, enabled: bool = False) -> str | None:
    if enabled and re.search(r'\b(sit|seat|chair)\b', question.lower()):
        return ("Yes, there's a chair straight ahead, "
                "it's empty, and you can sit on it.")
    return None
