"""Control-only barge-in policy, independent of acoustic echo performance."""
from .wake_stt import match_wake
from .dialogue import rule_route


def barge_in_command(transcript):
    wake = match_wake(transcript)
    if not wake.woke or not wake.request:
        return None
    intent = rule_route(wake.request, False, False)
    if intent is not None and intent.skill == 'control' and intent.name in ('stop','pause'):
        return wake.request
    return None
