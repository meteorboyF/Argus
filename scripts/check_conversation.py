#!/usr/bin/env python3
"""Real local-model conversation checks; no cameras, speech or tool actions.

Uses production routing and session history. Unexpected tool routes are logged
without executing them. Outputs are evidence for review, not an automatic
naturalness score or a voice-pipeline acceptance test.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argus.agent import GemmaAgent
from argus.config import load_config
from argus.dialogue import DialogueManager, Turn

CASES = [
    ('greeting', 'Hello, how are you today?'),
    ('preference', 'My name is Sami'),
    ('memory', 'What is my name?'),
    ('advice', 'I am nervous about presenting my project tomorrow. Any advice?'),
    ('followup', 'Can you give me one concrete example of that?'),
    ('knowledge', 'Why does ice float on water?'),
    ('abstract_text', 'What is the difference between a book and a movie?'),
    ('abstract_title', 'Can you suggest a title for my presentation about technology?'),
    ('capability', 'Can you open YouTube for me?'),
    ('visual_boundary', 'What colour is the shirt I am wearing?'),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args=parser.parse_args()
    agent=GemmaAgent(load_config().agent)
    if not agent.healthy():
        raise SystemExit('Local model server is unavailable')
    orch=SimpleNamespace(agent=agent, _reader=None,
                         speaker=SimpleNamespace(speak=lambda *a: None))
    dialogue=DialogueManager(orch)
    rows=[]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for label,text in CASES:
        start=time.monotonic()
        intent=dialogue.route(text)
        answer=''
        # Execute only speech/memory paths. Never activate cameras or skills.
        if intent.skill in ('chat','answer','preference_name'):
            answer=dialogue.run(intent,text)
            dialogue.history.append(Turn(text,answer,intent.skill))
        row=dict(case=label,user=text,intent=asdict(intent),answer=answer,
                 seconds=round(time.monotonic()-start,3))
        rows.append(row)
        args.out.write_text(json.dumps(dict(kind='local-text-conversation-check',
            voice_pipeline_tested=False,rows=rows),indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
