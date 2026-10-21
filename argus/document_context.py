"""Bound masked OCR context for local-model questions without inventing text."""
import re
from .sensitive import mask_text


def excerpt(text, question, max_chars=5000):
    text = mask_text(text)
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    terms = set(re.findall(r'\w+',question.lower()))-{'what','is','the','of','a','on','this','it','about','does','say','and','to','me'}
    ranked = sorted(range(len(lines)),key=lambda i: (-len(terms & set(re.findall(r'\w+',lines[i].lower()))),i))
    selected, used = set(),0
    for i in ranked:
        for j in range(max(0,i-1),min(len(lines),i+2)):
            if j not in selected and used+len(lines[j])+1 <= max_chars-80:
                selected.add(j)
                used += len(lines[j])+1
        if used >= max_chars-300:
            break
    if not selected:
        return '[Excerpt only; document is longer.]\n'+text[:max_chars-80]
    return '[Excerpt only; document is longer.]\n'+'\n'.join(lines[i] for i in sorted(selected))
