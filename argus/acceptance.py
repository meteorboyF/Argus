"""Acceptance scoring without inventing evidence or retaining source text."""
from __future__ import annotations

import math
import unicodedata


def cer(reference, hypothesis):
    reference, hypothesis = (unicodedata.normalize('NFC', s) for s in (reference, hypothesis))
    if not reference:
        raise ValueError('CER requires nonempty reference text')
    previous = list(range(len(hypothesis)+1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]/len(reference)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def score(kind, data):
    """Only literal observed outcomes count; absent evidence cannot pass."""
    checks = {}
    metrics = {}
    if kind == 'reading':
        samples = data.get('samples', [])
        metrics['samples'] = []
        for i, sample in enumerate(samples):
            value = cer(sample['reference'], sample['hypothesis'])
            metrics['samples'].append({'sample': i+1, 'cer': value,
                                       'first_word_s': sample.get('first_word_s')})
        english = [s for s in metrics['samples'] if samples[s['sample']-1].get('language') == 'eng']
        checks['english_cer'] = bool(english) and all(s['cer'] < .03 for s in english)
        checks['first_word'] = bool(samples) and all(finite(s.get('first_word_s')) and 0 <= s['first_word_s'] < 4 for s in samples)
        checks['title_author'] = data.get('title_correct') is True and data.get('author_correct') is True
        modes = ('backlit', 'glare', 'upside_down', 'two_columns', 'cut_off', 'hand', 'no_text', 'long_page_controls', 'sensitive_numbers')
        checks['failure_guidance'] = all(data.get('cases', {}).get(m) is True for m in modes)
    elif kind == 'seat':
        trials = data.get('trials', [])
        valid = [t for t in trials if isinstance(t.get('success'), bool) and finite(t.get('seconds')) and t['seconds'] >= 0 and t.get('layout')]
        metrics.update(trials=len(valid), successes=sum(t['success'] for t in valid))
        checks['ten_trials_three_layouts'] = len(valid) >= 10 and len({t['layout'] for t in valid}) >= 3
        checks['success_rate'] = len(valid) >= 10 and metrics['successes']/len(valid) >= .8
        checks['corner_cases'] = all(data.get('cases', {}).get(c) is True for c in ('no_seat', 'two_seats', 'occupied', 'behind', 'lost'))
    elif kind == 'wake':
        attempts = data.get('attempts', [])
        hours, false = data.get('ambient_hours'), data.get('false_wakes')
        metrics['recall'] = sum(v is True for v in attempts)/len(attempts) if attempts else None
        checks['recall'] = len(attempts) >= 50 and all(isinstance(v, bool) for v in attempts) and metrics['recall'] >= .8
        checks['ambient'] = finite(hours) and hours >= 2 and finite(false) and false >= 0 and float(false).is_integer() and false/hours <= .5
        checks['sound_coverage'] = all(data.get('sounds', {}).get(c) is True for c in ('bangla', 'english', 'tv'))
        metrics['false_wakes_per_hour'] = false/hours if finite(hours) and hours > 0 and finite(false) else None
    elif kind == 'demo':
        seconds = data.get('seconds')
        checks['duration'] = finite(seconds) and 110 <= seconds <= 130
        checks['segments'] = all(data.get('segments', {}).get(c) is True for c in ('startup', 'find_object', 'reading', 'approach_interrupt'))
        checks['recording_exists'] = data.get('recording_verified') is True
    else:
        raise ValueError(f'Unsupported acceptance kind: {kind}')
    return {'kind': kind, 'status': 'passed' if all(checks.values()) else 'partial',
            'checks': checks, 'metrics': metrics,
            'note': 'Scored observations supplied by the evaluator; does not replace device evidence review.'}
