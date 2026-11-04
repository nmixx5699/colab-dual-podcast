"""Structured Chinese dialogue validation and punctuation-aware splitting."""
import re


def split_text(text, limit=80):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be a nonempty string')
    if not isinstance(limit, int) or not 1 <= limit <= 80:
        raise ValueError('limit must be 1..80')
    text = text.strip()
    chunks = []
    while len(text) > limit:
        cut = max((text.rfind(p, 0, limit) for p in '。！？；，、!?;,.'), default=-1)
        cut = cut + 1 if cut >= 0 else limit
        if text[:cut].strip():
            chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def prepare_dialogue(dialogue, limit=80):
    if not isinstance(dialogue, list) or not dialogue:
        raise ValueError('DIALOGUE must be a nonempty list')
    result = []
    for index, row in enumerate(dialogue):
        if not isinstance(row, dict) or row.get('speaker') not in ('A', 'B'):
            raise ValueError(f'Row {index}: speaker must be A or B')
        text = row.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Row {index}: text is empty')
        tone = re.match(r'^\s*[（(]([^()（）]*)[)）]\s*', text)
        prefix = f'({tone.group(1)})' if tone else ''
        body = text[tone.end():] if tone else text
        for part in split_text(body, limit):
            result.append({'speaker': row['speaker'], 'text': part,
                           'synthesis_text': prefix + part, 'turn': index})
    return result
