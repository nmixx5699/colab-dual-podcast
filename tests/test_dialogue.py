import pytest
from src.dialogue import prepare_dialogue, split_text


@pytest.mark.parametrize('value', [None, [], {}, [{'speaker': 'C', 'text': '你好'}], [{'speaker': 'A', 'text': ''}], [{'speaker': 'B', 'text': '(開心)'}]])
def test_invalid_dialogue(value):
    with pytest.raises(ValueError):
        prepare_dialogue(value)


def test_long_chinese_and_tone_preserved():
    body = '這是一段很長的對話，' * 20
    result = prepare_dialogue([{'speaker': 'B', 'text': '(開心)' + body}])
    assert ''.join(r['text'] for r in result) == body
    assert all(len(r['text']) <= 80 and r['speaker'] == 'B' and r['synthesis_text'].startswith('(開心)') for r in result)


def test_no_punctuation_and_boundary():
    assert list(map(len, split_text('甲' * 161))) == [80, 80, 1]
    assert split_text('甲' * 79 + '。乙') == ['甲' * 79 + '。', '乙']
