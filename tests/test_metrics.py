import pytest

from goftar.metrics import cer, edit_distance, tokenize, wer


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a", "b"]) == 2
    assert edit_distance(["a"], ["a"]) == 0


def test_tokenize_ignores_punctuation_digits_style_and_half_space():
    assert tokenize("سلام، دنیا! ۱۲۳") == tokenize("سلام دنیا 123")
    assert tokenize("می‌روم") == tokenize("می روم")
    assert tokenize("كتاب") == tokenize("کتاب")


def test_wer():
    assert wer("من کتاب می خوانم", "من کتاب می خوانم") == 0.0
    assert wer("من کتاب می خوانم", "من کتاب خوانم") == pytest.approx(0.25)
    assert wer("a b", "c d") == 1.0


def test_wer_empty_reference():
    assert wer("", "") == 0.0
    assert wer("", "something") == 1.0


def test_cer():
    assert cer("سلام", "سلام") == 0.0
    assert cer("سلام", "سلان") == pytest.approx(0.25)
