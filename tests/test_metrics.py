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


def test_word_errors():
    from goftar.metrics import word_errors

    assert word_errors("من کتاب می خوانم", "من کتاب می خوانم") == []
    assert word_errors("من کتاب می خوانم", "من کتاب خوانم") == [("می", "")]
    assert word_errors("من کتاب خوانم", "من یک کتاب خوانم") == [("", "یک")]
    assert word_errors("اسم پروژه وب", "اسم پروژه ویب") == [("وب", "ویب")]


def test_word_errors_ignores_style_differences():
    from goftar.metrics import word_errors

    assert word_errors("می‌روم ۱۲۳.", "می روم 123") == []


def test_error_spans_indices():
    from goftar.metrics import ErrorSpan, error_spans

    spans = error_spans("یک دو سه چهار پنج", "یک تو سه چهار")
    assert spans == [ErrorSpan(1, 2, "دو", "تو"), ErrorSpan(4, 5, "پنج", "")]
    assert error_spans("یک دو", "یک دو") == []
