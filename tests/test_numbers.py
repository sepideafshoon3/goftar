import pytest

from goftar.core.config import PostprocessConfig
from goftar.metrics import wer
from goftar.text.normalize import normalize_text
from goftar.text.numbers import words_to_digits


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("صد و بیست و پنج", "125"),
        ("سه هزار و پونصد", "3500"),
        ("دو هزار و هفتاد و پنج", "2075"),
        ("هزار و دویست", "1200"),
        ("سی و پنج", "35"),
        ("ده هزار", "10000"),
        ("دو ممیز هفتاد و پنج", "2.75"),
        ("سه میلیون و پونصد هزار", "3 میلیون و 500 هزار"),
        ("یک میلیارد و دویست میلیون", "1 میلیارد و 200 میلیون"),
    ],
)
def test_number_words_become_digits(words, digits):
    assert words_to_digits(words) == digits


def test_numbers_inside_a_sentence():
    text = "در آن عددهای صد و بیست و پنج، سه هزار و پونصد، و دو ممیز هفتاد و پنج"
    assert words_to_digits(text) == "در آن عددهای 125، 3500، و 2.75"


@pytest.mark.parametrize(
    "text",
    [
        "یه چیزی بگم",  # single number words stay words
        "بیست تا",
        "صد درصد",
        "کانسیومر بیست باشه",  # English "based"
        "دو تا کتاب",
        "ده دقیقه",
        "نه نه نه نه",  # "no no no no"
        "هزار تا کار",  # a bare scale word
        "سی پلاس پلاس",  # "سی" alone is also the letter C
        "دو و سه",  # two numbers joined by "و", not one number
        "پنج و بیست",  # wrong order
        "دو سه هزار",  # "two or three thousand": a range, not 23000
        "دوستان و همکاران",  # words that merely start like a number
        "میلیون‌ها نفر",
        "بیست و هشتا کاتالوگ",  # the compound goes on with a word we do not know
        "سی و پنجم",
        "بیست و سه و هشتا",
        "ساعت پنج و بیست دقیقه",
        "نهصدمین",
    ],
)
def test_non_numbers_left_alone(text):
    assert words_to_digits(text) == text


def test_normalizer_converts_then_applies_digit_style():
    assert normalize_text("سه هزار و پونصد تومن") == "۳۵۰۰ تومن"
    assert normalize_text("دو ممیز پنج", PostprocessConfig(digits="latin")) == "2.5"
    assert normalize_text("دو ممیز پنج") == "۲٫۵"


def test_number_words_can_be_turned_off():
    assert normalize_text("صد و بیست", PostprocessConfig(number_words=False)) == "صد و بیست"


def test_scoring_treats_number_words_and_digits_alike():
    assert wer("عددهای ۱۲۵ و ۳۵۰۰", "عددهای صد و بیست و پنج و سه هزار و پونصد") == 0.0
