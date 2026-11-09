import pytest

from goftar.core.config import PostprocessConfig
from goftar.core.types import Segment, Transcript
from goftar.text.normalize import ZWNJ, normalize_text, postprocess_transcript


def test_arabic_letters_become_persian():
    assert normalize_text("كتاب ي") == "کتاب ی"


def test_diacritics_and_tatweel_removed():
    assert normalize_text("سَلاـم") == "سلام"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("persian", "۱۲۳ ۴۵۶"), ("latin", "123 456"), ("keep", "123 ٤٥٦")],
)
def test_digit_modes(mode, expected):
    assert normalize_text("123 ٤٥٦", PostprocessConfig(digits=mode, punctuation=False)) == (
        expected if mode != "keep" else "123 ٤٥٦"
    )


def test_half_space_for_mi_prefix():
    assert normalize_text("او می رود و نمی داند") == f"او می{ZWNJ}رود و نمی{ZWNJ}داند"


def test_half_space_not_inserted_inside_words():
    # "اسمی خوب": the "می" here is the end of a word, not the verb prefix
    assert normalize_text("اسمی خوب") == "اسمی خوب"


def test_half_space_for_plural_and_comparative_suffixes():
    assert normalize_text("کتاب ها بزرگ تر") == f"کتاب{ZWNJ}ها بزرگ{ZWNJ}تر"


def test_persian_punctuation():
    assert normalize_text("سلام, خوبی ?") == "سلام، خوبی؟"


def test_spacing_around_punctuation():
    assert normalize_text("سلام ،خوبی ؟بله") == "سلام، خوبی؟ بله"


def test_numbers_with_decimal_comma_are_untouched_by_punctuation_rules():
    assert normalize_text("3,5 درصد", PostprocessConfig(digits="keep")) == "3,5 درصد"


def test_whitespace_collapsed_and_trimmed():
    assert normalize_text("  سلام    دنیا   ") == "سلام دنیا"


def test_zwnj_cleanup():
    assert normalize_text(f"می{ZWNJ}{ZWNJ}رود {ZWNJ}") == f"می{ZWNJ}رود"


@pytest.mark.parametrize(
    "text",
    ["سلام, حال شما چطوره ?", "من 3 تا كتاب ها می خرم", "او می رود ،اما نمی داند ؟"],
)
def test_idempotent(text):
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_disabled_returns_input_unchanged():
    assert normalize_text("كتاب ,", PostprocessConfig(enabled=False)) == "كتاب ,"


def test_use_hazm_without_hazm_installed_does_not_crash():
    assert normalize_text("كتاب", PostprocessConfig(use_hazm=True))


def test_postprocess_transcript_drops_empty_segments():
    t = Transcript(segments=[Segment(0, 1, "كتاب"), Segment(1, 2, "  ")], duration=2)
    out = postprocess_transcript(t, PostprocessConfig())
    assert [s.text for s in out.segments] == ["کتاب"]
    assert out.duration == 2


def test_clean_text_is_plain_normalisation_by_default():
    from goftar.text.normalize import clean_text, normalize_text

    assert clean_text("می رود ,", None) == normalize_text("می رود ,")


def test_clean_text_applies_glossary(tmp_path):
    from goftar.core.config import PostprocessConfig
    from goftar.text.normalize import clean_text

    g = tmp_path / "g.yaml"
    g.write_text("ویب: وب\n", encoding="utf-8")
    assert clean_text("ویب سایت و ویب", PostprocessConfig(glossary=str(g))) == "وب سایت و وب"
