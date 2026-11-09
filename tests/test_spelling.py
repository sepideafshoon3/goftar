import pytest

from goftar.core.config import Config, PostprocessConfig
from goftar.errors import ConfigError
from goftar.text.spelling import SpellFixer, file_digest, load_glossary, load_wordlist

WORDS = {"صبح": 50, "ضبط": 40, "سلام": 100, "کتاب": 80, "قلم": 5, "غلم": 1}


def test_fixes_homophone_letter_when_result_is_known():
    fixer = SpellFixer(WORDS)
    assert fixer.fix_word("سبح") == "صبح"
    assert fixer.fix("سبح بخیر") == "صبح بخیر"


def test_leaves_known_short_and_unfixable_words_alone():
    fixer = SpellFixer(WORDS)
    assert fixer.fix_word("سلام") == "سلام"
    assert fixer.fix_word("کتابها") == "کتابها"  # known with a suffix
    assert fixer.fix_word("به") == "به"  # too short
    assert fixer.fix_word("پروژه") == "پروژه"  # no candidate


def test_ambiguous_candidates_are_not_guessed():
    fixer = SpellFixer({"صبر": 10, "سبر": 0, "ثبر": 10})
    assert fixer.fix_word("زبر") == "زبر"
    assert SpellFixer({"صبر": 10, "ثبر": 10}).fix_word("سبر") == "سبر"


def test_highest_frequency_candidate_wins():
    assert SpellFixer({"قلم": 5, "غلم": 1}).fix_word("غلم") == "غلم"  # known itself
    assert SpellFixer({"قلم": 5, "غلم": 1}).fix_word("قلم") == "قلم"


def test_glossary_replaces_whole_words_only_and_protects_output():
    fixer = SpellFixer(WORDS, {"ویب": "وب", "گیت هاب": "گیت‌هاب"})
    assert fixer.fix("ویب و گیت هاب") == "وب و گیت‌هاب"
    assert fixer.fix("ویبی") == "ویبی"
    assert fixer.fix("می‌ویب") == "می‌ویب"  # glued to a ZWNJ word: not a whole word


def test_load_wordlist_and_glossary(tmp_path):
    wl = tmp_path / "w.txt"
    wl.write_text("کتاب 3\nسلام\n\n", encoding="utf-8")
    assert load_wordlist(wl) == {"کتاب": 3, "سلام": 1}
    g = tmp_path / "g.yaml"
    g.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_glossary(g)
    with pytest.raises(ConfigError, match="not found"):
        load_glossary(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="not found"):
        load_wordlist(tmp_path / "nope.txt")


def test_pipeline_option_uses_wordlist(tmp_path):
    from goftar.text.normalize import clean_text

    wl = tmp_path / "w.txt"
    wl.write_text("صبح\nبخیر\n", encoding="utf-8")
    opts = PostprocessConfig(spelling=True, wordlist=str(wl))
    assert clean_text("سبح بخیر", opts) == "صبح بخیر"
    assert clean_text("سبح بخیر", PostprocessConfig()) == "سبح بخیر"


def test_fingerprint_tracks_wordlist_and_glossary_content(tmp_path):
    g = tmp_path / "g.yaml"
    g.write_text("الف: ب\n", encoding="utf-8")
    cfg = Config.from_dict({"postprocess": {"glossary": str(g)}})
    before = cfg.fingerprint()
    assert before != Config().fingerprint()
    g.write_text("الف: ج\n", encoding="utf-8")
    assert cfg.fingerprint() != before
    assert file_digest(tmp_path / "missing") == "missing"
