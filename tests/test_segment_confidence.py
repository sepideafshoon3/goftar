import math
from types import SimpleNamespace as NS

import pytest

from goftar.engines.faster_whisper_engine import segment_confidence


def test_confidence_is_mean_word_probability():
    seg = NS(words=[NS(probability=0.9), NS(probability=0.7)], avg_logprob=-0.07)
    assert segment_confidence(seg) == pytest.approx(0.8)


def test_segments_of_one_window_get_their_own_confidence():
    # avg_logprob is shared by every segment decoded in the same 30 s window
    a = NS(words=[NS(probability=0.99)], avg_logprob=-0.07)
    b = NS(words=[NS(probability=0.60)], avg_logprob=-0.07)
    assert segment_confidence(a) != segment_confidence(b)


def test_falls_back_to_avg_logprob_without_words():
    assert segment_confidence(NS(words=None, avg_logprob=-0.1)) == pytest.approx(math.exp(-0.1))
    assert segment_confidence(NS(words=[], avg_logprob=None)) is None
