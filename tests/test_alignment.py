import numpy as np
import pytest

from moe_interp.alignment import (align_tokens, common_prefix_length, common_suffix_length,
                                  intersect_on_reference)


def test_suffix_and_prefix_lengths():
    assert common_suffix_length([1, 2, 3, 4], [9, 3, 4]) == 2
    assert common_suffix_length([1, 2], [1, 2]) == 2
    assert common_suffix_length([1, 2], [3]) == 0
    assert common_suffix_length([], [1]) == 0
    assert common_prefix_length([1, 2, 3], [1, 2, 9, 9]) == 2
    assert common_prefix_length([5], [5, 6]) == 1


def test_common_suffix_preserves_original_indices():
    a = [10, 11, 12, 7, 8, 9]          # len 6
    b = [20, 21, 22, 23, 24, 7, 8, 9]  # len 8
    al = align_tokens(a, b, "common_suffix")
    assert al.ok and al.n == 3
    np.testing.assert_array_equal(al.positions_a, [3, 4, 5])
    np.testing.assert_array_equal(al.positions_b, [5, 6, 7])
    np.testing.assert_array_equal(np.asarray(a)[al.positions_a], np.asarray(b)[al.positions_b])


def test_last_n_requires_identical_tokens():
    a, b = [1, 2, 3, 4, 5], [9, 9, 3, 4, 5]
    assert align_tokens(a, b, "last_n", n=3).ok
    bad = align_tokens(a, b, "last_n", n=4)
    assert not bad.ok and bad.n == 0 and "differ" in bad.reason and "common suffix is only 3" in bad.reason
    short = align_tokens([1], b, "last_n", n=3)
    assert not short.ok and "shorter" in short.reason


def test_no_silent_alignment_when_nothing_matches():
    al = align_tokens([1, 2, 3], [4, 5, 6], "common_suffix")
    assert not al.ok and al.n == 0 and "0 tokens" in al.reason
    assert not align_tokens([1, 2, 3, 4], [9, 3, 4], "common_suffix", min_length=3).ok


def test_span_alignment():
    a = [0, 0, 5, 6, 7, 1]
    b = [0, 5, 6, 7, 1, 1, 1]
    sa = {"q": {"start": 2, "end": 5}}
    sb = {"q": {"start": 1, "end": 4}}
    al = align_tokens(a, b, "span", span_name="q", spans_a=sa, spans_b=sb)
    assert al.ok
    np.testing.assert_array_equal(al.positions_a, [2, 3, 4])
    np.testing.assert_array_equal(al.positions_b, [1, 2, 3])
    mism = align_tokens(a, [0, 5, 9, 7], "span", span_name="q", spans_a=sa, spans_b={"q": {"start": 1, "end": 4}})
    assert not mism.ok and "different ids" in mism.reason
    missing = align_tokens(a, b, "span", span_name="q", spans_a=sa, spans_b={})
    assert not missing.ok and "not annotated" in missing.reason


def test_common_prefix():
    al = align_tokens([1, 2, 3, 9], [1, 2, 3, 8, 8], "common_prefix")
    np.testing.assert_array_equal(al.positions_a, [0, 1, 2])


def test_unknown_method():
    with pytest.raises(ValueError):
        align_tokens([1], [1], "fuzzy")


def test_intersect_on_reference():
    ref = [1, 2, 3, 4, 5, 6]
    treat = [9, 9, 9, 4, 5, 6]          # suffix of 3
    ctrl = [7, 3, 4, 5, 6]              # suffix of 4
    a1, a2 = align_tokens(ref, treat), align_tokens(ref, ctrl)
    r, t, c = intersect_on_reference(a1, a2)
    np.testing.assert_array_equal(r, [3, 4, 5])
    np.testing.assert_array_equal(t, [3, 4, 5])
    np.testing.assert_array_equal(c, [2, 3, 4])
    np.testing.assert_array_equal(np.asarray(ref)[r], np.asarray(ctrl)[c])
