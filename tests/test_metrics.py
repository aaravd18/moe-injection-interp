import numpy as np
import pytest

from moe_interp import metrics as M


def test_topk_overlap_basic():
    a = np.array([[[0, 1, 2, 3]], [[0, 1, 2, 3]]])  # [L=2, N=1, k=4]
    b = np.array([[[3, 2, 1, 0]], [[0, 9, 8, 7]]])
    np.testing.assert_allclose(M.topk_overlap(a, b), [[1.0], [0.25]])
    assert M.topk_overlap(a, a).min() == 1.0


def test_topk_overlap_order_invariant_and_disjoint():
    a = np.array([[5, 6]])
    assert M.topk_overlap(a, np.array([[7, 8]]))[0] == 0.0
    assert M.topk_overlap(a, np.array([[6, 5]]))[0] == 1.0
    with pytest.raises(ValueError):
        M.topk_overlap(a, np.array([[1, 2, 3]]))


def test_expert_counts_and_distribution():
    ids = np.array([[[0, 1], [1, 2]]])  # L=1, N=2, k=2
    c = M.expert_counts(ids, 4)
    np.testing.assert_array_equal(c, [[1, 2, 1, 0]])
    np.testing.assert_allclose(M.selection_distribution(c), [[0.25, 0.5, 0.25, 0.0]])
    with pytest.raises(ValueError):
        M.expert_counts(np.array([[[5]]]), 4)
    with pytest.raises(ValueError):
        M.selection_distribution(np.zeros((1, 3)))


def test_jsd_known_values():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert M.js_divergence(p, q) == pytest.approx(1.0)          # disjoint -> 1 bit
    assert M.js_divergence(p, p) == pytest.approx(0.0)
    assert M.js_divergence(p, q, base=np.e) == pytest.approx(np.log(2))
    # Hand-computed: p=[.5,.5], q=[1,0] -> m=[.75,.25]
    expected = 0.5 * (0.5 * np.log2(0.5 / 0.75) + 0.5 * np.log2(0.5 / 0.25)) + 0.5 * (np.log2(1 / 0.75))
    assert M.js_divergence([0.5, 0.5], [1, 0]) == pytest.approx(expected)


def test_jsd_properties():
    rng = np.random.default_rng(0)
    p, q = rng.random((5, 16)), rng.random((5, 16))
    p[:, :4] = 0  # zeros must be handled exactly, no NaN/inf
    j = M.js_divergence(p, q)
    assert j.shape == (5,) and np.all(np.isfinite(j)) and np.all((0 <= j) & (j <= 1))
    np.testing.assert_allclose(j, M.js_divergence(q, p))           # symmetric
    np.testing.assert_allclose(M.js_divergence(p * 7, q), j)        # scale invariant (normalised internally)
    with pytest.raises(ValueError):
        M.js_divergence([-1, 2], [1, 1])


def test_cosine_and_relative_l2():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    b = np.array([[0.0, 2.0], [2.0, 2.0], [1.0, 0.0]])
    cos = M.cosine_similarity(a, b)
    assert cos[0] == pytest.approx(0.0) and cos[1] == pytest.approx(1.0) and np.isnan(cos[2])
    rl2 = M.relative_l2(a, b)
    assert rl2[0] == pytest.approx(np.sqrt(5)) and rl2[1] == pytest.approx(1.0) and np.isnan(rl2[2])
    assert M.relative_l2(a[:2], a[:2]).max() == 0.0
    assert M.cosine_similarity(-a[:1], a[:1])[0] == pytest.approx(-1.0)


def test_control_adjusted_delta():
    np.testing.assert_allclose(M.control_adjusted_delta([0.5, 0.2], [0.1, 0.2]), [0.4, 0.0])
