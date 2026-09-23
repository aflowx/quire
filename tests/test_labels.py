import pytest

from quire.labels import CANDIDATE_POOL, build_label_pool, pick_labels


class FakeTokenizer:
    """Encodes ' X' as one token for single chars, two tokens for digits.

    Deliberately unlike any real tokenizer: the point is that build_label_pool
    must DERIVE the pool rather than assume a fixed alphabet.
    """

    def encode(self, text, add_special_tokens=False):
        stripped = text.strip()
        if stripped.isdigit():
            return [1, 2]
        return [ord(stripped)]


def test_build_label_pool_excludes_multi_token_candidates():
    pool = build_label_pool(FakeTokenizer())
    assert "A" in pool
    assert "z" in pool
    assert "0" not in pool, "digits encode as 2 tokens for this tokenizer"


def test_build_label_pool_preserves_candidate_order():
    pool = build_label_pool(FakeTokenizer())
    assert pool == [c for c in CANDIDATE_POOL if c in set(pool)]


def test_pick_labels_returns_prefix_of_pool():
    pool = build_label_pool(FakeTokenizer())
    assert pick_labels(3, pool) == pool[:3]


def test_pick_labels_raises_when_pool_too_small():
    with pytest.raises(ValueError, match="exceeds"):
        pick_labels(5, ["A", "B"])


def test_build_label_pool_raises_when_nothing_is_single_token():
    class AllMultiToken:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

    with pytest.raises(RuntimeError, match="no single-token labels"):
        build_label_pool(AllMultiToken())


def test_pick_labels_rejects_negative_n():
    with pytest.raises(ValueError, match="must be >= 1"):
        pick_labels(-1, ["A", "B", "C", "D"])


def test_pick_labels_rejects_zero():
    with pytest.raises(ValueError, match="must be >= 1"):
        pick_labels(0, ["A", "B", "C", "D"])
