import mlx.core as mx
import pytest

from quire.fanout import answer_slot_logits, cache_is_rewindable, replay_fanout

MODEL_REPO = "mlx-community/Qwen3.5-0.8B-8bit"


@pytest.fixture(scope="module")
def loaded():
    from mlx_lm import load

    return load(MODEL_REPO)


@pytest.mark.model
def test_qwen35_cache_is_not_rewindable(loaded):
    """Documents why replay exists. If this ever fails, revisit fanout.py."""
    model, _ = loaded
    assert cache_is_rewindable(model.make_cache()) is False


@pytest.mark.model
def test_cache_is_heterogeneous_linear_and_full_attention(loaded):
    """3 linear layers per full-attention layer is the fact fanout.py is built on."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    model, _ = loaded
    cache = model.make_cache()
    n_linear = sum(isinstance(c, ArraysCache) for c in cache)
    n_full = sum(isinstance(c, KVCache) for c in cache)
    assert n_linear + n_full == len(cache)
    assert n_full > 0 and n_linear > 0, "expected a hybrid cache"
    assert n_linear == 3 * n_full, f"expected 3:1 linear:full, got {n_linear}:{n_full}"


@pytest.mark.model
def test_answer_slot_logits_returns_one_row_per_sequence(loaded):
    model, tokenizer = loaded
    ids = mx.array([tokenizer.encode("The capital of France is")])
    logits = answer_slot_logits(model, ids, cache=None)
    assert logits.shape[0] == 1
    assert logits.shape[1] == model.args.text_config["vocab_size"]


@pytest.mark.model
def test_answer_slot_is_read_at_the_correct_position(loaded):
    """The readout must come from the LAST position, not an off-by-one.

    An off-by-one here would silently score the wrong slot and every
    downstream probability would be garbage while still looking well-formed,
    so assert against a fact the model definitely knows.
    """
    model, tokenizer = loaded
    ids = mx.array([tokenizer.encode("The capital of France is")])
    logits = answer_slot_logits(model, ids, cache=None)
    top5 = mx.argsort(-logits[0])[:5].tolist()
    decoded = [tokenizer.decode([t]).strip().lower() for t in top5]
    assert "paris" in decoded, f"expected 'paris' in top-5, got {decoded}"


@pytest.mark.model
def test_replay_fanout_matches_standalone_forward_passes(loaded):
    model, tokenizer = loaded
    state_ids = tokenizer.encode("<state>\nthe build failed at step 3\n</state>\n")
    suffixes = [
        tokenizer.encode(s, add_special_tokens=False)
        for s in ("Did it fail? Answer:", "Was it step 3? Answer:")
    ]

    fanned = replay_fanout(model, state_ids, suffixes)

    assert len(fanned) == len(suffixes)
    for i, suffix in enumerate(suffixes):
        standalone = answer_slot_logits(
            model, mx.array([state_ids + suffix]), cache=model.make_cache()
        )
        assert mx.allclose(fanned[i], standalone[0], atol=1e-3)


@pytest.mark.model
def test_replay_fanout_gives_different_answers_to_different_questions(loaded):
    """Guards against a wiring bug where every suffix returns the same row."""
    model, tokenizer = loaded
    state_ids = tokenizer.encode("<state>\nthe sky is blue and grass is green\n</state>\n")
    suffixes = [
        tokenizer.encode(s, add_special_tokens=False)
        for s in ("What colour is the sky? Answer:", "What colour is grass? Answer:")
    ]
    fanned = replay_fanout(model, state_ids, suffixes)
    assert not mx.allclose(fanned[0], fanned[1], atol=1e-2), (
        "different questions produced identical logits; suffixes are not reaching the model"
    )


def _equal_length_suffixes(tokenizer):
    """Three suffixes padded by construction to identical token length."""
    raw = ["Is it red? Answer:", "Is it big? Answer:", "Is it new? Answer:"]
    encoded = [tokenizer.encode(s, add_special_tokens=False) for s in raw]
    assert len({len(e) for e in encoded}) == 1, f"lengths differ: {[len(e) for e in encoded]}"
    return encoded


def _label_probs(logits, tokenizer, letters="ABCDEFGHIJ"):
    """Softmax over the option-label tokens -- what the engine actually consumes."""
    ids = [tokenizer.encode(" " + c, add_special_tokens=False)[0] for c in letters]
    return mx.softmax(mx.array([logits[i] for i in ids]))


def _assert_same_decision(
    actual, expected, tokenizer, context, max_dp=0.10, margin_floor=0.05
):
    """Batched fan-out must agree with replay on DECISIONS, not bit patterns.

    Raw logits cannot match bitwise: any prefill-then-continue split changes
    accumulation order in the 18 recurrent layers, which on this 8-bit model
    shifts logits by ~1-2% of their scale (bounded, plateauing with state
    length). That is numerical, not semantic.

    Two separate claims are asserted:

      * probabilities always agree within `max_dp`; and
      * the chosen option agrees whenever the decision is not a near-tie.

    `max_dp` is 0.10 by derivation, not by fitting. For two labels at a tie,
    d(softmax)/d(logit) is 1/4, so the measured logit drift of up to ~0.4
    admits a probability swing of ~0.1. A 50-pair sample saw max 0.0586, all
    of it on exact ties -- consistent with that bound. Setting it at the
    observed maximum instead would flake on the next tied pair.

    The margin gate is not a fudge. Measured over 50 varied pairs, argmax
    agreement is 96%, and both disagreements were near-ties (|dp| 0.014 and
    0.027) where a ~1% logit shift reorders two labels already neck-and-neck.
    Production gates abstention on the top1-top2 margin for exactly this
    reason, so a flip below the floor is a decision the engine refuses anyway.
    A flip ABOVE the floor would be a genuine bug, and this still catches it.
    """
    for i, (got, want) in enumerate(zip(actual, expected)):
        pa, pb = _label_probs(got, tokenizer), _label_probs(want, tokenizer)
        mx.eval(pa, pb)

        dp = float(mx.max(mx.abs(pa - pb)))
        assert dp < max_dp, f"{context} row {i}: label probabilities differ by {dp:.4f}"

        ordered = mx.sort(pb)[::-1]
        mx.eval(ordered)
        margin = float(ordered[0] - ordered[1])
        if margin > margin_floor:
            assert int(mx.argmax(pa)) == int(mx.argmax(pb)), (
                f"{context} row {i}: chose a different option than replay at "
                f"margin {margin:.4f}, which is above the abstention floor"
            )


@pytest.mark.model
def test_batch_fanout_agrees_with_replay_equal_lengths(loaded):
    """No padding involved. A failure here means cache replication is wrong."""
    from quire.fanout import batch_fanout

    model, tokenizer = loaded
    state_ids = tokenizer.encode("<state>\nthe build failed at step 3\n</state>\n")
    suffixes = _equal_length_suffixes(tokenizer)

    expected = replay_fanout(model, state_ids, suffixes)
    actual = batch_fanout(model, state_ids, suffixes)

    assert len(actual) == len(expected)
    _assert_same_decision(actual, expected, tokenizer, "equal-length")


@pytest.mark.model
def test_batch_fanout_agrees_with_replay_ragged(loaded):
    """Ragged lengths are grouped, never padded, so this must also hold."""
    from quire.fanout import batch_fanout

    model, tokenizer = loaded
    state_ids = tokenizer.encode("<state>\nthe build failed at step 3\n</state>\n")
    suffixes = [
        tokenizer.encode(s, add_special_tokens=False)
        for s in (
            "Did it fail? Answer:",
            "Was the failure at step three of the build pipeline? Answer:",
            "Green? Answer:",
        )
    ]
    assert len({len(s) for s in suffixes}) > 1, "this test needs ragged lengths"

    expected = replay_fanout(model, state_ids, suffixes)
    actual = batch_fanout(model, state_ids, suffixes)
    _assert_same_decision(actual, expected, tokenizer, "ragged")


@pytest.mark.model
def test_batch_fanout_preserves_suffix_order_under_length_grouping(loaded):
    """Grouping reorders work internally; results must come back in input order.

    A silent permutation here would attach every answer to the wrong question
    while every probability still looked perfectly well-formed.
    """
    from quire.fanout import batch_fanout

    model, tokenizer = loaded
    state_ids = tokenizer.encode("<state>\nthe sky is blue and grass is green\n</state>\n")
    # Deliberately interleaved lengths so grouping cannot coincide with input order.
    suffixes = [
        tokenizer.encode(s, add_special_tokens=False)
        for s in (
            "What colour is the sky? Answer:",
            "Grass? Answer:",
            "What colour is the grass in this state? Answer:",
            "Sky? Answer:",
        )
    ]
    expected = replay_fanout(model, state_ids, suffixes)
    actual = batch_fanout(model, state_ids, suffixes)

    assert len(actual) == len(suffixes)
    _assert_same_decision(actual, expected, tokenizer, "order-preservation")


@pytest.mark.model
def test_batch_fanout_respects_batch_size_boundaries(loaded):
    """Decisions must not depend on how suffixes are chunked.

    Chunking changes the row count of the continuation forward pass, which
    perturbs logits by the same ~1% as any other prefill/continue split. What
    must not change is which option is chosen or what probability it carries.
    """
    from quire.fanout import batch_fanout

    model, tokenizer = loaded
    state_ids = tokenizer.encode("<state>\nalpha bravo charlie delta\n</state>\n")
    suffixes = _equal_length_suffixes(tokenizer)

    one_batch = batch_fanout(model, state_ids, suffixes, batch_size=8)
    split_batches = batch_fanout(model, state_ids, suffixes, batch_size=2)

    _assert_same_decision(split_batches, one_batch, tokenizer, "batch-size")


@pytest.mark.model
def test_prefill_continue_diverges_from_one_shot_but_stays_bounded(loaded):
    """Documents the numerical property that forced decision-level assertions.

    Splitting a forward pass changes accumulation order in the 18 recurrent
    layers. The shift is real but bounded at a few percent of logit scale, and
    it must not flip the argmax. If this ever exceeds the bound, the
    decision-level tolerances above need revisiting.
    """
    model, tokenizer = loaded
    state = tokenizer.encode("<state>\n" + "the build failed at step 3. " * 20 + "\n</state>\n")
    suffix = tokenizer.encode("Did it fail? Answer:", add_special_tokens=False)

    cache_a = model.make_cache()
    one_shot = model(mx.array([state + suffix]), cache=cache_a)[:, -1, :]

    cache_b = model.make_cache()
    model(mx.array([state]), cache=cache_b)
    continued = model(mx.array([suffix]), cache=cache_b)[:, -1, :]
    mx.eval(one_shot, continued)

    scale = float(mx.max(mx.abs(one_shot[0])))
    drift = float(mx.max(mx.abs(one_shot[0] - continued[0])))
    assert drift / scale < 0.05, f"drift {drift:.4f} is {100*drift/scale:.2f}% of scale {scale:.1f}"
    assert int(mx.argmax(one_shot[0])) == int(mx.argmax(continued[0])), "drift flipped the argmax"


@pytest.mark.model
def test_cache_snapshot_survives_later_in_place_writes(loaded):
    """Lazy-aliasing regression.

    KVCache.update_and_fetch writes in place into its buffer, and KVCache.state
    returns a slice of that buffer. Under MLX's lazy evaluation an unevaluated
    snapshot can be corrupted by a later write. snapshot_cache must mx.eval().
    """
    from quire.fanout import snapshot_cache

    model, tokenizer = loaded
    cache = model.make_cache()
    state_ids = tokenizer.encode("<state>\nalpha bravo charlie\n</state>\n")
    model(mx.array([state_ids]), cache=cache)

    snap = snapshot_cache(cache)
    kv_index = next(i for i, entry in enumerate(snap) if len(entry) == 2 and entry[0] is not None)
    before = [mx.array(a) for a in snap[kv_index]]

    # Advance the ORIGINAL cache; the snapshot must not move with it.
    model(mx.array([tokenizer.encode(" delta", add_special_tokens=False)]), cache=cache)

    for b, s in zip(before, snap[kv_index]):
        assert mx.array_equal(b, s), "snapshot was corrupted by a later in-place write"


@pytest.mark.model
def test_ragged_suffixes_do_not_re_prefill_a_large_state(loaded):
    """The regression test for the architecture's whole point.

    Suffixes are grouped by length, so ragged lengths produce many batches.
    If the state is prefilled per batch rather than once, a large shared state
    is re-paid for every distinct suffix length -- which measured 6x SLOWER
    than a tiny state, exactly inverting the claimed speedup.

    Compares a LARGE state against a TINY one over the same ragged suffixes.
    Prefilled once, the large state costs a fixed extra amount; prefilled per
    batch, it scales with the number of distinct lengths.
    """
    import time

    from quire.fanout import batch_fanout

    model, tokenizer = loaded
    big_state = tokenizer.encode("<state>\n" + "the build failed at step 3. " * 120 + "\n</state>\n")
    small_state = tokenizer.encode("<state>\nbuild failed\n</state>\n")
    # Deliberately ragged: 12 distinct lengths => 12 separate batches.
    suffixes = [
        tokenizer.encode("q " * n + "Answer:", add_special_tokens=False)
        for n in range(1, 13)
    ]

    t = time.perf_counter()
    batch_fanout(model, small_state, suffixes)
    small_seconds = time.perf_counter() - t

    t = time.perf_counter()
    batch_fanout(model, big_state, suffixes)
    big_seconds = time.perf_counter() - t

    ratio = big_seconds / small_seconds
    print(f"\nragged fan-out: small_state={small_seconds:.2f}s big_state={big_seconds:.2f}s "
          f"ratio={ratio:.2f}x  ({len(suffixes)} distinct lengths)")
    assert ratio < 6.0, (
        f"large state cost {ratio:.1f}x the small one over {len(suffixes)} "
        "length buckets; the state is likely being re-prefilled per batch"
    )
