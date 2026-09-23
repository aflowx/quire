"""Shared-state fan-out over a hybrid linear/full-attention cache.

Qwen3.5 interleaves 3 linear-attention layers per full-attention layer, so
`model.make_cache()` returns a heterogeneous list:

    [ArraysCache(size=2) if l.is_linear else KVCache() for l in layers]

An ArraysCache holds a RECURRENT state -- fixed size regardless of sequence
length, and not rewindable, because processing tokens advances it destructively
with no per-token history. That rules out any "answer a question then rewind"
strategy for 3 of every 4 layers.

Two modes:

  replay -- re-prefill the state per question. Correct by construction, slow.
            The correctness anchor, not a shipping path.
  batch  -- prefill once, replicate the cache B ways via the merge() classmethods
            mlx-lm already provides, run B suffixes in one forward pass.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import can_trim_prompt_cache


def cache_is_rewindable(cache) -> bool:
    """False for Qwen3.5. Kept as a named check so the reason stays visible."""
    return can_trim_prompt_cache(cache)


def answer_slot_logits(model, input_ids: mx.array, cache=None) -> mx.array:
    """Logits at the final position. Shape (B, vocab).

    This is the whole "no decode loop" trick: one forward pass, read the
    logits where the answer would have been generated.
    """
    logits = model(input_ids, cache=cache)
    out = logits[:, -1, :]
    mx.eval(out)
    return out


def replay_fanout(model, state_ids: list[int], suffix_ids: list[list[int]]) -> list[mx.array]:
    """Answer each suffix by re-prefilling the state. Reference implementation.

    Returns one (vocab,) logit vector per suffix.
    """
    results = []
    for suffix in suffix_ids:
        cache = model.make_cache()
        full = mx.array([list(state_ids) + list(suffix)])
        results.append(answer_slot_logits(model, full, cache=cache)[0])
    return results


def snapshot_cache(cache) -> list[tuple]:
    """Materialise the cache state so later in-place writes cannot corrupt it.

    Two distinct aliasing hazards, one per cache type:

    - KVCache.state can return a slice of the buffer update_and_fetch writes
      into. MLX is lazy, so the mx.eval() here is load-bearing, not a
      micro-optimisation: it forces that slice to materialise into its own
      values before the buffer underneath it is mutated again.
    - ArraysCache.state returns `self.cache` itself -- the live, mutable list
      the layer reassigns items into on every forward pass (`cache[0] = ...`).
      Copying it into a tuple is what decouples the snapshot from that list;
      without the tuple() the snapshot would silently follow later writes
      regardless of how thoroughly the arrays inside it are eval'd.
    """
    snap = [tuple(c.state) for c in cache]
    for entry in snap:
        for array in entry:
            if array is not None:
                mx.eval(array)
    return snap


def replicate_cache(cache, batch_size: int) -> list:
    """Replicate a batch-1 state cache `batch_size` ways.

    Uses mlx-lm's own merge() classmethods: KVCache.merge returns a
    BatchKVCache with left-padding, ArraysCache.merge stacks the recurrent
    states. Both copy into fresh buffers, so passing the same cache repeatedly
    is safe.
    """
    return [type(c).merge([c] * batch_size) for c in cache]


def batch_fanout(
    model, state_ids: list[int], suffix_ids: list[list[int]], batch_size: int = 8
) -> list[mx.array]:
    """Prefill the state once, then answer suffixes in batches.

    Suffixes are grouped by token length so that no padding is ever needed.
    That is not an optimisation: 3 of every 4 layers are recurrent and have no
    attention mask, so a pad token would destructively advance the SSM state
    and the batched result would not equal replay.
    """
    by_length: dict[int, list[int]] = {}
    for index, suffix in enumerate(suffix_ids):
        by_length.setdefault(len(suffix), []).append(index)

    # Prefill ONCE, outside the batch loop. Doing it inside meant the state was
    # re-prefilled per length-bucket per batch: free for a 20-token state,
    # ruinous for a whole-log one. Measured on a real pruning workload, that
    # mistake made the shared-state shape 6x SLOWER than per-chunk judgement,
    # when the whole architectural claim is that it should be far faster.
    state_cache = model.make_cache()
    model(mx.array([list(state_ids)]), cache=state_cache)
    mx.eval([c.state for c in state_cache])

    results: list[mx.array | None] = [None] * len(suffix_ids)
    for indices in by_length.values():
        for start in range(0, len(indices), batch_size):
            group = indices[start : start + batch_size]
            rows = _one_batch(model, state_cache, [suffix_ids[i] for i in group])
            for slot, row in zip(group, rows):
                results[slot] = row
    return [r for r in results if r is not None]


def _one_batch(model, state_cache, chunk: list[list[int]]) -> list[mx.array]:
    """One forward pass over `len(chunk)` equal-length suffixes.

    Takes an ALREADY-PREFILLED state cache and replicates it. It must not
    prefill -- that is the caller's job, done once. `replicate_cache` copies
    into fresh buffers, so the source cache survives every batch unmutated.
    """
    batched = replicate_cache(state_cache, len(chunk))
    logits = answer_slot_logits(model, mx.array(chunk), cache=batched)
    return [logits[i] for i in range(len(chunk))]
