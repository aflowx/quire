"""Label items with a frozen teacher model through llama.cpp (CUDA or CPU).

The same prompt as the student (the quire style: letters, thinking closed,
answer read as the first assistant token), two option orders averaged, the
prefix state saved once and restored for each order. Writes one line per item:
the teacher's distribution, the per-order distributions, and whether its argmax
agrees with the gold label. train_lora.py uses a teacher distribution as a soft
target only where it agrees with gold.

    python training/teacher_label_llamacpp.py --gguf Qwen3.8-27B-Q8_0.gguf --suite synth-v1 --split train \
        --out runs/teacher/synth-train.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from jinja2 import Environment  # noqa: E402

from quire.prompt import QUIRE_SYSTEM  # noqa: E402
import corpus  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Teacher:
    def __init__(self, gguf: str, n_ctx: int = 8192):
        from llama_cpp import Llama
        self.llm = Llama(model_path=gguf, n_ctx=n_ctx, n_gpu_layers=-1, n_batch=2048, logits_all=False,
                         verbose=False, flash_attn=True)
        tmpl = self.llm.metadata.get("tokenizer.chat_template")
        if not tmpl:
            raise RuntimeError("GGUF has no chat template")
        self.template = Environment().from_string(tmpl)
        self.letter_ids = {}
        for L in LETTERS:
            ids = self.llm.tokenize(L.encode(), add_bos=False, special=False)
            if len(ids) != 1:
                break
            self.letter_ids[L] = ids[0]

    def render(self, state: str, instructions: str, options: list[tuple[str, str]]) -> tuple[str, str]:
        """(prefix, suffix) text in the quire layout; the prefix ends with the state, shared across orders."""
        msgs = [{"role": "system", "content": QUIRE_SYSTEM}, {"role": "user", "content": "\x00"}]
        rendered = self.template.render(messages=msgs, add_generation_prompt=True, enable_thinking=False,
                                        bos_token="", eos_token="")
        head, tail = rendered.split("\x00", 1)
        lines = "\n".join(f"{L}. {d}" for L, d in options)
        return (f"{head}<state>\n{state}\n</state>\n",
                f"<test>{instructions}</test>\n<options>\n{lines}\n</options>{tail}")

    def slot_logits(self, suffix_tokens: list[int], state) -> np.ndarray:
        import ctypes
        import llama_cpp
        self.llm.load_state(state)
        self.llm.eval(suffix_tokens)
        # Logits of the last evaluated position, read directly from the context:
        # the Python-side `scores` buffer is batch-sized in this build.
        ptr = llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, -1)
        n_vocab = self.llm.n_vocab()
        return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(n_vocab,)).copy()

    def decide(self, state: str, instructions: str, option_ids: list[str], descriptions: list[str]) -> dict:
        n = len(option_ids)
        letters = LETTERS[:n]
        prefix, _ = self.render(state, instructions, list(zip(letters, descriptions)))
        ptoks = self.llm.tokenize(prefix.encode(), add_bos=False, special=True)  # Qwen has no BOS
        self.llm.reset()
        self.llm.eval(ptoks)
        saved = self.llm.save_state()
        runs = []
        suffix_tokens = 0
        for k in range(min(2, n)):
            order = option_ids[k:] + option_ids[:k]
            descs = [descriptions[option_ids.index(o)] for o in order]
            _, suffix = self.render(state, instructions, list(zip(letters, descs)))
            stoks = self.llm.tokenize(suffix.encode(), add_bos=False, special=True)
            suffix_tokens += len(stoks)
            logits = self.slot_logits(stoks, saved)
            picked = np.array([logits[self.letter_ids[L]] for L in letters])
            probs = np.exp(picked - picked.max()); probs /= probs.sum()
            by = dict(zip(order, probs.tolist()))
            runs.append([by[o] for o in option_ids])
        mean = np.mean(runs, axis=0)
        return {"probs": dict(zip(option_ids, mean.tolist())), "per_order": [dict(zip(option_ids, r)) for r in runs],
                "input_tokens": len(ptoks) + suffix_tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", action="append", default=None, help="only these item sources (families)")
    ap.add_argument("--max-tokens", type=int, default=7600, help="skip items whose prompt exceeds this")
    args = ap.parse_args()
    items = corpus.load(args.suite, args.split)
    if args.source:
        items = [i for i in items if i.source in args.source]
    items = [i for i in items if len(i.option_ids) <= 26]  # single-token letter pool; kev has 77-option items
    if args.limit:
        items = items[: args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["uid"] for l in open(args.out) if l.strip()}
        print(f"resuming: {len(done)} already labelled", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t = Teacher(args.gguf)
    print(f"{len(items)} items, {len(t.letter_ids)} single-token letters", flush=True)
    started = time.perf_counter(); n = agree = skipped = 0
    with open(args.out, "a") as f:
        for it in items:
            if it.uid in done:
                continue
            descs = [d or o for o, d in zip(it.option_ids, it.descriptions)]
            try:
                out = t.decide(it.state, it.instructions, list(it.option_ids), descs)
            except ValueError as e:  # context overflow
                skipped += 1
                continue
            top = max(out["probs"], key=out["probs"].get)
            ok = top == it.option_ids[it.label]
            agree += ok; n += 1
            f.write(json.dumps({"uid": it.uid, "source": it.source, "kind": it.kind, "gold": it.option_ids[it.label],
                                "teacher_probs": out["probs"], "per_order": out["per_order"], "agree": ok,
                                "input_tokens": out["input_tokens"]}) + "\n")
            if n % 50 == 0:
                el = time.perf_counter() - started
                print(f"  [{el:6.0f}s] {n} done, {el / n:.2f} s/item, teacher agrees {agree / n:.3f}, skipped {skipped}", flush=True)
    el = time.perf_counter() - started
    print(f"done: {n} items in {el:.0f}s ({el / max(n, 1):.2f} s/item); agreement {agree / max(n, 1):.3f}; skipped {skipped}")


if __name__ == "__main__":
    main()
