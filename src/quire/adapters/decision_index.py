"""Quire as a Decision Index engine (github.com/apolinario/decision-index).

    python -m decision_index run --engine quire.adapters.decision_index:QuireEngine \\
        --option model=mlx-community/Qwen3.5-4B-MLX-8bit --rows ... --out runs/quire

Unmodified state and questions go to one Engine.decide() call (state prefilled
once, every question a suffix in the fan-out). Choices wider than the 26-letter
bare-letter pool take the per-option fan-out in wide.py, so no request is
refused for width and none is truncated or filtered. Rendering is the fixed
quire-style prompt for every benchmark: no per-benchmark prompts.
"""

from __future__ import annotations

import json
import time

from decision_index.engines.base import Engine, text

from ..engine import Engine as Decider
from ..schema import Question


class QuireEngine(Engine):
    name = "quire-mlx"
    latency = "In-process wall time of one Engine.decide() call per request, including prompt construction; excludes model loading."

    def __init__(self, model="mlx-community/Qwen3.5-4B-MLX-8bit", permutations=2, style="quire", **options):
        super().__init__(model=model, permutations=permutations, style=style, **options)
        from ..calibration import TypeTemperature
        self.calibration = TypeTemperature.default()
        self.decider = Decider(model_repo=model, n_permutations=int(permutations), use_debias=False,
                               prompt_style=style)
        self.provenance = {
            "kind": "inference technique", "model": model, "permutations": int(permutations), "prompt_style": style,
            "policy": "Frozen model; letter logits at the first assistant token, orderings averaged; choices wider "
                      "than 26 options answered by a per-option yes/no fan-out normalised to one distribution. "
                      "No truncation, no option filtering, no per-benchmark prompt.",
        }

    def __call__(self, state, questions):
        keys = list(questions)
        qs = []
        for k in keys:
            q = questions[k]
            crit = q.get("criteria") or {}
            qs.append(Question(instructions=text(q["instructions"]),
                               criteria={o: text(d) if d is not None else o for o, d in crit.items()},
                               kind="choice"))
        started = time.perf_counter()
        answers = self.decider.decide(text(state) if state not in (None, "") else "(no state)", qs)
        wall_ms = (time.perf_counter() - started) * 1000
        out, used = {}, (answers[0].state_tokens if answers else 0)
        for k, a in zip(keys, answers):
            probs = {o: float(p) for o, p in a.probabilities.items()}
            total = sum(probs.values())
            probs = {o: p / total for o, p in probs.items()}
            probs = self.calibration.apply(probs, "choice")   # every Decision Index question is a choice
            out[k] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
            used += a.suffix_tokens
        response = {"model": "quire", "answers": out, "usage": {"input_tokens": used, "output_tokens": 0}}
        return response, {"wall_ms": wall_ms, "margins": [a.margin for a in answers]}

    def runtime(self):
        return {"model": self.options["model"], "permutations": self.options["permutations"],
                "style": self.options["style"], "wide_cap": self.decider.wide_cap}


class QuireTorchEngine(QuireEngine):
    """The same adapter on CUDA (torch_engine.TorchEngine): batched fan-out, wide path."""

    name = "quire-torch"

    def __init__(self, model="Qwen/Qwen3.5-4B", permutations=2, style="quire", revision=None, **options):
        from ..torch_engine import TorchEngine
        Engine.__init__(self, model=model, permutations=permutations, style=style, revision=revision, **options)
        from ..calibration import TypeTemperature
        self.calibration = TypeTemperature.default()
        self.decider = TorchEngine(model_repo=model, revision=revision, n_permutations=int(permutations),
                                   prompt_style=style)
        self.provenance = {
            "kind": "inference technique", "model": model, "revision": revision,
            "permutations": int(permutations), "prompt_style": style,
            "runtime": "torch bf16, prefix prefilled once, orderings and sub-questions batched over the expanded cache",
            "policy": "Frozen model; letter logits at the first assistant token, orderings averaged; choices wider "
                      "than 26 options answered by a per-option yes/no fan-out normalised to one distribution. "
                      "No truncation, no option filtering, no per-benchmark prompt.",
        }

    def synchronize(self):
        import torch
        torch.cuda.synchronize()
