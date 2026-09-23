"""Quire on CUDA: the same decisions as engine.py, in PyTorch.

The state is prefilled once; every question suffix (and every ordering of
its options) runs as a batch over the prefix cache expanded to that batch, at
most `max_batch` at a time. Choices wider than the letter pool take the
per-option fan-out in wide.py. The reported token count (state once plus
every suffix) is the compute.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from .ensemble import combine
from .prefix import share_question_prefix
from .labels import build_label_pool, label_token_ids, pick_labels
from .prompt import WORD_LABELS, first_token_style, question_suffix, rotate, state_prefix
from .schema import Answer, Question


def expand_cache(cache, repeats: int) -> None:
    """Repeat every layer's state `repeats` times along the batch axis.

    DynamicCache does this for attention layers; Qwen3.5's linear-attention
    layers keep conv/recurrent state dicts and lack the method, so they are
    expanded here.
    """
    for layer in cache.layers:
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(repeats)
            continue
        for states in (layer.conv_states, layer.recurrent_states):
            for k, v in states.items():
                if v is not None:
                    states[k] = v.repeat_interleave(repeats, dim=0)


@dataclass
class TorchEngine:
    """The decision engine on CUDA; the same surface as engine.Engine for the server."""

    model_repo: str = "Qwen/Qwen3.5-4B"
    revision: str | None = None
    dtype: str = "bf16"
    n_permutations: int = 2
    prompt_style: str = "quire"
    use_debias: bool = False
    single_turn: bool = True
    close_thinking: bool = False
    noul_words: bool = False
    # Choices wider than this take the per-option fan-out (wide.py).
    wide_cap: int = 26
    # Suffixes per forward; the expanded prefix cache grows with the batch.
    max_batch: int = 32
    # Optional LoRA adapter, merged at load so the forward pass is the plain
    # model's. Not part of the released configuration: on JevBench's public
    # hard tier every adapter tried lost to the frozen model (docs/RESULTS.md).
    adapter: str | None = None
    adapter_scale: float = 1.0
    # One question with several option orderings: the tokens its suffixes
    # share (the question text) join the prefill once, and only the
    # ordering-specific remainders are batched. The split is taken on the
    # already-tokenised suffixes, so every sequence the model reads is
    # token-for-token what it read before; only the compute (and the reported
    # token count) drops.
    share_question: bool = True

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo, revision=self.revision, dtype=dt, device_map="cuda").eval()
        if self.adapter:
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(self.model, self.adapter)
            if self.adapter_scale != 1.0:
                # Interpolate between the frozen model (0) and the adapter (1)
                # by scaling the LoRA delta.
                for module in peft_model.modules():
                    if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                        for key in module.scaling:
                            module.scaling[key] *= self.adapter_scale
            self.model = peft_model.merge_and_unload().eval()
        self.label_pool = build_label_pool(self.tokenizer)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def decide(self, state: str, questions: list[Question]) -> list[Answer]:
        from . import wide
        return wide.decide(self._decide, state, questions, self.wide_cap)

    @torch.no_grad()
    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        started = time.perf_counter()
        prefix = state_prefix(self.tokenizer, state, single_turn=self.single_turn, style=self.prompt_style)
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        bare = first_token_style(self.prompt_style)
        plans = []  # (question, labels, [(order, suffix_ids)])
        for q in questions:
            if self.noul_words and q.kind == "noul" and q.n_options == 2:
                labels, orders = list(WORD_LABELS), [list(q.option_ids)]
            else:
                labels = pick_labels(q.n_options, self.label_pool)
                orders = [rotate(q.option_ids, k) for k in range(min(self.n_permutations, q.n_options))]
            suffixes = [(order, self.tokenizer(question_suffix(
                self.tokenizer, q, order, labels, close_thinking=self.close_thinking,
                single_turn=self.single_turn, style=self.prompt_style), add_special_tokens=False).input_ids)
                for order in orders]
            plans.append((q, labels, suffixes))
        if self.share_question:
            prefix_ids, plans = share_question_prefix(prefix_ids, plans)
        n_suffixes = sum(len(p[2]) for p in plans)
        # The prefix is prefilled once; its cache is expanded to the batch of
        # suffixes and they run in ONE forward. The reported token count
        # (state once + suffixes) is therefore the compute. A single suffix
        # (one question, one ordering) is one forward pass over the whole text.
        flat = [(qi, order, sids) for qi, (_, _, sfx) in enumerate(plans) for order, sids in sfx]
        slot = {}
        if n_suffixes == 1:
            qi, order, sids = flat[0]
            slot[(qi, tuple(order))] = self.model(torch.tensor([prefix_ids + sids]).to(self.device),
                                                  logits_to_keep=1).logits[0, -1].float()
        else:
            cache = self.model(torch.tensor([prefix_ids]).to(self.device), use_cache=True).past_key_values
            by_len: dict[int, list] = {}
            for entry in flat:
                by_len.setdefault(len(entry[2]), []).append(entry)
            for length, whole in by_len.items():
                # At most max_batch suffixes per forward: the expanded prefix
                # cache grows with the batch, and a 255-option wide question on
                # a long state would not fit otherwise.
                for i in range(0, len(whole), self.max_batch):
                    group = whole[i:i + self.max_batch]
                    past = copy.deepcopy(cache)
                    expand_cache(past, len(group))
                    batch = torch.tensor([sids for _, _, sids in group]).to(self.device)
                    logits = self.model(batch, past_key_values=past, use_cache=True,
                                        logits_to_keep=1).logits[:, -1].float()
                    for row, (qi, order, _) in zip(logits, group):
                        slot[(qi, tuple(order))] = row

        answers = []
        for qi, (q, labels, suffixes) in enumerate(plans):
            token_ids = label_token_ids(self.tokenizer, labels, bare=bare)
            runs = []
            for order, suffix_ids in suffixes:
                picked = slot[(qi, tuple(order))][token_ids].cpu().numpy()
                probs = np.exp(picked - picked.max()); probs /= probs.sum()
                by_option = dict(zip(order, probs))
                runs.append(np.array([by_option[o] for o in q.option_ids]))
            out = combine(runs)
            p = dict(zip(q.option_ids, out["probabilities"].tolist()))
            answers.append(Answer(choice=max(p, key=p.get), probabilities=p,
                                  aleatoric=float(out.get("aleatoric", 0.0)),
                                  epistemic=float(out["epistemic"]), margin=float(out["margin"]),
                                  latency_ms=(time.perf_counter() - started) * 1000,
                                  state_tokens=len(prefix_ids),
                                  suffix_tokens=sum(len(s) for _, s in suffixes)))
        return answers
