"""The decision engine.

One forward pass per (question, ordering); no decode loop. The state is
prefilled once and the question suffixes fan out against a replicated cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from mlx_lm import load

from .calibrate import BucketedTemperature
from .debias import CONTENT_FREE_STATE, PriorCache, debias, shape_key
from .ensemble import combine
from .fanout import batch_fanout
from .labels import build_label_pool, label_token_ids, pick_labels
from . import wide
from .prefix import common_prefix_len
from .prompt import WORD_LABELS, first_token_style, question_suffix, rotate, state_prefix
from .schema import Answer, Question


@dataclass
class Engine:
    model_repo: str
    n_permutations: int = 2
    use_debias: bool = False
    # Injected into any Choice schema that lacks an explicit escape route.
    # Logit readout is worse at abstention than generation, so an explicit
    # "none of these" option plus margin gating is the documented mitigation.
    escape_option: tuple[str, str] | None = None
    batch_size: int = 8
    calibrator: BucketedTemperature | None = None
    # Instruction rephrasings to ensemble over, in addition to option orderings.
    # Binary questions have only two orderings, so option-order ensembling alone
    # yields epistemic ~0.01-0.04 (measured). Perturbing the phrasing restores
    # the signal at any cardinality: epistemic is disagreement under
    # perturbation, and option order is only one kind.
    paraphrases: tuple[str, ...] | None = None
    # Score a separate content-free prior for every option ordering, instead of
    # reusing the canonical one.
    #
    # The shared prior is indexed by label POSITION but was measured with the
    # canonical option in each position, so under rotation it subtracts one
    # option's lexical prior from another option's logit. That is only harmless
    # if the content-free prior is near-flat across descriptions -- an empirical
    # claim. per_rotation_prior keys the prior on the full rendered suffix
    # instead. Off by default because it costs one extra forward pass per
    # ordering rather than one per schema.
    per_rotation_prior: bool = False
    # Close the chat template's auto-opened <think> block so the answer slot is
    # not inside an unclosed reasoning block. (The quire style always closes it.)
    close_thinking: bool = False
    # Binary `noul` questions are read from the logits of " yes" / " no"
    # instead of letter labels over two rendered options. On JevBench's
    # standard-tier noul items the letter path answers by label position
    # (rotating the options flips the answer), which the word readout cannot
    # do -- but it recovered only 2 of those items, so it is off by default.
    noul_words: bool = False
    # "quire" (the released configuration) or "plain"; see prompt.py.
    prompt_style: str = "quire"
    # Choices with more options than this are answered by the per-option
    # fan-out in wide.py. 26 is the bare-letter pool of the quire style.
    wide_cap: int = 26
    # Render the state and its question as ONE user turn. The original split
    # emitted two consecutive user turns with no assistant turn between them,
    # which no chat model is trained on; single_turn keeps the state as a shared
    # prefix so the fan-out is unaffected.
    #
    # Default True on hygiene, not evidence: on kev's transfer-v4 (764 items)
    # it is +0.0092 accuracy with p = 0.30 and ECE unchanged.
    single_turn: bool = True
    # One question with several option orderings: the tokens its suffixes
    # share (the question text) join the prefill once, and only the
    # ordering-specific remainders are batched. The split is taken on the
    # already-tokenised suffixes, so every sequence the model reads is
    # token-for-token what it read before; only the compute (and the reported
    # token count) drops.
    share_question: bool = True

    model: object = field(init=False)
    tokenizer: object = field(init=False)
    label_pool: list[str] = field(init=False)
    _priors: PriorCache = field(init=False)
    # shape_key -> the question that first used it. The key is a digest the
    # scorer cannot invert, and the prior must be scored against the REAL
    # schema, so the engine keeps the mapping.
    _schemas: dict[str, tuple[Question, list[str]]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.model, self.tokenizer = load(self.model_repo)
        self.label_pool = build_label_pool(self.tokenizer)
        self._priors = PriorCache(self._score_content_free)

    def decide(self, state: str, questions: list[Question]) -> list[Answer]:
        """Answer every question against one shared state.

        The state is prefilled once; each question contributes n_permutations
        short suffixes to a single fan-out. `latency_ms` on every returned
        Answer is the wall-clock for the WHOLE call, not a per-question share.
        """
        started = time.perf_counter()
        state_ids = self.tokenizer.encode(
            state_prefix(self.tokenizer, state, single_turn=self.single_turn, style=self.prompt_style))

        # Wide choices are expanded into per-option yes/no sub-questions that
        # ride the same fan-out, then collapsed back (see wide.py).
        cap = self.wide_cap
        flat: list[Question] = []
        groups: list[tuple[int, int]] = []  # (start, count) into `flat` per original question
        for q in questions:
            if wide.is_wide(q, cap):
                subs = wide.expand(q)
                groups.append((len(flat), len(subs)))
                flat.extend(subs)
            else:
                groups.append((len(flat), 1))
                flat.append(q)
        questions_in, questions = questions, flat

        plans = [self._plan(q) for q in questions]
        if self.share_question and len(plans) == 1 and len(plans[0]["suffix_ids"]) > 1:
            seqs = plans[0]["suffix_ids"]
            n = common_prefix_len(seqs)
            state_ids = state_ids + seqs[0][:n]
            plans[0]["suffix_ids"] = [s[n:] for s in seqs]
        suffixes = [s for plan in plans for s in plan["suffix_ids"]]
        logit_rows = batch_fanout(
            self.model, state_ids, suffixes, batch_size=self.batch_size
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        answers, cursor = [], 0
        for question, plan in zip(questions, plans):
            take = len(plan["suffix_ids"])
            answer = self._assemble(
                question, plan, logit_rows[cursor : cursor + take], elapsed_ms
            )
            answer.state_tokens = len(state_ids)
            answer.suffix_tokens = sum(len(s) for s in plan["suffix_ids"])
            answers.append(answer)
            cursor += take
        out = []
        for q, (start, count) in zip(questions_in, groups):
            if count == 1 and not wide.is_wide(q, cap):
                out.append(answers[start])
            else:
                out.append(wide.collapse(q, answers[start:start + count], elapsed_ms))
        return out

    # -- internals ------------------------------------------------------

    def _with_escape(self, question: Question) -> Question:
        if self.escape_option is None or question.kind != "choice":
            return question
        option_id, description = self.escape_option
        if option_id in question.criteria:
            return question
        return Question(
            instructions=question.instructions,
            criteria={**question.criteria, option_id: description},
            kind=question.kind,
        )

    def _effective_options(self, question: Question) -> list[str]:
        return self._with_escape(question).option_ids

    def _paraphrased_instructions(self, question: Question) -> list[str]:
        """The question's instructions, once per paraphrase.

        Only the FIRST LINE is substituted. Everything after it -- including
        the candidate under judgement -- is carried through verbatim, so the
        perturbation is to the phrasing and nothing else.
        """
        if not self.paraphrases:
            return [question.instructions]
        head, _, tail = question.instructions.partition("\n")
        return [p + ("\n" + tail if tail else "") for p in self.paraphrases]

    def _plan(self, question: Question) -> dict:
        question = self._with_escape(question)
        if self.noul_words and question.kind == "noul" and question.n_options == 2:
            labels = list(WORD_LABELS)
            rotations = [list(question.option_ids)]  # the words fix the mapping
        else:
            labels = pick_labels(question.n_options, self.label_pool)
            rotations = [
                rotate(question.option_ids, k)
                for k in range(min(self.n_permutations, question.n_options))
            ]
        token_ids = label_token_ids(self.tokenizer, labels, bare=first_token_style(self.prompt_style))
        variants = self._paraphrased_instructions(question)

        orders, suffix_ids = [], []
        for instructions in variants:
            probe = Question(
                instructions=instructions,
                criteria=question.criteria,
                kind=question.kind,
            )
            for order in rotations:
                orders.append(order)
                suffix_ids.append(
                    self.tokenizer.encode(
                        question_suffix(self.tokenizer, probe, order, labels,
                                        close_thinking=self.close_thinking,
                                        single_turn=self.single_turn, style=self.prompt_style),
                        add_special_tokens=False,
                    )
                )
        return {
            "labels": labels,
            "token_ids": token_ids,
            "orders": orders,
            "suffix_ids": suffix_ids,
        }

    def _assemble(
        self, question: Question, plan: dict, rows: list, elapsed_ms: float
    ) -> Answer:
        shared_prior = None
        if self.use_debias and not self.per_rotation_prior:
            shared_prior = self._prior_for(question, question.option_ids)

        canonical = self._effective_options(question)
        runs = []
        for order, row in zip(plan["orders"], rows):
            logits = np.array([float(row[i]) for i in plan["token_ids"]])
            prior = (
                self._prior_for(question, order)
                if self.per_rotation_prior and self.use_debias
                else shared_prior
            )
            if prior is not None:
                logits = debias(logits, prior[: len(logits)])
            probs = self._to_probs(logits, question.n_options)
            # The prior is indexed by LABEL POSITION, which is what carries the
            # bias, so it is subtracted before re-indexing back to option order.
            by_option = dict(zip(order, probs))
            runs.append(np.array([by_option[o] for o in canonical]))

        result = combine(runs)
        probabilities = dict(zip(canonical, result["probabilities"].tolist()))
        return Answer(
            choice=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
            aleatoric=result["aleatoric"],
            epistemic=result["epistemic"],
            margin=result["margin"],
            latency_ms=elapsed_ms,
            per_permutation=[dict(zip(canonical, r.tolist())) for r in runs],
        )

    def _to_probs(self, logits: np.ndarray, n_options: int) -> np.ndarray:
        if self.calibrator is not None:
            return self.calibrator.transform(logits[None, :], n_options)[0]
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def _prior_for(self, question: Question, order: list[str]) -> np.ndarray:
        key = shape_key(question, order)
        self._schemas.setdefault(key, (question, list(order)))
        return self._priors.get(key)

    def _score_content_free(self, key: str) -> np.ndarray:
        """This schema's content-independent label preference.

        Contextual calibration scores the SAME prompt with a content-free
        state: same instructions, same option descriptions, only the state
        blanked. A generic probe would measure bias on a prompt nobody asks,
        while still being cached under this schema's key.
        """
        question, order = self._schemas[key]
        labels = pick_labels(question.n_options, self.label_pool)
        token_ids = label_token_ids(self.tokenizer, labels, bare=first_token_style(self.prompt_style))
        state_ids = self.tokenizer.encode(
            state_prefix(self.tokenizer, CONTENT_FREE_STATE, single_turn=self.single_turn,
                         style=self.prompt_style)
        )
        suffix = self.tokenizer.encode(
            question_suffix(self.tokenizer, question, order, labels,
                            close_thinking=self.close_thinking,
                            single_turn=self.single_turn, style=self.prompt_style),
            add_special_tokens=False,
        )
        row = batch_fanout(self.model, state_ids, [suffix], batch_size=1)[0]
        return np.array([float(row[i]) for i in token_ids])
