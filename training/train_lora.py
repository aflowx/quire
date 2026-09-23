"""LoRA + readout, on CUDA: the arm the Mac cannot run.

The frozen-backbone ablation found the parameter-free letter readout beats a
trained pointer head everywhere (0.761 vs 0.543 on transfer-v4). That does not
refute kev's design, because kev trains its pointer head JOINTLY with a LoRA
adapter, and MLX cannot backprop through Qwen3.5's GatedDeltaNet layers. This
script runs that arm properly and answers whether adaptation rescues it.

Two arms, same base, same data, same LoRA (rank 16, the kev recipe), differing
only in the readout that consumes the backbone:

  lora+letter   read the frozen LM head at the answer slot
  lora+pointer  bilinear between each option's last token and the answer slot

Every other axis is held: prompt, label pool, permutation averaging at eval,
optimiser, schedule, seeds.

    python scripts/train_lora_readout.py --arm lora+pointer --seed 0
"""

import argparse
import datetime
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from quire.labels import build_label_pool, label_token_ids, pick_labels  # noqa: E402
from quire.prompt import first_token_style, question_suffix, state_prefix  # noqa: E402
from quire.schema import Question  # noqa: E402
import corpus  # noqa: E402

ARMS = ("lora+letter", "lora+pointer")


class PointerHead(nn.Module):
    """kev's mechanism: (K h_opt) . (Q h_decide) / sqrt(dp)."""

    def __init__(self, hidden: int, dp: int = 256):
        super().__init__()
        self.q = nn.Linear(hidden, dp)
        self.k = nn.Linear(hidden, dp)
        self.scale = 1.0 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):          # [d], [K, d] -> [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


def encode(tok, pool, item, offsets=True, style="plain"):
    """Prompt ids plus the positions the readouts need."""
    question = Question(instructions=item.instructions,
                        criteria={o: (d or o) for o, d in zip(item.option_ids, item.descriptions)},
                        kind="choice")
    labels = pick_labels(question.n_options, pool)
    prefix = state_prefix(tok, item.state, single_turn=True, style=style)
    suffix = question_suffix(tok, question, list(question.option_ids), labels, single_turn=True, style=style)
    text = prefix + suffix
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=offsets)
    ids = enc["input_ids"]
    positions = []
    if offsets:
        # last token of each option line, found by character span
        cursor = len(prefix)
        body = suffix
        for label, option_id, description in zip(labels, item.option_ids, item.descriptions):
            line = f"{label}. {description or option_id}"
            start = cursor + body.index(line)
            end = start + len(line)
            inside = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
            positions.append(inside[-1])
    return ids, positions, label_token_ids(tok, labels, bare=first_token_style(style)), item.label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None, help="dev items to score")
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--style", default="quire", choices=["quire", "plain"],
                   help="prompt the adapter is trained on; must match how it is served")
    p.add_argument("--suite", action="append", default=None,
                   help="suite:split[:limit], repeatable; default decision-v7:train")
    p.add_argument("--dev-suite", action="append", default=None,
                   help="suite:split[:limit] for evaluation; default decision-v7:development")
    p.add_argument("--teacher", action="append", default=None,
                   help="teacher_label.py output; agreement-gated soft targets for items it covers")
    p.add_argument("--soft-weight", type=float, default=1.0, help="weight of the soft-target CE term")
    p.add_argument("--max-tokens", type=int, default=8192, help="skip longer prompts")
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt if present")
    p.add_argument("--ckpt-every", type=int, default=1000, help="items between checkpoints")
    args = p.parse_args()
    out = args.out or pathlib.Path("runs/lora") / f"{args.arm.replace('+', '-')}-s{args.seed}"
    if (out / "result.json").exists():
        sys.exit(f"{out} exists; remove it to re-run")
    resume_from = None
    if args.resume and (out / "ckpt" / "progress.json").exists():
        resume_from = json.loads((out / "ckpt" / "progress.json").read_text())
        print(f"resuming from epoch {resume_from['epoch']} item {resume_from['k']} (seen {resume_from['seen']})", flush=True)

    torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    base.config.use_cache = False
    hidden = base.config.text_config.hidden_size if hasattr(base.config, "text_config") else base.config.hidden_size

    # Target every projection the architecture has, attention and DeltaNet
    # alike, as kev does ("the adapter also covers the DeltaNet projections").
    names = {n.split(".")[-1] for n, m in base.named_modules() if isinstance(m, nn.Linear)}
    targets = sorted(n for n in names if any(k in n for k in
                     ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                      "down_proj", "in_proj", "out_proj")))
    model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0, bias="none",
        target_modules=targets, task_type="CAUSAL_LM"))
    if resume_from is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(out / "ckpt" / "adapter" / "adapter_model.safetensors")))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = PointerHead(hidden).cuda().to(torch.bfloat16) if args.arm == "lora+pointer" else None
    if head is not None:
        trainable += sum(p.numel() for p in head.parameters())
    print(f"{args.arm} seed {args.seed}: LoRA targets {targets}\n  trainable {trainable/1e6:.2f}M", flush=True)

    pool = build_label_pool(tok)
    need_positions = args.arm == "lora+pointer"
    def load_specs(specs, default):
        out = []
        for spec in specs or [default]:
            parts = spec.split(":")
            suite, split = parts[0], parts[1]
            picked = [i for i in corpus.load(suite, split) if len(i.option_ids) <= len(pool)]
            if len(parts) > 2 and parts[2]:
                rng0 = np.random.default_rng(args.seed)
                picked = [picked[j] for j in sorted(rng0.permutation(len(picked))[: int(parts[2])])]
            print(f"  {suite}:{split}: {len(picked)} items", flush=True)
            out.extend(picked)
        return out

    train = load_specs(args.suite, "decision-v7:train")
    if args.max_train:
        train = train[: args.max_train]
    dev = load_specs(args.dev_suite, "decision-v7:development")
    if args.eval_limit:
        dev = dev[: args.eval_limit]
    teacher = {}
    for path in args.teacher or []:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                if row.get("agree"):
                    teacher[row["uid"]] = row["teacher_probs"]
    print(f"  train {len(train)}  dev {len(dev)}  teacher-covered {sum(i.uid in teacher for i in train)}  "
          f"exact-soft {sum(i.soft is not None for i in train)}", flush=True)
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    params = [p for p in model.parameters() if p.requires_grad] + \
             (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = args.epochs * len(train) // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1),
                                                pct_start=0.03)

    # The pointer arm needs the final hidden state. output_hidden_states=True
    # materialises all 32 layers; a hook on the final norm keeps one.
    captured = {}
    if need_positions:
        inner = model.base_model.model
        for attr in ("language_model", "model"):
            inner = getattr(inner, attr, inner)
        target = inner.norm if hasattr(inner, "norm") else inner
        target.register_forward_hook(lambda _m, _i, o: captured.__setitem__("h", o))

    def logits_for(ids, positions, label_tokens, grad=True):
        t = torch.tensor([ids], device="cuda")
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = model(t)
            if need_positions:
                h = captured["h"][0]
                return head(h[-1], h[torch.tensor(positions, device="cuda")])
            return out.logits[0, -1][torch.tensor(label_tokens, device="cuda")].float()

    rng = np.random.default_rng(args.seed)
    started, seen, skipped_long = time.perf_counter(), 0, 0
    if resume_from is not None:
        seen = resume_from["seen"]
        # Replay the optimizer schedule to where it was; the permutation is
        # seeded, so the item order is identical.
        for _ in range(seen // args.accum):
            sched.step()

    def checkpoint(epoch, k):
        (out / "ckpt").mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out / "ckpt" / "adapter"))
        (out / "ckpt" / "progress.json").write_text(json.dumps({"epoch": epoch, "k": k, "seen": seen}))

    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        if resume_from is not None and epoch < resume_from["epoch"]:
            continue
        for k, idx in enumerate(order):
            if resume_from is not None and epoch == resume_from["epoch"] and k <= resume_from["k"]:
                continue
            item = train[idx]
            ids, positions, label_tokens, gold = encode(tok, pool, item, need_positions, style=args.style)
            if len(ids) > args.max_tokens:
                skipped_long += 1
                continue
            logits = logits_for(ids, positions, label_tokens)
            loss = F.cross_entropy(logits[None, :].float(), torch.tensor([gold], device="cuda"))
            # Soft target: the exact gold distribution where the item has one
            # (probability family), else the teacher's distribution where it
            # agreed with gold (Winnow's gate). Never on items the teacher got wrong.
            soft = item.soft
            if soft is None and item.uid in teacher:
                tp = teacher[item.uid]
                soft = tuple(float(tp.get(o, 0.0)) for o in item.option_ids)
            if soft is not None and args.soft_weight > 0:
                target = torch.tensor(soft, device="cuda").float()
                target = target / target.sum()
                loss = loss + args.soft_weight * F.cross_entropy(logits[None, :].float(), target[None, :])
            loss = loss / args.accum
            loss.backward()
            seen += 1
            if seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if seen % 500 == 0:
                print(f"  [{time.perf_counter()-started:6.0f}s] epoch {epoch} {k+1}/{len(train)} "
                      f"loss {loss.item()*args.accum:.4f}", flush=True)
            if seen % args.ckpt_every == 0:
                checkpoint(epoch, k)

    model.eval()
    if head is not None:
        head.eval()
    correct, rows = 0, []
    for item in dev:
        ids, positions, label_tokens, gold = encode(tok, pool, item, need_positions, style=args.style)
        if len(ids) > args.max_tokens:
            continue
        logits = logits_for(ids, positions, label_tokens, grad=False).float().cpu().numpy()
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        correct += int(probs.argmax()) == gold
        rows.append({"uid": item.uid, "kind": item.kind, "source": item.source, "suite": item.suite,
                     "probs": probs.tolist(), "label": gold})
    accuracy = correct / max(1, len(rows))
    by_source = {}
    for row in rows:
        b = by_source.setdefault(f"{row['suite']}/{row['source']}", [0, 0])
        b[0] += int(np.argmax(row["probs"]) == row["label"]); b[1] += 1
    print("  dev by source: " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in sorted(by_source.items())), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({
        "_provenance": {"arm": args.arm, "model": args.model, "seed": args.seed, "rank": args.rank,
                        "lr": args.lr, "epochs": args.epochs, "accum": args.accum,
                        "lora_targets": targets, "trainable_params": trainable,
                        "n_train": len(train), "device": torch.cuda.get_device_name(0),
                        "style": args.style, "suites": args.suite, "dev_suites": args.dev_suite,
                        "teacher": args.teacher, "soft_weight": args.soft_weight, "max_tokens": args.max_tokens,
                        "skipped_long": skipped_long,
                        "dev_by_source": by_source,
                        "train_seconds": time.perf_counter() - started,
                        "generated": datetime.datetime.now(datetime.UTC).isoformat()},
        "dev_accuracy": accuracy, "rows": rows}, indent=2))
    model.save_pretrained(str(out / "adapter"))
    if head is not None:
        torch.save(head.state_dict(), out / "head.pt")
    print(f"\n{args.arm} seed {args.seed}: dev accuracy {accuracy:.4f} "
          f"({time.perf_counter()-started:.0f}s)\nwrote {out}")


if __name__ == "__main__":
    main()
