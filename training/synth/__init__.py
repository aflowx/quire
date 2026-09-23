"""Synthetic long-state decision scenarios with exact gold (docs/PLAN.md W1).

Every scenario is built facts-first: a small structured world is sampled,
the answer is computed by an evaluator over that world, and only then is
the world rendered into prose. The twin of each scenario flips one fact
through the same evaluator, so every pair is contrastive by construction.
No LLM writes anything here; the templates are disjoint from JevBench's
authored items by origin.
"""
