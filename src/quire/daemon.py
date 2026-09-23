"""The Quire HTTP server.

  POST /v1/systemone  the System One wire format (state + typed questions ->
                      probabilities), which JevBench's `typesafe` adapter and
                      the Decision Index `http` engine both speak, so the
                      benchmarks' own runners measure Quire directly.
  POST /decide        Quire's own request shape.
  GET  /health, /v1/models

    quire-serve --backend torch --model Qwen/Qwen3.5-4B --port 8778     # CUDA
    quire-serve --backend mlx --model mlx-community/Qwen3.5-4B-MLX-8bit # Apple Silicon

No authentication: bind it to a private interface or put it behind a proxy.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .schema import Question


class QuestionBody(BaseModel):
    instructions: str
    criteria: dict[str, str]
    kind: str = "choice"


class DecideRequest(BaseModel):
    # a string, a JSON object or a JSON array, as in the System One format
    state: str | dict | list
    questions: list[QuestionBody]


class SystemOneQuestion(BaseModel):
    type: str
    instructions: str
    # choice/noul: {option_id: description or null}. score: an ordered list.
    criteria: dict[str, str | None] | list[str] | None = None


class SystemOneRequest(BaseModel):
    # a string, a JSON object or a JSON array, as in the System One format
    state: str | dict | list
    questions: dict[str, SystemOneQuestion]
    model: str | None = None


def _criteria_for(question: SystemOneQuestion) -> tuple[dict[str, str], list[str]]:
    """Engine criteria plus the option order the response must report back."""
    if question.type == "score":
        levels = list(question.criteria or [])
        if len(levels) < 2:
            raise ValueError("score needs at least two ordered levels")
        ids = [str(i) for i in range(len(levels))]
        return {i: f"{i}: {text}" for i, text in zip(ids, levels)}, ids
    if question.type == "noul":
        given = question.criteria if isinstance(question.criteria, dict) else {}
        ids = ["true", "false"]
        return ({i: f"{i}: {given.get(i) or f'The proposition is {i}.'}" for i in ids}, ids)
    if question.type == "choice":
        given = question.criteria if isinstance(question.criteria, dict) else {}
        if len(given) < 2:
            raise ValueError("choice needs at least two options")
        ids = list(given)
        return {i: f"{i}: {given[i] or i}" for i in ids}, ids
    raise ValueError(f"unknown question type {question.type!r}")


def build_app(engine, calibration=None) -> FastAPI:
    """`calibration`: an optional quire.calibration.TypeTemperature applied per answer type."""
    app = FastAPI(title="quire")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": engine.model_repo}

    @app.post("/decide")
    def decide(request: DecideRequest) -> dict:
        try:
            questions = [
                Question(instructions=q.instructions, criteria=q.criteria, kind=q.kind)
                for q in request.questions
            ]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        answers = engine.decide(request.state, questions)
        return {
            "answers": [
                {
                    "choice": a.choice,
                    # Log every distribution, not just the argmax: you cannot
                    # calibrate or re-threshold without them, and they are
                    # unrecoverable after the fact.
                    "probabilities": a.probabilities,
                    "aleatoric": a.aleatoric,
                    "epistemic": a.epistemic,
                    "margin": a.margin,
                    "latency_ms": a.latency_ms,
                }
                for a in answers
            ]
        }

    @app.get("/v1/models")
    def models() -> dict:
        return {"data": [{"id": "quire-latest", "checkpoint": engine.model_repo,
                          "permutations": engine.n_permutations,
                          "debias": engine.use_debias,
                          "single_turn": engine.single_turn,
                          "close_thinking": engine.close_thinking}]}

    @app.post("/v1/systemone")
    def system_one(request: SystemOneRequest) -> dict:
        try:
            prepared = {qid: _criteria_for(q) for qid, q in request.questions.items()}
            questions = [
                Question(instructions=request.questions[qid].instructions,
                         criteria=criteria, kind="choice")
                for qid, (criteria, _) in prepared.items()
            ]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        answers = engine.decide(request.state, questions)
        out = {}
        # The state is prefilled once for the whole request; every question
        # adds only its suffixes. This is what the engine computed, which is
        # what JevBench prices ("hosted list price x measured tokens").
        total_input = (answers[0].state_tokens if answers else 0) + sum(a.suffix_tokens for a in answers)
        for (qid, (_, ids)), answer in zip(prepared.items(), answers):
            kind = request.questions[qid].type
            probabilities = {i: float(answer.probabilities[i]) for i in ids}
            if calibration is not None:
                probabilities = calibration.apply(probabilities, kind)
            if kind == "noul":
                # TypeSafe reports one number: P(the proposition holds).
                out[qid] = {"type": "noul", "noul": probabilities["true"]}
            elif kind == "choice":
                out[qid] = {"type": "choice", "choice": answer.choice,
                            "probabilities": probabilities,
                            "confidence": float(answer.margin)}
            else:
                mean = sum(int(i) * p for i, p in probabilities.items())
                out[qid] = {"type": "score", "score": mean,
                            "legend": {i: request.questions[qid].criteria[int(i)] for i in ids},
                            "probabilities": probabilities,
                            "confidence": float(answer.margin)}
        return {
            "model": request.model or "quire-latest",
            "answers": out,
            # `confidence` here is this engine's top1-top2 margin. It is NOT a
            # claim to reproduce TypeSafe's confidence, which is undocumented.
            "usage": {"input_tokens": total_input, "output_tokens": 0},
            "latency_ms": answers[0].latency_ms if answers else 0.0,
        }

    return app


def main() -> None:
    import argparse

    import uvicorn

    from .schema import Question

    ap = argparse.ArgumentParser(prog="quire-serve")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument("--model", default=None,
                    help="default: Qwen/Qwen3.5-4B (torch), mlx-community/Qwen3.5-4B-MLX-8bit (mlx)")
    ap.add_argument("--revision", default=None, help="pin the Hub revision (torch backend)")
    ap.add_argument("--permutations", type=int, default=2, help="option orderings averaged per question")
    ap.add_argument("--style", choices=["quire", "plain"], default="quire")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter directory (torch backend); not the released config")
    ap.add_argument("--adapter-scale", type=float, default=1.0)
    ap.add_argument("--calibration", default="default",
                    help="per-answer-type temperatures: 'default' (the released map), 'off', or a JSON file "
                         "from bench/calibration/fit.py; it changes no answer, only the probabilities")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8778)
    args = ap.parse_args()

    if args.backend == "torch":
        from .torch_engine import TorchEngine
        engine = TorchEngine(model_repo=args.model or "Qwen/Qwen3.5-4B", revision=args.revision,
                             n_permutations=args.permutations, prompt_style=args.style,
                             adapter=args.adapter, adapter_scale=args.adapter_scale)
    else:
        if args.adapter:
            ap.error("--adapter is supported on the torch backend only")
        from .engine import Engine
        engine = Engine(model_repo=args.model or "mlx-community/Qwen3.5-4B-MLX-8bit",
                        n_permutations=args.permutations, use_debias=False, prompt_style=args.style)
    engine.decide("warm up", [Question(instructions="ready?", criteria={"a": "yes", "b": "no"})])
    print(f"quire ready on {args.host}:{args.port}  backend={args.backend} model={engine.model_repo} "
          f"orderings={args.permutations} style={args.style} adapter={args.adapter}", flush=True)
    from .calibration import TypeTemperature
    calibration = (None if args.calibration == "off" else TypeTemperature.default() if args.calibration == "default"
                   else TypeTemperature.load(args.calibration))
    if calibration:
        print(f"calibration: {calibration.temperatures} ({args.calibration})", flush=True)
    uvicorn.run(build_app(engine, calibration), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
