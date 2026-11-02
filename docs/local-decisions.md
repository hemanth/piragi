# Local evidence decisions with SemIf / OpenJev

Piragi follows the retrieval → evidence judgment → generation design described in
[Akshay Pachaar's RAG post](https://x.com/akshay_pachaar/status/2105281570898849903).
The local implementation uses [SemIf (formerly OpenJev)](https://github.com/TheoLeeCJ/SemIf-OpenJev),
an independent open-model project. It is not TypeSafe's hosted Jev model and does
not inherit its latency or calibration claims.

```python
from piragi import Ragi

kb = Ragi("./docs", config={"decision": {"engine": "openjev"}})
answer = kb.ask("How can I restrict search to my team's documents?")
print(answer.text)
print(answer.decision)  # relevance, retained IDs, answerability, status, timing
```

## Runtime

Install SemIf separately into a compatible environment, following its
[upstream installation instructions](https://github.com/TheoLeeCJ/SemIf-OpenJev#quick-start).
Use the MLX extra on Apple Silicon. SemIf pins ML dependencies, so a separate
runtime is recommended when its versions differ from your existing application.

The adapter lazily loads `Qwen/Qwen3.5-4B`, pinned at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Missing assets download on the first
nonempty evidence judgment. The source checkpoint is approximately 9 GB even
when using MLX's in-memory 4-bit quantization (the default). Quantization reduces
loaded weights, not download size; loading and batched activations still need
additional memory. No performance improvement has been established for this flow.

Automatic selection prefers installed MLX/Metal on Apple Silicon, then Torch on
CUDA, MPS or CPU. Torch does not use the MLX quantization setting. Model loading
and scoring are serialized per engine to support shared sync/async callers.

Optional overrides include `model_path` (existing checkpoint directory),
`revision`, `backend`, `device`, `dtype`, `cache_dir`, `local_files_only`,
`max_tokens`, and `mlx_bits` (4, 8, or None). A prequantized MLX checkpoint requires
`mlx_bits=None`; the upstream loader only accepts native Qwen3.5 checkpoints.
`local_files_only=True` prevents downloading a missing default checkpoint.
An explicit local path bypasses downloads and is validated at construction.

## Evidence flow

1. Selecting the engine enables hybrid retrieval and disables query expansion by
   default. Explicit retrieval/expansion settings take precedence. The preset is
   disabled unless selected; explicit `decision.enabled=False` wins.
2. `ask()` retrieves up to `max(decision.candidate_k, top_k)` candidates, with
   `candidate_k=20` by default. The default Lance store retrieves BM25 candidates
   independently of dense hits and combines ranks using RRF. Metadata filters
   apply to both candidate pools. It reads the current local corpus per query;
   this implementation is intended for modest local collections. Custom stores
   without `get_all_citations(filters=...)` retain their existing hybrid behavior.
3. Independent yes/no relevance questions are sent to SemIf's shared-state scorer.
   This reuses a query prefix and batches suffixes on MLX/CUDA. Upstream Torch MPS
   uses cached serial suffixes. It is not a loop of full model calls per passage.
4. Code retains positive judgments meeting `min_score` (default 0.7) and
   `min_margin` (default 0.2), sorts by their yes scores, and caps at `top_k`.
   These thresholds are provisional, uncalibrated settings.
5. A second, dependent decision checks answerability against exactly those retained
   passages, after hierarchical parent expansion and the final context cap.
   Graph context, if enabled, goes through the same judge as a candidate.
6. Insufficient or uncertain evidence returns a fixed insufficient-evidence answer
   with empty citations and no answer-model call. A valid positive decision passes
   the surviving passages to the configured answer LLM.

Empty retrieval skips model loading. Plain `retrieve()` returns search results
without evidence judgments or answer generation. Explicit HyDE runs before search;
judgments always use the original question. Citation retrieval scores remain intact;
`citation.metadata['evidence_score']` contains the judge's relevance score.

`answer.decision['status']` distinguishes `accepted`, `abstained`, and `unavailable`.
Missing runtime, oversized prompts, malformed results, or scoring errors produce
an explicit `unavailable` report and fall back to the original candidate ranking.
They are not recorded as successful model decisions or claims of absent evidence.
No separate prompt-injection classifier is included. These semantic judgments are
not an access-control or security boundary.

The exported `DecisionEngine` protocol supports both `decide()` and
`decide_many(state=..., questions=...)`. Each `DecisionQuestion` declares its
question and option descriptions, and each response is a validated `DecisionResult`.
Standalone calls raise errors; the evidence integration records them and falls back.
A batch's elapsed time is shared across results and must not be summed per question.

## Evaluation

`evals/eval_openjev.py` retrieves a frozen hybrid candidate pool once per query and
passes the same candidates to both conditions. It records passage recall and
precision, programmatic abstentions, false abstentions, context size, answer calls,
latency, raw judgments, and generated answers. It rejects inference fallbacks and
summarizes only completed pairs. Retrieval time is measured once and added equally
to each condition's separately measured judgment/generation time.

```sh
python evals/eval_openjev.py --output /tmp/piragi-evidence-eval --repeats 1
```

Use `--model-path` for an existing checkpoint and `--skip-generation` to evaluate
real evidence decisions without running an answer model. The authored development
fixture includes adjacent distractors and unanswerable queries. It is not a held-out
production benchmark, and the script does not automatically grade answer correctness.
Tune thresholds on separate data before drawing quality conclusions.

The earlier results under `evals/results/openjev-20260930-run2` measure the abandoned
query-expansion gate. Their saved runner and fixture preserve that experiment; they
do not evaluate the new evidence flow. Integration tests use fake scores and establish
behavior only. Downloaded evaluation models were removed after the earlier pilot.
