# ProofWriter VGCF-2

An individual AI course project / research-style engineering project by
**Ye Lei, Nanjing University**. It studies whether selective formalization,
structural validation, bounded repair and fallback improve reasoning reliability
on ProofWriter. This is a technical portfolio, not a peer-reviewed publication,
a SOTA claim, a trained-model contribution or evidence of universal symbolic
reasoning superiority.

[Public course report](docs/report_public.pdf) — public display copy with only the student ID removed; scientific content is unchanged.

## Pipeline

Natural-language theory + question → constrained formalization → deterministic
structural validator → signed forward-chaining solver. Full VGCF-2 accepts a
hard-valid program; otherwise it attempts one repair and uses the same model's
CoT fallback if the repaired program is still invalid. The model sees exactly
`{theory, question}`. Gold labels, proofs and reference representations are
evaluator-only. Failure to derive a query means Unknown; False requires its
explicit complement. Structural validity does not establish semantic fidelity.

See [architecture](docs/architecture.md) and [IR contract](docs/v2_ir.md).

## Frozen evaluation

VGCF-2.3.1 used `qwen3.5-9b`, non-Thinking mode, temperature 0 and a 3,000-token
response cap on a frozen 300-example sample from ProofWriter OWA depth-5
meta-test. Seven methods were evaluated on the same examples. These are
historical real-model results, not the synthetic smoke results.

| Method | Correct / 300 | Accuracy |
|---|---:|---:|
| Direct | 114 | 38.00% |
| CoT | 236 | 78.67% |
| CoT-Refine | 245 | 81.67% |
| Constrained | 173 | 57.67% |
| Gate+CoT | 263 | 87.67% |
| Repair-only | 224 | 74.67% |
| Full VGCF-2 | 271 | 90.33% |

The frozen primary paired comparisons were H1: Full versus CoT-Refine
(+8.67 percentage points; 41 wins / 15 losses; Holm-adjusted exact McNemar
p = 0.0013711283), and H2: Full versus Gate+CoT (+2.67 points; 10 wins /
2 losses; adjusted p = 0.03857421875). The correction family contains only H1
and H2. Frozen paired bootstrap 95% intervals are [4.00, 13.33] and
[0.67, 5.00] percentage points respectively. Other comparisons are descriptive.

[Minimal frozen summaries](results/frozen_summary.json) support these numbers.
The public verifier checks counts and paired-test arithmetic from those
aggregates; it cannot independently authenticate the underlying predictions.

## Run locally

Use Python 3.12+ from the repository root. No package installation, network,
GPU, LM Studio or LLM API is needed for these commands:

```sh
python -B scripts/check_environment.py
python -B scripts/run_core_tests.py
python -B scripts/run_public_tests.py
python -B scripts/verify_summary.py
python -B scripts/portfolio/run_fixture_smoke.py --project-root .
```

The core suite has 69 deterministic tests. The mock smoke runs 30 synthetic
examples × 5 legacy smoke methods twice (150 records per run), checking zero
errors and identical semantic/aggregate hashes. It is software validation,
not a replication of the seven-method real-model evaluation. Temporary smoke
files are removed automatically; core tests may leave ignored `outputs/` files.

## Reproducibility and limits

Publicly reproducible: parser/solver/validator tests, model-input leakage
sentinels, synthetic routing/persistence/analysis, and aggregate arithmetic.
Raw Final inference is neither bundled nor reproduced. Official data, model
weights, full evidence and archival execution locks are absent. Missing evidence
is reported as unavailable; the smoke fixture is never substituted for it.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for sources and boundaries, and
[EVIDENCE_MAP.md](EVIDENCE_MAP.md) for provenance.

One model, one benchmark and one 300-example frozen sample limit generality.
Static acceptance can admit semantically wrong formalizations. Repair can regress;
fallback and shared calls complicate cost attribution. Exact model revision,
quantization and runtime/weight hashes were not frozen, so bit-identical new
inference is not supported. No new model was trained.

MIT applies to the exported project-authored code and documentation.
See [LICENSE](LICENSE) and [third-party review](THIRD_PARTY.md).
