# Architecture and invariants

## Components

1. `data.py` adapts known layouts into evaluator `Example` objects, creates a
   gold-free `ModelExample`, and samples with a fixed seed and one question per
   theory by default.
2. `protocol.py` identifies release splits by SHA-256, enforces phase
   provenance, loads fixed sample sets, and applies question/theory quarantine.
3. `build_model_input()` keeps exactly natural-language `{theory, question}`;
   `render_model_input()` makes literal `s1..sN/q1` lines visible.
4. `ir_v2.py` parses the single compact model-facing contract and converts it
   deterministically to the internal `schema.py` dataclasses.
5. `validator.py` separates hard invariants from soft lexical warnings.
6. `solver.py` computes the signed forward-chaining least fixed point.
7. `methods.py` implements Direct, CoT, two-call CoT-Refine, Plain, and the
   shared-initial Constrained/Gate/Repair/VGCF-2 family.
8. `gold.py` independently parses evaluator-only official representations and
   is never imported by prompt/method code.
9. `experiment.py` joins method outputs with labels/gold programs only after a
   method returns, then persists split operational and semantic metrics.
10. `analysis.py` consumes saved JSONL offline and never calls a model.
11. `audit_v2.py` checks complete v2 artifacts, routing, reuse, leakage, and
    accounting without changing the frozen historical auditor.

## VGCF-2 routing

```text
one constrained initial response
        |
        +-- Constrained baseline: executable -> solver (hard status recorded)
        +-- Gate + CoT:      hard-valid -> solver; otherwise CoT fallback
        +-- Repair only:     incomplete -> one repair -> best executable stage
        +-- Full VGCF-2:     hard-valid -> solver
                              otherwise -> one repair
                              repaired hard-valid -> solver
                              otherwise -> same-model CoT fallback
```

The four branches share the same response object/cache key and record one
`initial_response_hash`. VGCF-2 has no independent base-generation prompt.

## Non-negotiable invariants

- Model-visible dataset fields equal `{theory, question}`.
- No unarmed phase can use the known `meta-test` byte identity.
- Probe/dev use fixed, example- and theory-disjoint manifest sets.
- Final-test sampling filters both observed questions and every exposed theory.
- Final test also requires frozen config/prompt/quarantine hashes and an
  explicit environment arm before client creation.
- The solver never adds a query to facts and never uses inverse implication or
  contraposition.
- Failure to derive `q` is `Unknown`; only its explicit complement gives
  `False`.
- Variables occur only in rules; facts and query are ground; binary order and
  shared conjunctive substitutions are preserved.
- A failed repair never overwrites a better executable initial stage.
- Gold labels/representations never influence formalization, validation,
  repair, fallback, or solving.
- Hidden reasoning is rejected for a non-Thinking run, including cache hits.

## Validation and proof boundary

JSON parse validity, v2 schema validity, hard-validator pass, and solver
executability are four separate facts. Soft lexical warnings affect only issue
counts/trust score. Neither static validity nor repair improvement establishes
semantic correctness; evaluator-only gold precision/recall measures that after
the method completes.

`solver_executable` is a shadow-program property. Only `answer_from_solver`
plus `route_source` establishes whether the final label came from the solver.

Proof steps record the ground fact, given/rule kind, original source ID,
matched premises, and shared variable binding. Premise steps precede their
dependent conclusion.

## Public export scope

The historical Final guards described above remain in the implementation, but
their private manifests and execution artifacts are not distributed. The public
workflow supports synthetic smoke and aggregate checks, not Final inference.
