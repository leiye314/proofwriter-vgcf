# Reproducibility

Run every README command with Python 3.12+ from this repository root. Runtime
and offline verification use only the standard library. Optional packaging is
`python -m pip install -e .`; package installation may access the package index.
It is unnecessary for the documented offline workflow.

The public core code is copied byte-for-byte from the reviewed portfolio source;
documentation, package metadata, test entry points and aggregate verification
are curated for this independent repository. Internal Git history is not included.
The synthetic smoke retains its pre-existing deterministic expected hashes.

## Data and model sources

Obtain data from the [recorded AllenAI V2020.12.3 archive](https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip).
Historical archive SHA-256:
`bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26`.
Historical OWA depth-5 meta-test SHA-256:
`c09fad796aaf546d6fcbfc77ecf91f935ffed3c936c2b0e96f4aa57211fad842`.
Respect the upstream terms. Dataset adapters require actual schema inspection
using `scripts/inspect_dataset.py`; model inputs must pass `build_model_input()`.

The frozen runtime recorded `qwen3.5-9b` through an OpenAI-compatible local
LM Studio endpoint, context 16,384, parallelism 1, timeout 360 seconds,
one retry, temperature 0, non-Thinking mode and 3,000 response tokens.
Consult the [Qwen project](https://github.com/QwenLM) for upstream model releases.
The exact weight revision, quantization and runtime identity are unavailable;
this repository does not claim that any current download is the frozen model.

## Evidence boundaries

Full evidence remains in the author's private archive and is not downloaded by
any public command. Access would require a separate privacy/rights review.
The public aggregate is a minimal derived transcription, not a replacement for
raw records, sample manifests, prompts/config locks or dataset provenance.
No evaluator-only official packets are bundled. Synthetic gold programs in unit
tests exercise evaluator isolation and do not originate from held-out examples.

`python -B scripts/check_environment.py --require-final-evidence` intentionally
fails with an explicit unavailable status. It does not search private paths.
The preserved generic runner contains real-provider capabilities, but its
historical Final path requires omitted locks/evidence and is not a supported
public rerun recipe. Do not interpret that path as an exact reproduction recipe.

The summary verifier recomputes accuracy and exact McNemar/Holm arithmetic.
Bootstrap intervals are preserved from the frozen summary; they are not
recomputed from undisclosed example-level predictions. Hashes provide provenance
anchors for later authorized comparison, not independent public authentication.
