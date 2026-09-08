# Evidence map

| Claim | Public support | Boundary |
|---|---|---|
| Seven-method frozen accuracies | `results/frozen_summary.json` | Transcribed aggregate; raw predictions absent |
| H1/H2 paired comparisons | Discordant counts, exact p values and frozen intervals in summary | McNemar/Holm recomputed; bootstrap intervals transcribed |
| Software invariants | Core implementation and 69 compact tests | Synthetic deterministic validation |
| Smoke reproducibility | Existing two-run smoke and fixed expected hashes | Five-method synthetic smoke only |
| Input leakage boundary | `build_model_input()` and sentinel tests | Exactly theory and question |
| Scientific archive identity | Original archive checked before export | Complete archive is private |

The summary records SHA-256 identities of the two frozen source aggregates.
`PUBLIC_EXPORT_MANIFEST.json` lists every copied source file with its original
byte hash, transformed entry points, and new public files. No raw Final records
were read into the aggregate transformation. No scientific number was changed.

The archived report, rejected exploratory packets, case-level analyses and
historical intermediate results are not public evidence in this repository.
