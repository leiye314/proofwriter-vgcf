# Dataset adaptation and leakage prevention

Only a project-generated synthetic fixture is bundled. Official data and
historical quarantine/sample manifests are not distributed. Obtain upstream data
separately and inspect its actual schema before using a new adapter.

## Supported explicit profiles

- `vgcf_flat_v1`: checked-in fixture fields `example_id`, `depth`, `theory`,
  `question`, and `gold_label`; unknown fields are rejected.
- `proofwriter_flat`: requires `id`, `theory`, `question`, and `answer`; depth
  may come from `depth` or `QDep`.
- `proofwriter_nested`: requires `id`, `triples`, `rules`, and `questions`.
  Only `triples[*].text`, `rules[*].text`, and `questions[*].question` become
  natural-language input. Answers and every representation/proof field remain
  evaluator-only.

Detection uses complete required-key sets. Mixed or unknown schemas fail.
`scripts/inspect_dataset.py` reports actual fields and types before a new
adapter is added; field meanings are never guessed from names alone.

## Leakage controls

- `MODEL_INPUT_FIELDS = {"theory", "question"}` is enforced at runtime.
- `assert_safe_model_input()` rejects missing/extra keys and non-string data.
- Evaluator `Example.model_view()` creates a separate `ModelExample` type with
  no depth, label, answer, proof, or representation fields.
- `MethodRunner` accepts only `ModelExample`; it accepts neither a dataset path
  nor the gold module's types.
- All visible requests are rendered from the whitelist as literal `s1..sN`
  and `q1` lines.
- Repair receives only those numbered lines, the model's original output, the
  public IR instructions/toy example, at most five grouped hard issues, and a
  separately labelled non-mandatory soft-advisory section.
- Gold parsing and semantic comparison happen in `experiment.py` after method
  execution through the independent `gold.py` module.
- Sentinel regression tests inspect model-visible messages, saved raw request
  payloads, and cache files and require the planted gold marker to be absent.
- Non-Thinking runs reject hidden reasoning from both current and compatible
  legacy cache entries; provider payloads containing hidden reasoning are also
  rejected before caching.

## Evaluator-only gold representations

The official representation parser lives in `gold.py` and is exercised by
synthetic unit programs. Its outputs are used after model execution only.
Historical Final phase guards still require omitted manifests and locks; the
public export does not supply a Final inference recipe.
