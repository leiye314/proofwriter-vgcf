# VGCF IR v2.2

`vgcf-ir-v2` is the single model-facing formalization contract used by Plain,
Constrained, and the VGCF-2 family. Parsing is deterministic and never
invents a predicate, removes an article, changes polarity, swaps arguments, or
reverses a rule.

## Project-authored worked example

This example was written for this project and is not from any ProofWriter
split.

```text
s1: Bob visits the bald eagle.
s2: The bald eagle is not quiet.
s3: If someone visits the bald eagle then they admire the bald eagle.
q1: Bob admires the bald eagle.
```

```json
{
  "facts": [
    {"id": "s1", "atom": "+visits(bob,the_bald_eagle)"},
    {"id": "s2", "atom": "-quiet(the_bald_eagle)"}
  ],
  "rules": [
    {
      "id": "s3",
      "if": ["+visits(X,the_bald_eagle)"],
      "then": "+admires(X,the_bald_eagle)"
    }
  ],
  "query": {"id": "q1", "atom": "+admires(bob,the_bald_eagle)"}
}
```

The example covers a proper name, an article-preserving multiword entity, a
binary relation, a rule variable, and explicit negation.

## Atom grammar and naming

```text
atom       := sign predicate "(" argument ("," argument)? ")"
sign       := "+" | "-"
predicate  := lowercase_identifier
argument   := lowercase_identifier | uppercase_identifier
```

- `+` is positive polarity; `-` is explicit negation. Missing evidence is not
  negative evidence.
- Predicates and constants match `[a-z][a-z0-9_]*`.
- Constants are lowercase. Proper-name case is normalized: `Bob -> bob` and
  `Charlie -> charlie`.
- Multiword constants preserve every word and article and join them with
  underscores: `the bald eagle -> the_bald_eagle`, `the cat -> the_cat`.
- Variables match `[A-Z][A-Za-z0-9_]*` and are allowed only inside rules.
- Facts and queries are ground. A ground title-case constant is safely
  lowercased; an all-uppercase identifier such as `X` remains a variable and
  is rejected outside a rule.
- Only unary and binary predicates are supported. Argument order is semantic.

## Structural invariants

- Every `s1..sN` occurs exactly once as a fact or rule ID; `q1` occurs only as
  the query.
- Rule premises are a non-empty conjunction and `then` is the forward
  conclusion. Conclusion variables must be bound in the same rule body.
- One predicate has one arity within a program.
- The parser strips only surrounding whitespace and one complete optional JSON
  fence. It performs no semantic auto-repair.

## Strict and singleton-normalized views

Every model response is measured twice. `strict_schema_valid` describes the
raw JSON before normalization. `normalized_schema_valid` describes the same
JSON after the only permitted safe transformation: if and only if a
`rules[*].if` value is a string that fully matches one signed unary or binary
atom, the parser wraps that exact string in a one-element array.

For example, `"if":"-needs(X,the_cat)"` may become
`"if":["-needs(X,the_cat)"]`. Existing arrays are no-ops. The normalizer does
not split strings, repair JSON, infer predicates, change signs or arguments,
move facts/rules, or reverse implications. Strings such as `+p(a),+q(b)`,
`+p(a) and +q(b)`, `+p(a b)`, `+p()`, `p(a)`, and `+p(a)->+q(a)` remain
invalid.

`strict_solver_executable` and `normalized_solver_executable` retain the same
two views. The operational route uses the normalized view, but reports must
always show the strict view alongside it.

## Entity warnings

The validator recognizes exact contiguous article-preserving forms such as
`the_bald_eagle`. A likely article-dropping alias such as `bald_eagle` for
“the bald eagle” is exposed as `inconsistent_entity_alias`, a soft warning.
Because reliable alias resolution cannot be guaranteed from visible text, it
never becomes a hard gate. Broad token-subset matching is not used as a hard
semantic decision.
