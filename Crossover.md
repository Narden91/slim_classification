# SLIM Crossover (Block-Level) — Context, Problem, Solutions

## Context
SLIM-GSGP individuals are not a single GP tree: they are an **ordered list of blocks** (trees) whose outputs (semantics) are aggregated by an operator (typically **sum** or **product**). Mutation (inflate/deflate) naturally adds/removes blocks. Historically, the codebase had a crossover probability (`p_xo`) but the SLIM crossover branch was a placeholder.

## Problem
Without a SLIM-aware crossover:

- `p_xo > 0` does nothing (no genetic recombination).
- The algorithm relies exclusively on inflate/deflate mutations.
- You cannot study/compare crossover vs mutation contributions.

Classic GP subtree crossover is not a good fit here because SLIM’s representation is **block/ensemble-like**: swapping random subtrees inside blocks changes semantics in uncontrolled ways and breaks the “incremental semantics” design.

## Solutions implemented (2 alternatives)
Both solutions operate at the **block level** and are implemented in:
- slim_gsgp/algorithms/SLIM_GSGP/operators/crossover_operators.py

They work with `reconstruct=True` (keeps block collections) and `reconstruct=False` (semantics-only mode) by recombining cached per-block semantics.

### 1) One-point block swap crossover (`one_point`)
**Idea:** treat each individual as a list of blocks and do a one-point recombination.

- Child = `prefix(parent1)` + `suffix(parent2)`
- Preserves “building blocks” and produces meaningful semantic recombination.

**Pros**
- Simple and fast.
- Preserves ordering and can keep early “foundation” blocks from one parent.

**Cons**
- Recombination is coarse; may keep large contiguous segments.

### 2) Uniform block mix crossover (`uniform`)
**Idea:** for each block position, pick the block from either parent.

- Child length = `max(len(p1), len(p2))`
- For each index `i` where both parents have blocks, choose from either parent with probability 0.5.
- If one parent is shorter, remaining blocks come from the longer parent.

**Pros**
- Higher mixing rate, more exploratory.
- Still respects SLIM’s block representation.

**Cons**
- Can disrupt block ordering patterns (more “shuffling” effect).

## Integration details (how it is used)
- The evolution loop in slim_gsgp/algorithms/SLIM_GSGP/slim_gsgp.py now executes crossover when `random() < p_xo`.
- Depth constraint is enforced: if offspring exceeds `max_depth`, it falls back to copying the parent (or deflating when copy isn’t allowed).

## CLI usage
These flags are available in the main experiment runner:
- slim_gsgp/example_binary_classification.py

New SLIM flags:
- `--p-xo` (float): crossover probability (0 disables crossover)
- `--crossover-operator` (`one_point|uniform|none`)

## Examples

### Example 1 — Run SLIM with one-point crossover
```bash
python slim_gsgp/example_binary_classification.py \
  --dataset=breast_cancer \
  --algorithm=slim \
  --slim-version=SLIM+SIG2 \
  --pop-size=100 \
  --n-iter=50 \
  --p-inflate=0.5 \
  --p-xo=0.3 \
  --crossover-operator=one_point
```

### Example 2 — Run SLIM with uniform crossover
```bash
python slim_gsgp/example_binary_classification.py \
  --dataset=breast_cancer \
  --algorithm=slim \
  --slim-version=SLIM+SIG2 \
  --pop-size=100 \
  --n-iter=50 \
  --p-inflate=0.5 \
  --p-xo=0.3 \
  --crossover-operator=uniform
```

---

## Notes on testing
A dedicated test suite validates:
- Crossover operators produce structurally consistent offspring (sizes, semantics shapes).
- The SLIM solver runs end-to-end with crossover enabled.
- The `slim()` wrapper accepts the crossover arguments.

Tests live in:
- tests/test_slim_crossover.py
