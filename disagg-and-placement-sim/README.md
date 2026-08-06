# Phase 4 Simulator — Disaggregated Prefill/Decode Discrete-Event Sim

Discrete-event simulator (SimPy) for the dense (Llama-3-70B, TPU 8i, FP4) leg
of [`disagg_and_placement_notes.md`](../disagg_and_placement_notes.md).
Answers the one thing Phases 1–3's closed-form math structurally can't: how
often the finite intermediate KV pool actually hits capacity under bursty
arrival, and what real admission-queueing latency looks like.
Throughput/chip-ratio numbers are not re-derived here — they're inputs, taken
from `../disagg_and_placement_notes.md` §2 (post-QKVO-correction figures), and
used as regression-test targets.

Design rationale, formula derivations, and flagged assumptions are summarized
in §4 of the notes doc above — this README is just setup + run instructions.

## Setup

```
cd disagg-and-placement-sim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run order (each step gates the next)

1. `pytest tests/` — regression tests, must pass exactly against the two
   reference throughput numbers (142.70 req/s/chip prefill, 830.59 req/s/chip
   decode). If these don't pass exactly, nothing downstream can be trusted.
2. `python validate.py` — stripped-down end-to-end SimPy runs (not just the
   formula module) must converge to the same two numbers within 2%.
3. Run a single pilot sweep cell manually (see `run_sweep.py --help`) to
   check wall-clock cost before committing to the full grid.
4. `python run_sweep.py` — full 4λ × 5 pool-capacity × 5 seed grid, writes
   `results/summary.csv` and `results/raw_requests.csv`.

## What to look for in `results/summary.csv`

The smallest pool-capacity multiplier (0.25×) should show meaningfully higher
`frac_pool_blocked` / pool-wait than the largest (6×), and higher λ should
show more contention than lower λ at fixed pool size. If the shape isn't
monotonic and mechanistically sensible, that's a bug to chase before trusting
any number from the sweep.
