"""Final consolidated sweep. Supersedes the exploratory scripts
(explore_pool_threshold.py, explore_dynamic_admission.py, explore_occupancy.py)
with one internally-consistent run at full methodology (2,000-completion
warm-up, 20,000 recorded per replication) across everything those scripts
found worth reporting:

- lambda: 500 -> 10,000 req/s (spans below, at, and well past the ~4,138
  req/s prefill-side ceiling discovered during exploration).
- pool capacity: three sizes safely above the real contention threshold
  (found empirically to sit between 500MB and 1GB) -- 1GB and 2GB bracket
  it closely, ROUND_UNIT_BYTES (~38.92GB) is the original "one round"
  reference point. Sub-threshold sizes (200MB collapses outright, 500MB is
  unstable) are NOT re-run here at full cost -- explore_pool_threshold.py's
  own lighter methodology already characterized them adequately, and
  re-running known-pathological cells at 20,000-completion target would
  mostly burn wall-clock time hitting the safety valve.
- decode admission cap: 320 (the real hard-cap policy) is the primary sweep
  dimension; 4,558 (dynamic admission) is additionally run at the
  ROUND_UNIT_BYTES pool size across all lambdas, to reconfirm under full
  methodology that it makes no measurable difference (explore_dynamic_
  admission.py found this, but with a shorter warm-up).

Also collects decode occupancy stats (mean, max, fraction of time above the
~296 compute-bound crossover) per cell -- explore_occupancy.py showed this
is the real story behind "decode is never the bottleneck": it saturates
against prefill's own throughput ceiling, not against its own admission cap.

Calls validate.py first and refuses to proceed on failure by default --
--skip-validate is a fine escape hatch for fast iteration, not for a real run.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import validate
from disagg_sim import constants
from disagg_sim import metrics as metrics_mod
from disagg_sim.config import SimConfig
from disagg_sim.run_single import run_replication_with_system

RESULTS_DIR = Path(__file__).resolve().parent / "results"

LAMBDAS = [500, 2000, 3500, 4500, 6000, 8000, 10000]
POOL_BYTES = {"1GB": 1e9, "2GB": 2e9, "1round_38.92GB": constants.ROUND_UNIT_BYTES}
DECODE_CAPS = {"hard_cap_320": 320}
DECODE_CAPS_EXTRA_AT_DEFAULT_POOL = {"dynamic_4558": 4558}  # only run at the default pool size
CROSSOVER_N = 296  # notes.md SS2.5


def build_configs(lambdas=None, n_seeds=None, target_completions=None, warmup_completions=None, max_sim_time_s=None):
    lambdas = lambdas if lambdas is not None else LAMBDAS
    n_seeds = n_seeds if n_seeds is not None else constants.N_SEEDS
    target_completions = target_completions if target_completions is not None else constants.TARGET_COMPLETIONS
    warmup_completions = warmup_completions if warmup_completions is not None else constants.WARMUP_COMPLETIONS
    max_sim_time_s = max_sim_time_s if max_sim_time_s is not None else 300.0

    configs = []
    for lam in lambdas:
        for pool_label, pool_bytes in POOL_BYTES.items():
            for cap_label, cap in DECODE_CAPS.items():
                for seed in range(n_seeds):
                    configs.append(
                        (
                            pool_label,
                            cap_label,
                            SimConfig(
                                arrival_rate=lam,
                                pool_capacity_bytes=pool_bytes,
                                decode_cap_per_machine=cap,
                                seed=seed,
                                target_completions=target_completions,
                                warmup_completions=warmup_completions,
                                max_sim_time_s=max_sim_time_s,
                            ),
                        )
                    )
        # Dynamic-admission comparison, only at the default (1-round) pool size.
        for cap_label, cap in DECODE_CAPS_EXTRA_AT_DEFAULT_POOL.items():
            for seed in range(n_seeds):
                configs.append(
                    (
                        "1round_38.92GB",
                        cap_label,
                        SimConfig(
                            arrival_rate=lam,
                            pool_capacity_bytes=constants.ROUND_UNIT_BYTES,
                            decode_cap_per_machine=cap,
                            seed=seed,
                            target_completions=target_completions,
                            warmup_completions=warmup_completions,
                            max_sim_time_s=max_sim_time_s,
                        ),
                    )
                )
    return configs


def _occupancy_stats(system, warmup_cutoff_time: float) -> dict:
    """Time-weighted occupancy stats across all decode machines, restricted
    to samples at/after warmup_cutoff_time for a fair steady-state read."""
    means, maxes, frac_aboves = [], [], []
    for m in system.decode_machines:
        samples = [(t, o) for t, o in m.occupancy_samples if t >= warmup_cutoff_time]
        if not samples:
            continue
        end_time = system.env.now
        times = [t for t, _ in samples] + [end_time]
        occs = [o for _, o in samples]
        total_time = end_time - times[0]
        if total_time <= 0:
            continue
        weighted_sum = sum(occs[i] * (times[i + 1] - times[i]) for i in range(len(occs)))
        above_time = sum((times[i + 1] - times[i]) for i in range(len(occs)) if occs[i] >= CROSSOVER_N)
        means.append(weighted_sum / total_time)
        maxes.append(max(occs))
        frac_aboves.append(above_time / total_time)
    if not means:
        return {"mean_occupancy": None, "max_occupancy": None, "frac_time_above_crossover": None}
    return {
        "mean_occupancy": sum(means) / len(means),
        "max_occupancy": max(maxes),
        "frac_time_above_crossover": sum(frac_aboves) / len(frac_aboves),
    }


def run_one(args) -> tuple[pd.DataFrame, dict]:
    pool_label, cap_label, config = args
    reqs, system = run_replication_with_system(config)
    occ_stats = None
    if reqs:
        warmup_cutoff = reqs[0].arrival_time  # first post-warmup request's arrival is a reasonable steady-state proxy
        occ_stats = _occupancy_stats(system, warmup_cutoff)
        occ_stats.update(
            {
                "lambda": config.arrival_rate,
                "pool_label": pool_label,
                "decode_cap_label": cap_label,
                "seed": config.seed,
            }
        )

    if not reqs:
        df = pd.DataFrame(
            [{"lambda": config.arrival_rate, "pool_label": pool_label, "decode_cap_label": cap_label,
              "seed": config.seed, "incomplete": True}]
        )
        return df, occ_stats

    records = [metrics_mod.request_to_record(r) for r in reqs]
    df = pd.DataFrame(records)
    df["lambda"] = config.arrival_rate
    df["pool_label"] = pool_label
    df["decode_cap_label"] = cap_label
    df["seed"] = config.seed
    df["incomplete"] = False
    return df, occ_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validate", action="store_true", help="skip validate.py (fast iteration only)")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds per cell")
    parser.add_argument("--target-completions", type=int, default=None)
    parser.add_argument("--warmup-completions", type=int, default=None)
    parser.add_argument("--max-sim-time", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None, help="default: os.cpu_count()")
    parser.add_argument("--out-prefix", type=str, default="", help="prefix for output filenames")
    args = parser.parse_args()

    if not args.skip_validate:
        print("Running validate.py before sweeping...")
        if validate.main() != 0:
            print("validate.py failed -- refusing to run the sweep. Fix the failures or pass --skip-validate.")
            return 1
        print()

    configs = build_configs(
        lambdas=args.lambdas,
        n_seeds=args.seeds,
        target_completions=args.target_completions,
        warmup_completions=args.warmup_completions,
        max_sim_time_s=args.max_sim_time,
    )
    print(f"Running {len(configs)} replications...")

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, configs))
    elapsed = time.time() - start
    print(f"Sweep finished in {elapsed:.1f}s ({elapsed / len(configs):.2f}s/replication avg)")

    dfs = [r[0] for r in results]
    occ_rows = [r[1] for r in results if r[1] is not None]

    raw = pd.concat(dfs, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    raw_path = RESULTS_DIR / f"{args.out_prefix}raw_requests.csv"
    raw.to_csv(raw_path, index=False)
    print(f"Wrote {raw_path} ({len(raw)} rows)")

    incomplete = raw[raw.get("incomplete", False) == True]  # noqa: E712
    if len(incomplete):
        print("\nCells that hit the safety valve before finishing:")
        print(incomplete.groupby(["lambda", "pool_label", "decode_cap_label"]).size().to_string())

    complete = raw[raw.get("incomplete", False) == False]  # noqa: E712
    summary = metrics_mod.summarize(complete, group_cols=["lambda", "pool_label", "decode_cap_label"])

    occ_df = pd.DataFrame(occ_rows)
    if len(occ_df):
        occ_summary = (
            occ_df.groupby(["lambda", "pool_label", "decode_cap_label"])
            .agg(
                mean_occupancy=("mean_occupancy", "mean"),
                max_occupancy=("max_occupancy", "max"),
                frac_time_above_crossover=("frac_time_above_crossover", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(occ_summary, on=["lambda", "pool_label", "decode_cap_label"], how="left")

    summary_path = RESULTS_DIR / f"{args.out_prefix}summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary)} rows)")
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
