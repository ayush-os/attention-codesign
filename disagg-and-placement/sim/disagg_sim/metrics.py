"""Per-request record conversion + summary aggregation helpers."""

import pandas as pd

from disagg_sim.workload import Request


def request_to_record(req: Request) -> dict:
    return {
        "id": req.id,
        "arrival_time": req.arrival_time,
        "prompt_len": req.prompt_len,
        "output_len": req.output_len,
        "prefill_wait": req.prefill_wait,
        "pool_wait": req.pool_wait,
        "decode_admission_wait": req.decode_admission_wait,
        "end_to_end_latency": req.end_to_end_latency,
    }


def records_to_df(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """One row per group_cols combination (e.g. [lambda, pool_capacity_gb]),
    aggregated across all seeds pooled into that group."""

    def p95(s):
        return s.quantile(0.95)

    def p99(s):
        return s.quantile(0.99)

    return (
        df.groupby(group_cols)
        .agg(
            n_requests=("id", "count"),
            frac_pool_blocked=("pool_wait", lambda s: (s > 1e-9).mean()),
            prefill_wait_mean=("prefill_wait", "mean"),
            prefill_wait_p95=("prefill_wait", p95),
            prefill_wait_p99=("prefill_wait", p99),
            pool_wait_mean=("pool_wait", "mean"),
            pool_wait_p95=("pool_wait", p95),
            pool_wait_p99=("pool_wait", p99),
            decode_admit_wait_mean=("decode_admission_wait", "mean"),
            decode_admit_wait_p95=("decode_admission_wait", p95),
            decode_admit_wait_p99=("decode_admission_wait", p99),
            e2e_latency_mean=("end_to_end_latency", "mean"),
            e2e_latency_p50=("end_to_end_latency", "median"),
            e2e_latency_p95=("end_to_end_latency", p95),
            e2e_latency_p99=("end_to_end_latency", p99),
        )
        .reset_index()
    )
