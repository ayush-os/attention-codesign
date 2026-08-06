"""Phase 4 self-consistency check -- run before trusting any sweep output.

Two checks, each combining (a) a direct formula-module assertion and (b) an
actual stripped-down SimPy run through the real dispatcher/machine code (not
just the formula in isolation) -- confirms the wiring, not just the math.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import simpy

from disagg_sim import service_time
from disagg_sim.config import SimConfig
from disagg_sim.constants import KV_BYTES_PER_TOKEN
from disagg_sim.decode import DecodeDispatcher, DecodeMachine
from disagg_sim.pool import IntermediateKVPool
from disagg_sim.prefill import PrefillDispatcher
from disagg_sim.workload import Request

TOL = 0.02  # 2%
REF_PREFILL_TPUT = 142.70  # req/s/chip
REF_DECODE_TOKEN_TPUT = 53_157.6  # tok/s
REF_DECODE_REQ_TPUT = 830.59  # req/s/chip


def _check(name, got, expected, tol=TOL):
    rel_err = abs(got - expected) / expected
    status = "PASS" if rel_err < tol else "FAIL"
    print(f"[{status}] {name}: got {got:.4f}, expected ~{expected:.4f} (rel err {rel_err:.2%})")
    return rel_err < tol


def validate_prefill_formula() -> bool:
    t = service_time.prefill_batch_time([512] * 32)
    return _check("prefill formula throughput", 32 / t, REF_PREFILL_TPUT)


def validate_decode_formula() -> bool:
    t = service_time.decode_token_time(320, avg_context_len=544.0)
    tok_tput = 320 / t
    return _check("decode formula token throughput", tok_tput, REF_DECODE_TOKEN_TPUT)


def validate_prefill_sim(n_batches: int = 200) -> bool:
    """Single prefill machine, backlog kept saturated (a large fixed backlog
    submitted at t=0, all at the anchor prompt length) -- the machine always
    has >=32 waiting until near the very end, so realized throughput should
    converge to the reference number."""
    env = simpy.Environment()
    pool = IntermediateKVPool(env, capacity_bytes=1e15)  # effectively unlimited -- isolates prefill alone
    config = SimConfig(arrival_rate=1.0, pool_capacity_bytes=1e15, n_prefill_machines=1)

    handed_off: list[Request] = []

    class _DummyDecodeDispatcher:
        def submit(self, req):
            handed_off.append(req)

    dispatcher = PrefillDispatcher(env, config, pool, _DummyDecodeDispatcher())

    n_requests = n_batches * config.prefill_batch_cap
    for i in range(n_requests):
        req = Request(
            id=i, arrival_time=0.0, prompt_len=512, output_len=64, kv_bytes=512 * KV_BYTES_PER_TOKEN
        )
        dispatcher.submit(req)

    env.run()  # no `until` -- runs until the backlog is fully drained

    last_done = max(r.prefill_done_time for r in handed_off)
    throughput = n_requests / last_done
    return _check("prefill stripped-down sim throughput", throughput, REF_PREFILL_TPUT)


def validate_decode_sim(n_completions_target: int = 5_000, output_len: int = 50) -> bool:
    """Single decode machine, occupancy pinned at exactly 320 for the whole
    run: 320 synthetic requests submitted at t=0; each time one completes, a
    fresh replacement is submitted immediately, keeping occupancy at 320
    throughout. Throughput is measured from real completions (actual
    accumulated timeout events through the real _run_request code path), not
    reconstructed analytically from the formula -- a tautology would defeat
    the point of a "stripped-down real sim" check distinct from
    validate_decode_formula above."""
    env = simpy.Environment()
    machine = DecodeMachine(machine_id=0, cap=320)
    pool = IntermediateKVPool(env, capacity_bytes=1e15)
    config = SimConfig(arrival_rate=1.0, pool_capacity_bytes=1e15, n_decode_machines=1, decode_cap_per_machine=320)

    completed: list[Request] = []
    next_id = [320]  # first 320 ids (0..319) used below
    done_event = env.event()

    def on_complete(req: Request):
        completed.append(req)
        if len(completed) >= n_completions_target:
            if not done_event.triggered:
                done_event.succeed()
            return
        replacement = Request(id=next_id[0], arrival_time=env.now, prompt_len=512, output_len=output_len, kv_bytes=1.0)
        next_id[0] += 1
        dispatcher.submit(replacement)

    dispatcher = DecodeDispatcher(env, config, [machine], pool, on_complete=on_complete)

    for i in range(320):
        req = Request(id=i, arrival_time=0.0, prompt_len=512, output_len=output_len, kv_bytes=1.0)
        dispatcher.submit(req)

    assert machine.occupancy == 320, f"expected pinned occupancy 320, got {machine.occupancy}"

    env.run(until=done_event)
    total_tokens = sum(r.output_len for r in completed)
    tok_tput = total_tokens / env.now
    return _check("decode stripped-down sim token throughput", tok_tput, REF_DECODE_TOKEN_TPUT)


def main() -> int:
    checks = [
        validate_prefill_formula,
        validate_prefill_sim,
        validate_decode_formula,
        validate_decode_sim,
    ]
    results = [fn() for fn in checks]
    if all(results):
        print("\nAll validation checks passed -- safe to trust run_sweep.py output.")
        return 0
    print("\nSome validation checks FAILED -- do not trust sweep output until fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
