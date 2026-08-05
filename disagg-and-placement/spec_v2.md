# Project Spec v2: Disaggregated Serving + Weight/KV-Cache Placement — MoE Extension

**Supersedes:** original `spec.md`. Does not invalidate it — Phase 0, Phase 1,
and the dense-transformer half of Phase 2 are complete, validated, and stay
exactly as derived. This doc's job is narrow: add the MoE leg that Phase 3
was always going to need (spec's original Phase 3 already asked "should hot
experts get preferential SRAM residency" — that question presupposes a MoE
decode-throughput story that doesn't exist yet) and pull it forward into
Phase 2, where the chip-ratio number actually lives.

Keep `spec.md` (v1) and the working notes untouched as the historical record,
same convention as keeping the rejected Groq/heterogeneous design on file in
Phase 0 rather than scrubbing it.

**Legend:** 🔧 = boilerplate/setup. 🧠 = your job. ✅ = already done, reused
as-is, not re-derived here.

---

## Why this changed (one paragraph, for anyone who wasn't in the room)

Phase 2a's dense-transformer chip ratio (~5.5 prefill : 1 decode, Llama-3-70B)
is real and stays as the project's **validated baseline** — it's the case
that matches DistServe's own reported direction, and that agreement is what
gives you confidence the hand-built simulator/analytical model is sound
before trusting it on anything harder. But it's a dense-only artifact, and
the stated target (codesign for workloads that are "all MoE nowadays")
means shipping a project whose headline ratio is dense-only would undersell
the harder, more relevant question. The mechanism, not just the numbers,
changes for MoE: dense decode is memory-bound because *every* token touches
*every* weight, full stop, regardless of batch size. MoE decode only touches
top-k experts per token — which means, unlike dense, decode's regime
position is potentially a function of batching/routing policy, not a fixed
property of the workload. That's a structurally different question, not a
parameter swap, which is why it gets its own phase rather than a footnote.

---

## Phase 0 — Setup ✅ (done, unchanged)

TPU 8i homogeneous (prefill + decode), FP4 uniform, discrete-event
simulator design (request lifecycle, N-concurrent-per-decode-machine,
finite intermediate KV pool, Poisson arrivals, lognormal DistServe-anchored
request shape). Reference reading (PagedAttention, DistServe, Mooncake)
complete with checkpoint numbers banked for Phase 4. Nothing here needs
touching for the MoE leg — the simulator's entity/lifecycle model is
workload-agnostic; only the per-phase service-time formulas plugged into it
change (Phase 2).

## Phase 1 — Memory hierarchy characterization ✅ (done, unchanged)

Llama-3-70B weight footprint, KV-cache-bytes/token growth curve, N≈320–339
decode concurrency ceiling, static-weights-vs.-elastic-KV-cache tension —
all stays as the dense reference case. **Not re-run for MoE in this
version.** DeepSeek-V2's own weight/HBM footprint exists already (project
#3) but a full MoE-specific Phase-1-style capacity derivation (KV cache +
*sparse* expert-weight residency, which is a different shape of problem
than dense's single fixed weight blob) is flagged as a candidate for a
**future, separate pass** if Phase 2b's findings make it clearly necessary
— not assumed necessary up front. Don't do this unless Phase 2b's numbers
demand it; adding it preemptively is exactly the shallow-breadth trap the
original spec's "Note on scope" already warned against.

## Phase 2a — Dense chip ratio ✅ (done, unchanged)

~5.5 prefill chips per decode chip, Llama-3-70B, TPU 8i/FP4, real workload
parameters (seq_len=512 prompt, seq_len_kv=544 avg context, N=320). Root
cause of the initial backwards result (FFN weight-byte batch-invariance,
under-amortized at batch=32) is the load-bearing mechanistic finding here —
**keep this finding front and center in the MoE leg below**, because it's
exactly the axis MoE breaks.

---

## Phase 2b — MoE chip ratio (🧠, new)

**Reuse, don't re-derive from scratch:** project #3's DeepSeek-V2 FLOPs/
bytes-per-expert formulas, its 8-of-162 active-experts-per-token routing,
and its structural imbalance-floor finding (~21,065 FLOPs/byte) are your
starting inputs. Same discipline as reusing `prefill_notes.md`/
`decode_notes.md` into 2a: check reference-chip/precision compatibility
before combining (project #3 was TPU 8i/FP4 already — confirm this still
holds, don't assume).

**The core question, stated precisely:** dense FFN's weight bytes are
batch-invariant *because every token touches every weight* — batching
doesn't change what fraction of weights gets loaded, only how many tokens
share that one load. MoE FFN's weight bytes are only batch-invariant **per
expert that actually gets touched** — the real lever is *how many distinct
experts a batch of N tokens touches*, which depends on routing distribution,
not just N. Two boundary cases to hand-derive before touching any model:

- **Low-diversity limit**: if a batch of N tokens is small/homogeneous
  enough that its top-k choices cluster on a small expert subset, each
  touched expert's weight load amortizes over many tokens — same mechanism
  as dense's N=320 crossover, applied per-expert instead of globally.
- **High-diversity limit**: if N is large/diverse enough that the batch
  collectively touches most or all of the 162 experts anyway, you're back to
  paying for (nearly) the full expert table's weight bytes regardless of N —
  no amortization win, MoE decode looks memory-bound the same way dense does,
  just with a different fixed cost.

**Derive, don't assume, which limit realistic batch sizes land in.** Use
project #3's own routing-skew/imbalance model (already derived, don't
re-invent) rather than assuming uniform random routing — uniform routing
would systematically overstate how many distinct experts a small batch
touches, understating the amortization win that's the whole point of this
phase.

**Concretely, Phase 2b asks for:**
1. Expected number of distinct experts touched as a function of batch size
   N (using project #3's routing model, not a uniform assumption).
2. From that: effective FFN weight bytes moved per decode step as a
   function of N — this replaces dense's flat "352 MB every step regardless
   of N" with something that should look more like a saturating curve.
3. The regime crossover (if one exists) — is there an N where MoE decode's
   FFN flips compute-bound the way dense's did at N≈296? Higher, lower, or
   does it not cross within a realistic N range at all (i.e., is DeepSeek-V2
   decode memory-bound-proof the way `decode_notes.md`'s attention numerics
   floor was quantization-proof)? Any of these three outcomes is a real,
   reportable finding — don't go in assuming the crossover exists.
4. Decode throughput/chip at whatever N the derivation lands on, same
   service-time → throughput → ratio chain as 2a (§2.1's own reframing:
   regime label isn't the ratio driver, service time is).
5. **Prefill side**: project #3 didn't derive a prefill/decode split (it
   was a routing/topology project, not a serving-latency one) — you'll need
   DeepSeek-V2's prefill service time freshly, same seq_len=512 anchoring as
   2a for comparability. Sparse FFN at prefill should be structurally
   simpler than the decode amortization question (prefill's batch dimension
   is tokens-within-a-sequence plus batch-of-sequences, all present
   simultaneously — check whether that changes the distinct-experts-touched
   math from the decode case, or whether it's the same formula with a bigger
   effective N).
6. Final MoE chip-ratio hypothesis, stated the same way as 2a's "~5.5:1" —
   and explicitly compare direction and magnitude against 2a. Higher ratio
   (needs even more prefill chips relative to decode)? Lower? Say why,
   mechanistically, tied to whichever of the two boundary-case limits above
   the real numbers landed near.

**Explicitly out of scope for 2b** (flag as future work if it comes up,
don't chase it now): routing-*aware* batch co-scheduling — deliberately
grouping requests to increase expert overlap and push amortization further
than natural routing would give you. That's a real, published lever (it's
part of why production MoE serving systems care about routing-aware
batching) but it's a *policy* question layered on top of the *mechanism*
question 2b is asking. Solving the mechanism first, the same "characterize
before you optimize" order every prior project in this repo has followed.

**Open thread this phase will either close or hand to Phase 3:** if hot
experts really do warrant preferential SRAM residency (spec v1's original
Phase 3 question), the amortization curve from step 2 above is the direct
input to answering it — an expert that's disproportionately likely to be
one of the "distinct experts touched" in any batch is exactly the
SRAM-residency candidate. Don't answer this in 2b; just make sure 2b's
output is shaped so Phase 3 can consume it directly.

---

## Phase 2c — KV handoff mechanism (🧠, carried over from spec v1, still not started)

Unchanged from the original spec: transfer cost (bytes = Phase 1's
KV-cache-bytes/token, dense case; needs its own number if Phase 1 ever gets
a MoE-specific pass per the note above) and interconnect fabric choice
(same Boardfly fabric MoE project #3 already validated as latency-dominated,
or a dedicated channel — spec v1's open question, still open). No MoE-specific
change needed here *unless* Phase 2b's findings suggest KV-cache handling
itself differs for MoE (it shouldn't — KV cache is an attention-layer
artifact, MoE only touches the FFN sublayer — but worth a one-line sanity
check once 2b is done rather than assuming).

## Phase 3 — KV/weight placement policy (🧠, unchanged scope, now has real inputs)

Same as spec v1: eviction/placement policy research (PagedAttention paging,
quantization, sliding window, CPU offload), plus the hot-expert SRAM
residency question — which Phase 2b now actually supplies an answer to
instead of leaving as speculative.

## Phase 4 — Validate (🔧 build, 🧠 interpret)

Unchanged mechanically. One honest caveat to flag going in, not discover
mid-phase: DistServe and Mooncake, your Phase 0 reference numbers, are both
dense-transformer systems (OPT-175B). There's no equivalent public
production-scale MoE-disaggregation benchmark to sanity-check Phase 2b's
ratio against the way 2a was checked against DistServe's direction — 2b's
validation will have to lean more on internal consistency (does the
amortization curve behave sensibly at the boundary cases you hand-derived
first?) than external cross-checking. Say this explicitly in the writeup
rather than implying 2b got the same grade of validation 2a did.

## Phase 5 — Full synthesis (🧠, capstone, now three-legged)

Unchanged structurally from spec v1's ask (rack-scale budget across SRAM/
bandwidth/handoff), but now genuinely spans attention → MoE routing →
disaggregation-dense → disaggregation-MoE, four pieces of derived evidence
instead of three. The dense-vs-MoE chip-ratio contrast from Phase 2 is
itself worth a paragraph in this synthesis on its own, independent of the
rack-budget question — it's a direct, quotable answer to "does disaggregation
work the same way once you're serving a MoE model," which is the question
that started this whole revision.

---

## Fallback

Same as spec v1: Phases 1–2 (now including 2b) stand alone as a complete
artifact if time runs short. Given where you already are, the realistic
fallback line has moved — 2a is done, so the fallback floor is now
"2a + 2b without 2c/3/4/5," which is already a real, complete, comparable
dense-vs-MoE chip-ratio result even in the worst case.

## Note on scope (extended from v1)

v1's own scope argument — depth per decision beats breadth — is the reason
2b is scoped as an *addition to Phase 2*, not a fourth sequential project.
The temptation this revision has to actively resist: turning 2b into a full
MoE-serving project in miniature (routing-aware batching, dynamic expert
placement across the whole cluster, expert-parallelism-aware chip-ratio
math) instead of the one focused question it's actually asking — does the
FFN-batch-amortization mechanism that made dense's ratio non-obvious behave
the same way, better, or worse once the FFN is sparse. Everything flagged
"explicitly out of scope for 2b" above is real, legitimate material — for
whatever comes after this project, not smuggled into it.