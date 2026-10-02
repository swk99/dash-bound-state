# DASH: implementation-grounded theory and submission plan

The supplied repository is the July 2 snapshot. No Raju manuscript, memory CSV,
10-stream breakpoint experiment or gating rates were supplied. Their claims
cannot be verified from this snapshot. Preserve the cascade and model artifacts.

## T1: bounded in stream length, not in window length

Invariant: after each completed atomic update, the Redis list contains the most
recent min(N,W) canonical float32 vectors, each d values. A and B1 both use this
layout. Logical payload is 4 d min(N,W) bytes; for N >= W it is Theta(W d).
Allocator/list/key overhead is implementation dependent and must be measured;
it is not generally o(W). For fixed W and d the state is O(1) in N.
A avoids full-window retrieval for tabular inference; B1 retrieves the window.
This is an access-cost distinction, not a state-footprint distinction.
976 bytes at one W is evidence of N independence, not W independence.
RSS includes runtime, models and loaded historical arrays and is a different metric.
B2 holds the whole supplied raw array, so its total resident input is O(N d).

E2: `benchmark_2.run_memory_vs_nw` measures A/B1 with MEMORY USAGE SAMPLES 0,
W=10/30/60/120 and five repeats, reports actual key names, list length and RSS,
and deletes only its own keys. No model retraining or architecture changes.
At equal N/W/d expect equal payload; key lengths can affect measured overhead.

## T2: mixture quantiles

Use lower generalized inverse Q(q)=inf{x:F(x)>=q}. Assume 0<p<1 and 0<q<1.
With separated ordered supports (hot below warm), q <= 1-p belongs to the hot
component: Q_A(q)=Q_h(q/(1-p)); q > 1-p belongs to the warm component:
Q_A(q)=Q_w((q-1+p)/p). At the boundary use the hot upper endpoint; do not
substitute an undefined Q_w(0). Handle p=0 and p=1 as pure components.

For overlapping supports, F_A=(1-p)F_h+pF_w implies hot-component bounds
Q_h((q-p)/(1-p)) <= Q_A(q) <= Q_h(q/(1-p)) wherever the respective arguments
are strictly within (0,1). Likewise warm-component bounds are
Q_w((q-1+p)/p) <= Q_A(q) <= Q_w(q/p) on valid arguments. Outside those
ranges the corresponding bound is vacuous (-infinity or +infinity).
These bounds cover the high-miss region but need not be informative there.
Mixture results concern A, not L=A+B; quantiles do not add in general.
If b_min<=B<=b_max almost surely, Q_A(q)+b_min<=Q_L(q)<=Q_A(q)+b_max.

## T3: conditional capacity approximation, not an implementation theorem

K_h=floor(C_h/s), where s is measured per-symbol state and C_h is usable
state budget after Redis overhead. p=max(0,1-K_h/K) additionally assumes uniform
independent symbol requests and a full cache; it is not universal for LRU,
skewed requests or partial-list eviction. Solving p=1-q gives K_crit~K_h/q.
Redis evicts entire list keys; the present engines neither detect cache misses
nor reconstruct missing windows. Therefore a maxmemory sweep alone is not a
valid fast/slow experiment. E1, breakpoint CIs and workload invariance remain
deferred until a faithful reconstruction path exists. Do not create kdb+,
AKS or other components merely to produce a knee.

## T4: bounded pathwise savings

If L'=L-Delta with 0<=Delta<=delta almost surely, L-delta<=L'<=L.
Monotonicity and translation equivariance imply
0<=Q_L(q)-Q_L'(q)<=delta. For Q_L(q)>0 the relative bound is delta/Q_L(q).
This requires a deterministic upper bound on savings and a coupling preserving
all other costs. Observed maximum Stage-2 duration is not an almost-sure bound;
separate noisy benchmark timings do not establish the coupling.
For always-Stage-2 versus gated execution, savings occur when the gate CLOSES,
not when it opens. Forced Stage-2 can change predictions for those requests;
E0 compares A/B1/B2 with the SAME gating policy, not force versus normal.
High gate firing rates imply fewer avoided calls but do not alone validate T4.

## Statistics and utility

`analyze_latency.py` computes P50/P90/P95/P99 95% IID BCa intervals (10,000
resamples default), records degeneracy, and sweeps deadlines from 1 to 1000 ms.
Streaming dependence requires block/run-level resampling before treating these
as publication CIs. Five repeats should be independent runs and analysed at
run level; seed reproducibility does not remove host jitter.
Failure to reject unequal breakpoints is not proof of invariance; use equivalence
margins and sufficient power. Holm correction is for a predefined hypothesis family.
Report the mean/variance of measured footprint by N/W/d, not pooled across W.
Actual deadline-correct utility is P(L<=deadline AND correct). Coverage*Accuracy
requires correctness independent of meeting the deadline, which E0 does not prove.

## Lua prototype and remaining validation

Ingestion formerly used two sequential commands. Engines already pipelined
LPUSH/LTRIM in one round trip, without atomicity. Both now use the same cached
Lua script. Steady-state EVALSHA uses one round trip; initial load/NOSCRIPT may
need more. Expect ingestion RTT savings and engine atomicity, not guaranteed
engine speedup or an unchanged slow-tail regime.

Thresholds: xgb 0.5002538562, rf 0.5032152391, lr 0.5136591624 (Stage 1).
Defaults now preserve calibrated thresholds in benchmarks. Legacy root artifacts
are accepted when absent under artifacts/. sklearn is pinned to 1.6.1; this is
not a complete historical environment lock. Generated sym_* workloads are
controlled replication, not independent real market symbols.
Existing latency CSVs predate these fixes: retain them as historical results and
rerun before claiming calibrated-gate performance. Manuscript anonymity,
2.38x claims, 90 measurements and reported firing rates remain unverified.

Priority: Lua target-host test; E2; E0 on real canonical input; rerun gating with
separate matched streams; E6 and dependence-aware CIs. E1/T2/T3 are optional.
