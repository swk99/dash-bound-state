const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, HeadingLevel, AlignmentType, WidthType, ShadingType, LevelFormat, BorderStyle, Footer, PageNumber, TableOfContents } = require('docx');
const FONT = 'Arial', W = 9026; // A4 content width (DXA) with 1" margins
const p = (t, o = {}) => new Paragraph({ spacing: { after: 120 }, ...o, children: (Array.isArray(t) ? t : [t]).map(x => typeof x === 'string' ? new TextRun({ text: x, font: FONT, size: 21 }) : x) });
const b = (t) => new TextRun({ text: t, bold: true, font: FONT, size: 21 });
const code = (t) => new TextRun({ text: t, font: 'Consolas', size: 18 });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 140 }, children: [new TextRun({ text: t, font: FONT })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 }, children: [new TextRun({ text: t, font: FONT })] });
const li = (t, lvl = 0) => new Paragraph({ numbering: { reference: 'bul', level: lvl }, spacing: { after: 60 }, children: (Array.isArray(t) ? t : [t]).map(x => typeof x === 'string' ? new TextRun({ text: x, font: FONT, size: 21 }) : x) });
const cp = (t) => new Paragraph({ spacing: { after: 0 }, shading: { type: ShadingType.CLEAR, fill: 'F3F4F6' }, children: [code(t)] });
const border = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
function table(header, rows, widths) {
  const tot = widths.reduce((a, c) => a + c, 0);
  const cell = (t, i, hd) => new TableCell({ width: { size: widths[i], type: WidthType.DXA }, borders: { top: border, bottom: border, left: border, right: border },
    shading: hd ? { type: ShadingType.CLEAR, fill: 'DCE6F1' } : undefined, margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), font: FONT, size: 18, bold: !!hd })] })] });
  return new Table({ width: { size: tot, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }), ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))] });
}
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });
const C = [];
C.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'DASH Bounded-State Execution', font: FONT, size: 40, bold: true })] }));
C.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'Final Experiment Specification: Two-Environment P0 Replication and Stage-2 Prediction Equivalence', font: FONT, size: 26 })] }));
C.push(p([b('Target venue: '), 'Journal of Systems and Software (JSS) submission.   ', b('Version: '), '1.2, 2 October 2026.   ', b('Author: '), 'Seonwoo Kim.']));
C.push(p([b('Status: '), 'both environments completed: E1 cloud (rep_cloud_20261002T141735Z) and E2 author laptop (rep_laptop_20261002T201141Z); single-host robustness conditions completed (Section 13).']));

C.push(h1('1. Purpose and Scope'));
C.push(p('This specification fixes the final, minimal experiment that closes the empirical evidence for the JSS submission. It has two goals:'));
C.push(li([b('P0 replication: '), 'reproduce the same small latency experiment, with no added network delay, in two independent environments under identical code, artifacts, data and run settings, and test whether the direction of the A versus B1 tail-latency difference is maintained.']));
C.push(li([b('Stage-2 prediction equivalence (E0): '), 'show that the three execution paths (A, B1, B2) produce identical labels on a pre-registered window in which the Stage-2 gate actually opens.']));
C.push(p('The algorithm is frozen at the current version. Changing it now would require re-validating the new version and would enlarge the submission scope. Limitations that cannot be controlled are documented (Section 14); controllable conditions are fixed in the run configuration.'));
C.push(h2('1.1 Explicitly out of scope'));
['Model retraining or threshold recalibration.', 'New data collection or new datasets.', 'Network impairment (P1–P4 jitter/delay profiles). This P0 replication does not validate the earlier proxy-based jitter results.',
 'Capacity/E1 experiments (T2 cache-miss knee, T3 capacity law), maxmemory or eviction sweeps.', 'Open-loop arrival-rate or multi-client concurrency experiments.',
 'New storage layers (kdb+, SQL, graph DB), Kubernetes or multi-region deployment.', 'Breakpoints, capacity laws or p-values not supported by an executed experiment.'].forEach(t => C.push(li(t)));

C.push(h1('2. Fixed Experimental Conditions'));
C.push(table(['Item', 'Fixed value'], [
  ['Environments', 'E1 = current cloud container (reference); E2 = independent experiment PC or separate VM'],
  ['Code / models / data', 'Identical files, verified by SHA256 before measurement (Section 4)'],
  ['Model (latency)', 'XGBoost two-stage cascade, calibrated thresholds from thresholds_H30_a2_L94.json'],
  ['Inference threads', 'Pinned to 1 (Section 6)'],
  ['State', 'W = 30, d = 6 features, float32'],
  ['Resident streams K', '1 and 100'],
  ['Engines', 'A: ProposedStatefulEngine; B1: RedisFetchBaselineEngine; B2: InMemoryRecomputeEngine (no Redis; control)'],
  ['Network', 'P0_direct only: client connects directly to Redis over loopback TCP; no proxy, no netem'],
  ['Gating policy', 'Normal gating (no force_stage2)'],
  ['Repeats', '5 independent state initialisations per K'],
  ['Samples', '200 warmup + 5,000 measured requests per engine x K x repeat (plus W fill requests per stream)'],
  ['Execution order', 'Engines run sequentially; order randomised per repeat with a recorded seed'],
  ['Client concurrency', '1 (closed loop)'],
  ['Request timeout', '1,000 ms (redis-py socket_timeout = 1.0 s)'],
  ['Outputs', 'Per-repeat P99, A/B1 difference and ratio, failure rate, deadline coverage, host CPU utilisation'],
], [2600, 6426]));

C.push(h1('3. Environments'));
C.push(h2('3.1 E1: cloud reference environment (as measured)'));
C.push(table(['Property', 'Recorded value'], [
  ['Host', 'Ephemeral cloud container (Firecracker VM), shared host, co-tenancy unknown'],
  ['CPU / vCPU / RAM', 'Intel Xeon @ 2.10 GHz / 2 vCPU / 8.4 GB'],
  ['OS', 'Linux 6.18.44 (x86_64), glibc 2.39'],
  ['Python', '3.13.15 (README target 3.11 not met)'],
  ['Key packages', 'scikit-learn 1.6.1, xgboost 3.4.1, redis-py 8.1.0, numpy 2.5.3, pandas 3.0.5, torch 2.14.1'],
  ['Redis', '7.0.15, same host, loopback, default configuration (noeviction, maxmemory unset); restarted in an empty data directory after a container reboot, before this run'],
  ['Redis client', 'redis-py default ConnectionPool; one connection used sequentially'],
], [2600, 6426]));
C.push(h2('3.2 E2: author laptop (as measured)'));
C.push(table(['Property', 'Recorded value'], [
  ['Host', 'Author laptop, Windows with WSL2 Ubuntu (kernel 6.18.35.2-microsoft-standard-WSL2, glibc 2.43); mains power, other applications closed'],
  ['CPU / logical CPUs / RAM', 'Intel Core 5 210H / 12 / 8.1 GB visible to WSL2'],
  ['Python', '3.13.16 in a uv-managed venv (system Python too new for scikit-learn 1.6.1 wheels)'],
  ['Key packages', 'scikit-learn 1.6.1, xgboost 3.4.1, redis-py 8.1.0, numpy 2.5.3, pandas 3.0.6, torch 2.14.1+cpu'],
  ['Redis', '8.0.5 (Ubuntu package), same host, loopback, default configuration'],
  ['Integrity', 'SHA256 check all_match (artifacts, data manifest, input array, repository code, run script)'],
  ['Host CPU during run', '7–8 % mean'],
], [2600, 6426]));
C.push(gap());
C.push(p('E1 and E2 differ in hardware, virtualisation and Redis version (7.0.15 vs 8.0.5). They are therefore two heterogeneous environments rather than identical instances; results are reported per environment and never pooled.'));
C.push(h2('3.3 Requirements for any further environment'));
['Same repository, artifact and data files (SHA256 check must report all_match; the script stops otherwise).', 'Redis reachable directly (no proxy, no netem); an experiment-only Redis instance is preferred.',
 'scikit-learn 1.6.1 (artifact compatibility); other versions are recorded, not changed.', 'No other heavy workload during the run (about 4 minutes on E1).',
 'Record CPU model, vCPU, RAM, OS, Redis version and Redis location (automatically written to run_meta.json).'].forEach(t => C.push(li(t)));

C.push(h1('4. Software, Artifacts and Integrity'));
C.push(p('The run script computes SHA256 for every artifact, the data-file manifest, the canonical float32 input array, the repository code and itself, and compares them with expected_sha256.json (generated on E1). A mismatch aborts the run unless DASH_ALLOW_MISMATCH=1 is set, in which case the mismatch is recorded in the output.'));
C.push(table(['Object', 'SHA256'], [
  ['run_replication.py', 'f39f9fb86ffe4d8f80ebc9fd513d16e992e581a58abcec11dc615a21f80c02b3'],
  ['assemble_replication.py', '4cb32e30c64ef5dc4519998519b9496d6b63890a05cbc4a55736e9fa2250a785'],
  ['Repository *.py (combined)', '7a47b365f9e6f648e1f29334ffe9e9af30a96e6c74bdecc05e585a1d51b850ea'],
  ['Input array (float32, canonical order)', 'ab56a05058af6310fbb694e02c729d8268d23a45f0bf459f4a1586f2cdc676d2'],
  ['Data file manifest (75 parquet files)', 'dff1a6053ed595a9c7d37eb403207ec73524cbcf0642642bd6482297c67ae79b'],
  ['s1_xgb_H30_a2_L94.json', '1404dd6d6c7a1eb38f98e8d82dfc071b4db81d3853c84b7fa31b3ab9e5f9977a'],
  ['s2_xgb_H30_a2_L94.json', '987e58456f730c44bf48d669680405dcb79d291338ece805642ece8576a0ee9f'],
  ['s1_rf_H30_a2_L94.pkl', '5051a3329ec9b4ad0facd44201f63c7db4353a0edb31dc248af91b0108f70914'],
  ['s2_rf_H30_a2_L94.pkl', '863a778357023a267ed00f737577e3756efd2d857c08dfa9985a263b0f755aaf'],
  ['artifacts/s1_lr_H30_a2_L94.pkl', '925de0eb9d460e401bd70d0bfcf615bfc35ba9714af666ab0cbe6694998c7bc7'],
  ['s2_lr_H30_a2_L94.pkl', 'afcc15fb8e0d6965e8fbd302b12ab3d111b333d45b6d3618a829a528faad09f9'],
  ['scaler_H30_a2_L94.pkl', 'bb747f7456f13f17f21334e19c33849b1725604f5b06a75b4a7a3b28340e37ff'],
  ['thresholds_H30_a2_L94.json', '9c567b7d82502b18a9ea9ed1599c7683c1f9d2a6005dbb5eef81d045dddfdd4f'],
], [3000, 6026]));
C.push(gap());
C.push(p([b('Calibrated thresholds applied (read back from the loaded harness): '), 'XGBoost tau_s1 = 0.5002538561820984, tau_s2 = 0.386002; RandomForest tau_s1 = 0.5032152391133569; Logistic tau_s1 = 0.5136591624213244. cfg.TAU_CONF = 0.6731 is a fallback only and does not override the JSON values.']));

C.push(h1('5. Data and Workload'));
C.push(h2('5.1 Data'));
C.push(table(['Property', 'Value'], [
  ['Symbol', 'btcusdt (real 1-second bars, MinIO flush export; not newly collected)'],
  ['Files', 'flush_0 … flush_74 parquet, concatenated in flush-index order'],
  ['Rows', '74,855; timestamps monotonic increasing; 0 NaN, 0 Inf in features'],
  ['Feature order', 'r_t, sigma_hat, OFI_t, Imbalance_t, VolSpike_t, msg_count (d = 6)'],
  ['Preparation', 'Converted once to a contiguous float32 array before any timing'],
], [2600, 6426]));
C.push(h2('5.2 Workload definition (controlled replication)'));
C.push(p('K is the number of resident logical streams/states. Each stream j uses a deterministic, contiguous, non-overlapping segment of the series starting at offset j x L, with L = W + ceil(5,200 / K). The client is single-threaded and closed-loop (client_concurrency = 1). Requests are issued in global round-robin order across streams; stream-local tick order is preserved.'));
['Fill: W = 30 requests per stream (stream-major), excluded from samples.', 'Warmup: 200 global round-robin requests, excluded.', 'Measurement: the next 5,000 global requests (K = 100: about 50 per stream).',
 'A, B1 and B2 use exactly the same schedule and inputs within a repeat.', 'K streams are replicas from one symbol. They are not K real symbols, not concurrent in-flight requests and not an arrival-load experiment.'].forEach(t => C.push(li(t)));

C.push(h1('6. Engines, Timing and Thread Control'));
C.push(table(['Engine', 'Per-tick state work', 'state_ms (A_ms) covers'], [
  ['A ProposedStateful', 'Atomic Lua LPUSH+LTRIM (1 Redis round trip)', 'RedisWindowWriter.push including feat.tobytes(); no LRANGE for non-LSTM models'],
  ['B1 RedisFetch', 'Lua push + LRANGE(W) + float32 decode (2 round trips)', 'push + LRANGE + decode of W entries'],
  ['B2 InMemoryRecompute', 'In-memory window slice + z-score/diff recompute; no Redis', 'slice + recompute'],
], [2200, 3400, 3426]));
C.push(gap());
C.push(p('latency_ms is the total process_tick wall time (perf_counter_ns). stage1_ms and stage2_ms are measured inside the model wrapper. Feature conversion (_to_feat_vec) lies outside state_ms but inside latency_ms, so components do not sum exactly to the total.'));
C.push(h2('6.1 Thread pinning'));
['OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = NUMEXPR_NUM_THREADS = 1, set before numpy import.', 'torch.set_num_threads(1).',
 'XGBoost: set_params(n_jobs=1) and booster.set_param(nthread=1) on both stages, then nthread is read back from the booster configuration and recorded (E1 readback: s1 = 1, s2 = 1).',
 'Host CPU utilisation is sampled every second by a background thread during each engine run and recorded per engine/K/repeat.'].forEach(t => C.push(li(t)));

C.push(h1('7. Procedure'));
['Verify SHA256 against expected_sha256.json; abort on mismatch.', 'Write the pre-registered decision rules (Section 10) to preregistered_rules_<env>.json before any measurement.',
 'Load the XGBoost harness once; pin threads; verify readback.', 'For K in (1, 100), for repeat r in 0..4: draw engine order = default_rng([20260928, r, K, 0, 4]).permutation([A, B1, B2]); for each engine, use fresh keys under the run prefix, run fill + warmup + 5,000 measured requests, delete that engine\'s keys, record host CPU.',
 'Save each completed K/repeat as a checkpoint (parquet) outside the timed region; completed results survive interruption.', 'Run the Stage-2 E0 (Section 8).',
 'Cleanup: delete only keys under the run prefix and confirm 0 remain.', 'Assemble dash_replication_results.csv and run the validation checks (Section 11).'].forEach(t => C.push(new Paragraph({ numbering: { reference: 'num', level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text: t, font: FONT, size: 21 })] })));
C.push(h2('7.1 Safety rules'));
['Experiment-only key prefix jss4:<run_id>:.', 'Never FLUSHALL, FLUSHDB or SCRIPT FLUSH; never change maxmemory, eviction or persistence settings.', 'No network configuration changes.',
 'Models are never run in parallel; no MCP or polling inside the measured loop.'].forEach(t => C.push(li(t)));

C.push(h1('8. Stage-2 Prediction Equivalence (E0)'));
C.push(p([b('Window selection rule (pre-registered): '), 'for each model, compute Stage-1 probability pi_1 only (scaler + s1.predict_proba) over the full array. Select the first contiguous 1,000-row window whose start is a multiple of 100 and in which the fraction of pi_1 >= tau_s1 is at least 0.20. Labels are not inspected before selection.']));
['Models: Logistic, RandomForest, XGBoost; W = 30; normal gating; separate Redis keys for A and B1; B2 on the same rows.', 'Record per tick: prediction per path, stage2_fired, reference_prediction (= A-path label) and label_match.',
 'Report: mismatch count vs A, Stage-2 fired ticks, counts of each Stage-2 output label, window and whole-dataset Stage-1 open fractions.',
 'Interpretation: agreement between execution paths only. There is no ground truth: no accuracy, F1 or AUC is computed. The window firing rate is a property of the selected window and must not be reported as the representative firing rate; use the whole-dataset rate for that.'].forEach(t => C.push(li(t)));

C.push(h1('9. Metrics and Statistics'));
['Per engine x K x repeat: measured count, success/failure counts, failure rate, mean, SD, P50/P90/P95/P99 (numpy linear quantile, Hyndman–Fan type 7), Stage-2 firing rate, mean state_ms and stage1_ms, host CPU mean/max.',
 'Deadline coverage at 1, 5, 10, 20, 50, 100 and 1,000 ms = count(success and latency <= deadline) / all measured requests. 1,000 ms is a 1-second bar reference, not an SLA.',
 'A/B1 pairing by repeat and schedule: P99_B1 / P99_A and P99_B1 − P99_A per repeat. Engines run sequentially, so tick-level noise is not shared.',
 'Across repeats: mean, median, SD, min and max of the repeat-level values. A pooled P99 within one environment is a separately named metric.',
 'Raw data from different environments are never pooled into one quantile.',
 'Uncertainty: with 5 runs, results are descriptive and no confidence interval is claimed. P99 of 5,000 requests rests on about 50 largest observations.'].forEach(t => C.push(li(t)));

C.push(h1('10. Pre-Registered Decision Rules'));
C.push(table(['Rule', 'Definition'], [
  ['Direction (per environment, per K)', 'Number of repeats (of 5) with P99_B1 / P99_A > 1, and sign of (median ratio − 1)'],
  ['Cross-environment direction maintained', 'For each K, the median-ratio sign agrees in E1 and E2'],
  ['Noise gate', 'B2 (no Redis) per-repeat P99 coefficient of variation; CV > 0.25 flags the environment as noisy and A/B1 differences there are reported as inconclusive'],
  ['Scope reduction', 'If the median ratio <= 1.10 or signs disagree, report "benefit is limited without added network delay"'],
  ['Pooling', 'Never pool raw data across environments'],
], [3000, 6026]));
C.push(gap());
C.push(p('A small difference or a reversal is reported as observed. Scope is narrowed accordingly; results are never selected after the fact.'));

C.push(h1('11. Output and Validation'));
C.push(p('A single UTF-8 CSV, dash_replication_results.csv, with one header (the same column schema as dash_paper_results.csv), long format, rows grouped by a section column:'));
C.push(table(['Section', 'Content'], [
  ['00_run_manifest', 'Design, data manifest, pre-registered rules, artifact SHA256'], ['01_environment', 'Versions, hardware, Redis location, thread readback, SHA check result'],
  ['02_artifact_checks', 'Thread pinning checks per model'], ['05_prediction_equivalence', 'Stage-2 E0 raw ticks and summaries'],
  ['07_latency_raw', 'One row per environment/K/engine/repeat/request'], ['08_latency_run_summary', 'Per-repeat metrics and paired A/B1 ratio/difference'],
  ['09_latency_condition_summary', 'Across-repeat aggregates and pre-registered decisions'], ['10_deadline_coverage', 'Per-repeat and aggregated coverage'],
  ['13_historical_results', 'Prior P0-via-proxy reference rows (different path, not pooled)'], ['14_experiment_status', 'done / not_run / deferred per planned item'], ['15_limitations', 'One row per limitation'],
], [3000, 6026]));
C.push(gap());
C.push(p([b('Validation before release: '), 'write to a temporary file, re-read, then check column match, row count, section order, no duplicate primary keys, finite non-negative latencies, consistent true/false booleans, coverage in [0, 1], ISO-8601 UTC timestamps, and recompute counts and quantiles from raw rows. The final file is written only if every check passes (E1: 9/9 pass, 160,311 rows).']));

C.push(h1('12. Two-Environment Results'));
C.push(h2('12.1 E1: cloud'));
C.push(p('XGBoost, W = 30, P0_direct, threads pinned to 1, 5 repeats, 5,000 measured requests each; 0 failures; host CPU mean 45–47 %.'));
C.push(table(['K', 'Engine', 'P50 mean (ms)', 'P99 mean (SD) (ms)'], [
  ['1', 'A', '0.83', '1.78 (0.15)'], ['1', 'B1', '1.07', '2.42 (0.42)'], ['1', 'B2 (control)', '0.62', '1.36 (0.02)'],
  ['100', 'A', '0.86', '1.89 (0.20)'], ['100', 'B1', '1.07', '2.16 (0.13)'], ['100', 'B2 (control)', '0.63', '1.48 (0.11)'],
], [1200, 2400, 2400, 3026]));
C.push(gap());
C.push(table(['K', 'Per-repeat P99_B1 / P99_A', 'Repeats > 1', 'Median', 'B2 CV', 'Decision'], [
  ['1', '1.56, 1.23, 1.50, 1.28, 1.19', '5 / 5', '1.28', '0.014', 'Direction B1 > A; not noisy; scope not reduced'],
  ['100', '1.24, 1.17, 1.25, 1.15, 0.95', '4 / 5', '1.17', '0.072', 'Direction B1 > A (one reversal); not noisy; scope not reduced'],
], [700, 2600, 1100, 900, 900, 2826]));
C.push(gap());
C.push(p('An earlier E1 run with an identical protocol but a pre-final script (before the SHA-check patch) gave median ratios 1.18 (K = 1) and 1.14 (K = 100). It is preserved as superseded. The effect size varies between runs, so conclusions should rest on direction rather than magnitude.'));
C.push(table(['Model', 'Window start', 'Stage-2 fired ticks', 'Labels exercised', 'Mismatches vs A (B1, B2)', 'Dataset Stage-1 open fraction'], [
  ['Logistic', '700', '214 (21.4 %)', '2: 141, 1: 73', '0, 0', '18.1 %'], ['RandomForest', '0', '224 (22.4 %)', '2: 192, 1: 32', '0, 0', '28.8 %'], ['XGBoost', '0', '307 (30.7 %)', '2: 261, 1: 46', '0, 0', '27.2 %'],
], [1400, 1100, 1500, 1500, 1700, 1826]));
C.push(gap());
C.push(h2('12.2 E2: author laptop'));
C.push(p('Same protocol and files; 0 failures; host CPU mean 7–8 %.'));
C.push(table(['K', 'Engine', 'P50 mean (ms)', 'P99 mean (SD) (ms)'], [
  ['1', 'A', '0.98', '4.07 (1.76)'], ['1', 'B1', '1.20', '4.52 (2.04)'], ['1', 'B2 (control)', '0.60', '2.32 (0.20)'],
  ['100', 'A', '0.90', '2.81 (0.55)'], ['100', 'B1', '1.22', '4.27 (1.08)'], ['100', 'B2 (control)', '0.60', '2.58 (0.61)'],
], [1200, 2400, 2400, 3026]));
C.push(gap());
C.push(table(['K', 'Per-repeat P99_B1 / P99_A', 'Repeats > 1', 'Median', 'B2 CV', 'Decision'], [
  ['1', '0.73, 2.76, 0.89, 0.75, 1.26', '2 / 5', '0.89', '0.087', 'Direction reversed at median; not noisy by rule; scope rule applies (benefit limited)'],
  ['100', '1.44, 2.31, 0.91, 1.34, 1.96', '4 / 5', '1.43', '0.234', 'Direction B1 > A; noise gate passed but close to 0.25'],
], [700, 2600, 1100, 900, 900, 2826]));
C.push(gap());
C.push(p('E0 on E2 selected the same windows (rows 700, 0, 0) with identical Stage-2 counts (214, 224, 307) and 0 mismatches for all three models.'));
C.push(h2('12.3 Cross-environment decision (pre-registered rule)'));
C.push(table(['K', 'E1 median ratio', 'E2 median ratio', 'Direction maintained?'], [
  ['1', '1.28', '0.89', 'No (not_maintained)'], ['100', '1.17', '1.43', 'Yes (maintained)'],
], [1500, 2500, 2500, 2526]));
C.push(gap());
C.push(p([b('Interpretation. '), 'With a single resident stream and no added network delay, the A versus B1 P99 relationship is not stable across environments: it holds on E1 and reverses at the median on E2, where per-repeat P99 is dominated by host noise (A P99 SD 1.76 ms). With 100 resident streams the direction is maintained in both environments. Following the pre-registered scope rule, the paper states that the tail-latency benefit of A is limited without added network delay at K = 1, and is observed in both environments at K = 100.']));
C.push(p([b('Descriptive, not pre-registered: '), 'P50_B1 / P50_A exceeded 1 in all 20 repeats across both environments and both K (range 1.08–1.47). This is consistent with B1 performing one extra Redis round trip per tick, but because it was not a pre-registered decision metric it is reported as supporting description only.']));

C.push(h1('13. Controlled-Condition Robustness (Single Host)'));
C.push(p('Same container, code, artifacts and data as Section 12, varying one controlled condition at a time. This is a robustness observation on one host, not cross-environment replication. Rules and predictions were written to preregistered_rules_robustness.json before measurement.'));
C.push(h2('13.1 Design'));
C.push(table(['Item', 'Value'], [
  ['Common', 'XGBoost, K = 1, P0_direct, inference threads = 1, A/B1/B2, 5 repeats, 200 warmup + 5,000 measured'],
  ['Exp RW (window)', 'W in {10, 30, 60, 120}; client affinity 2 cores'],
  ['Exp RC (CPU allocation)', 'W = 30; client process affinity {cpu0} (1 core) vs {cpu0, cpu1} (2 cores) via os.sched_setaffinity; Redis affinity unchanged (all CPUs)'],
  ['Ordering', 'Per repeat, conditions run in seeded permuted order (RW: default_rng([20260928, r, 11]); RC: default_rng([20260928, r, 12])); engines permuted within each condition'],
  ['Script SHA256', 'run_robustness.py; recorded in dash_robustness_results.csv'],
], [2600, 6426]));
C.push(h2('13.2 Pre-registered predictions'));
['B1 state_ms (push + LRANGE + decode) increases monotonically with W.', 'A state_ms (push only for non-LSTM models) does not depend on W: max/min of mean state_ms across W <= 1.25.',
 'B2 state_ms increases with W (O(W) recompute).', 'Median P99_B1/P99_A across repeats increases with W (Spearman sign, descriptive, 4 points).',
 'Restricting the client to 1 core does not reverse the A/B1 direction at W = 30 (median ratio > 1 in both).', 'Noise gate: B2 P99 CV > 0.25 flags a condition as noisy.'].forEach(t => C.push(li(t)));
C.push(h2('13.3 Results'));
C.push(p('90 engine runs, 0 failed requests, 0 leftover keys. Noise gate passed for all 6 conditions (B2 P99 CV 0.03–0.11). Values are means over 5 repeats (ms).'));
C.push(table(['W', 'A state_ms', 'B1 state_ms', 'B2 state_ms', 'A P99', 'B1 P99', 'Median B1/A (repeats > 1)'], [
  ['10', '0.214', '0.387', '0.068', '1.54', '1.93', '1.30 (5/5)'], ['30', '0.226', '0.444', '0.072', '1.67', '2.15', '1.30 (5/5)'],
  ['60', '0.231', '0.505', '0.074', '1.73', '2.19', '1.14 (5/5)'], ['120', '0.233', '0.614', '0.078', '1.69', '2.34', '1.48 (5/5)'],
], [900, 1200, 1200, 1200, 1100, 1100, 2326]));
C.push(gap());
C.push(table(['CPU allocation (W = 30)', 'A P99', 'B1 P99', 'Median B1/A', 'Repeats with B1 > A'], [
  ['1 core (client)', '1.89', '2.22', '1.27', '4 / 5'], ['2 cores (client)', '1.79', '2.13', '1.19', '3 / 5'],
], [2800, 1300, 1300, 1600, 2026]));
C.push(gap());
C.push(table(['Prediction', 'Outcome'], [
  ['B1 state_ms monotonic in W', 'Supported (0.387 → 0.444 → 0.505 → 0.614 ms)'],
  ['A state_ms independent of W (<= 1.25)', 'Supported (max/min = 1.09)'],
  ['B2 state_ms increases with W', 'Supported (0.068 → 0.078 ms)'],
  ['Tail ratio increases with W', 'Not supported (Spearman 0.40; medians 1.30, 1.30, 1.14, 1.48)'],
  ['1 core does not reverse direction', 'Supported at median level (1.27 and 1.19 > 1); per-repeat reversals: 1/5 (1 core), 2/5 (2 cores)'],
], [3600, 5426]));
C.push(h2('13.4 Interpretation'));
['Mechanism: the structural prediction is confirmed. B1 state cost grows with W, while A state cost is W-independent for non-LSTM models.',
 'Tail latency: without added network delay, the A advantage in P99 is modest (median 1.1–1.5x) and can reverse in individual repeats. It does not grow monotonically with W, so no W-driven tail claim is made.',
 'Operating regime: combined with the earlier proxy-based delay profiles (reported separately, with their caveats), the evidence is consistent with "the A advantage grows with the per-round-trip Redis cost". This is a regime description, not a guarantee.',
 'All statements are qualified as single host, controlled conditions. Independent deployment instances (same image on separate VMs) remain not_run.'].forEach(t => C.push(li(t)));
C.push(h1('14. Constraints, Limitations and Claim Scope'));
C.push(p('This section consolidates every constraint on what the evidence supports. Items are grouped by origin; each maps to a limitation row in the CSV outputs and to the Limitations sheet of the results workbook.'));
C.push(h2('14.1 Fixed scope constraints (by design)'));
['No model retraining or threshold recalibration; calibrated thresholds from thresholds_H30_a2_L94.json only.', 'Single symbol (btcusdt), existing data only; no new data collection.',
 'Closed-loop, single client (client_concurrency = 1); K = resident replicated streams, not real symbols, concurrent requests or arrival load.', 'No E1 capacity, maxmemory/eviction, open-loop or multi-client experiments; no T2/T3 validation.',
 'Algorithm frozen at the current version (Lua push+trim writer, unchanged engines).'].forEach(t => C.push(li(t)));
C.push(h2('14.2 Environment constraints'));
['E1: shared 2 vCPU cloud container; container-local Redis 7.0.15 restarted after a reboot; not the original experiment Redis.', 'E2: one laptop under WSL2 with Redis 8.0.5; heterogeneous with E1 (hardware, virtualisation, Redis version).',
 'Only two environments; no homogeneous multi-instance replication (cloud VM attempts on Azure for Students failed for capacity/policy reasons).', 'Python 3.13 rather than the README target 3.11 in both environments.',
 'Network impairment experiments (P1–P4) used a user-space Python TCP proxy because netem was unavailable; they also used default XGBoost threading and are not comparable with the P0-direct runs.'].forEach(t => C.push(li(t)));
C.push(h2('14.3 Statistical constraints'));
['Five repeats per condition; repeat-level descriptive statistics only; no confidence intervals and no dependence-aware (block) inference.', 'P99 of 5,000 requests rests on about 50 largest observations per repeat.',
 'A/B1 pairing is by repeat and schedule; engines run sequentially, so tick-level noise is not shared.', 'Effect sizes vary between runs of the same protocol (E1 superseded run: medians 1.18 / 1.14 versus 1.28 / 1.17); conclusions rest on direction, not magnitude.',
 'P50 consistency is descriptive and was not a pre-registered decision metric.'].forEach(t => C.push(li(t)));
C.push(h2('14.4 Claim scope'));
['E1 is a shared 2 vCPU cloud container with a container-local Redis (restarted before the run); absolute latencies are not transferable.', 'Python 3.13 rather than 3.11.',
 'Closed-loop single client and K replicated streams from one symbol: no multi-client, arrival-load, queueing-SLA or capacity inference. No saturation was observed, so no scalability limit is claimed.',
 'Five repeats; P99 depends on about 50 observations per repeat; no confidence intervals; no dependence-aware (block) inference.',
 'The background CPU monitor adds small, engine-independent interference.', 'E0 windows are rule-selected to exercise Stage 2 and are not representative; E0 shows path agreement, not accuracy.',
 'The P0 replication does not validate the earlier jitter results, and jitter-induced tail growth is not evidence for T2/T3.',
 'Observed Stage-2 wall-clock times do not establish a deterministic bound (T4); T4 is stated only structurally (Stage 2 is skipped when the gate is closed).',
 'State size is bounded in N for fixed W and d, with logical payload 4 x d x min(N, W) bytes; it is not W-independent.',
 'Robustness conditions (Section 13) vary one factor on one host; the 1-core condition restricts only the client process.', 'Lua is claimed for atomicity of push+trim; its latency against a non-transactional pipeline (same round trips) is not distinguishable.'].forEach(t => C.push(li(t)));

C.push(h1('Appendix A. Run Commands (E2)'));
['export DASH_REPO=/path/to/dash-bound-state-main', "export DASH_DATA_GLOB='/path/to/btcusdt/flush_*.parquet'", 'export DASH_ENV_LABEL=labpc REDIS_HOST=localhost REDIS_PORT=6379',
 'python run_replication.py        # writes ckpt_labpc/, progress_labpc.log', 'python assemble_replication.py    # merges ckpt_cloud/ + ckpt_labpc/'].forEach(t => C.push(cp(t)));
C.push(gap());
C.push(p('Return dash_replication_results.csv, the ckpt_labpc/ folder and progress_labpc.log. The package dash_labpc_replication_package.zip contains the scripts, columns.txt, expected_sha256.json, the E1 checkpoints and a README with Windows PowerShell equivalents.'));
C.push(h1('Appendix B. Seeds'));
['Base seed 20260928; repeat label seed = 20260928 + repeat.', 'Engine order: numpy default_rng([20260928, repeat, K, 0, 4]).permutation([A, B1, B2]).', 'E0 window selection is deterministic (rule-based, no randomness).'].forEach(t => C.push(li(t)));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [{ id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 30, bold: true, color: '1F3864', font: FONT }, paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 24, bold: true, color: '2E5597', font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } }] },
  numbering: { config: [{ reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: 'num', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ font: FONT, size: 16, children: ['DASH final experiment specification · page ', PageNumber.CURRENT] })] })] }) },
    children: C }] });
Packer.toBuffer(doc).then(buf => fs.writeFileSync('DASH_Final_Experiment_Specification.docx', buf));
