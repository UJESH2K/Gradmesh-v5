/**
 * `npm run test:scheduler` — the scheduler's behaviour, asserted.
 *
 * These are the claims the paper makes, so they are worth failing loudly. The
 * suite needs only the standard library, so it runs before torch is installed.
 */

import { ENGINE_DIR, findSystemPython, paint, venvPython, venvReady } from "./lib/env.mjs";
import { spawnSync } from "node:child_process";
import { writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const SUITE = String.raw`
import sys, time, json
sys.path.insert(0, ".")
from coordinator.scheduler import (
    MeshPolicy, admit, aggregation_weights, calibration_factor, deadline_for,
    efficiency, imbalance, plan_round, safe_batch_size, should_abort_round,
    straggler_action, update_reliability, update_throughput, update_fixed, affine_split,
    PARTITION_EQUAL, PARTITION_LINEAR, PARTITION_PROPORTIONAL,
)

failures = []

def expect(name, condition, detail=""):
    if condition:
        print("  ok   " + name)
    else:
        failures.append(name)
        print("  FAIL " + name + ("  " + detail if detail else ""))

now = time.time()

def node(nid, gflops, mem=8192, thr=0.0, rel=0.85, lat=5.0, active=True, training=True, fixed=0.0, backend="cuda"):
    return {
        "node_id": nid, "capability": {"gflops": gflops}, "gpu_memory_mb": mem,
        "allocated_memory_mb": 0, "last_seen": now, "active": active,
        "supports_training": training, "reliability": rel, "latency_ms": lat,
        "throughput_sps": thr, "fixed_seconds": fixed, "backend": backend,
    }

# 1. A heterogeneous mesh finishes together rather than waiting on the slowest.
pool = [node("fast", 4200), node("mid", 1800), node("slow", 700)]
plan = plan_round(pool, 1200)
times = [a.predicted_seconds for a in plan.assignments]
expect("shards equalise predicted finish time", max(times) - min(times) <= 0.5 * max(times) * 0.1,
       json.dumps(times))
expect("every sample is assigned exactly once",
       sum(a.samples for a in plan.assignments) == 1200)
expect("the fastest node carries the largest shard",
       max(plan.assignments, key=lambda a: a.samples).node_id == "fast")
expect("predicted speedup beats a single GPU", plan.predicted_speedup > 1.0,
       str(plan.predicted_speedup))

# 2. Adding a weak machine must not slow the mesh down.
plan_without = plan_round([node("fast", 4200), node("mid", 1800)], 1200)
expect("adding a weak node does not raise makespan",
       plan.predicted_makespan_seconds <= plan_without.predicted_makespan_seconds + 1e-6,
       "%.2f vs %.2f" % (plan.predicted_makespan_seconds, plan_without.predicted_makespan_seconds))

# 3. Admission control keeps unusable hardware out, with a reason.
gate = admit([node("tiny", 4000, mem=512), node("cpu-only", 0, training=False), node("good", 3000)])
expect("a device below the memory floor is rejected", gate["tiny"].tier == "rejected")
expect("a device with no accelerator is rejected", gate["cpu-only"].tier == "rejected")
expect("every decision carries a human reason", all(d.reason for d in gate.values()))
expect("a capable device is admitted", gate["good"].tier == "full")

# 4. A single node still forms a valid mesh.
solo = plan_round([node("only", 2500)], 300)
expect("a single node receives the whole dataset",
       len(solo.assignments) == 1 and solo.assignments[0].samples == 300)

# 5. Straggler policy escalates in the right order.
d = deadline_for(60.0)
expect("a worker inside its deadline is left alone", straggler_action(30, d, True) == "wait")
expect("a soft miss with an idle peer speculates", straggler_action(d.soft_seconds + 1, d, True) == "speculate")
expect("a soft miss with no idle peer waits", straggler_action(d.soft_seconds + 1, d, False) == "wait")
expect("a hard miss drops the shard", straggler_action(d.hard_seconds + 1, d, True) == "drop")
cold = deadline_for(60.0, cold=True)
expect("a first round gets a much longer deadline",
       cold.soft_seconds >= MeshPolicy().cold_start_grace_seconds and cold.soft_seconds > d.soft_seconds,
       "%.0f vs %.0f" % (cold.soft_seconds, d.soft_seconds))
expect("a warm round is unaffected by the cold grace", d.soft_seconds == deadline_for(60.0).soft_seconds)
expect("a first round is not declared late at its own predicted time",
       straggler_action(70, cold, True) == "wait")

expect("losing most of the data aborts the round", should_abort_round(700, 1000))
expect("losing a little data does not abort", not should_abort_round(50, 1000))

# 6. Aggregation weights follow contribution.
weights = aggregation_weights([
    {"batch_id": "big", "samples": 800, "reliability": 1.0},
    {"batch_id": "small", "samples": 200, "reliability": 1.0},
])
expect("weights sum to one", abs(sum(weights.values()) - 1.0) < 1e-9)
expect("the larger shard carries more weight", weights["big"] > weights["small"])
expect("weights are proportional to samples", abs(weights["big"] - 0.8) < 1e-9)

penalised = aggregation_weights([
    {"batch_id": "trusted", "samples": 500, "reliability": 1.0},
    {"batch_id": "flaky", "samples": 500, "reliability": 0.4},
])
expect("an unreliable worker is damped", penalised["trusted"] > penalised["flaky"])

# 7. The throughput model learns from observation.
learner = node("learner", 1000)
first = update_throughput(learner, 100, 10.0)
expect("the first observation is taken at face value", abs(first - 10.0) < 1e-9)
learner["throughput_sps"] = first
second = update_throughput(learner, 100, 5.0)
expect("later observations are smoothed", 10.0 < second < 20.0, str(second))

expect("success raises reliability", update_reliability(node("n", 1, rel=0.5), True) > 0.5)
expect("failure lowers reliability", update_reliability(node("n", 1, rel=0.5), False) < 0.5)
expect("reliability stays bounded", update_reliability(node("n", 1, rel=0.99), True) <= 1.0)

# 8. An empty or offline mesh degrades safely rather than crashing.
expect("an empty mesh plans nothing", plan_round([], 500).assignments == [])
expect("an all-offline mesh plans nothing",
       plan_round([node("gone", 3000, active=False)], 500).assignments == [])
expect("a zero-sample dataset plans nothing", plan_round(pool, 0).assignments == [])

# 9. Policy is honoured.
strict = MeshPolicy(min_memory_mb=16000)
expect("a stricter memory floor rejects more devices",
       all(d.tier == "rejected" for d in admit(pool, strict).values()))

expect("efficiency is speedup over worker count", abs(efficiency(2.0, 4) - 0.5) < 1e-9)

# 11. The equal-split control arm, which the load-balancing ablation compares against.
mixed = [node("fast", 4200), node("mid", 1800), node("slow", 700)]
prop = plan_round(mixed, 1200, strategy=PARTITION_PROPORTIONAL)
flat = plan_round(mixed, 1200, strategy=PARTITION_EQUAL)
expect("equal split gives every worker the same shard",
       len(set(a.samples for a in flat.assignments)) == 1)
expect("proportional split does not",
       len(set(a.samples for a in prop.assignments)) > 1)
expect("equal split is predicted to finish later",
       flat.predicted_makespan_seconds > prop.predicted_makespan_seconds,
       "%.1f vs %.1f" % (flat.predicted_makespan_seconds, prop.predicted_makespan_seconds))
expect("equal split is predicted to be more imbalanced",
       flat.predicted_imbalance > prop.predicted_imbalance * 10,
       "%.3f vs %.3f" % (flat.predicted_imbalance, prop.predicted_imbalance))
expect("both arms still assign every sample",
       sum(a.samples for a in flat.assignments) == 1200)

expect("observed imbalance is zero when shards finish together", imbalance([10.0, 10.0, 10.0]) == 0.0)
expect("observed imbalance rises with spread", imbalance([5.0, 10.0, 30.0]) > 0.5)

# 12. Measured and probe-derived throughput must share one scale.
#     A machine that has finished a round reports far fewer samples per second
#     than its probe implies, because a round carries fixed per-call costs. Left
#     uncorrected, an identical machine that had never run looked much faster and
#     took nearly the whole dataset.
veteran = node("veteran", 2780, thr=2.62)
newcomer = node("newcomer", 2776)
expect("calibration is 1.0 before anything has been measured",
       abs(calibration_factor([newcomer]) - 1.0) < 1e-9)
expect("calibration reflects the measured-to-probe ratio",
       0.0 < calibration_factor([veteran, newcomer]) < 0.5)

pair = plan_round([veteran, newcomer], 1000)
expect("a measured and an unmeasured twin are given equal shards",
       len(pair.assignments) == 2
       and abs(pair.assignments[0].samples - pair.assignments[1].samples) <= 2,
       str([a.samples for a in pair.assignments]))
expect("neither twin is dropped for being too slow", pair.rejected == [])

unmeasured = plan_round([node("fast", 4200), node("slow", 700)], 1000)
expect("with nothing measured the probe still sets the shape",
       unmeasured.assignments[0].samples > unmeasured.assignments[1].samples)

# 10. Batch size is capped per device, so one contributor's smaller GPU does not
#     crash on a batch chosen for somebody else's card.
expect("a small card gets a smaller batch than requested",
       safe_batch_size(4096, 640, 16) < 16)
expect("a large card gets what was asked for",
       safe_batch_size(24576, 640, 16) == 16)
expect("the guard never returns less than one",
       safe_batch_size(1024, 1280, 16) >= 1)
expect("the guard never exceeds the request",
       safe_batch_size(24576, 320, 4) == 4)
expect("a larger image size lowers the ceiling",
       safe_batch_size(4096, 640, 16) < safe_batch_size(4096, 416, 16))
expect("a node's own maximum is respected",
       safe_batch_size(24576, 320, 16, node_max=2) == 2)

# 13. The v5 affine cost model: a fixed overhead per round plus a per-image rate.
#     Leg 1 fitted T(n) = 18.77 s + 0.0418 s/image * n on one RTX 5070, about
#     24 images per second after a 19 second fixed cost.
LEG1_FIXED, LEG1_RATE = 18.77, 1.0 / 0.0418
twins = [node("a", 24000, thr=LEG1_RATE, fixed=LEG1_FIXED), node("b", 24000, thr=LEG1_RATE, fixed=LEG1_FIXED)]
two = plan_round(twins, 1000)
expect("the affine model reproduces leg 1's two-machine ceiling at 1000 images (1.53x)",
       abs(two.predicted_speedup - 1.53) < 0.03, "%.3f" % two.predicted_speedup)
small = plan_round(twins, 100)
expect("and its near-flat result at 100 images (about 1.1x)",
       1.0 <= small.predicted_speedup < 1.2, "%.3f" % small.predicted_speedup)

uneven = [node("quick", 9000, thr=10.0, fixed=5.0), node("slow-start", 9000, thr=10.0, fixed=30.0)]
big = plan_round(uneven, 2000)
finish = [a.predicted_seconds for a in big.assignments]
expect("affine shards equalise finish time including overhead",
       len(finish) == 2 and max(finish) - min(finish) < 0.5, json.dumps(finish))
expect("the machine with less overhead takes more of the data",
       {a.node_id: a.samples for a in big.assignments}["quick"] > {a.node_id: a.samples for a in big.assignments}["slow-start"])
linear = plan_round(uneven, 2000, strategy=PARTITION_LINEAR)
expect("the affine planner beats v4's rate-only split when overheads differ",
       big.predicted_makespan_seconds < linear.predicted_makespan_seconds - 5,
       "%.1f vs %.1f" % (big.predicted_makespan_seconds, linear.predicted_makespan_seconds))

tiny = plan_round(uneven, 200)
expect("a machine whose overhead exceeds the round sits it out",
       [a.node_id for a in tiny.assignments] == ["quick"] and any("overhead" in r["reason"] for r in tiny.rejected),
       json.dumps(tiny.as_dict()["assignments"]))
expect("sitting it out is faster than including it",
       tiny.predicted_makespan_seconds <= plan_round(uneven, 200, strategy=PARTITION_LINEAR).predicted_makespan_seconds)

import random
rng = random.Random(7)
monotone = True
for trial in range(200):
    pool_size = rng.randint(1, 5)
    machines = [node("m%d" % i, 5000, thr=rng.uniform(2, 60), fixed=rng.uniform(1, 40)) for i in range(pool_size)]
    extra = node("extra", 5000, thr=rng.uniform(2, 60), fixed=rng.uniform(1, 40))
    n = rng.randint(50, 5000)
    before = plan_round(machines, n, MeshPolicy(max_shard_skew=1000.0, min_shard_samples=1))
    after = plan_round(machines + [extra], n, MeshPolicy(max_shard_skew=1000.0, min_shard_samples=1))
    if after.predicted_makespan_seconds > before.predicted_makespan_seconds + 0.05:
        monotone = False
        break
expect("adding any machine never lengthens an affine plan (200 random meshes)", monotone)

shares, finish_time, excluded = affine_split({"x": 10.0, "y": 10.0}, {"x": 5.0, "y": 5.0}, 1000, {"x": 100.0})
expect("a probation cap holds under the affine split", abs(shares["x"] - 100.0) < 1e-6 and abs(shares["y"] - 900.0) < 1e-6,
       json.dumps(shares))

veteran_fixed = node("v", 3000, thr=20.0, fixed=12.0)
expect("a measured fixed cost is learned and smoothed", 12.0 < update_fixed(veteran_fixed, 20.0) < 20.0)
expect("one wild observation cannot move an estimate more than the step limit",
       update_throughput(node("w", 3000, thr=20.0), 1000, 1.0) <= 20.0 * MeshPolicy().max_estimate_step)
unmeasured_pair = plan_round([node("p", 4000, thr=20.0, fixed=8.0), node("q", 4000)], 1000)
expect("an unmeasured machine borrows the mesh's measured overhead",
       {a.node_id: a.fixed_seconds for a in unmeasured_pair.assignments}.get("q") == 8.0,
       json.dumps(unmeasured_pair.as_dict()["assignments"]))

# 14. Mixed vendors plan on measured rates, not vendor names.
mixed_vendors = [node("rtx", 20000, thr=30.0, fixed=8.0, backend="cuda"),
                 node("arc", 9000, thr=14.0, fixed=10.0, backend="xpu"),
                 node("m3", 6000, thr=9.0, fixed=9.0, backend="mps")]
vendor_plan = plan_round(mixed_vendors, 3000)
expect("NVIDIA, Intel and Apple machines all receive work",
       sorted(a.backend for a in vendor_plan.assignments) == ["cuda", "mps", "xpu"])
vendor_finish = [a.predicted_seconds for a in vendor_plan.assignments]
expect("and are predicted to finish together", max(vendor_finish) - min(vendor_finish) < 0.5, json.dumps(vendor_finish))
expect("unified memory gets a more conservative batch than the same dedicated memory",
       safe_batch_size(16384, 640, 64, unified_memory=True) < safe_batch_size(16384, 640, 64))

print("")
if failures:
    print("%d assertion(s) failed" % len(failures))
    sys.exit(1)
print("scheduler suite passed")
`;

const python = venvReady() ? venvPython() : null;
const fallback = findSystemPython();

if (!python && !fallback) {
  console.error(paint("red", "No Python found. Run `npm run setup` first."));
  process.exit(1);
}

console.log(paint("bold", "\nGradMesh scheduler\n"));

// Writing the suite to a file rather than passing it with -c keeps Windows
// argument quoting out of the picture entirely.
const suitePath = path.join(tmpdir(), `gradmesh-scheduler-${process.pid}.py`);
writeFileSync(suitePath, SUITE, "utf8");

const interpreter = python ? { command: python, args: [] } : fallback;
const result = spawnSync(interpreter.command, [...interpreter.args, suitePath], {
  cwd: ENGINE_DIR,
  stdio: "inherit",
});

rmSync(suitePath, { force: true });
process.exit(result.status ?? 1);
