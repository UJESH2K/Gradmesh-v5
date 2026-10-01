"use client";

import { vendorInfo } from "@/lib/format";
import { useMesh } from "./MeshProvider";
import { Empty } from "./ui";

function vendorName(backend: string | undefined): string {
  if (!backend) return "an unknown device";
  const info = vendorInfo(backend);
  return info.backend === "cpu" ? "CPU only" : `${info.name} (${info.api})`;
}

/**
 * A plain-language log of what the scheduler decided.
 *
 * This exists because "the mesh got faster" is not a claim anyone should take
 * on faith. Every admission, shard, speculation and drop shows up here with the
 * numbers behind it.
 */
const RENDERERS: Record<string, (data: Record<string, any>) => string> = {
  "coordinator.ready": () => "Coordinator started.",
  "node.joined": (d) =>
    `${d.name || d.node_id} ${d.restarted ? "rejoined after a restart" : "joined"} on ${vendorName(d.backend)}${d.gflops ? ` at ${Math.round(d.gflops)} GFLOP/s` : ""}. ${d.reason || ""}`,
  "node.reset": (d) => `${d.name || d.node_id} will be measured again from scratch.`,
  "node.batch_capped": (d) => `A machine ran out of GPU memory, so its batch is now capped at ${d.batch_size}.`,
  "node.version": (d) => `${d.name || d.node_id}: ${d.detail}`,
  "run.deleted": () => "A run record was deleted.",
  "round.evaluated": (d) =>
    d.ok
      ? `Round ${(d.round ?? 0) + 1} scored mAP50 ${Number(d.map50 ?? 0).toFixed(4)} in ${d.seconds}s.`
      : `Scoring round ${(d.round ?? 0) + 1} failed: ${d.error}`,
  "suite.created": (d) => `Sweep planned with ${d.trials} trials.`,
  "suite.started": (d) => `Sweep started: ${d.trials} trials.`,
  "suite.trial": (d) => `Sweep trial ${d.trial?.label ?? ""}: ${d.status}${d.completed != null ? ` (${d.completed}/${d.total})` : ""}.`,
  "suite.finished": (d) => `Sweep ${d.status}${d.completed != null ? `, ${d.completed} of ${d.total} trials done` : ""}.`,
  "dataset.importing": (d) => `Importing ${d.key}: ${d.message}`,
  "dataset.import_failed": (d) => `Import of ${d.key} failed: ${d.error}`,
  "dataset.subset": (d) => `Prepared a ${d.images}-image subset.`,
  "internal.error": (d) => `Internal error in ${d.where}: ${d.error}`,
  "node.offline": (d) => `${d.name || d.node_id} stopped responding.`,
  "node.left": (d) => `${d.name || d.node_id} left the mesh.`,
  "node.evicted": (d) => `${d.name || d.node_id} was removed by the owner.`,
  "dataset.added": (d) => `Dataset "${d.name}" added with ${d.images} training images.`,
  "dataset.removed": () => "A dataset was removed.",
  "dataset.default": () => "The starting dataset changed.",
  "policy.updated": () => "Scheduling policy updated.",
  "run.created": (d) => `Run "${d.name}" created for ${d.rounds} rounds.`,
  "run.waiting": (d) => `Run is waiting: ${d.reason}.`,
  "run.stopped": () => "Run stopped by the owner.",
  "run.failed": (d) => `Run failed: ${d.error}`,
  "run.completed": (d) =>
    `Run finished with ${d.summary?.speedup ?? "?"}x speedup over a single GPU estimate.`,
  "round.started": (d) =>
    `Round ${(d.round ?? 0) + 1} planned across ${d.plan?.assignments?.length ?? 0} machines, ${d.plan?.total_samples ?? 0} images, predicted ${d.plan?.predicted_makespan_seconds ?? "?"}s.`,
  "round.completed": (d) =>
    `Round ${(d.round ?? 0) + 1} done in ${d.wall_clock_seconds}s. Straggler gap ${d.straggler_gap_seconds}s, aggregation ${d.aggregation_seconds}s, speedup ${d.speedup}x.`,
  "round.failed": (d) => `Round failed: ${d.reason}${d.retrying ? " Retrying." : ""}`,
  "shard.assigned": (d) =>
    `${d.name || d.node_id} took ${d.samples} images, predicted ${d.predicted_seconds}s.`,
  "shard.completed": (d) =>
    `${d.node_id} finished ${d.samples} images in ${d.seconds}s${d.epoch_seconds ? ` (${d.epoch_seconds}s of it training)` : ""}.`,
  "shard.failed": (d) => `A shard failed on ${d.node_id}: ${d.error}`,
  "shard.dropped": (d) => `Shard dropped: ${d.reason}`,
  "shard.speculated": (d) =>
    `${d.from_node} passed its soft deadline at ${d.elapsed_seconds}s, so a copy was started on ${d.to_node}.`,
  "shard.superseded": (d) =>
    `A duplicate shard on ${d.node_id} was retired because the other copy finished first.`,
  "mesh.token_rotated": (d) =>
    `The mesh token was rotated. ${d.removed} machine(s) were disconnected.`,
  "visitor.arrived": (d) =>
    `${d.name || "A device"} opened the join page from ${d.ip || "this network"}${d.gpu ? ` with ${d.gpu}` : ""}.`,
  "network.scanned": (d) => `Network sweep finished. ${d.devices} device(s) on this Wi-Fi.`,
  "supervisor.error": (d) => `Supervisor error: ${d.error}`,
};

const TONE: Record<string, string> = {
  "shard.dropped": "var(--danger)",
  "shard.failed": "var(--danger)",
  "run.failed": "var(--danger)",
  "round.failed": "var(--danger)",
  "supervisor.error": "var(--danger)",
  "internal.error": "var(--danger)",
  "dataset.import_failed": "var(--danger)",
  "node.batch_capped": "var(--warn)",
  "round.evaluated": "var(--cyan)",
  "shard.speculated": "var(--warn)",
  "run.waiting": "var(--warn)",
  "node.offline": "var(--warn)",
  "round.completed": "var(--accent)",
  "run.completed": "var(--accent)",
  "node.joined": "var(--live)",
  "visitor.arrived": "var(--cyan)",
};

export default function EventFeed({ limit = 60 }: { limit?: number }) {
  const { events } = useMesh();
  const recent = [...events].reverse().slice(0, limit);

  if (recent.length === 0) {
    return <Empty>Nothing has happened yet. Events appear here the moment they do.</Empty>;
  }

  return (
    <div className="feed">
      {recent.map((event) => {
        const render = RENDERERS[event.kind];
        const text = render ? render(event.data) : event.kind;
        return (
          <div className="feed-item" key={`${event.id}-${event.at}`}>
            <span className="feed-time">
              {new Date(event.at * 1000).toLocaleTimeString(undefined, {
                hour12: false,
                minute: "2-digit",
                second: "2-digit",
                hour: "2-digit",
              })}
            </span>
            <span style={{ color: TONE[event.kind] || "var(--text-dim)" }}>{text}</span>
          </div>
        );
      })}
    </div>
  );
}
