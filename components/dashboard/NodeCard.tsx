"use client";

import { ago, gflops, memory, seconds } from "@/lib/format";
import type { MeshNode } from "@/lib/types";
import { Meter, TierBadge } from "./ui";
import VendorBadge from "./VendorBadge";

const PHASE_LABEL: Record<string, string> = {
  downloading: "fetching data",
  loading: "loading model",
  training: "training",
  uploading: "sending weights",
};

export default function NodeCard({
  node,
  onEvict,
  onReset,
  canEvict = false,
}: {
  node: MeshNode;
  onEvict?: (nodeId: string) => void;
  onReset?: (nodeId: string) => void;
  canEvict?: boolean;
}) {
  const training = node.active_batches > 0;
  const suspect = node.liveness === "suspect";
  const progress =
    node.progress && node.progress.batches > 0 ? node.progress.batch / node.progress.batches : null;
  const warnings = node.warnings || [];
  const diag = node.diagnostics || {};

  return (
    <article
      className={`node-card${training ? " is-training" : ""}${node.active ? "" : " is-offline"}`}
    >
      <div className="node-head">
        <div className="stack-sm" style={{ gap: 4, minWidth: 0 }}>
          <div className="row" style={{ gap: 8 }}>
            <span
              className={`dot${training ? " dot-live" : ""}`}
              style={
                training
                  ? undefined
                  : { background: suspect ? "var(--warn)" : node.active ? "var(--cyan)" : "var(--text-faint)" }
              }
              title={suspect ? "No heartbeat for a while; waiting before giving up on it" : undefined}
            />
            <span className="node-name truncate">{node.display_name || node.node_id}</span>
          </div>
          <span className="small faint truncate" title={node.gpu}>
            {node.gpu}
          </span>
        </div>
        <div className="stack-sm" style={{ gap: 6, alignItems: "flex-end" }}>
          <VendorBadge backend={node.backend} compact />
          <TierBadge tier={node.tier} title={node.admission_reason} />
        </div>
      </div>

      {training ? (
        <div className="stack-sm" style={{ gap: 6 }}>
          <div className="row-between small">
            <span className="phase">{PHASE_LABEL[node.phase || "training"] || node.phase}</span>
            {progress !== null ? (
              <span className="mono faint">
                batch {node.progress?.batch} / {node.progress?.batches}
              </span>
            ) : null}
          </div>
          <Meter value={progress ?? 0.05} tone="cyan" />
        </div>
      ) : null}

      <div className="node-facts">
        <div className="node-fact">
          <span>Memory</span>
          <span>
            {memory(node.gpu_memory_mb)}
            {node.capability?.unified_memory ? " shared" : ""}
          </span>
        </div>
        <div className="node-fact">
          <span>Measured</span>
          <span>{gflops(node.capability?.gflops)}</span>
        </div>
        <div className="node-fact">
          <span>Training rate</span>
          <span>{node.throughput_sps ? `${node.throughput_sps.toFixed(1)} img/s` : "unmeasured"}</span>
        </div>
        <div className="node-fact">
          <span>Overhead per round</span>
          <span>{node.fixed_seconds ? seconds(node.fixed_seconds) : "unmeasured"}</span>
        </div>
        <div className="node-fact">
          <span>Rounds</span>
          <span>
            {node.completed_rounds}
            {node.failed_rounds ? ` / ${node.failed_rounds} failed` : ""}
          </span>
        </div>
        <div className="node-fact">
          <span>Last seen</span>
          <span>{node.active ? (suspect ? "waiting" : "now") : ago(node.last_seen)}</span>
        </div>
      </div>

      <div className="stack-sm" style={{ gap: 6 }}>
        <div className="row-between small faint" style={{ gap: 8 }}>
          <span>Fitness</span>
          <span className="mono">{node.fitness.toFixed(3)}</span>
        </div>
        <Meter value={node.fitness} tone={node.tier === "probation" ? "warn" : "accent"} />
      </div>

      {node.admission_reason && node.tier !== "full" ? (
        <p className="small faint">{node.admission_reason}</p>
      ) : null}

      {warnings.length > 0 ? (
        <ul className="warn-list" title="Why this machine may train slower than its GPU suggests">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}

      {diag.cpu ? (
        <p className="small faint truncate" title={diag.cpu}>
          {diag.cpu}
          {diag.os ? ` · ${diag.os}` : ""}
          {node.capability?.torch_version ? ` · torch ${node.capability.torch_version}` : ""}
        </p>
      ) : null}

      <div className="row-between">
        <span className="small faint">
          {node.samples_trained ? `${node.samples_trained} images trained` : "no work yet"}
          {node.seconds_trained ? ` in ${seconds(node.seconds_trained)}` : ""}
        </span>
        {canEvict ? (
          <div className="row" style={{ gap: 4 }}>
            {onReset && node.workloads && Object.keys(node.workloads).length > 0 ? (
              <button
                className="btn btn-ghost btn-sm"
                type="button"
                onClick={() => onReset(node.node_id)}
                title="Forget the learned speed and overhead, after a driver update or hardware change"
              >
                Re-measure
              </button>
            ) : null}
            {onEvict ? (
              <button className="btn btn-ghost btn-sm" type="button" onClick={() => onEvict(node.node_id)}>
                Remove
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}
