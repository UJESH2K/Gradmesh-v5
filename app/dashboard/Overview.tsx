"use client";

import Link from "next/link";

import EventFeed from "@/components/dashboard/EventFeed";
import HardwareMix from "@/components/dashboard/HardwareMix";
import { nodeSoftware } from "@/components/dashboard/SoftwareStack";
import NodeCard from "@/components/dashboard/NodeCard";
import { VendorDot } from "@/components/dashboard/VendorBadge";
import { useMesh } from "@/components/dashboard/MeshProvider";
import { Empty, Meter, Panel, StatTile, StatusBadge } from "@/components/dashboard/ui";
import { compact, gflops, memory, seconds } from "@/lib/format";
import type { MeshState } from "@/lib/types";

export default function Overview({ canManage }: { canManage: boolean }) {
  const { mesh, request, refresh } = useMesh();

  if (!mesh) {
    return (
      <div className="panel">
        <Empty>Connecting to the coordinator…</Empty>
      </div>
    );
  }

  const { metrics, plan_preview: plan, nodes, active_runs: activeRuns } = mesh;
  const online = nodes.filter((node) => node.active);
  const admitted = plan.assignments.length;

  async function evict(nodeId: string) {
    await request(`/api/mesh/nodes/${nodeId}`, { method: "DELETE" }).catch(() => {});
    await refresh();
  }

  async function reset(nodeId: string) {
    await request(`/api/mesh/nodes/${nodeId}/reset`, { method: "POST" }).catch(() => {});
    await refresh();
  }

  return (
    <>
      <div className="grid grid-4">
        <StatTile
          label="Machines online"
          value={metrics.nodes_active}
          foot={`${metrics.nodes_admitted} eligible to train`}
          accent={metrics.nodes_active > 0}
        />
        <StatTile
          label="Mesh compute"
          value={gflops(metrics.total_gflops)}
          foot="Measured, not advertised"
        />
        <StatTile label="Device memory" value={memory(metrics.total_memory_mb)} foot="Across the mesh" />
        <StatTile
          label="Predicted speedup"
          value={plan.predicted_speedup ? `${plan.predicted_speedup.toFixed(2)}x` : "—"}
          foot={
            plan.total_samples
              ? `on ${compact(plan.total_samples)} images vs the best single GPU`
              : "upload a dataset to see this"
          }
          accent={plan.predicted_speedup > 1}
        />
      </div>

      <Panel
        title="Hardware mix"
        action={
          <Link className="btn btn-sm" href="/dashboard/setup">
            Setup and health
          </Link>
        }
      >
        <HardwareMix backends={mesh.backends} />
        <StackSummary mesh={mesh} />
      </Panel>

      {activeRuns.length > 0 ? (
        <Panel
          title="Running now"
          action={
            <Link className="btn btn-sm" href="/dashboard/runs">
              All runs
            </Link>
          }
          flush
        >
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Round</th>
                  <th className="num">Machines</th>
                  <th className="num">Elapsed</th>
                  <th className="num">Speedup</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {activeRuns.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <Link href={`/dashboard/runs/${run.id}`} className="accent">
                        {run.name}
                      </Link>
                      <div className="small faint">{run.dataset_name}</div>
                    </td>
                    <td>
                      <StatusBadge status={run.status} />
                    </td>
                    <td style={{ minWidth: 140 }}>
                      <div className="small mono" style={{ marginBottom: 5 }}>
                        {run.current_round} / {run.rounds}
                      </div>
                      <Meter value={run.rounds ? run.current_round / run.rounds : 0} />
                    </td>
                    <td className="num mono">{run.peak_workers || "—"}</td>
                    <td className="num mono">{seconds(run.wall_clock_seconds)}</td>
                    <td className="num mono accent">{run.speedup ? `${run.speedup}x` : "—"}</td>
                    <td className="num">
                      <Link className="btn btn-sm" href={`/dashboard/runs/${run.id}`}>
                        Open
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.55fr) minmax(0, 1fr)" }}>
        <div className="stack">
          <Panel
            title={`Machines (${online.length} online)`}
            action={
              <Link className="btn btn-sm" href="/dashboard/nodes">
                Details
              </Link>
            }
          >
            {nodes.length === 0 ? (
              <Empty>
                No machines yet. Open{" "}
                <Link href="/dashboard/invite" className="accent">
                  Invite a GPU
                </Link>{" "}
                and send someone the one-line command, or contribute this machine from the top bar.
              </Empty>
            ) : (
              <div className="grid grid-2">
                {nodes.slice(0, 6).map((node) => (
                  <NodeCard key={node.node_id} node={node} canEvict={canManage} onEvict={evict} onReset={reset} />
                ))}
              </div>
            )}
          </Panel>

          {admitted > 0 ? (
            <Panel title="How the next round would be split">
              <p className="small faint" style={{ marginBottom: 12 }}>
                Recomputed continuously from what each machine has measured on{" "}
                <span className="mono">{mesh.workload || "this workload"}</span>: a fixed overhead per round
                plus a per-image rate. Shards are sized so every machine finishes at the same moment.
              </p>
              <div className="row" style={{ gap: 14, marginBottom: 14 }}>
                <span className="legend is-fixed">overhead</span>
                <span className="legend">training</span>
              </div>
              <div className="stack-sm" style={{ gap: 12 }}>
                {plan.assignments.map((assignment) => {
                  const node = nodes.find((item) => item.node_id === assignment.node_id);
                  const longest = Math.max(...plan.assignments.map((item) => item.predicted_seconds), 1);
                  const fixed = Math.min(assignment.fixed_seconds ?? 0, assignment.predicted_seconds);
                  const compute = Math.max(0, assignment.predicted_seconds - fixed);
                  return (
                    <div key={assignment.node_id} className="stack-sm" style={{ gap: 5 }}>
                      <div className="row-between small">
                        <span className="row truncate" style={{ gap: 7 }}>
                          <VendorDot backend={assignment.backend || node?.backend} />
                          <span className="truncate">{node?.display_name || assignment.node_id}</span>
                        </span>
                        <span className="mono faint">
                          {assignment.samples} images · {assignment.predicted_seconds}s
                        </span>
                      </div>
                      <div className="split-bar" title={`${fixed.toFixed(1)}s overhead, ${compute.toFixed(1)}s training`}>
                        <span className="is-fixed" style={{ width: `${(fixed / longest) * 100}%` }} />
                        <span className="is-compute" style={{ width: `${(compute / longest) * 100}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>

              {plan.rejected.length > 0 ? (
                <div className="stack-sm" style={{ marginTop: 20 }}>
                  <span className="eyebrow">Not scheduled</span>
                  {plan.rejected.map((item) => (
                    <div key={item.node_id} className="small faint">
                      <span className="mono">{item.node_id.slice(0, 8)}</span> — {item.reason}
                    </div>
                  ))}
                </div>
              ) : null}
            </Panel>
          ) : null}
        </div>

        <Panel title="Activity" flush>
          <EventFeed />
        </Panel>
      </div>
    </>
  );
}

/** One line under the hardware mix: is every online machine on the pinned stack? */
function StackSummary({ mesh }: { mesh: MeshState }) {
  const online = mesh.nodes.filter((node) => node.active);
  if (online.length === 0 || !mesh.reference_stack) return null;
  const stacks = online.map((node) => ({ node, software: nodeSoftware(node) }));
  const off = stacks.filter((item) => item.software.on_reference === false);
  const unknown = stacks.filter((item) => item.software.on_reference === null);
  const reference = mesh.reference_stack;
  return (
    <p className="small" style={{ marginTop: 14, color: off.length ? "var(--warn)" : "var(--text-dim)" }}>
      <span className="faint">Software · </span>
      {off.length === 0 && unknown.length === 0
        ? `all ${online.length} online machine${online.length === 1 ? "" : "s"} on the reference stack`
        : `${online.length - off.length - unknown.length} of ${online.length} on the reference stack`}
      <span className="mono faint">
        {" "}
        (torch {reference.torch} · torchvision {reference.torchvision} · ultralytics {reference.ultralytics})
      </span>
      {off.length > 0 ? `. Off it: ${off.map((item) => `${item.node.display_name} (${item.software.drift.join(", ")})`).join("; ")}` : ""}
      {unknown.length > 0 ? `. Not reported: ${unknown.map((item) => item.node.display_name).join(", ")}` : ""}
    </p>
  );
}
