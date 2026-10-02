"use client";

import Link from "next/link";
import { useState } from "react";

import NodeCard from "@/components/dashboard/NodeCard";
import { nodeSoftware, ReferenceBadge } from "@/components/dashboard/SoftwareStack";
import VendorBadge from "@/components/dashboard/VendorBadge";
import { useMesh } from "@/components/dashboard/MeshProvider";
import { Empty, Panel, TierBadge } from "@/components/dashboard/ui";
import { ago, gflops, memory, seconds, VENDORS } from "@/lib/format";

type Layout = "cards" | "table";

export default function NodesView({ canManage }: { canManage: boolean }) {
  const { mesh, request, refresh } = useMesh();
  const [layout, setLayout] = useState<Layout>("cards");
  const [vendor, setVendor] = useState<string>("all");
  const [error, setError] = useState<string | null>(null);

  if (!mesh) {
    return (
      <div className="panel">
        <Empty>Connecting…</Empty>
      </div>
    );
  }

  const nodes = [...mesh.nodes]
    .filter((node) => vendor === "all" || node.backend === vendor)
    .sort((a, b) => {
      if (a.active !== b.active) return a.active ? -1 : 1;
      return b.fitness - a.fitness;
    });
  const present = new Set(mesh.nodes.map((node) => node.backend));

  async function reset(nodeId: string) {
    setError(null);
    try {
      await request(`/api/mesh/nodes/${nodeId}/reset`, { method: "POST" });
      await refresh();
    } catch (cause) {
      setError((cause as Error).message);
    }
  }

  async function evict(nodeId: string) {
    setError(null);
    try {
      await request(`/api/mesh/nodes/${nodeId}`, { method: "DELETE" });
      await refresh();
    } catch (cause) {
      setError((cause as Error).message);
    }
  }

  return (
    <>
      <div className="row-between">
        <div>
          <h1 className="page-title">Machines</h1>
          <p className="small faint" style={{ marginTop: 4 }}>
            Every device that has offered its GPU, ranked by the fitness score the scheduler
            actually uses.
          </p>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <button
            className={`btn btn-sm${layout === "cards" ? " btn-primary" : ""}`}
            onClick={() => setLayout("cards")}
            type="button"
          >
            Cards
          </button>
          <button
            className={`btn btn-sm${layout === "table" ? " btn-primary" : ""}`}
            onClick={() => setLayout("table")}
            type="button"
          >
            Table
          </button>
        </div>
      </div>

      {error ? <div className="notice notice-danger">{error}</div> : null}

      {mesh.nodes.length > 0 ? (
        <div className="chip-row" role="tablist" aria-label="Filter by GPU vendor">
          <button type="button" className={`chip${vendor === "all" ? " is-on" : ""}`} onClick={() => setVendor("all")}>
            All ({mesh.nodes.length})
          </button>
          {VENDORS.filter((item) => present.has(item.backend)).map((item) => (
            <button
              key={item.backend}
              type="button"
              className={`chip${vendor === item.backend ? " is-on" : ""}`}
              onClick={() => setVendor(item.backend)}
            >
              {item.name} ({mesh.nodes.filter((node) => node.backend === item.backend).length})
            </button>
          ))}
          {present.has("cpu") ? (
            <button type="button" className={`chip${vendor === "cpu" ? " is-on" : ""}`} onClick={() => setVendor("cpu")}>
              No GPU ({mesh.nodes.filter((node) => node.backend === "cpu").length})
            </button>
          ) : null}
        </div>
      ) : null}

      {nodes.length === 0 ? (
        <div className="panel">
          <Empty>
            No machines have joined. Send someone the command on the{" "}
            <Link href="/dashboard/invite" className="accent">
              Invite a GPU
            </Link>{" "}
            page.
          </Empty>
        </div>
      ) : layout === "cards" ? (
        <div className="grid grid-3">
          {nodes.map((node) => (
            <NodeCard key={node.node_id} node={node} canEvict={canManage} onEvict={evict} onReset={reset} />
          ))}
        </div>
      ) : (
        <Panel title={`${nodes.length} machines`} flush>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Machine</th>
                  <th>Vendor</th>
                  <th>Software</th>
                  <th className="num">Measured</th>
                  <th className="num">Memory</th>
                  <th className="num">Rate</th>
                  <th className="num">Overhead</th>
                  <th className="num">Fitness</th>
                  <th className="num">Reliability</th>
                  <th className="num">Rounds</th>
                  <th>Status</th>
                  {canManage ? <th /> : null}
                </tr>
              </thead>
              <tbody>
                {nodes.map((node) => (
                  <tr key={node.node_id}>
                    <td>
                      <div className="row" style={{ gap: 8 }}>
                        <span
                          className={`dot${node.active_batches > 0 ? " dot-live" : ""}`}
                          style={
                            node.active_batches > 0
                              ? undefined
                              : { background: node.active ? "var(--cyan)" : "var(--text-faint)" }
                          }
                        />
                        <div style={{ minWidth: 0 }}>
                          <div className="truncate">{node.display_name}</div>
                          <div className="small faint truncate">{node.gpu}</div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <VendorBadge backend={node.backend} compact />
                    </td>
                    <td>
                      {(() => {
                        const software = nodeSoftware(node);
                        return (
                          <div className="stack-sm" style={{ gap: 3 }}>
                            <span className="mono small">
                              {software.torch ? `torch ${software.torch}` : "—"}
                              {software.torch_build ? ` · ${software.torch_build}` : ""}
                            </span>
                            <span className="small faint truncate">
                              {[software.python && `py ${software.python}`, software.os].filter(Boolean).join(" · ")}
                            </span>
                          </div>
                        );
                      })()}
                    </td>
                    <td className="num">{gflops(node.capability?.gflops)}</td>
                    <td className="num">{memory(node.gpu_memory_mb)}</td>
                    <td className="num">
                      {node.throughput_sps ? `${node.throughput_sps.toFixed(1)} img/s` : "—"}
                    </td>
                    <td className="num">{node.fixed_seconds ? seconds(node.fixed_seconds) : "—"}</td>
                    <td className="num">{node.fitness.toFixed(3)}</td>
                    <td className="num">{node.reliability.toFixed(2)}</td>
                    <td className="num">
                      {node.completed_rounds}
                      {node.failed_rounds ? (
                        <span style={{ color: "var(--danger)" }}> / {node.failed_rounds}</span>
                      ) : null}
                    </td>
                    <td>
                      <div className="stack-sm" style={{ gap: 4 }}>
                        <TierBadge tier={node.tier} title={node.admission_reason} />
                        <span className="small faint">
                          {node.liveness === "suspect" ? "waiting" : node.active ? "online" : ago(node.last_seen)}
                        </span>
                        {node.warnings?.length ? (
                          <span className="small" style={{ color: "var(--warn)" }} title={node.warnings.join("; ")}>
                            {node.warnings.length} warning{node.warnings.length === 1 ? "" : "s"}
                          </span>
                        ) : null}
                      </div>
                    </td>
                    {canManage ? (
                      <td className="num">
                        <button
                          className="btn btn-ghost btn-sm"
                          type="button"
                          onClick={() => evict(node.node_id)}
                        >
                          Remove
                        </button>
                      </td>
                    ) : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {nodes.length > 0 ? (
        <Panel
          title="Software on each machine"
          action={
            mesh.reference_stack ? (
              <span className="small faint mono">
                reference: torch {mesh.reference_stack.torch} · torchvision {mesh.reference_stack.torchvision} ·
                ultralytics {mesh.reference_stack.ultralytics}
              </span>
            ) : null
          }
          flush
        >
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Machine</th>
                  <th>OS</th>
                  <th>Python</th>
                  <th>PyTorch</th>
                  <th>torchvision</th>
                  <th>Ultralytics</th>
                  <th>Runtime</th>
                  <th>Agent</th>
                  <th>Stack</th>
                </tr>
              </thead>
              <tbody>
                {nodes.map((node) => {
                  const software = nodeSoftware(node);
                  return (
                    <tr key={node.node_id} title={software.drift.join("; ") || undefined}>
                      <td>
                        <div className="row" style={{ gap: 8 }}>
                          <VendorBadge backend={node.backend} compact />
                          <span className="truncate">{node.display_name}</span>
                        </div>
                      </td>
                      <td className="small" style={{ whiteSpace: "nowrap" }}>{software.os || "—"}</td>
                      <td className="mono small">{software.python || "—"}</td>
                      <td className="mono small" style={{ whiteSpace: "nowrap" }}>
                        {software.torch || "—"}
                        {software.torch_build ? <span className="faint"> · {software.torch_build}</span> : null}
                      </td>
                      <td className="mono small">{software.torchvision || "—"}</td>
                      <td className="mono small">{software.ultralytics || "—"}</td>
                      <td className="small" style={{ whiteSpace: "nowrap" }}>
                        {software.runtime || "—"}
                        {software.driver ? <span className="faint"> · driver {software.driver}</span> : null}
                      </td>
                      <td
                        className="mono small"
                        style={software.agent_behind ? { color: "var(--warn)" } : undefined}
                        title={software.agent_behind ? "Older than the host; re-run the join command on this machine" : undefined}
                      >
                        {software.agent || "—"}
                      </td>
                      <td>
                        <ReferenceBadge software={software} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      <Panel title="Contribution totals">
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Machine</th>
                <th>Contributed by</th>
                <th className="num">Images trained</th>
                <th className="num">GPU time given</th>
                <th className="num">Joined</th>
              </tr>
            </thead>
            <tbody>
              {nodes.map((node) => (
                <tr key={node.node_id}>
                  <td className="truncate">{node.display_name}</td>
                  <td className="small faint">{node.owner || "anonymous"}</td>
                  <td className="num">{node.samples_trained || "—"}</td>
                  <td className="num">{node.seconds_trained ? seconds(node.seconds_trained) : "—"}</td>
                  <td className="num small faint">{ago(node.joined_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
