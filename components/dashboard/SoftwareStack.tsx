"use client";

import type { MeshNode, NodeSoftware } from "@/lib/types";

/**
 * The software one machine trains with, next to its GPU.
 *
 * A cross-vendor result is only fair if every machine ran the same PyTorch,
 * torchvision and Ultralytics, so each card says what this machine runs and
 * whether that matches the pinned reference stack, instead of leaving it to
 * a log file on the contributor's machine.
 */

/** The coordinator's view, or one pieced together from the raw capability for an older host. */
export function nodeSoftware(node: MeshNode): NodeSoftware {
  if (node.software) return node.software;
  const capability = node.capability || {};
  const torch = String(capability.torch_version || "");
  return {
    python: capability.python_version || node.diagnostics?.python || "",
    torch: torch.split("+")[0],
    torch_build: torch.includes("+") ? torch.split("+")[1] : node.backend === "mps" ? "macOS" : "",
    torchvision: String(capability.torchvision_version || "").split("+")[0],
    ultralytics: capability.ultralytics_version || "",
    runtime: capability.runtime || "",
    os: capability.os_version || node.diagnostics?.os || capability.platform || "",
    driver: node.diagnostics?.driver || "",
    agent: node.agent_version || "",
    agent_behind: false,
    on_reference: null,
    drift: [],
  };
}

export function ReferenceBadge({ software }: { software: NodeSoftware }) {
  if (software.on_reference === null) {
    return (
      <span className="badge" title="This agent did not report its full stack; re-run the join command to update it">
        stack unknown
      </span>
    );
  }
  return software.on_reference ? (
    <span className="badge badge-live" title="torch, torchvision and Ultralytics match the reference stack">
      reference stack
    </span>
  ) : (
    <span className="badge badge-warn" title={software.drift.join("\n")}>
      off reference
    </span>
  );
}

export default function SoftwareStack({ node }: { node: MeshNode }) {
  const software = nodeSoftware(node);
  const rows: [string, string, string?][] = [
    ["Python", software.python],
    ["PyTorch", software.torch ? `${software.torch}${software.torch_build ? ` · ${software.torch_build}` : ""}` : ""],
    ["torchvision", software.torchvision],
    ["Ultralytics", software.ultralytics],
    [
      "Runtime",
      [software.runtime, software.driver ? `driver ${software.driver}` : ""].filter(Boolean).join(" · "),
    ],
    ["OS", software.os],
    [
      "Agent",
      software.agent,
      software.agent_behind ? "Older than the host; run the join command again on this machine to update" : undefined,
    ],
  ];
  const shown = rows.filter(([, value]) => value);
  if (shown.length === 0) return null;

  return (
    <div className="sw-block">
      <div className="row-between">
        <span className="sw-title">Software</span>
        <ReferenceBadge software={software} />
      </div>
      <dl className="sw-grid">
        {shown.map(([label, value, note]) => (
          <div key={label} className={note ? "is-behind" : undefined} title={note}>
            <dt>{label}</dt>
            <dd className="mono truncate" title={value}>
              {value}
            </dd>
          </div>
        ))}
      </dl>
      {software.drift.length > 0 ? <p className="small sw-drift">{software.drift.join(" · ")}</p> : null}
    </div>
  );
}
