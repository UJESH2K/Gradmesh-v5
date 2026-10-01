"use client";

import Link from "next/link";

import { gflops, memory, VENDORS } from "@/lib/format";
import type { BackendSummary } from "@/lib/types";

/**
 * The mesh by GPU family: NVIDIA, Intel and Apple, always all three.
 *
 * An empty column is information too. It says which kind of machine the mesh
 * has not got yet and what that machine needs, which is the first thing to
 * know when the experiment is a cross-vendor one.
 */
export default function HardwareMix({ backends }: { backends: BackendSummary[] | undefined }) {
  const byBackend = new Map((backends || []).map((item) => [item.backend, item]));
  const cpu = byBackend.get("cpu");

  return (
    <div className="stack">
      <div className="mix-grid">
        {VENDORS.map((vendor) => {
          const summary = byBackend.get(vendor.backend);
          const online = summary?.online ?? 0;
          return (
            <div
              key={vendor.backend}
              className={`mix-card vendor-${vendor.vendor}${online === 0 ? " is-empty" : ""}`}
            >
              <div className="row-between">
                <span className={`vendor vendor-${vendor.vendor}`}>
                  <span className="vendor-dot" aria-hidden="true" />
                  {vendor.name} · {vendor.api}
                </span>
                <span className="small faint">
                  {summary?.eligible ?? 0} eligible
                </span>
              </div>
              <div className="row" style={{ gap: 8, alignItems: "baseline" }}>
                <span className="mix-count mono">{online}</span>
                <span className="small faint">online{summary && summary.nodes > online ? `, ${summary.nodes - online} away` : ""}</span>
              </div>
              {online > 0 ? (
                <div className="small faint">
                  {gflops(summary?.gflops)} · {memory(summary?.memory_mb)}
                  {summary?.throughput_sps ? ` · ${summary.throughput_sps.toFixed(1)} img/s` : ""}
                </div>
              ) : (
                <div className="small faint">
                  {vendor.needs}{" "}
                  <Link href="/dashboard/invite" className="accent">
                    Add one
                  </Link>
                </div>
              )}
            </div>
          );
        })}
      </div>
      {cpu && cpu.online > 0 ? (
        <p className="small faint">
          {cpu.online} machine{cpu.online === 1 ? "" : "s"} joined without a usable GPU. They are measured but never
          given shards; the Machines page says why for each.
        </p>
      ) : null}
    </div>
  );
}
