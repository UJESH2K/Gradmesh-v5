"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import EventFeed from "@/components/dashboard/EventFeed";
import { useMesh } from "@/components/dashboard/MeshProvider";
import { Empty, Meter, Panel, Spark, StatTile, StatusBadge, TierBadge } from "@/components/dashboard/ui";
import { compact, percent, seconds } from "@/lib/format";
import type { RunDetail as Run } from "@/lib/types";

export default function RunDetail({ runId, canManage }: { runId: string; canManage: boolean }) {
  const { request, events, refresh } = useMesh();
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const load = useCallback(async () => {
    try {
      setRun(await request<Run>(`/api/mesh/runs/${runId}`));
      setError(null);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }, [request, runId]);

  useEffect(() => {
    void load();
  }, [load, events.length]);

  // A second ticker so in-flight shard bars advance smoothly between events.
  useEffect(() => {
    const live = run?.status === "running" || run?.status === "planning";
    if (!live) return;
    const timer = setInterval(() => setTick((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [run?.status]);

  async function stop() {
    try {
      await request(`/api/mesh/runs/${runId}/stop`, { method: "POST" });
      await load();
      await refresh();
    } catch (cause) {
      setError((cause as Error).message);
    }
  }

  if (error && !run) return <div className="notice notice-danger">{error}</div>;
  if (!run) {
    return (
      <div className="panel">
        <Empty>Loading run…</Empty>
      </div>
    );
  }

  const history = run.round_history || [];
  const live = ["running", "planning", "waiting"].includes(run.status);
  const lastRound = history.at(-1);

  return (
    <>
      <div className="row-between">
        <div>
          <div className="row" style={{ gap: 10 }}>
            <h1 className="page-title">{run.name}</h1>
            <StatusBadge status={run.status} />
            {run.mode === "solo" ? <span className="badge">Single GPU baseline</span> : null}
          </div>
          <p className="small faint" style={{ marginTop: 4 }}>
            {run.dataset_name} · {run.base_model} · {run.imgsz}px · batch {run.batch_size} ·{" "}
            {compact(run.total_samples)} images
          </p>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <Link className="btn btn-sm" href="/dashboard/runs">
            All runs
          </Link>
          {run.has_artifact ? (
            <a className="btn btn-sm" href={`/api/artifact/${run.id}`}>
              Download weights
            </a>
          ) : null}
          {canManage && live ? (
            <button className="btn btn-danger btn-sm" type="button" onClick={stop}>
              Stop run
            </button>
          ) : null}
        </div>
      </div>

      {run.error ? <div className="notice notice-danger">{run.error}</div> : null}
      {run.status === "waiting" ? (
        <div className="notice notice-warn">
          Waiting for an eligible machine. The round starts on its own as soon as one joins.
        </div>
      ) : null}

      <div className="grid grid-4">
        <StatTile
          label="Progress"
          value={`${run.current_round} / ${run.rounds}`}
          foot="rounds completed"
        />
        <StatTile label="Wall clock" value={seconds(run.wall_clock_seconds)} foot="across all rounds" />
        <StatTile
          label="Speedup"
          value={run.speedup ? `${run.speedup.toFixed(2)}x` : "—"}
          foot="versus the same work run serially"
          accent={run.speedup > 1}
        />
        <StatTile
          label="Efficiency"
          value={run.efficiency ? percent(run.efficiency) : "—"}
          foot={`over ${run.peak_workers || 0} machines`}
        />
      </div>

      {run.live_shards.length > 0 ? (
        <Panel title={`Round ${run.current_round + 1} in flight`}>
          <p className="small faint" style={{ marginBottom: 16 }}>
            Bars show elapsed time against each shard&apos;s own prediction. The marker is the soft
            deadline, past which the coordinator clones the work onto an idle machine.
          </p>
          <div>
            {run.live_shards.map((shard) => {
              const elapsed = shard.elapsed_seconds ?? 0;
              const scale = Math.max(shard.hard_deadline_seconds, shard.predicted_seconds * 1.2, 1);
              const fraction = Math.min(1, elapsed / scale);
              const softMark = Math.min(1, shard.soft_deadline_seconds / scale);
              const tone =
                shard.status === "done"
                  ? "is-done"
                  : shard.status === "dropped" || shard.status === "failed"
                    ? "is-lost"
                    : shard.status === "queued"
                      ? "is-queued"
                      : elapsed > shard.soft_deadline_seconds
                        ? "is-late"
                        : "is-running";
              return (
                <div className="shard-row" key={shard.batch_id} data-tick={tick}>
                  <div style={{ minWidth: 0 }}>
                    <div className="truncate small">{shard.node_name || shard.node_id}</div>
                    <div className="row small faint" style={{ gap: 6 }}>
                      <span className="mono">{shard.samples} images</span>
                      <TierBadge tier={shard.tier} />
                    </div>
                  </div>
                  <div className="shard-track">
                    <div className={`shard-fill ${tone}`} style={{ width: `${fraction * 100}%` }} />
                    <div className="shard-marker" style={{ left: `${softMark * 100}%` }} />
                  </div>
                  <div className="num small mono">
                    {shard.status === "queued"
                      ? "queued"
                      : `${elapsed.toFixed(0)}s / ${shard.predicted_seconds.toFixed(0)}s`}
                  </div>
                </div>
              );
            })}
          </div>
          {run.live_shards.some((shard) => shard.error) ? (
            <div className="stack-sm" style={{ marginTop: 16 }}>
              {run.live_shards
                .filter((shard) => shard.error)
                .map((shard) => (
                  <div key={shard.batch_id} className="small" style={{ color: "var(--danger)" }}>
                    {shard.node_name || shard.node_id}: {shard.error}
                  </div>
                ))}
            </div>
          ) : null}
        </Panel>
      ) : null}

      {history.length > 0 ? (
        <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.5fr) minmax(0, 1fr)" }}>
          <Panel title="Round timings">
            <div className="grid grid-2" style={{ marginBottom: 22 }}>
              <div className="stack-sm">
                <span className="eyebrow">Wall clock per round</span>
                <Spark values={history.map((round) => round.wall_clock_seconds)} />
                <span className="small faint">
                  Latest {seconds(lastRound?.wall_clock_seconds)}, aggregation{" "}
                  {seconds(lastRound?.aggregation_seconds)}
                </span>
              </div>
              <div className="stack-sm">
                <span className="eyebrow">Straggler gap per round</span>
                <Spark values={history.map((round) => round.straggler_gap_seconds)} />
                <span className="small faint">
                  Gap between the first and last machine to finish. Lower is a better split.
                </span>
              </div>
            </div>

            <div className="table-scroll">
              <table className="table">
                <thead>
                  <tr>
                    <th>Round</th>
                    <th className="num">Machines</th>
                    <th className="num">Images</th>
                    <th className="num">Makespan</th>
                    <th className="num">Gap</th>
                    <th className="num">Aggregation</th>
                    <th className="num">Speedup</th>
                    <th className="num">Lost</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((round) => (
                    <tr key={round.round}>
                      <td className="mono">{round.round + 1}</td>
                      <td className="num">{round.workers}</td>
                      <td className="num">{round.samples}</td>
                      <td className="num">{seconds(round.makespan_seconds)}</td>
                      <td className="num">{seconds(round.straggler_gap_seconds)}</td>
                      <td className="num">{seconds(round.aggregation_seconds)}</td>
                      <td className="num accent">{round.speedup.toFixed(2)}x</td>
                      <td className="num">
                        {round.dropped_shards ? (
                          <span style={{ color: "var(--danger)" }}>{round.dropped_shards}</span>
                        ) : (
                          "—"
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="Activity" flush>
            <EventFeed limit={40} />
          </Panel>
        </div>
      ) : (
        <Panel title="Activity" flush>
          <EventFeed limit={40} />
        </Panel>
      )}

      {lastRound ? (
        <Panel title={`Contribution in round ${lastRound.round + 1}`}>
          <p className="small faint" style={{ marginBottom: 16 }}>
            Aggregation weight is the share of the global model each machine&apos;s update
            contributed, proportional to the images it trained and damped by its track record.
          </p>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Machine</th>
                  <th className="num">Images</th>
                  <th className="num">Predicted</th>
                  <th className="num">Actual</th>
                  <th className="num">Error</th>
                  <th style={{ width: 180 }}>Aggregation weight</th>
                </tr>
              </thead>
              <tbody>
                {lastRound.shards.map((shard) => {
                  const drift = shard.predicted_seconds
                    ? (shard.seconds - shard.predicted_seconds) / shard.predicted_seconds
                    : 0;
                  return (
                    <tr key={shard.node_id}>
                      <td className="truncate">{shard.node_name || shard.node_id}</td>
                      <td className="num">{shard.samples}</td>
                      <td className="num">{seconds(shard.predicted_seconds)}</td>
                      <td className="num">{seconds(shard.seconds)}</td>
                      <td
                        className="num"
                        style={{ color: Math.abs(drift) > 0.35 ? "var(--warn)" : undefined }}
                      >
                        {drift >= 0 ? "+" : ""}
                        {Math.round(drift * 100)}%
                      </td>
                      <td>
                        <div className="row" style={{ gap: 10 }}>
                          <div className="grow">
                            <Meter value={shard.weight} />
                          </div>
                          <span className="small mono faint">{percent(shard.weight, 1)}</span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      {run.notes ? (
        <Panel title="Notes">
          <p className="muted">{run.notes}</p>
        </Panel>
      ) : null}
    </>
  );
}
