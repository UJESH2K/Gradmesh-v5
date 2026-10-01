"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { useMesh } from "@/components/dashboard/MeshProvider";
import { Empty, Panel, StatTile, StatusBadge } from "@/components/dashboard/ui";
import { clock, compact, seconds } from "@/lib/format";
import type { Dataset, RunSummary } from "@/lib/types";

export default function RunsView({ canManage }: { canManage: boolean }) {
  const { mesh, request, refresh } = useMesh();
  const router = useRouter();
  const params = useSearchParams();

  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<{ name: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [showForm, setShowForm] = useState(params.get("new") === "1");

  const load = useCallback(async () => {
    try {
      const [runPayload, datasetPayload, modelPayload] = await Promise.all([
        request<{ runs: RunSummary[] }>("/api/mesh/runs"),
        request<{ datasets: Dataset[] }>("/api/mesh/datasets"),
        request<{ models: { name: string }[] }>("/api/mesh/models"),
      ]);
      setRuns(runPayload.runs);
      setDatasets(datasetPayload.datasets);
      setModels(modelPayload.models);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }, [request]);

  useEffect(() => {
    void load();
  }, [load, mesh?.active_runs.length]);

  async function create(formData: FormData) {
    setCreating(true);
    setError(null);
    try {
      const body = {
        name: String(formData.get("name") || "mesh-run"),
        dataset_id: String(formData.get("dataset_id") || "") || undefined,
        base_model: String(formData.get("base_model") || "yolov8n.pt"),
        rounds: Number(formData.get("rounds") || 4),
        imgsz: Number(formData.get("imgsz") || 640),
        batch_size: Number(formData.get("batch_size") || 8),
        mode: String(formData.get("mode") || "mesh"),
        notes: String(formData.get("notes") || "") || undefined,
      };
      const run = await request<RunSummary>("/api/mesh/runs", {
        method: "POST",
        body: JSON.stringify(body),
      });
      await refresh();
      router.push(`/dashboard/runs/${run.id}`);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setCreating(false);
    }
  }

  const eligible = mesh?.plan_preview.assignments.length ?? 0;
  const defaultDataset = datasets.find((item) => item.is_default) || datasets[0];
  const finished = (runs || []).filter((run) => ["done", "failed", "stopped"].includes(run.status));
  const best = finished.reduce<RunSummary | null>(
    (winner, run) => (!winner || run.speedup > winner.speedup ? run : winner),
    null
  );

  return (
    <>
      <div className="row-between">
        <div>
          <h1 className="page-title">Training runs</h1>
          <p className="small faint" style={{ marginTop: 4 }}>
            Each run is a sequence of synchronised rounds. Every round is replanned from what the
            machines actually delivered in the last one.
          </p>
        </div>
        {canManage ? (
          <button className="btn btn-primary" type="button" onClick={() => setShowForm((v) => !v)}>
            {showForm ? "Close" : "New run"}
          </button>
        ) : null}
      </div>

      {error ? <div className="notice notice-danger">{error}</div> : null}

      {showForm && canManage ? (
        <Panel title="Start a run">
          {eligible === 0 ? (
            <div className="notice notice-warn" style={{ marginBottom: 18 }}>
              No machine is currently eligible to train. Contribute this machine from the top bar,
              or invite a GPU, then start the run.
            </div>
          ) : (
            <div className="notice notice-accent" style={{ marginBottom: 18 }}>
              {eligible} machine{eligible === 1 ? "" : "s"} ready.
              {mesh?.plan_preview.predicted_speedup
                ? ` Predicted ${mesh.plan_preview.predicted_speedup.toFixed(2)}x against the best single GPU.`
                : ""}
            </div>
          )}

          <form action={create} className="stack">
            <div className="grid grid-2">
              <div className="field">
                <label className="label" htmlFor="name">
                  Run name
                </label>
                <input
                  className="input"
                  id="name"
                  name="name"
                  defaultValue={`run-${new Date().toISOString().slice(5, 16).replace("T", "-")}`}
                  required
                />
              </div>

              <div className="field">
                <label className="label" htmlFor="dataset_id">
                  Dataset
                </label>
                <select className="select" id="dataset_id" name="dataset_id" defaultValue={defaultDataset?.id}>
                  {datasets.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name} · {dataset.train_count} images
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label className="label" htmlFor="base_model">
                  Base checkpoint
                </label>
                <select className="select" id="base_model" name="base_model" defaultValue="yolov8n.pt">
                  {(models.length ? models : [{ name: "yolov8n.pt" }]).map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label className="label" htmlFor="mode">
                  Mode
                </label>
                <select className="select" id="mode" name="mode" defaultValue="mesh">
                  <option value="mesh">Mesh — use every eligible machine</option>
                  <option value="solo">Single GPU baseline — strongest machine only</option>
                </select>
                <span className="hint">
                  Run the same job in both modes to get the comparison the paper needs.
                </span>
              </div>

              <div className="field">
                <label className="label" htmlFor="rounds">
                  Rounds
                </label>
                <input className="input" id="rounds" name="rounds" type="number" min={1} max={200} defaultValue={4} />
                <span className="hint">One local epoch per machine per round, then aggregation.</span>
              </div>

              <div className="field">
                <label className="label" htmlFor="imgsz">
                  Image size
                </label>
                <select className="select" id="imgsz" name="imgsz" defaultValue={640}>
                  <option value={320}>320</option>
                  <option value={416}>416</option>
                  <option value={512}>512</option>
                  <option value={640}>640</option>
                  <option value={800}>800</option>
                </select>
              </div>

              <div className="field">
                <label className="label" htmlFor="batch_size">
                  Batch size per machine
                </label>
                <input
                  className="input"
                  id="batch_size"
                  name="batch_size"
                  type="number"
                  min={1}
                  max={64}
                  defaultValue={4}
                />
                <span className="hint">Lower this if a contributor runs out of GPU memory.</span>
              </div>

              <div className="field">
                <label className="label" htmlFor="notes">
                  Notes
                </label>
                <textarea
                  className="textarea"
                  id="notes"
                  name="notes"
                  placeholder="What is this run testing?"
                />
              </div>
            </div>

            <div className="row">
              <button className="btn btn-primary" type="submit" disabled={creating || eligible === 0}>
                {creating ? "Starting…" : "Start run"}
              </button>
              <button className="btn btn-ghost" type="button" onClick={() => setShowForm(false)}>
                Cancel
              </button>
            </div>
          </form>
        </Panel>
      ) : null}

      {finished.length > 0 ? (
        <div className="grid grid-4">
          <StatTile label="Completed runs" value={finished.length} />
          <StatTile
            label="Best speedup"
            value={best?.speedup ? `${best.speedup.toFixed(2)}x` : "—"}
            foot={best ? best.name : undefined}
            accent
          />
          <StatTile
            label="Best efficiency"
            value={best?.efficiency ? `${Math.round(best.efficiency * 100)}%` : "—"}
            foot="speedup per machine"
          />
          <StatTile
            label="Images trained"
            value={compact(finished.reduce((sum, run) => sum + run.total_samples, 0))}
            foot="across all runs"
          />
        </div>
      ) : null}

      <Panel title="All runs" flush>
        {!runs ? (
          <Empty>Loading…</Empty>
        ) : runs.length === 0 ? (
          <Empty>No runs yet. Start one to see the mesh train.</Empty>
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Mode</th>
                  <th className="num">Rounds</th>
                  <th className="num">Machines</th>
                  <th className="num">Wall clock</th>
                  <th className="num">Speedup</th>
                  <th className="num">Started</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <Link href={`/dashboard/runs/${run.id}`} className="accent">
                        {run.name}
                      </Link>
                      <div className="small faint truncate">
                        {run.dataset_name} · {run.base_model}
                      </div>
                    </td>
                    <td>
                      <StatusBadge status={run.status} />
                    </td>
                    <td className="small">{run.mode === "solo" ? "Single GPU" : "Mesh"}</td>
                    <td className="num">
                      {run.current_round} / {run.rounds}
                    </td>
                    <td className="num">{run.peak_workers || "—"}</td>
                    <td className="num">{seconds(run.wall_clock_seconds)}</td>
                    <td className="num accent">{run.speedup ? `${run.speedup.toFixed(2)}x` : "—"}</td>
                    <td className="num small faint">{clock(run.created_at)}</td>
                    <td className="num">
                      {run.has_artifact ? (
                        <a className="btn btn-sm" href={`/api/artifact/${run.id}`}>
                          Weights
                        </a>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}
