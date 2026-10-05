"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useMesh } from "@/components/dashboard/MeshProvider";
import { Empty, Meter, Panel } from "@/components/dashboard/ui";
import { bytes, clock } from "@/lib/format";
import type { Dataset } from "@/lib/types";
import { filesFromEntries, filesFromFolderInput, zipEntries, type ZipEntry } from "@/lib/zip";

const IMAGE = /\.(jpe?g|png|bmp|webp)$/i;

export default function DatasetManager({ canManage }: { canManage: boolean }) {
  const { request, refresh } = useMesh();
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [phase, setPhase] = useState<"packing" | "uploading">("uploading");
  const [uploadName, setUploadName] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  const folderRef = useRef<HTMLInputElement | null>(null);

  const load = useCallback(async () => {
    try {
      const payload = await request<{ datasets: Dataset[] }>("/api/mesh/datasets");
      setDatasets(payload.datasets);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }, [request]);

  useEffect(() => {
    void load();
  }, [load]);

  function upload(file: File) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setError("Upload a .zip archive, or a folder, containing images/train and labels/train.");
      return;
    }
    send(file, file.name, file.name.replace(/\.zip$/i, ""));
  }

  /**
   * A dataset folder, zipped here first. On a Mac this is the usual case:
   * Safari unzips what it downloads, so the export is a folder by now.
   */
  async function uploadFolder(entries: ZipEntry[]) {
    const zipped = entries.find((entry) => entry.path.toLowerCase().endsWith(".zip") && entries.length === 1);
    if (zipped) return upload(zipped.file);
    const images = entries.filter((entry) => IMAGE.test(entry.path)).length;
    const labels = entries.filter((entry) => entry.path.toLowerCase().endsWith(".txt")).length;
    if (images === 0 || labels === 0) {
      setError(
        `That folder has ${images} images and ${labels} label files. Choose the dataset folder itself, the one ` +
          "containing images/ and labels/ (or train/ and valid/)."
      );
      return;
    }
    const folder = entries[0].path.includes("/") ? entries[0].path.split("/")[0] : "dataset";
    setError(null);
    setPhase("packing");
    setProgress(0);
    try {
      const blob = await zipEntries(entries, (done, total) => setProgress(total ? done / total : 1));
      send(blob, `${folder}.zip`, folder);
    } catch (cause) {
      setProgress(null);
      setError(`Could not package that folder: ${(cause as Error).message}`);
    }
  }

  function send(file: Blob, filename: string, fallbackName: string) {
    setError(null);
    setPhase("uploading");
    setProgress(0);

    const form = new FormData();
    form.append("file", file, filename);
    form.append("name", uploadName.trim() || fallbackName);
    // The first dataset uploaded becomes the mesh default automatically on the
    // coordinator, so this only forces it for later ones.
    form.append("make_default", String(!datasets || datasets.length === 0));

    // XHR rather than fetch: a multi-gigabyte YOLO export needs a progress bar,
    // and fetch still has no upload progress event.
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/datasets");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) setProgress(event.loaded / event.total);
    };
    xhr.onload = async () => {
      setProgress(null);
      let payload: any = {};
      try {
        payload = JSON.parse(xhr.responseText);
      } catch {
        payload = {};
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        setUploadName("");
        if (inputRef.current) inputRef.current.value = "";
        if (folderRef.current) folderRef.current.value = "";
        await load();
        await refresh();
      } else {
        setError(payload?.detail || "The upload failed.");
      }
    };
    xhr.onerror = () => {
      setProgress(null);
      setError("The upload could not reach the host.");
    };
    xhr.send(form);
  }

  async function makeDefault(id: string) {
    try {
      await request(`/api/mesh/datasets/${id}/default`, { method: "POST" });
      await load();
      await refresh();
    } catch (cause) {
      setError((cause as Error).message);
    }
  }

  async function remove(id: string) {
    try {
      await request(`/api/mesh/datasets/${id}`, { method: "DELETE" });
      await load();
      await refresh();
    } catch (cause) {
      setError((cause as Error).message);
    }
  }

  return (
    <>
      <div>
        <h1 className="page-title">Datasets</h1>
        <p className="small faint" style={{ marginTop: 4 }}>
          Upload a YOLO export once. It becomes the starting dataset for every machine that joins,
          and the coordinator ships each worker only the slice it was assigned, so a contributing
          machine never needs the dataset itself.
        </p>
      </div>

      {error ? <div className="notice notice-danger">{error}</div> : null}

      {canManage ? (
        <Panel title="Add a dataset">
          <div className="stack">
            <div className="field">
              <label className="label" htmlFor="dataset-name">
                Name
              </label>
              <input
                className="input"
                id="dataset-name"
                value={uploadName}
                onChange={(event) => setUploadName(event.target.value)}
                placeholder="Strawberries, 640px"
              />
              <span className="hint">Optional. Defaults to the file name.</span>
            </div>

            <div
              className={`dropzone${dragOver ? " is-over" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragOver(false);
                // Entries must be read during the event; the list is emptied after it.
                const roots = Array.from(event.dataTransfer.items || [])
                  .map((item) => item.webkitGetAsEntry?.())
                  .filter((entry): entry is FileSystemEntry => Boolean(entry));
                if (roots.some((entry) => entry.isDirectory)) {
                  void filesFromEntries(roots).then(uploadFolder, (cause: Error) => setError(cause.message));
                  return;
                }
                const file = event.dataTransfer.files?.[0];
                if (file) upload(file);
              }}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
              }}
            >
              {progress === null ? (
                <>
                  <div style={{ fontWeight: 540, marginBottom: 6 }}>
                    Drop a dataset .zip or folder here, or click to choose a zip
                  </div>
                  <div className="small faint">
                    It should contain images/train and labels/train, and optionally images/val and
                    labels/val. A data.yaml inside it supplies the class names. Safari unzips
                    downloads, so on a Mac drop the folder it made.
                  </div>
                </>
              ) : (
                <div className="stack-sm">
                  <div className="small">
                    {phase === "packing" ? "Packaging the folder" : "Uploading"}… {Math.round(progress * 100)}%
                  </div>
                  <Meter value={progress} />
                </div>
              )}
              <input
                ref={inputRef}
                type="file"
                accept=".zip"
                className="sr-only"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) upload(file);
                }}
              />
            </div>
            <div className="row" style={{ gap: 10 }}>
              <button
                className="btn btn-sm"
                type="button"
                disabled={progress !== null}
                onClick={() => folderRef.current?.click()}
              >
                Choose a folder instead
              </button>
              <span className="small faint">For an export that is already unzipped.</span>
              <input
                ref={folderRef}
                type="file"
                multiple
                className="sr-only"
                {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
                onChange={(event) => {
                  if (event.target.files?.length) void uploadFolder(filesFromFolderInput(event.target.files));
                }}
              />
            </div>
          </div>
        </Panel>
      ) : null}

      <Panel title={`Registered datasets${datasets ? ` (${datasets.length})` : ""}`} flush>
        {!datasets ? (
          <Empty>Loading…</Empty>
        ) : datasets.length === 0 ? (
          <Empty>No datasets yet. Upload one above to start training.</Empty>
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Classes</th>
                  <th className="num">Train</th>
                  <th className="num">Val</th>
                  <th className="num">Size</th>
                  <th className="num">Added</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset) => (
                  <tr key={dataset.id}>
                    <td>
                      <div className="row" style={{ gap: 8 }}>
                        <span className="truncate">{dataset.name}</span>
                        {dataset.is_default ? (
                          <span className="badge badge-accent">Starting dataset</span>
                        ) : null}
                      </div>
                      <div className="small faint truncate">{dataset.filename}</div>
                    </td>
                    <td className="small">
                      {dataset.class_names.slice(0, 4).join(", ")}
                      {dataset.class_names.length > 4 ? ` +${dataset.class_names.length - 4}` : ""}
                    </td>
                    <td className="num">{dataset.train_count}</td>
                    <td className="num">{dataset.val_count || "—"}</td>
                    <td className="num">{bytes(dataset.bytes)}</td>
                    <td className="num small faint">{clock(dataset.created_at)}</td>
                    <td className="num">
                      <div className="row" style={{ gap: 6, justifyContent: "flex-end" }}>
                        {dataset.available !== false ? (
                          <a
                            className="btn btn-sm"
                            href={`/api/datasets/${encodeURIComponent(dataset.id)}/download`}
                            download
                            title="Download as a zip in the layout this page accepts"
                          >
                            Download
                          </a>
                        ) : null}
                        {canManage && !dataset.is_default ? (
                          <button
                            className="btn btn-sm"
                            type="button"
                            onClick={() => makeDefault(dataset.id)}
                          >
                            Make default
                          </button>
                        ) : null}
                        {canManage ? (
                          <button
                            className="btn btn-ghost btn-sm"
                            type="button"
                            onClick={() => remove(dataset.id)}
                          >
                            Delete
                          </button>
                        ) : null}
                      </div>
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
