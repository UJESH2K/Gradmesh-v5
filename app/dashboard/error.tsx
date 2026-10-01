"use client";

import Link from "next/link";
import { useEffect } from "react";

/**
 * A dashboard page threw while rendering. Say so, offer the two things that
 * usually fix it, and keep the rest of the dashboard usable rather than
 * replacing it with a blank screen.
 */
export default function DashboardError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="panel">
      <div className="stack">
        <h1 className="page-title">This page hit an error</h1>
        <p className="muted small">
          {error.message || "Something went wrong while rendering."}
          {error.digest ? <span className="faint mono"> ({error.digest})</span> : null}
        </p>
        <p className="small faint">
          If the coordinator was restarting, trying again is usually enough. If it keeps happening, run{" "}
          <code className="code-inline">npm run doctor</code> on the host.
        </p>
        <div className="row" style={{ gap: 8 }}>
          <button className="btn btn-primary btn-sm" type="button" onClick={reset}>
            Try again
          </button>
          <Link className="btn btn-sm" href="/dashboard/setup">
            Setup and health
          </Link>
        </div>
      </div>
    </div>
  );
}
