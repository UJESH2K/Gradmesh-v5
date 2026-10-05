import { NextResponse } from "next/server";

import { currentUser } from "@/lib/auth";
import { coordinatorStream } from "@/lib/coordinator";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 3600;

/**
 * Streams a registered dataset to the browser as a zip.
 *
 * The browser asking may be on another machine, a Mac on the same Wi-Fi for
 * instance, so the file comes through this origin rather than from the
 * coordinator's own address, which only the host can be relied on to reach.
 * Separate from the general mesh proxy because that one parses JSON.
 */
export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });

  const { id } = await context.params;
  let upstream: Response;
  try {
    upstream = await coordinatorStream(`/datasets/${encodeURIComponent(id)}/download`);
  } catch {
    return NextResponse.json({ detail: "The coordinator is not reachable." }, { status: 503 });
  }
  if (!upstream.ok || !upstream.body) {
    const payload = await upstream.json().catch(() => ({ detail: "The dataset could not be packaged." }));
    return NextResponse.json(payload, { status: upstream.status || 502 });
  }

  const headers = new Headers({ "Content-Type": "application/zip" });
  for (const name of ["content-disposition", "content-length"]) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  if (!headers.has("content-disposition")) headers.set("Content-Disposition", `attachment; filename="${id}.zip"`);
  return new NextResponse(upstream.body, { headers });
}
