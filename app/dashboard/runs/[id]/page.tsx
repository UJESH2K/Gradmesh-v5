import { currentUser } from "@/lib/auth";

import RunDetail from "./RunDetail";

export const dynamic = "force-dynamic";

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const user = await currentUser();
  return <RunDetail runId={id} canManage={user?.role === "owner"} />;
}
