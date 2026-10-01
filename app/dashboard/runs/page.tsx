import { currentUser } from "@/lib/auth";

import RunsView from "./RunsView";

export const metadata = { title: "Training runs" };
export const dynamic = "force-dynamic";

export default async function RunsPage() {
  const user = await currentUser();
  return <RunsView canManage={user?.role === "owner"} />;
}
