import { vendorInfo } from "@/lib/format";

/**
 * Which kind of GPU a machine is, at a glance: NVIDIA (CUDA), Intel (XPU) or
 * Apple (Metal). Used wherever a machine, a shard or a share of a round is
 * shown, so a mixed-vendor mesh reads the same on every page.
 */
export default function VendorBadge({
  backend,
  compact = false,
  title,
}: {
  backend: string | null | undefined;
  compact?: boolean;
  title?: string;
}) {
  const info = vendorInfo(backend);
  return (
    <span className={`vendor vendor-${info.vendor}`} title={title ?? `${info.name} via ${info.api}`}>
      <span className="vendor-dot" aria-hidden="true" />
      {compact ? info.name : `${info.name} · ${info.api}`}
    </span>
  );
}

/** Just the coloured dot, for dense tables and bars. */
export function VendorDot({ backend }: { backend: string | null | undefined }) {
  const info = vendorInfo(backend);
  return <span className={`vendor-dot vendor-${info.vendor}`} title={`${info.name} ${info.api}`} aria-label={info.name} />;
}
