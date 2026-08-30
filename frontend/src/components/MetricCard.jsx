export default function MetricCard({ label, value, sub, accent }) {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <p className="text-xs uppercase tracking-wide text-ink-muted mb-1">{label}</p>
      <p className={`text-2xl font-semibold ${accent || "text-ink"}`}>{value}</p>
      {sub && <p className="text-xs text-ink-muted mt-1">{sub}</p>}
    </div>
  );
}
