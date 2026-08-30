const STYLES = {
  Low: "bg-emerald-950 text-emerald-400 border-emerald-800",
  Medium: "bg-amber-950 text-amber-400 border-amber-800",
  High: "bg-red-950 text-red-400 border-red-800",
};

export default function RiskBadge({ level }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${
        STYLES[level] || STYLES.Medium
      }`}
    >
      {level} Risk
    </span>
  );
}
