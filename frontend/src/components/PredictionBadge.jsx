import { TrendingUp, TrendingDown, Minus } from "lucide-react";

const CONFIG = {
  Positive: { icon: TrendingUp, className: "bg-emerald-950 text-emerald-400 border-emerald-800" },
  Negative: { icon: TrendingDown, className: "bg-red-950 text-red-400 border-red-800" },
  Neutral: { icon: Minus, className: "bg-slate-800 text-slate-400 border-slate-700" },
};

export default function PredictionBadge({ signal }) {
  const { icon: Icon, className } = CONFIG[signal] || CONFIG.Neutral;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium border ${className}`}>
      <Icon size={12} />
      {signal}
    </span>
  );
}
