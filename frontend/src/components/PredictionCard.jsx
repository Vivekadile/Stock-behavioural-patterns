import { Link } from "react-router-dom";
import PredictionBadge from "./PredictionBadge";
import RiskBadge from "./RiskBadge";

export default function PredictionCard({ prediction, featured }) {
  const { symbol, industry, last_close, probability_beats_nifty50_median, signal, risk_level } =
    prediction;

  return (
    <Link
      to={`/company/${symbol}`}
      className={`block rounded-xl border transition-colors ${
        featured
          ? "bg-gradient-to-br from-surface to-surface-hover border-accent/30 p-6"
          : "bg-surface border-border p-4 hover:border-accent/40"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <p className={featured ? "text-lg font-bold" : "font-semibold"}>{symbol}</p>
          <p className="text-xs text-ink-muted">{industry}</p>
        </div>
        <PredictionBadge signal={signal} />
      </div>
      <p className={featured ? "text-2xl font-semibold my-2" : "text-lg font-medium my-1"}>
        ₹{last_close?.toLocaleString("en-IN")}
      </p>
      <div className="flex items-center justify-between mt-3">
        <div>
          <p className="text-xs text-ink-muted">Probability of beating NIFTY50</p>
          <p className="text-xl font-bold text-accent">
            {(probability_beats_nifty50_median * 100).toFixed(1)}%
          </p>
        </div>
        <RiskBadge level={risk_level} />
      </div>
    </Link>
  );
}
