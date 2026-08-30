import { Link } from "react-router-dom";

export default function CompanyCard({ symbol, industry, last_close, probability, signal }) {
  return (
    <Link
      to={`/company/${symbol}`}
      className="block bg-surface border border-border rounded-lg p-4 hover:border-accent/50 hover:bg-surface-hover transition-colors"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="font-semibold text-ink">{symbol}</p>
          <p className="text-xs text-ink-muted">{industry}</p>
        </div>
        {signal && (
          <span
            className={`text-xs px-2 py-0.5 rounded border ${
              signal === "Positive"
                ? "bg-emerald-950 text-emerald-400 border-emerald-800"
                : signal === "Negative"
                ? "bg-red-950 text-red-400 border-red-800"
                : "bg-slate-800 text-slate-400 border-slate-700"
            }`}
          >
            {signal}
          </span>
        )}
      </div>
      <p className="text-lg font-medium mt-3">₹{last_close?.toLocaleString("en-IN")}</p>
      {probability != null && (
        <p className="text-xs text-ink-muted mt-1">
          {(probability * 100).toFixed(1)}% probability of beating NIFTY50
        </p>
      )}
    </Link>
  );
}
