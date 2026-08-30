import { Link } from "react-router-dom";
import PredictionBadge from "./PredictionBadge";
import RiskBadge from "./RiskBadge";

export default function PredictionTable({ rows }) {
  return (
    <div className="overflow-x-auto border border-border rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-surface text-ink-muted text-xs uppercase tracking-wide">
            <th className="text-left px-4 py-3">Rank</th>
            <th className="text-left px-4 py-3">Company</th>
            <th className="text-right px-4 py-3">Price</th>
            <th className="text-right px-4 py-3">Probability</th>
            <th className="text-left px-4 py-3">Signal</th>
            <th className="text-left px-4 py-3">Risk</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.symbol} className="border-t border-border hover:bg-surface-hover">
              <td className="px-4 py-3 text-ink-muted">{row.rank}</td>
              <td className="px-4 py-3">
                <p className="font-medium">{row.symbol}</p>
                <p className="text-xs text-ink-muted">{row.industry}</p>
              </td>
              <td className="px-4 py-3 text-right tabular-nums">
                ₹{row.last_close?.toLocaleString("en-IN")}
              </td>
              <td className="px-4 py-3 text-right font-semibold text-accent tabular-nums">
                {(row.probability_beats_nifty50_median * 100).toFixed(1)}%
              </td>
              <td className="px-4 py-3">
                <PredictionBadge signal={row.signal} />
              </td>
              <td className="px-4 py-3">
                <RiskBadge level={row.risk_level} />
              </td>
              <td className="px-4 py-3 text-right">
                <Link to={`/company/${row.symbol}`} className="text-accent text-xs hover:underline">
                  View →
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
