import { useEffect, useState } from "react";
import { getMarketOverview } from "../api/market";
import MetricCard from "../components/MetricCard";

export default function MarketOverview() {
  const [overview, setOverview] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMarketOverview().then(setOverview).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="text-negative p-6">{error}</p>;
  if (!overview) return <p className="text-ink-muted p-6">Loading...</p>;

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-xl font-semibold text-ink mb-1">Market Overview</h1>
      <p className="text-sm text-ink-muted mb-6">As of {overview.as_of_date}</p>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
        <MetricCard
          label="NIFTY50 Close"
          value={overview.nifty50_close.toLocaleString("en-IN")}
          sub={`${overview.nifty50_daily_return_pct >= 0 ? "+" : ""}${overview.nifty50_daily_return_pct}% · ${
            overview.is_live ? "Live" : "Stored data"
          }`}
          accent={overview.nifty50_daily_return_pct >= 0 ? "text-accent" : "text-negative"}
        />
        <MetricCard label="Market Trend" value={overview.market_trend} />
        <MetricCard label="Stocks in Universe" value={overview.stocks_in_universe} />
        <MetricCard label="Stocks With Data Today" value={overview.stocks_with_data_as_of_date} />
        <MetricCard label="Advancing" value={overview.stocks_positive_today} accent="text-accent" />
        <MetricCard label="Declining" value={overview.stocks_negative_today} accent="text-negative" />
      </div>

      <p className="text-xs text-ink-muted bg-surface border border-border rounded-lg p-4">
        {overview.note}
      </p>
    </div>
  );
}
