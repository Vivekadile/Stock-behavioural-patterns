import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getMarketOverview } from "../api/market";
import { getTopPredictions } from "../api/predictions";
import { listCompanies } from "../api/companies";
import MetricCard from "../components/MetricCard";
import PredictionCard from "../components/PredictionCard";
import CompanyCard from "../components/CompanyCard";

export default function Dashboard() {
  const [overview, setOverview] = useState(null);
  const [top, setTop] = useState([]);
  const [companies, setCompanies] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getMarketOverview(), getTopPredictions(30, 6), listCompanies()])
      .then(([ov, preds, comps]) => {
        setOverview(ov);
        setTop(preds);
        setCompanies(comps.slice(0, 8));
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="text-negative p-6">Failed to load dashboard: {error}</p>;
  if (!overview) return <p className="text-ink-muted p-6">Loading...</p>;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-10">
      <section>
        <h1 className="text-xl font-semibold text-ink mb-4">Market Overview</h1>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="NIFTY50"
            value={overview.nifty50_close.toLocaleString("en-IN")}
            sub={`${overview.nifty50_daily_return_pct >= 0 ? "+" : ""}${overview.nifty50_daily_return_pct}% today · ${
              overview.is_live ? "Live" : "Stored data"
            }`}
            accent={overview.nifty50_daily_return_pct >= 0 ? "text-accent" : "text-negative"}
          />
          <MetricCard label="Market Trend" value={overview.market_trend} />
          <MetricCard label="Stocks Advancing" value={overview.stocks_positive_today} />
          <MetricCard label="Stocks Declining" value={overview.stocks_negative_today} />
        </div>
        <p className="text-xs text-ink-muted mt-3">{overview.note}</p>
      </section>

      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-ink">Top Predictions · 30 Day</h2>
          <Link to="/predictions" className="text-accent text-sm hover:underline">
            View all →
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {top.map((p, i) => (
            <PredictionCard key={p.symbol} prediction={p} featured={i === 0} />
          ))}
        </div>
      </section>

      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-ink">Companies</h2>
          <Link to="/companies" className="text-accent text-sm hover:underline">
            View all →
          </Link>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {companies.map((c) => (
            <CompanyCard key={c.symbol} {...c} />
          ))}
        </div>
      </section>
    </div>
  );
}
