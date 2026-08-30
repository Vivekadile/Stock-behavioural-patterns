import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getCompany, getCompanyHistory } from "../api/companies";
import { getPrediction } from "../api/predictions";
import StockChart from "../components/StockChart";
import PredictionBadge from "../components/PredictionBadge";
import RiskBadge from "../components/RiskBadge";
import MetricCard from "../components/MetricCard";

const HORIZONS = [7, 30, 60, 90];

export default function CompanyDetail() {
  const { symbol } = useParams();
  const [company, setCompany] = useState(null);
  const [history, setHistory] = useState([]);
  const [horizon, setHorizon] = useState(30);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setError(null);
    Promise.all([getCompany(symbol), getCompanyHistory(symbol, 180)])
      .then(([c, h]) => {
        setCompany(c);
        setHistory(h);
      })
      .catch((e) => setError(e.message));
  }, [symbol]);

  useEffect(() => {
    getPrediction(symbol, horizon).then(setPrediction).catch((e) => setError(e.message));
  }, [symbol, horizon]);

  if (error) return <p className="text-negative p-6">{error}</p>;
  if (!company) return <p className="text-ink-muted p-6">Loading...</p>;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-ink">{company.symbol}</h1>
          <p className="text-sm text-ink-muted">{company.industry}</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-semibold text-ink">
            ₹{company.last_close.toLocaleString("en-IN")}
          </p>
          <p className="text-xs text-ink-muted">
            {company.is_live ? "Live price" : `as of ${company.as_of_date}`}
          </p>
        </div>
      </div>

      <section className="bg-surface border border-border rounded-xl p-5">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <h2 className="font-semibold text-ink">Prediction</h2>
          <div className="flex gap-2">
            {HORIZONS.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-3 py-1 rounded-md text-xs font-medium border transition-colors ${
                  horizon === h
                    ? "bg-accent text-bg border-accent"
                    : "bg-bg text-ink-muted border-border hover:text-ink"
                }`}
              >
                {h}D
              </button>
            ))}
          </div>
        </div>

        {prediction && (
          <>
            <div className="flex items-center justify-between flex-wrap gap-4 mb-4">
              <div>
                <p className="text-xs text-ink-muted mb-1">Probability of beating NIFTY50 median</p>
                <p className="text-3xl font-bold text-accent">
                  {(prediction.probability_beats_nifty50_median * 100).toFixed(1)}%
                </p>
              </div>
              <div className="flex gap-2">
                <PredictionBadge signal={prediction.signal} />
                <RiskBadge level={prediction.risk_level} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <MetricCard
                label="Historical Accuracy"
                value={`${(prediction.historical_accuracy_at_horizon * 100).toFixed(1)}%`}
                sub={`at ${horizon}-day horizon`}
              />
            </div>
            <p className="text-xs text-ink-muted border-t border-border pt-4">
              {prediction.confidence_note}
            </p>
          </>
        )}
      </section>

      <section>
        <h2 className="font-semibold text-ink mb-4">Price History</h2>
        <StockChart data={history} />
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <section>
          <h2 className="font-semibold text-ink mb-4">Fundamentals</h2>
          <div className="grid grid-cols-2 gap-3">
            <MetricCard label="ROE" value={`${company.fundamentals.roe_pct}%`} />
            <MetricCard label="P/E Ratio" value={company.fundamentals.pe_ratio} />
            <MetricCard label="Net Profit Growth" value={`${company.fundamentals.net_profit_growth_pct}%`} />
            <MetricCard label="EPS Growth" value={`${company.fundamentals.eps_growth_pct}%`} />
          </div>
          <p className="text-xs text-ink-muted mt-3">{company.fundamentals.note}</p>
        </section>

        <section>
          <h2 className="font-semibold text-ink mb-4">Technicals</h2>
          <div className="grid grid-cols-2 gap-3">
            <MetricCard label="RSI (14)" value={company.technicals.rsi_14} />
            <MetricCard label="MACD" value={company.technicals.macd_signal} />
            <MetricCard label="vs SMA50" value={`${company.technicals.price_vs_sma50_pct}%`} />
            <MetricCard label="vs SMA200" value={`${company.technicals.price_vs_sma200_pct}%`} />
            <MetricCard label="Volatility (20d)" value={company.technicals.volatility_20d} />
            <MetricCard label="Volume Trend" value={company.technicals.volume_trend} />
          </div>
        </section>
      </div>
    </div>
  );
}
