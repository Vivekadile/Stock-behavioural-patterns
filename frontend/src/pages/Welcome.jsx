import { Link } from "react-router-dom";
import { LineChart, ShieldAlert, TrendingUp } from "lucide-react";

export default function Welcome() {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col items-center justify-center px-4 text-center">
      <div className="flex items-center gap-2 mb-4">
        <LineChart className="text-accent" size={32} />
        <span className="text-2xl font-bold text-ink">AlgoTraders</span>
      </div>
      <h1 className="text-3xl md:text-4xl font-bold text-ink max-w-2xl">
        A statistically honest read on NIFTY50 stocks
      </h1>
      <p className="text-ink-muted mt-4 max-w-xl">
        For each stock, we estimate the probability that it outperforms the NIFTY50
        median over the next 7, 30, 60, or 90 trading days — built on a walk-forward
        tested model, not a black-box price target.
      </p>

      <div className="flex gap-3 mt-8">
        <Link
          to="/dashboard"
          className="inline-flex items-center gap-2 bg-accent text-bg font-semibold px-6 py-3 rounded-lg hover:opacity-90 transition-opacity"
        >
          <TrendingUp size={18} />
          Enter Dashboard
        </Link>
        <Link
          to="/disclaimer"
          className="inline-flex items-center gap-2 bg-surface border border-border text-ink-muted px-6 py-3 rounded-lg hover:text-ink hover:border-accent/40 transition-colors"
        >
          <ShieldAlert size={18} />
          Read Disclaimer
        </Link>
      </div>

      <p className="text-xs text-ink-muted mt-10 max-w-md">
        This is a research and educational tool. It does not provide investment advice,
        buy/sell recommendations, or price targets. See the Disclaimer and Model
        Information pages before using it.
      </p>
    </div>
  );
}
