export default function About() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-6 text-sm text-ink-muted leading-relaxed">
      <div>
        <h1 className="text-xl font-semibold text-ink mb-3">About AlgoTraders</h1>
        <p>
          AlgoTraders is a research project studying whether machine learning can find
          an exploitable statistical edge in NIFTY50 stock behavior. It was built end to
          end — data collection, point-in-time-correct feature engineering, leakage-free
          time-series validation, baseline and deep learning modeling, and honest
          walk-forward evaluation — rather than optimized for a single impressive-looking
          backtest number.
        </p>
      </div>
      <div>
        <h2 className="text-lg font-semibold text-ink mb-2">Why probability, not price</h2>
        <p>
          Early designs for this platform displayed an "expected return %" and "target
          price" for each stock. Those numbers were dropped after testing showed
          absolute price/return prediction carries no reliable signal beyond the
          historical average — showing them would have overstated what the model can
          actually do. What survived testing was a much narrower, weaker claim: a
          modest, unstable tilt in whether a stock beats the market median. That is the
          only thing this platform reports.
        </p>
      </div>
      <div>
        <h2 className="text-lg font-semibold text-ink mb-2">Not a licensed advisory service</h2>
        <p>
          This project is not a SEBI-registered investment adviser and does not connect
          to any live brokerage or trading account. See the Disclaimer page for full
          terms.
        </p>
      </div>
    </div>
  );
}
