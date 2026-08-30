const ACCURACY_ROWS = [
  { horizon: "7 Day", accuracy: "51.3%", edge: "-0.4% to +1.3%" },
  { horizon: "30 Day", accuracy: "52.9%", edge: "-1.7% to +4.3%" },
  { horizon: "60 Day", accuracy: "54.1%", edge: "-1.6% to +2.4%" },
  { horizon: "90 Day", accuracy: "50.1%", edge: "-1.5% to +1.0%" },
];

export default function ModelInfo() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-8 text-sm text-ink-muted leading-relaxed">
      <div>
        <h1 className="text-xl font-semibold text-ink mb-3">Model Information</h1>
        <p>
          AlgoTraders predicts one thing: the probability that a stock outperforms the
          NIFTY50 median return over a given horizon (7, 30, 60, or 90 trading days). It
          does not predict a price target or an expected return percentage — extensive
          testing found no reliable signal for absolute price movement, only a weak,
          unstable edge in relative (cross-sectional) ranking.
        </p>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-ink mb-2">What the model is</h2>
        <p>
          A Logistic Regression classifier trained on 36 technical and point-in-time
          fundamental features (price momentum, volatility, RSI, MACD, moving-average
          ratios, sector and market regime, ROE, profit/EPS growth, India VIX) per
          horizon. It was chosen over more complex alternatives — including a 5-model
          ensemble of single-layer LSTMs — because those did not outperform it in
          walk-forward testing on this task, consistent with published research on
          individual stock return prediction (Gu, Kelly & Xiu, 2020).
        </p>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-ink mb-2">Walk-forward accuracy by horizon</h2>
        <div className="overflow-x-auto border border-border rounded-lg">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-surface text-ink-muted text-xs uppercase">
                <th className="text-left px-4 py-3">Horizon</th>
                <th className="text-left px-4 py-3">Historical Accuracy</th>
                <th className="text-left px-4 py-3">Yearly Edge Range (2021-2025)</th>
              </tr>
            </thead>
            <tbody>
              {ACCURACY_ROWS.map((r) => (
                <tr key={r.horizon} className="border-t border-border">
                  <td className="px-4 py-3 text-ink">{r.horizon}</td>
                  <td className="px-4 py-3 text-ink">{r.accuracy}</td>
                  <td className="px-4 py-3">{r.edge}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs mt-2">
          "Accuracy" here means beating a coin-flip (50%) at classifying whether a stock
          outperforms the NIFTY50 median. These numbers come from purged, embargoed
          walk-forward testing — not a single random train/test split — and the edge
          range shows how much that accuracy has varied year to year.
        </p>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-ink mb-2">Known limitations</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>The edge is weak (a few percentage points above 50%) and unstable across years.</li>
          <li>It did not survive realistic transaction costs in backtesting.</li>
          <li>Six independent remedies (outlier removal, IC-stable feature filtering, quintile re-ranking, rolling training windows, volatility-scaled targets, adding India VIX) were tested and none fixed the instability — it appears to be intrinsic to the regime relationship, not a fixable bug.</li>
          <li>Absolute return/price prediction showed no exploitable signal at all — this is why the platform reports a relative probability, not a price target.</li>
        </ul>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-ink mb-2">Data</h2>
        <p>
          Daily OHLCV from Yahoo Finance for NIFTY50 constituents; fundamentals scraped
          from Screener.in and matched to real NSE filing dates (not fiscal year-end) to
          avoid look-ahead bias; India VIX for implied volatility context.
        </p>
      </div>
    </div>
  );
}
