import { ShieldAlert } from "lucide-react";

export default function Disclaimer() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-6 text-sm text-ink-muted leading-relaxed">
      <div className="flex items-center gap-2 mb-2">
        <ShieldAlert className="text-neutral" size={22} />
        <h1 className="text-xl font-semibold text-ink">Disclaimer</h1>
      </div>

      <p>
        AlgoTraders is an educational and research tool. Nothing on this platform is
        investment advice, a recommendation to buy or sell any security, or a
        guarantee of future performance. It does not provide price targets, expected
        returns, entry zones, or stop-loss levels.
      </p>

      <p>
        The "probability" shown for each stock is the output of a statistical model
        estimating whether that stock is likely to outperform the NIFTY50 median return
        over a stated horizon. This edge is small (typically a few percentage points
        above a 50/50 coin flip), has varied — and in some years disappeared or
        reversed — across the walk-forward test period (2021-2025), and did not survive
        realistic transaction costs in backtesting. Past patterns in this data are not a
        promise of future results.
      </p>

      <p>
        AlgoTraders and its creator are not SEBI-registered Investment Advisers or
        Research Analysts. This platform does not connect to any brokerage account and
        cannot place trades. Any investment decision you make is your own responsibility
        — consult a qualified, registered financial adviser before investing.
      </p>

      <p>
        Data shown reflects a stored historical dataset, not a live market feed, and may
        be delayed or outdated. See the Model Information page for methodology, accuracy,
        and known limitations.
      </p>
    </div>
  );
}
