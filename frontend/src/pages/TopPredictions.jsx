import { useEffect, useState } from "react";
import { getTopPredictions } from "../api/predictions";
import PredictionTable from "../components/PredictionTable";

const HORIZONS = [7, 30, 60, 90];

export default function TopPredictions() {
  const [horizon, setHorizon] = useState(30);
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    getTopPredictions(horizon, 20)
      .then(setRows)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [horizon]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <h1 className="text-xl font-semibold text-ink mb-1">Top Predictions</h1>
      <p className="text-sm text-ink-muted mb-6">
        Ranked by probability of beating the NIFTY50 median return over the selected horizon.
      </p>

      <div className="flex gap-2 mb-6">
        {HORIZONS.map((h) => (
          <button
            key={h}
            onClick={() => setHorizon(h)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium border transition-colors ${
              horizon === h
                ? "bg-accent text-bg border-accent"
                : "bg-surface text-ink-muted border-border hover:text-ink"
            }`}
          >
            {h}D
          </button>
        ))}
      </div>

      {error && <p className="text-negative">{error}</p>}
      {loading ? (
        <p className="text-ink-muted">Loading...</p>
      ) : (
        <PredictionTable rows={rows} />
      )}
    </div>
  );
}
