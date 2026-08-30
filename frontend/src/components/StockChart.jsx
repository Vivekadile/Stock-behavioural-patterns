import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function StockChart({ data }) {
  if (!data || data.length === 0) return <p className="text-ink-muted text-sm">No history available.</p>;

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#232b37" />
          <XAxis dataKey="date" tick={{ fill: "#8a94a3", fontSize: 11 }} minTickGap={40} />
          <YAxis tick={{ fill: "#8a94a3", fontSize: 11 }} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#12171f", border: "1px solid #232b37", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "#8a94a3" }}
          />
          <Line type="monotone" dataKey="close" stroke="#22c55e" strokeWidth={2} dot={false} name="Close" />
          <Line type="monotone" dataKey="sma20" stroke="#60a5fa" strokeWidth={1.5} dot={false} name="SMA 20" />
          <Line type="monotone" dataKey="sma50" stroke="#f5b942" strokeWidth={1.5} dot={false} name="SMA 50" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
