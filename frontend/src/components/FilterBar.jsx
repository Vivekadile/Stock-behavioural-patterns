export default function FilterBar({ industries, selectedIndustry, onIndustryChange, sortKey, onSortChange }) {
  return (
    <div className="flex flex-wrap gap-3 items-center">
      <select
        value={selectedIndustry}
        onChange={(e) => onIndustryChange(e.target.value)}
        className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent/50"
      >
        <option value="">All Sectors</option>
        {industries.map((ind) => (
          <option key={ind} value={ind}>
            {ind}
          </option>
        ))}
      </select>
      <select
        value={sortKey}
        onChange={(e) => onSortChange(e.target.value)}
        className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent/50"
      >
        <option value="probability">Sort: Probability</option>
        <option value="symbol">Sort: Name</option>
        <option value="last_close">Sort: Price</option>
      </select>
    </div>
  );
}
