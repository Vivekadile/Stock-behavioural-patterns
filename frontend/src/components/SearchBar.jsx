import { Search } from "lucide-react";

export default function SearchBar({ value, onChange, placeholder = "Search company..." }) {
  return (
    <div className="relative">
      <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-muted" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-surface border border-border rounded-lg pl-9 pr-3 py-2 text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:border-accent/50"
      />
    </div>
  );
}
