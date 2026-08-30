import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { listCompanies } from "../api/companies";
import CompanyCard from "../components/CompanyCard";
import SearchBar from "../components/SearchBar";
import FilterBar from "../components/FilterBar";

export default function AllCompanies() {
  const [searchParams] = useSearchParams();
  const [companies, setCompanies] = useState([]);
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [industry, setIndustry] = useState("");
  const [sortKey, setSortKey] = useState("symbol");
  const [error, setError] = useState(null);

  useEffect(() => {
    listCompanies().then(setCompanies).catch((e) => setError(e.message));
  }, []);

  const industries = useMemo(
    () => [...new Set(companies.map((c) => c.industry))].sort(),
    [companies]
  );

  const filtered = useMemo(() => {
    let rows = companies;
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      rows = rows.filter((c) => c.symbol.toLowerCase().includes(q));
    }
    if (industry) rows = rows.filter((c) => c.industry === industry);
    return [...rows].sort((a, b) => {
      if (sortKey === "last_close") return b.last_close - a.last_close;
      return String(a[sortKey]).localeCompare(String(b[sortKey]));
    });
  }, [companies, query, industry, sortKey]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-xl font-semibold text-ink mb-6">All Companies</h1>

      <div className="flex flex-col md:flex-row gap-3 mb-6">
        <div className="flex-1">
          <SearchBar value={query} onChange={setQuery} />
        </div>
        <FilterBar
          industries={industries}
          selectedIndustry={industry}
          onIndustryChange={setIndustry}
          sortKey={sortKey}
          onSortChange={setSortKey}
        />
      </div>

      {error && <p className="text-negative">{error}</p>}
      <p className="text-xs text-ink-muted mb-4">{filtered.length} companies</p>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {filtered.map((c) => (
          <CompanyCard key={c.symbol} {...c} />
        ))}
      </div>
    </div>
  );
}
