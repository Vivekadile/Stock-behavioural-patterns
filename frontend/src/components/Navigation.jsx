import { useState, useRef, useEffect } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { LineChart, MoreVertical, Search } from "lucide-react";

const PRIMARY_LINKS = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/predictions", label: "Top Predictions" },
  { to: "/companies", label: "Companies" },
  { to: "/market", label: "Market Overview" },
];

const MENU_LINKS = [
  { to: "/model-info", label: "Model Information" },
  { to: "/about", label: "About" },
  { to: "/disclaimer", label: "Disclaimer" },
];

export default function Navigation() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [query, setQuery] = useState("");
  const menuRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    function onClickOutside(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function onSearchSubmit(e) {
    e.preventDefault();
    if (query.trim()) {
      navigate(`/companies?q=${encodeURIComponent(query.trim())}`);
    }
  }

  return (
    <header className="sticky top-0 z-20 bg-bg/95 backdrop-blur border-b border-border">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center gap-6">
        <Link to="/dashboard" className="flex items-center gap-2 shrink-0">
          <LineChart className="text-accent" size={22} />
          <span className="font-bold text-ink tracking-tight">AlgoTraders</span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {PRIMARY_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive ? "text-ink bg-surface" : "text-ink-muted hover:text-ink hover:bg-surface"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <form onSubmit={onSearchSubmit} className="flex-1 max-w-xs ml-auto relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-muted" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search company..."
            className="w-full bg-surface border border-border rounded-lg pl-8 pr-3 py-1.5 text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:border-accent/50"
          />
        </form>

        <div className="relative" ref={menuRef}>
          <button
            type="button"
            onClick={() => setMenuOpen((v) => !v)}
            className="p-2 rounded-md text-ink-muted hover:text-ink hover:bg-surface transition-colors"
            aria-label="More"
          >
            <MoreVertical size={18} />
          </button>
          {menuOpen && (
            <div className="absolute right-0 mt-2 w-48 bg-surface border border-border rounded-lg shadow-lg py-1">
              {MENU_LINKS.map((link) => (
                <Link
                  key={link.to}
                  to={link.to}
                  onClick={() => setMenuOpen(false)}
                  className="block px-4 py-2 text-sm text-ink-muted hover:text-ink hover:bg-surface-hover"
                >
                  {link.label}
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
