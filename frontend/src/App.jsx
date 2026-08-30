import { Routes, Route, Navigate } from "react-router-dom";
import Navigation from "./components/Navigation";
import Welcome from "./pages/Welcome";
import Dashboard from "./pages/Dashboard";
import TopPredictions from "./pages/TopPredictions";
import AllCompanies from "./pages/AllCompanies";
import CompanyDetail from "./pages/CompanyDetail";
import MarketOverview from "./pages/MarketOverview";
import ModelInfo from "./pages/ModelInfo";
import About from "./pages/About";
import Disclaimer from "./pages/Disclaimer";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Welcome />} />
      <Route
        path="/*"
        element={
          <div className="min-h-screen bg-bg text-ink">
            <Navigation />
            <Routes>
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="predictions" element={<TopPredictions />} />
              <Route path="companies" element={<AllCompanies />} />
              <Route path="company/:symbol" element={<CompanyDetail />} />
              <Route path="market" element={<MarketOverview />} />
              <Route path="model-info" element={<ModelInfo />} />
              <Route path="about" element={<About />} />
              <Route path="disclaimer" element={<Disclaimer />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </div>
        }
      />
    </Routes>
  );
}
