import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import LandingHero from './components/LandingHero';
import QuantSignals from './components/QuantSignals';
import Fundamentals from './components/Fundamentals';
import Auth from './pages/Auth';
import Home from './pages/Home';
import Watchlist from './pages/Watchlist';
import NewsSentiment from './pages/NewsSentiment';
import Simulator from './pages/Simulator';
import PaperTrading from './pages/PaperTrading';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';

// Protected Route Wrapper
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  if (loading) return null;
  if (!user) return <Navigate to="/auth" replace />;
  return children;
};

// Main Layout with Top Bar and Sidebar (Dashboard Shell)
const Shell = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="h-screen bg-background text-on-background font-body selection:bg-primary selection:text-on-primary transition-colors duration-300 flex flex-col overflow-hidden">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-primary text-on-primary px-4 py-2 z-50 rounded-sm outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-surface">
        Skip to main content
      </a>

      {/* Extracted TopBar */}
      <TopBar onMenuClick={() => setSidebarOpen(true)} />

      <div className="flex flex-1 overflow-hidden">
        {/* Extracted Sidebar */}
        <Sidebar isOpen={sidebarOpen} setIsOpen={setSidebarOpen} />

        {/* Main Content */}
        <main id="main-content" className="flex-1 overflow-y-auto overflow-x-hidden relative bg-background transition-colors duration-300">
          {children}
        </main>
      </div>
    </div>
  );
};

const AppRoutes = () => {
  const navigate = useNavigate();
  return (
    <Routes>
      <Route path="/" element={<LandingHero onEnter={() => navigate('/home')} />} />
      <Route path="/auth" element={<Auth />} />
      <Route path="/login" element={<Navigate to="/auth" replace />} />
      
      {/* Protected Routes inside Shell */}
      <Route path="/home" element={
        <ProtectedRoute>
          <Shell><Home /></Shell>
        </ProtectedRoute>
      } />
      
      <Route path="/watchlist" element={
        <ProtectedRoute>
          <Shell><Watchlist /></Shell>
        </ProtectedRoute>
      } />
      
      <Route path="/dashboard" element={<Navigate to="/dashboard/RELIANCE" replace />} />
      <Route path="/dashboard/:symbol" element={
        <ProtectedRoute>
          <Shell><Dashboard /></Shell>
        </ProtectedRoute>
      } />
      
      <Route path="/signals" element={
        <ProtectedRoute>
          <Shell><QuantSignals /></Shell>
        </ProtectedRoute>
      } />
      
      <Route path="/fundamentals/:symbol" element={
        <ProtectedRoute>
          <Shell><Fundamentals /></Shell>
        </ProtectedRoute>
      } />

      {/* NEW: News & Sentiment */}
      <Route path="/news/:symbol" element={
        <ProtectedRoute>
          <Shell><NewsSentiment /></Shell>
        </ProtectedRoute>
      } />

      {/* NEW: Trading Simulator / Backtest Lab */}
      <Route path="/simulator" element={
        <ProtectedRoute>
          <Shell><Simulator /></Shell>
        </ProtectedRoute>
      } />

      {/* Paper Trading Dashboard */}
      <Route path="/paper-trading" element={
        <ProtectedRoute>
          <Shell><PaperTrading /></Shell>
        </ProtectedRoute>
      } />
    </Routes>
  );
};

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <Router>
          <AppRoutes />
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}
