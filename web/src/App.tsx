import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Loading } from './components/ui'
import { LivePage } from './pages/LivePage'

// Live is the landing page and ships in the main chunk; everything else (Recharts-heavy Analytics and
// Results especially) loads on first visit.
const TrajectoriesPage = lazy(() => import('./pages/TrajectoriesPage').then((m) => ({ default: m.TrajectoriesPage })))
const SearchPage = lazy(() => import('./pages/SearchPage').then((m) => ({ default: m.SearchPage })))
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage').then((m) => ({ default: m.AnalyticsPage })))
const AlertsPage = lazy(() => import('./pages/AlertsPage').then((m) => ({ default: m.AlertsPage })))
const WatchlistPage = lazy(() => import('./pages/WatchlistPage').then((m) => ({ default: m.WatchlistPage })))
const ResultsPage = lazy(() => import('./pages/ResultsPage').then((m) => ({ default: m.ResultsPage })))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5_000, retry: 1, refetchOnWindowFocus: false },
  },
})

const page = (el: React.ReactNode) => <Suspense fallback={<Loading />}>{el}</Suspense>

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/live" replace />} />
            <Route path="live" element={<LivePage />} />
            <Route path="trajectories" element={page(<TrajectoriesPage />)} />
            <Route path="trajectories/:id" element={page(<TrajectoriesPage />)} />
            <Route path="search" element={page(<SearchPage />)} />
            <Route path="analytics" element={page(<AnalyticsPage />)} />
            <Route path="alerts" element={page(<AlertsPage />)} />
            <Route path="alerts/:id" element={page(<AlertsPage />)} />
            <Route path="watchlist" element={page(<WatchlistPage />)} />
            <Route path="results" element={page(<ResultsPage />)} />
            <Route path="*" element={<Navigate to="/live" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
