import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAlerts, useCameras, useCity } from '../api/hooks'
import type { Alert, HeatMetric } from '../api/types'
import { CityMap, type MapLine, type MapPoint } from '../components/CityMap'
import { HeatControls, HeatLegend } from '../components/HeatOverlay'
import { AlertTypeTag, cx, EmptyState, ErrorState, Loading, Plate, SeverityBadge, Swatch } from '../components/ui'
import { useLive } from '../hooks/liveStore'
import { toMapHeat, useHeat, type HeatSource } from '../hooks/useHeat'
import { fmtNum, fmtPct, fmtTime } from '../lib/format'
import { getRouter } from '../lib/geo'
import { formatPlate } from '../lib/plate'

const FADE_MS = 120_000
const HEAT_WINDOW_MIN = 15

/** Per-viewer convenience only: remembers the heatmap controls. Storage may be unavailable. */
function usePersisted<T extends string | boolean>(key: string, initial: T, valid: (v: unknown) => v is T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      const parsed: unknown = raw === null ? null : JSON.parse(raw)
      return valid(parsed) ? parsed : initial
    } catch {
      return initial
    }
  })
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(v))
    } catch {
      /* storage blocked: keep in memory */
    }
  }, [key, v])
  return [v, setV]
}
const isBool = (x: unknown): x is boolean => typeof x === 'boolean'
const isMetric = (x: unknown): x is HeatMetric => x === 'density' || x === 'speed'
const isSource = (x: unknown): x is HeatSource => x === 'live' || x === 'snapshot'

function AlertFeed() {
  const hist = useAlerts({ limit: 30 })
  const live = useLive((s) => s.alerts)
  const merged = useMemo(() => {
    const byId = new Map<string, Alert>()
    for (const a of [...live, ...(hist.data ?? [])]) if (!byId.has(a.alert_id)) byId.set(a.alert_id, a)
    return [...byId.values()].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)).slice(0, 30)
  }, [live, hist.data])
  const liveIds = useMemo(() => new Set(live.map((a) => a.alert_id)), [live])

  return (
    <section className="flex max-h-[42%] min-h-[140px] flex-col border-b border-ink-700">
      <header className="flex h-9 shrink-0 items-center justify-between border-b border-ink-700 px-3">
        <h2 className="text-[11px] font-semibold tracking-[0.08em] text-fg-muted uppercase">Alerts</h2>
        <Link to="/alerts" className="text-[11px] text-fg-dim hover:text-fg">
          All alerts →
        </Link>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto" aria-live="polite">
        {hist.isLoading && <Loading />}
        {hist.isError && <ErrorState error={hist.error} />}
        {!hist.isLoading && merged.length === 0 && <EmptyState title="No alerts" />}
        <ul>
          {merged.map((a) => (
            <li key={a.alert_id} className={cx('border-b border-ink-750', liveIds.has(a.alert_id) && 'alert-in')}>
              <Link to={`/alerts/${a.alert_id}`} className="block px-3 py-2 hover:bg-ink-800">
                <div className="flex items-center gap-2">
                  <SeverityBadge severity={a.severity} />
                  <AlertTypeTag type={a.type} />
                  {a.type === 'watchlist' && a.evidence.matched_on === 'trajectory_consensus' && <span className="text-[10px] font-medium text-fg-muted">via consensus</span>}
                  <span className="num ml-auto font-mono text-[11px] text-fg-dim">{fmtTime(a.created_at)}</span>
                </div>
                <div className="mt-1 flex items-start gap-2">
                  <Plate value={a.plate} size="sm" tone={a.type === 'anomaly' ? 'default' : 'alert'} />
                  <p className="line-clamp-2 text-[11px] leading-snug text-fg-muted">{a.summary}</p>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

function EventTicker() {
  const events = useLive((s) => s.events)
  const navigate = useNavigate()
  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <header className="flex h-9 shrink-0 items-center justify-between border-b border-ink-700 px-3">
        <h2 className="text-[11px] font-semibold tracking-[0.08em] text-fg-muted uppercase">Read ticker</h2>
        <span className="text-[10px] text-fg-dim">newest first</span>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {events.length === 0 && <EmptyState title="Waiting for reads">Start the replay from the top bar.</EmptyState>}
        <table className="w-full text-xs">
          <caption className="sr-only">Most recent ANPR reads</caption>
          <tbody>
            {events.map((e) => (
              <tr
                key={e.event_id}
                className={cx('ticker-in border-b border-ink-750', e.trajectory_id && 'cursor-pointer hover:bg-ink-800')}
                onClick={() => e.trajectory_id && navigate(`/trajectories/${e.trajectory_id}`)}
              >
                <td className="num py-1 pl-3 font-mono text-[11px] text-fg-dim">{fmtTime(e.timestamp)}</td>
                <td className="px-2 py-1 font-mono text-[11px] text-fg-muted">{e.camera_id}</td>
                <td className="py-1 font-mono text-[12px] font-semibold whitespace-nowrap text-fg-strong">{formatPlate(e.plate_argmax)}</td>
                <td className={cx('num px-2 py-1 text-right font-mono text-[11px]', e.plate_confidence < 0.7 ? 'text-caution' : 'text-fg-dim')} title="OCR confidence">
                  {fmtPct(e.plate_confidence, 0)}
                </td>
                <td className="py-1 pr-3">
                  <span className="flex items-center justify-end gap-1.5 text-[11px] text-fg-muted">
                    <Swatch color={e.color} />
                    <span className="w-8 truncate">{e.vehicle_type}</span>
                    {e.trajectory_id ? (
                      <span className="text-accent" title={`Linked to ${e.trajectory_id}`}>
                        ●
                      </span>
                    ) : (
                      <span className="text-fg-dim" title="Not yet linked">
                        ○
                      </span>
                    )}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export function LivePage() {
  const city = useCity()
  const cameras = useCameras(5_000)
  const liveTrajs = useLive((s) => s.trajectories)
  const readsSinceOpen = useLive((s) => s.eventsSinceLoad)
  const navigate = useNavigate()
  const [heatOn, setHeatOn] = usePersisted('sutra.heat.on', false, isBool)
  const [metric, setMetric] = usePersisted<HeatMetric>('sutra.heat.metric', 'density', isMetric)
  const [source, setSource] = usePersisted<HeatSource>('sutra.heat.source', 'live', isSource)
  const heat = useHeat(city.data, { enabled: heatOn, metric, source, windowMin: HEAT_WINDOW_MIN })
  const mapHeat = useMemo(() => toMapHeat(heat), [heat])

  const { lines, points } = useMemo(() => {
    if (!city.data) return { lines: [] as MapLine[], points: [] as MapPoint[] }
    const router = getRouter(city.data)
    const now = Date.now()
    const lines: MapLine[] = []
    const points: MapPoint[] = []
    // oldest first so the freshest tracks draw on top
    for (const t of [...liveTrajs].reverse()) {
      const age = Math.min(1, (now - t.updatedAt) / FADE_MS)
      const coords = router.line(t.summary.camera_sequence)
      const alert = t.summary.has_alert
      // With the heatmap on, individual tracks recede so the macro picture reads; alert tracks stay visible.
      const recede = heatOn && !alert ? 0.35 : 1
      lines.push({ id: t.summary.trajectory_id, coords, color: alert ? '#ef4444' : '#3cc4d8', width: (age < 0.05 ? 3.2 : 2.2) * (heatOn && !alert ? 0.6 : 1), opacity: (0.95 - 0.8 * age) * recede })
      const head = coords[coords.length - 1]
      if (head && age < 0.5) points.push({ id: t.summary.trajectory_id, lngLat: head, color: alert ? '#ef4444' : '#e8fbff', radius: 3.5, strokeWidth: 1.5 })
    }
    return { lines, points }
  }, [city.data, liveTrajs, heatOn])

  const activeTracks = liveTrajs.filter((t) => Date.now() - t.updatedAt < FADE_MS).length

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
      <div className="relative h-[55vh] min-h-[320px] lg:h-auto lg:min-w-0 lg:flex-1">
        {city.isLoading && <Loading label="Loading city" />}
        {city.isError && <ErrorState error={city.error} />}
        {city.data && (
          <CityMap
            city={city.data}
            cameraStats={cameras.data}
            lines={lines}
            points={points}
            livePulses
            heat={mapHeat}
            fit={{ key: 'city', padding: 40 }}
            onCameraClick={(id) => navigate(`/trajectories?camera_id=${encodeURIComponent(id)}`)}
            ariaLabel="Live city map: cameras sized by reads in the last hour, recent trajectories drawn along roads"
          />
        )}
        {/* overlay: readouts */}
        <div className="pointer-events-none absolute top-3 left-3 flex gap-2">
          {[
            ['Cameras', fmtNum(city.data?.cameras.length)],
            ['Tracks updating', fmtNum(activeTracks)],
            ['Reads since open', fmtNum(readsSinceOpen)],
          ].map(([k, v]) => (
            <div key={k} className="rounded-sm border border-ink-700 bg-ink-950/85 px-2.5 py-1.5 backdrop-blur-sm">
              <div className="text-[9px] font-semibold tracking-[0.1em] text-fg-dim uppercase">{k}</div>
              <div className="num font-mono text-[15px] text-fg-strong">{v}</div>
            </div>
          ))}
        </div>
        {/* overlay: heatmap controls + legend */}
        <div className="pointer-events-none absolute top-3 right-3 flex flex-col items-end gap-2">
          <HeatControls on={heatOn} onToggle={setHeatOn} metric={metric} onMetric={setMetric} source={source} onSource={setSource} />
        </div>
        {heat && (
          <div className="absolute right-3 bottom-3">
            <HeatLegend view={heat} />
          </div>
        )}
        {/* overlay: legend */}
        <div className="pointer-events-none absolute bottom-3 left-3 rounded-sm border border-ink-700 bg-ink-950/85 px-2.5 py-2 text-[11px] text-fg-muted backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <svg width="26" height="12" aria-hidden>
              <circle cx="4" cy="6" r="2.5" fill="#0f3940" stroke="#3cc4d8" />
              <circle cx="17" cy="6" r="5.5" fill="#0f3940" stroke="#3cc4d8" />
            </svg>
            Camera · size = reads in last hour
          </div>
          <div className="mt-1 flex items-center gap-2">
            <svg width="26" height="12" aria-hidden>
              <line x1="1" y1="6" x2="25" y2="6" stroke="#3cc4d8" strokeWidth="2.5" />
            </svg>
            Trajectory extending (fades over 2 min)
          </div>
          <div className="mt-1 flex items-center gap-2">
            <svg width="26" height="12" aria-hidden>
              <line x1="1" y1="6" x2="25" y2="6" stroke="#ef4444" strokeWidth="2.5" />
            </svg>
            Trajectory with an alert
          </div>
          <div className="mt-1 flex items-center gap-2">
            <svg width="26" height="12" aria-hidden>
              <line x1="1" y1="6" x2="25" y2="6" stroke="#3b4a5a" strokeWidth="3.5" />
            </svg>
            Road · width = speed limit
          </div>
        </div>
      </div>
      <aside className="flex min-h-[420px] w-full flex-col border-t border-ink-700 bg-ink-850 lg:min-h-0 lg:w-[360px] lg:border-t-0 lg:border-l">
        <AlertFeed />
        <EventTicker />
      </aside>
    </div>
  )
}
