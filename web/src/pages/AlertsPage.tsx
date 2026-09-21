import { useQueries } from '@tanstack/react-query'
import { useMemo } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { qk, useAlerts, useCity } from '../api/hooks'
import type { Alert, AlertType, PathPoint, TrajectoryDetail } from '../api/types'
import { CityMap, type MapLabel, type MapLine, type MapPoint } from '../components/CityMap'
import { alertTypeLabel, Button, cx, EmptyState, ErrorState, Loading, Panel, Plate, Segmented, SeverityBadge, Swatch } from '../components/ui'
import { fmtDateTime, fmtDistance, fmtDuration, fmtNum, fmtTime } from '../lib/format'
import { getRouter, haversineM, type LngLat } from '../lib/geo'
import { formatPlate } from '../lib/plate'

const TYPE_OPTS: { value: AlertType | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'clone', label: 'Clone' },
  { value: 'impossible_travel', label: 'Impossible' },
  { value: 'anomaly', label: 'Anomaly' },
]

const fin = (x: number | undefined): x is number => x !== undefined && x !== null && Number.isFinite(x)

// ------------------------------------------------------------------ physics block

function PhysicsCheck({ a }: { a: Alert }) {
  const { distance_m, delta_t_s, min_required_s } = a.evidence
  if (!fin(delta_t_s) || !fin(min_required_s)) {
    return (
      <div className="p-3 text-xs text-fg-dim">
        {fin(distance_m) ? `Sightings ${fmtDistance(distance_m)} apart. ` : ''}No travel-time evidence attached to this alert.
      </div>
    )
  }
  const impossible = delta_t_s < min_required_s
  const scaleMax = Math.max(delta_t_s, min_required_s) * 1.08
  const pct = (x: number) => `${Math.max(0, Math.min(100, (x / scaleMax) * 100))}%`
  const impliedKmh = fin(distance_m) && delta_t_s > 0 ? (distance_m / delta_t_s) * 3.6 : NaN
  const maxKmh = fin(distance_m) && min_required_s > 0 ? (distance_m / min_required_s) * 3.6 : NaN

  return (
    <div className="flex flex-col gap-3 p-3">
      <div
        role="status"
        className={cx('flex flex-wrap items-baseline gap-x-3 gap-y-1 rounded-sm border px-3 py-2', impossible ? 'border-alert bg-alert-faint' : 'border-ink-600 bg-ink-800')}
      >
        <span className={cx('text-sm font-bold tracking-[0.06em] uppercase', impossible ? 'text-alert' : 'text-fg')}>{impossible ? 'Physically impossible' : 'Physically possible'}</span>
        <span className="text-[13px] text-fg-strong">
          {impossible ? (
            <>
              Same plate seen {fin(distance_m) ? fmtDistance(distance_m) : ''} apart only <b>{fmtDuration(delta_t_s)}</b> later — at least <b>{fmtDuration(min_required_s)}</b> is needed.
            </>
          ) : (
            <>Observed gap {fmtDuration(delta_t_s)} exceeds the minimum {fmtDuration(min_required_s)}.</>
          )}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Straight-line distance</div>
          <div className="num font-mono text-2xl text-fg-strong">{fmtDistance(distance_m)}</div>
        </div>
        <div>
          <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Speed it would take</div>
          <div className={cx('num font-mono text-2xl', impossible ? 'text-alert' : 'text-fg-strong')}>{Number.isFinite(impliedKmh) ? `${fmtNum(impliedKmh)} km/h` : '—'}</div>
          {Number.isFinite(maxKmh) && <div className="text-[11px] text-fg-dim">physical gate ≈ {fmtNum(maxKmh)} km/h</div>}
        </div>
        <div>
          <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Time short by</div>
          <div className={cx('num font-mono text-2xl', impossible ? 'text-alert' : 'text-fg-strong')}>{impossible ? fmtDuration(min_required_s - delta_t_s) : '—'}</div>
          <div className="text-[11px] text-fg-dim">observed is {((delta_t_s / min_required_s) * 100).toFixed(0)}% of minimum</div>
        </div>
      </div>

      {/* time comparison on one shared scale */}
      <div className="flex flex-col gap-1.5" role="img" aria-label={`Observed gap ${fmtDuration(delta_t_s)} versus minimum required ${fmtDuration(min_required_s)}`}>
        <div className="grid grid-cols-[132px_1fr_64px] items-center gap-2">
          <span className="text-xs text-fg-muted">Minimum required</span>
          <div className="relative h-5 rounded-[2px] bg-ink-800">
            <div className="absolute inset-y-0 left-0 rounded-[2px] border border-fg-muted bg-ink-700" style={{ width: pct(min_required_s) }} />
          </div>
          <span className="num text-right font-mono text-xs text-fg">{fmtDuration(min_required_s)}</span>
        </div>
        <div className="grid grid-cols-[132px_1fr_64px] items-center gap-2">
          <span className="text-xs text-fg-muted">Observed gap</span>
          <div className="relative h-5 rounded-[2px] bg-ink-800">
            <div className={cx('absolute inset-y-0 left-0 rounded-[2px]', impossible ? 'bg-alert' : 'bg-accent')} style={{ width: pct(delta_t_s) }} />
            {impossible && (
              <div className="hatch absolute inset-y-0 flex items-center justify-center border-y border-r border-alert-dim" style={{ left: pct(delta_t_s), width: `calc(${pct(min_required_s)} - ${pct(delta_t_s)})` }}>
                <span className="truncate rounded-sm bg-ink-900 px-1 text-[10px] text-alert">missing {fmtDuration(min_required_s - delta_t_s)}</span>
              </div>
            )}
          </div>
          <span className={cx('num text-right font-mono text-xs font-semibold', impossible ? 'text-alert' : 'text-fg')}>{fmtDuration(delta_t_s)}</span>
        </div>
      </div>
    </div>
  )
}

function AppearanceGauge({ d }: { d: number | undefined }) {
  if (!fin(d)) return <p className="p-3 text-xs text-fg-dim">No appearance comparison attached.</p>
  const v = Math.max(0, Math.min(1, d))
  const verdict = v >= 0.5 ? 'Looks like a different vehicle' : v >= 0.3 ? 'Appearance inconclusive' : 'Looks like the same vehicle'
  return (
    <div className="p-3">
      <div className="flex items-baseline justify-between">
        <span className={cx('text-[13px] font-semibold', v >= 0.5 ? 'text-alert' : 'text-fg-strong')}>{verdict}</span>
        <span className="num font-mono text-lg text-fg-strong">{d.toFixed(3)}</span>
      </div>
      <div className="relative mt-2 h-3 rounded-full bg-gradient-to-r from-ink-600 via-ink-600 to-ink-600">
        <div className="absolute inset-y-0 left-0 w-[30%] rounded-l-full bg-ink-500/60" />
        <div className="absolute -top-1 h-5 w-1 -translate-x-1/2 rounded-full bg-fg-strong shadow" style={{ left: `${v * 100}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-fg-dim">
        <span>0 · identical</span>
        <span>cosine distance of Re-ID embeddings</span>
        <span>1 · unrelated</span>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ detail

function SightingCard({ label, p, traj, cameraName }: { label: string; p?: PathPoint; traj?: TrajectoryDetail; cameraName?: string }) {
  if (!p) return null
  const ev = traj?.events.find((e) => e.event_id === p.event_id)
  return (
    <div className="flex min-w-0 flex-1 items-start gap-2.5 rounded-sm border border-ink-700 bg-ink-900/60 p-2.5">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-sm bg-alert font-mono text-xs font-bold text-ink-950">{label}</span>
      <div className="min-w-0 text-xs">
        <div className="num font-mono text-[13px] text-fg-strong">{fmtTime(p.timestamp)}</div>
        <div className="truncate text-fg-muted">
          <span className="font-mono text-fg">{p.camera_id}</span> {cameraName}
        </div>
        {traj && (
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-fg-muted">
            <Swatch color={traj.color} /> {traj.color} {traj.vehicle_type}
            {ev && <span className="font-mono text-fg-dim">read {formatPlate(ev.plate_argmax)}</span>}
            <Link to={`/trajectories/${traj.trajectory_id}`} className="text-accent hover:underline">
              {traj.trajectory_id} →
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}

function AlertDetail({ alert }: { alert: Alert }) {
  const city = useCity()
  const trajs = useQueries({
    queries: alert.trajectory_ids.map((id) => ({ queryKey: qk.trajectory(id), queryFn: () => api.trajectory(id) })),
  })
  const trajData = trajs.map((q) => q.data)
  const points = useMemo(() => alert.evidence.points ?? [], [alert])
  const [pa, pb] = points
  const trajOf = (p?: PathPoint) => trajData.find((t) => t?.events.some((e) => e.event_id === p?.event_id))
  const camName = useMemo(() => new Map(city.data?.cameras.map((c) => [c.camera_id, c.name]) ?? []), [city.data])
  const isKinematic = alert.type !== 'anomaly' && points.length >= 2

  const layers = useMemo(() => {
    const lines: MapLine[] = []
    const pts: MapPoint[] = []
    const labels: MapLabel[] = []
    if (!city.data) return { lines, pts, labels }
    const router = getRouter(city.data)
    const palette = ['#3cc4d8', '#e3a008']
    trajData.forEach((t, i) => {
      if (!t) return
      lines.push({ id: `t-${t.trajectory_id}`, coords: router.line(t.path.map((p) => p.camera_id)), color: palette[i % 2], width: 2.5, opacity: 0.55 })
    })
    if (isKinematic && pa && pb) {
      lines.push({ id: 'gap', coords: [[pa.lon, pa.lat], [pb.lon, pb.lat]], color: '#ef4444', width: 2.5, opacity: 1, dashed: true })
    }
    points.forEach((p, i) => {
      const letter = String.fromCharCode(65 + i)
      pts.push({ id: `p-${i}`, lngLat: [p.lon, p.lat], color: isKinematic ? '#ef4444' : '#eef3f8', radius: 7, stroke: '#070a0e', strokeWidth: 2.5 })
      labels.push({ id: `l-${i}`, lngLat: [p.lon, p.lat], text: `${letter} · ${fmtTime(p.timestamp)}`, tone: isKinematic ? 'alert' : 'neutral' })
    })
    if (isKinematic && pa && pb && fin(alert.evidence.distance_m)) {
      labels.push({ id: 'dist', lngLat: [(pa.lon + pb.lon) / 2, (pa.lat + pb.lat) / 2], text: `${fmtDistance(alert.evidence.distance_m)} in ${fmtDuration(alert.evidence.delta_t_s)}`, tone: 'neutral' })
    }
    return { lines, pts, labels }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city.data, alert, ...trajData])

  const fitCoords = useMemo<LngLat[]>(() => points.map((p) => [p.lon, p.lat]), [points])
  const straight = pa && pb ? haversineM(pa.lat, pa.lon, pb.lat, pb.lon) : NaN

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className={cx('rounded border bg-ink-850 px-3 py-2.5', alert.type === 'anomaly' ? 'border-ink-700' : 'border-alert-dim')}>
        <div className="flex flex-wrap items-center gap-2.5">
          <SeverityBadge severity={alert.severity} />
          <span className={cx('text-sm font-semibold', alert.type === 'anomaly' ? 'text-fg-strong' : 'text-alert')}>{alertTypeLabel(alert.type)}</span>
          <Plate value={alert.plate} size="md" tone={alert.type === 'anomaly' ? 'default' : 'alert'} />
          <span className="num ml-auto font-mono text-xs text-fg-dim">
            {alert.alert_id} · {fmtDateTime(alert.created_at)} UTC
          </span>
        </div>
        <p className="mt-1.5 text-[13px] text-fg-strong">{alert.summary}</p>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <Panel title={isKinematic ? 'Conflicting sightings' : 'Sightings'} className="h-[380px]">
          {city.data ? (
            <CityMap city={city.data} lines={layers.lines} points={layers.pts} labels={layers.labels} highlightCameras={points.map((p) => p.camera_id)} fit={{ key: alert.alert_id, coords: fitCoords.length ? fitCoords : undefined, padding: 70, maxZoom: 15 }} ariaLabel="Map of the alert's sightings" />
          ) : city.isError ? (
            <ErrorState error={city.error} />
          ) : (
            <Loading />
          )}
        </Panel>
        <div className="flex flex-col gap-3">
          <Panel title="Physics check">
            <PhysicsCheck a={alert} />
          </Panel>
          <Panel title="Appearance distance">
            <AppearanceGauge d={alert.evidence.appearance_distance} />
          </Panel>
        </div>
      </div>

      {points.length > 0 && (
        <Panel title="Sightings" actions={Number.isFinite(straight) && points.length >= 2 ? <span className="num font-mono text-[11px] text-fg-dim">A↔B {fmtDistance(straight)} straight-line</span> : undefined}>
          <div className="flex flex-col gap-2 p-3 md:flex-row">
            {points.slice(0, 4).map((p, i) => (
              <SightingCard key={p.event_id} label={String.fromCharCode(65 + i)} p={p} traj={trajOf(p)} cameraName={camName.get(p.camera_id)} />
            ))}
          </div>
        </Panel>
      )}

      <Panel title="Trajectories involved">
        <ul className="divide-y divide-ink-750">
          {alert.trajectory_ids.map((id, i) => {
            const q = trajs[i]
            const t = q?.data
            return (
              <li key={id} className="flex flex-wrap items-center gap-3 px-3 py-2 text-xs">
                <span className="inline-block h-1 w-5 rounded-full" style={{ background: ['#3cc4d8', '#e3a008'][i % 2] }} aria-hidden />
                <Link to={`/trajectories/${id}`} className="font-mono text-accent hover:underline">
                  {id}
                </Link>
                {q?.isLoading && <span className="text-fg-dim">loading…</span>}
                {q?.isError && <span className="text-fg-dim">unavailable</span>}
                {t && (
                  <>
                    <Plate value={t.decoded_plate} size="sm" />
                    <span className="flex items-center gap-1.5 text-fg-muted">
                      <Swatch color={t.color} /> {t.color} {t.vehicle_type}
                    </span>
                    <span className="num font-mono text-fg-dim">
                      {fmtTime(t.start_time)}–{fmtTime(t.end_time)} · {t.n_events} hits
                    </span>
                  </>
                )}
              </li>
            )
          })}
        </ul>
      </Panel>
    </div>
  )
}

// ------------------------------------------------------------------ page

export function AlertsPage() {
  const { id } = useParams()
  const [sp, setSp] = useSearchParams()
  const navigate = useNavigate()
  const type = (sp.get('type') as AlertType | null) ?? undefined
  const alerts = useAlerts({ type, limit: 200 })
  const selected = alerts.data?.find((a) => a.alert_id === id)
  const search = sp.toString() ? `?${sp.toString()}` : ''

  return (
    <div className="flex h-full min-h-0 flex-col md:flex-row">
      <aside className={cx('flex min-h-0 flex-col border-ink-700 bg-ink-850 md:w-[340px] md:shrink-0 md:border-r', id ? 'hidden md:flex' : 'flex flex-1 md:flex-none')}>
        <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-ink-700 px-3">
          <Segmented
            label="Alert type"
            value={type ?? 'all'}
            options={TYPE_OPTS}
            onChange={(v) => {
              const next = new URLSearchParams(sp)
              if (v === 'all') next.delete('type')
              else next.set('type', v)
              setSp(next, { replace: true })
            }}
          />
          <span className="num text-[11px] text-fg-dim">{alerts.data ? alerts.data.length : ''}</span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {alerts.isLoading && <Loading />}
          {alerts.isError && <ErrorState error={alerts.error} />}
          {alerts.data?.length === 0 && <EmptyState title="No alerts" />}
          <ul>
            {alerts.data?.map((a) => {
              const active = a.alert_id === id
              const ev = a.evidence
              return (
                <li key={a.alert_id}>
                  <Link
                    to={`/alerts/${a.alert_id}${search}`}
                    aria-current={active ? 'page' : undefined}
                    className={cx('block border-b border-ink-750 px-3 py-2', active ? 'bg-ink-750 shadow-[inset_2px_0_0_var(--color-alert)]' : 'hover:bg-ink-800')}
                  >
                    <div className="flex items-center gap-2">
                      <SeverityBadge severity={a.severity} />
                      <span className={cx('text-xs font-semibold', a.type === 'anomaly' ? 'text-fg' : 'text-alert')}>{alertTypeLabel(a.type)}</span>
                      <span className="num ml-auto font-mono text-[11px] text-fg-dim">{fmtTime(a.created_at)}</span>
                    </div>
                    <div className="mt-1.5 flex items-center gap-2">
                      <Plate value={a.plate} size="sm" tone={a.type === 'anomaly' ? 'default' : 'alert'} />
                      {fin(ev.distance_m) && fin(ev.delta_t_s) && (
                        <span className="num truncate font-mono text-[11px] text-fg-muted">
                          {fmtDistance(ev.distance_m)} in {fmtDuration(ev.delta_t_s)}
                          {fin(ev.min_required_s) && <span className="text-fg-dim"> / ≥{fmtDuration(ev.min_required_s)}</span>}
                        </span>
                      )}
                    </div>
                    <p className="mt-1 line-clamp-2 text-[11px] text-fg-dim">{a.summary}</p>
                  </Link>
                </li>
              )
            })}
          </ul>
        </div>
      </aside>
      <section className={cx('min-h-0 min-w-0 flex-1 overflow-y-auto', !id && 'hidden md:block')}>
        {id && (
          <div className="border-b border-ink-700 px-3 py-1.5 md:hidden">
            <Button variant="ghost" onClick={() => navigate(`/alerts${search}`)}>
              ← Back to alerts
            </Button>
          </div>
        )}
        {!id && <EmptyState title="Select an alert">Clone and impossible-travel alerts show both sightings, the distance, and the time gap against the minimum physically possible.</EmptyState>}
        {id && alerts.isLoading && <Loading />}
        {id && alerts.data && !selected && <EmptyState title="Alert not found">It may be outside the current filter.</EmptyState>}
        {selected && <AlertDetail key={selected.alert_id} alert={selected} />}
      </section>
    </div>
  )
}
