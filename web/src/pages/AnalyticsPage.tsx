import { useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useCity, useCorridors, useFlowTrend, useOdMatrix, useSummary, useVolumes } from '../api/hooks'
import type { Corridor, FlowBucket } from '../api/types'
import { SPEED_V_MAX_KMH } from '../lib/speed'
import { cx, EmptyState, ErrorState, Loading, Panel, Segmented, inputCls } from '../components/ui'
import { fmtDistance, fmtDuration, fmtNum, fmtPct, fmtTime } from '../lib/format'

function Kpi({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: 'alert' | 'accent' }) {
  return (
    <div className="rounded border border-ink-700 bg-ink-850 px-3 py-2.5">
      <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">{label}</div>
      <div className={cx('num mt-0.5 font-mono text-[22px] leading-tight', tone === 'alert' ? 'text-alert' : tone === 'accent' ? 'text-accent-strong' : 'text-fg-strong')}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-fg-dim">{hint}</div>}
    </div>
  )
}

function KpiRow() {
  const s = useSummary()
  if (s.isError) return <ErrorState error={s.error} className="h-20" />
  const d = s.data
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <Kpi label="Plate reads" value={d ? fmtNum(d.n_events) : '…'} />
      <Kpi label="Trajectories" value={d ? fmtNum(d.n_trajectories) : '…'} />
      <Kpi label="Mean trip" value={d ? fmtDuration(d.mean_trip_duration_s) : '…'} />
      <Kpi label="Reads / trajectory" value={d ? fmtNum(d.mean_events_per_trajectory, 1) : '…'} />
      <Kpi label="Plate repair rate" value={d ? fmtPct(d.plate_repair_rate, 1) : '…'} hint="multi-read tracks where consensus corrected a read" tone="accent" />
      <Kpi label="Active alerts" value={d ? fmtNum(d.active_alerts) : '…'} tone={d && d.active_alerts > 0 ? 'alert' : undefined} />
    </div>
  )
}

function VolumeChart() {
  const [bucket, setBucket] = useState(30)
  const [camera, setCamera] = useState('')
  const city = useCity()
  const v = useVolumes({ bucket_minutes: bucket, camera_id: camera || undefined })
  const data = useMemo(() => {
    const m = new Map<string, number>()
    for (const r of v.data ?? []) m.set(r.bucket_start, (m.get(r.bucket_start) ?? 0) + r.count)
    return [...m.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([t, count]) => ({ t, count }))
  }, [v.data])

  return (
    <Panel
      title="Read volume over time"
      className="h-[320px]"
      actions={
        <>
          <select aria-label="Camera" className={cx(inputCls, 'h-7 w-32 text-xs')} value={camera} onChange={(e) => setCamera(e.target.value)}>
            <option value="">All cameras</option>
            {city.data?.cameras.map((c) => (
              <option key={c.camera_id} value={c.camera_id}>
                {c.camera_id}
              </option>
            ))}
          </select>
          <Segmented label="Bucket size" value={bucket} onChange={setBucket} options={[15, 30, 60].map((m) => ({ value: m, label: `${m}m` }))} />
        </>
      }
      bodyClassName="p-2"
    >
      {v.isLoading ? (
        <Loading />
      ) : v.isError ? (
        <ErrorState error={v.error} />
      ) : data.length === 0 ? (
        <EmptyState title="No volume data" />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
            <defs>
              <linearGradient id="volFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3cc4d8" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#3cc4d8" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#1e2732" vertical={false} />
            <XAxis dataKey="t" tickFormatter={(t: string) => fmtTime(t, false)} stroke="#566374" tick={{ fontSize: 11, fill: '#7d8b9b' }} tickLine={false} axisLine={{ stroke: '#2a3542' }} minTickGap={24} />
            <YAxis stroke="#566374" tick={{ fontSize: 11, fill: '#7d8b9b' }} tickLine={false} axisLine={false} allowDecimals={false} width={44} />
            <Tooltip
              cursor={{ stroke: '#7d8b9b', strokeDasharray: '3 3' }}
              contentStyle={{ background: '#131920', border: '1px solid #2a3542', borderRadius: 4, fontSize: 12 }}
              labelStyle={{ color: '#7d8b9b' }}
              itemStyle={{ color: '#eef3f8' }}
              labelFormatter={(t) => `${fmtTime(String(t), false)} UTC · ${bucket} min bucket`}
              formatter={(value) => [fmtNum(Number(value)), 'reads']}
            />
            <Area type="monotone" dataKey="count" stroke="#3cc4d8" strokeWidth={2} fill="url(#volFill)" isAnimationActive={false} activeDot={{ r: 4, stroke: '#0f141a', strokeWidth: 2 }} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </Panel>
  )
}

function OdHeatmap() {
  const od = useOdMatrix()
  const d = od.data
  const max = d ? Math.max(1, ...d.matrix.flat()) : 1
  const rowTotals = d?.matrix.map((r) => r.reduce((a, b) => a + b, 0)) ?? []
  return (
    <Panel title="Origin → destination (trips)" className="h-[320px]" bodyClassName="p-3 overflow-auto">
      {od.isLoading && <Loading />}
      {od.isError && <ErrorState error={od.error} />}
      {d && d.zones.length === 0 && <EmptyState title="No OD data" />}
      {d && d.zones.length > 0 && (
        <table className="w-full border-separate border-spacing-[3px] text-xs">
          <caption className="sr-only">Trips by origin zone (rows) and destination zone (columns)</caption>
          <thead>
            <tr>
              <th className="text-left text-[10px] font-normal text-fg-dim">from ↓ to →</th>
              {d.zones.map((z) => (
                <th key={z} scope="col" className="pb-1 text-center text-[11px] font-semibold text-fg-muted">
                  {z}
                </th>
              ))}
              <th className="text-right text-[10px] font-normal text-fg-dim">total</th>
            </tr>
          </thead>
          <tbody>
            {d.zones.map((z, i) => (
              <tr key={z}>
                <th scope="row" className="pr-1 text-left text-[11px] font-semibold text-fg-muted">
                  {z}
                </th>
                {d.zones.map((z2, j) => {
                  const v = d.matrix[i]?.[j] ?? 0
                  const a = v / max
                  return (
                    <td
                      key={z2}
                      title={`${z} → ${z2}: ${v} trips`}
                      className="num h-9 rounded-[3px] text-center font-mono text-[12px]"
                      style={{ background: `rgba(60,196,216,${0.06 + 0.8 * a})`, color: a > 0.5 ? '#070a0e' : '#d3dbe4' }}
                    >
                      {fmtNum(v)}
                    </td>
                  )
                })}
                <td className="num pl-1 text-right font-mono text-[11px] text-fg-dim">{fmtNum(rowTotals[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ speeds & congestion

/** avg speed as a fraction of free-flow speed; NaN when unknown. */
const speedRatio = (r: Corridor) => (r.free_flow_speed_kmh > 0 && Number.isFinite(r.avg_speed_kmh) ? r.avg_speed_kmh / r.free_flow_speed_kmh : NaN)

/**
 * Sequential single-hue congestion scale (caution amber, dim -> bright as the road slows).
 * 0 = at free flow, 1 = at or below 30% of free-flow speed. Always shown next to the number.
 */
function congestionLevel(ratio: number): number {
  if (!Number.isFinite(ratio)) return 0
  return Math.max(0, Math.min(1, (1 - ratio) / 0.7))
}
const congestionFill = (level: number) => `rgba(227,160,8,${(0.18 + 0.82 * level).toFixed(3)})`
const congestionWord = (ratio: number) => (!Number.isFinite(ratio) ? '—' : ratio >= 0.85 ? 'free flowing' : ratio >= 0.65 ? 'moderate' : ratio >= 0.45 ? 'heavy' : 'severe')

function SpeedVsFreeFlow({ r }: { r: Corridor }) {
  const ratio = speedRatio(r)
  const p85 = r.free_flow_speed_kmh > 0 ? r.p85_speed_kmh / r.free_flow_speed_kmh : NaN
  const scale = 1.2
  const pct = (x: number) => `${Math.max(0, Math.min(100, (x / scale) * 100))}%`
  return (
    <div className="flex items-center gap-2" title={`Average ${fmtNum(r.avg_speed_kmh)} km/h, P85 ${fmtNum(r.p85_speed_kmh)} km/h, free-flow ${fmtNum(r.free_flow_speed_kmh)} km/h`}>
      <div className="relative h-2.5 flex-1 rounded-[2px] bg-ink-700">
        {Number.isFinite(ratio) && <div className="absolute inset-y-0 left-0 rounded-[2px]" style={{ width: pct(ratio), background: congestionFill(congestionLevel(ratio)) }} />}
        {Number.isFinite(p85) && <div className="absolute -inset-y-0.5 w-0.5 rounded-full bg-fg-muted" style={{ left: pct(p85) }} aria-hidden />}
        <div className="absolute -inset-y-1 w-px bg-fg" style={{ left: pct(1) }} aria-hidden />
      </div>
      <span className="num w-10 text-right font-mono text-fg">{Number.isFinite(ratio) ? `${Math.round(ratio * 100)}%` : '—'}</span>
    </div>
  )
}

function CongestionChip({ idx, ratio }: { idx: number; ratio: number }) {
  const level = congestionLevel(ratio)
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="num inline-block w-12 rounded-[2px] px-1 text-center font-mono font-semibold" style={{ background: congestionFill(level), color: level > 0.45 ? '#070a0e' : '#eef3f8' }}>
        {Number.isFinite(idx) ? idx.toFixed(2) : '—'}
      </span>
      <span className="w-[74px] text-[11px] text-fg-muted">{congestionWord(ratio)}</span>
    </span>
  )
}

function CorridorTable({ q }: { q: ReturnType<typeof useCorridors> }) {
  const city = useCity()
  const names = useMemo(() => new Map(city.data?.cameras.map((x) => [x.camera_id, x.name]) ?? []), [city.data])
  const rows = useMemo(() => [...(q.data ?? [])].sort((a, b) => b.congestion_index - a.congestion_index).slice(0, 20), [q.data])
  return (
    <Panel
      title="Corridor speeds & congestion"
      actions={<span className="hidden text-[10px] text-fg-dim md:inline">speed = road distance / observed travel time · congestion index = median / free-flow time · worst first</span>}
      bodyClassName="overflow-x-auto"
    >
      {q.isLoading && <Loading />}
      {q.isError && <ErrorState error={q.error} />}
      {q.data && rows.length === 0 && <EmptyState title="No corridors with enough trips yet" />}
      {rows.length > 0 && (
        <table className="w-full min-w-[1040px] text-xs">
          <caption className="sr-only">Corridor travel times, speeds and congestion, worst congestion first</caption>
          <thead>
            <tr className="border-b border-ink-700 text-[10px] tracking-wider text-fg-dim uppercase">
              <th className="px-3 py-2 text-left font-semibold">Corridor</th>
              <th className="px-2 py-2 text-right font-semibold">Trips</th>
              <th className="px-2 py-2 text-right font-semibold">Road dist.</th>
              <th className="px-2 py-2 text-right font-semibold">Free-flow</th>
              <th className="px-2 py-2 text-right font-semibold">Median</th>
              <th className="px-2 py-2 text-right font-semibold">P90</th>
              <th className="px-2 py-2 text-right font-semibold" title="Mean speed of observed trips">
                Avg km/h
              </th>
              <th className="px-2 py-2 text-right font-semibold" title="85th percentile speed">
                P85 km/h
              </th>
              <th className="w-[200px] px-3 py-2 text-left font-semibold" title="Average speed as a share of free-flow speed; tick = P85, line = free flow">
                Speed vs free-flow
              </th>
              <th className="px-3 py-2 text-left font-semibold">Congestion idx</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const ratio = speedRatio(r)
              return (
                <tr key={`${r.from_camera}-${r.to_camera}`} className="border-b border-ink-750 hover:bg-ink-800">
                  <td className="max-w-[260px] px-3 py-1.5">
                    <div>
                      <span className="font-mono text-fg">{r.from_camera}</span>
                      <span className="px-1.5 text-fg-dim">→</span>
                      <span className="font-mono text-fg">{r.to_camera}</span>
                    </div>
                    <div className="truncate text-[10px] text-fg-dim" title={`${names.get(r.from_camera) ?? ''} → ${names.get(r.to_camera) ?? ''}`}>
                      {names.get(r.from_camera)} → {names.get(r.to_camera)}
                    </div>
                  </td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtNum(r.n_trips)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtDistance(r.distance_m)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtDuration(r.free_flow_s)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg">{fmtDuration(r.median_travel_s)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtDuration(r.p90_travel_s)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-strong">{fmtNum(r.avg_speed_kmh)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtNum(r.p85_speed_kmh)}</td>
                  <td className="px-3 py-1.5">
                    <SpeedVsFreeFlow r={r} />
                  </td>
                  <td className="px-3 py-1.5">
                    <CongestionChip idx={r.congestion_index} ratio={ratio} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
      {rows.length > 0 && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-ink-700 px-3 py-2 text-[10px] text-fg-dim">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-16 rounded-[2px]" style={{ background: `linear-gradient(90deg, ${congestionFill(0)}, ${congestionFill(1)})` }} aria-hidden />
            free flow → ≤ 30% of free-flow speed
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-px bg-fg" aria-hidden /> free-flow speed (100%)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-0.5 rounded-full bg-fg-muted" aria-hidden /> P85 speed
          </span>
          <span>Links faster than {SPEED_V_MAX_KMH} km/h are excluded from speeds (they are clone evidence).</span>
        </div>
      )}
    </Panel>
  )
}

function Bottlenecks({ q }: { q: ReturnType<typeof useCorridors> }) {
  const city = useCity()
  const names = useMemo(() => new Map(city.data?.cameras.map((x) => [x.camera_id, x.name]) ?? []), [city.data])
  const ranked = useMemo(
    () =>
      [...(q.data ?? [])]
        .filter((r) => Number.isFinite(speedRatio(r)))
        .sort((a, b) => speedRatio(a) - speedRatio(b))
        .slice(0, 7),
    [q.data],
  )
  return (
    <Panel title="Congestion bottlenecks" actions={<span className="text-[10px] text-fg-dim">lowest avg speed ÷ free-flow speed</span>} className="h-[340px]" bodyClassName="overflow-y-auto">
      {q.isLoading && <Loading />}
      {q.isError && <ErrorState error={q.error} />}
      {q.data && ranked.length === 0 && <EmptyState title="No corridor speeds yet" />}
      <ol>
        {ranked.map((r, i) => {
          const ratio = speedRatio(r)
          const level = congestionLevel(ratio)
          return (
            <li key={`${r.from_camera}-${r.to_camera}`} className="grid grid-cols-[22px_1fr_auto] items-center gap-x-2.5 border-b border-ink-750 px-3 py-2">
              <span className="num text-center font-mono text-[13px] text-fg-dim">{i + 1}</span>
              <div className="min-w-0">
                <div className="flex items-baseline gap-1.5 text-xs">
                  <span className="font-mono text-fg">{r.from_camera}</span>
                  <span className="text-fg-dim">→</span>
                  <span className="font-mono text-fg">{r.to_camera}</span>
                  <span className="num ml-1 text-[10px] text-fg-dim">{fmtNum(r.n_trips)} trips</span>
                </div>
                <div className="truncate text-[10px] text-fg-dim">
                  {names.get(r.from_camera)} → {names.get(r.to_camera)}
                </div>
                <div className="mt-1 h-1.5 rounded-full bg-ink-700">
                  <div className="h-full rounded-full" style={{ width: `${Math.min(100, ratio * 100)}%`, background: congestionFill(level) }} />
                </div>
              </div>
              <div className="text-right">
                <div className="num font-mono text-[15px] leading-tight text-fg-strong">{Math.round(ratio * 100)}%</div>
                <div className="num font-mono text-[10px] text-fg-dim">
                  {fmtNum(r.avg_speed_kmh)} / {fmtNum(r.free_flow_speed_kmh)} km/h
                </div>
              </div>
            </li>
          )
        })}
      </ol>
    </Panel>
  )
}

// ------------------------------------------------------------------ flow trend

type FlowRow = FlowBucket & { t: string }

function FlowTooltip({ active, payload, bucket }: { active?: boolean; payload?: { payload?: FlowRow }[]; bucket: number }) {
  const row = payload?.[0]?.payload
  if (!active || !row) return null
  return (
    <div className="rounded border border-ink-600 bg-ink-800 px-2.5 py-1.5 text-xs shadow-lg">
      <div className="text-fg-muted">
        {fmtTime(row.bucket_start, false)} UTC · {bucket} min
      </div>
      <div className="num mt-0.5 font-mono text-fg-strong">{fmtNum(row.events)} reads</div>
      <div className="num font-mono text-fg">{fmtNum(row.active_trajectories)} active tracks</div>
      <div className="num font-mono text-fg">{row.mean_speed_kmh === null ? 'no speed data' : `${fmtNum(row.mean_speed_kmh, 1)} km/h mean speed`}</div>
    </div>
  )
}

function FlowTrend() {
  const [bucket, setBucket] = useState(15)
  const f = useFlowTrend({ bucket_minutes: bucket })
  const data: FlowRow[] = useMemo(() => (f.data ?? []).map((b) => ({ ...b, t: b.bucket_start })), [f.data])
  const axis = { stroke: '#566374', tick: { fontSize: 11, fill: '#7d8b9b' }, tickLine: false }
  const speeds = data.map((d) => d.mean_speed_kmh).filter((v): v is number => v !== null)
  return (
    <Panel
      title="Traffic flow trend"
      className="h-[340px]"
      actions={<Segmented label="Flow bucket size" value={bucket} onChange={setBucket} options={[15, 30, 60].map((m) => ({ value: m, label: `${m}m` }))} />}
      bodyClassName="flex flex-col px-2 pt-1 pb-2"
    >
      {f.isLoading ? (
        <Loading />
      ) : f.isError ? (
        <ErrorState error={f.error} />
      ) : data.length === 0 ? (
        <EmptyState title="No flow data" />
      ) : (
        <>
          <div className="flex items-baseline justify-between px-2 text-[10px] text-fg-dim">
            <span className="font-semibold tracking-[0.08em] uppercase">Reads per {bucket} min</span>
            <span className="num font-mono">peak {fmtNum(Math.max(...data.map((d) => d.events)))}</span>
          </div>
          <div className="min-h-0 flex-[1.15]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} syncId="flow" margin={{ top: 4, right: 12, bottom: 0, left: -8 }}>
                <defs>
                  <linearGradient id="flowFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3cc4d8" stopOpacity={0.32} />
                    <stop offset="100%" stopColor="#3cc4d8" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1e2732" vertical={false} />
                <XAxis dataKey="t" hide />
                <YAxis {...axis} axisLine={false} allowDecimals={false} width={44} />
                <Tooltip cursor={{ stroke: '#7d8b9b', strokeDasharray: '3 3' }} content={<FlowTooltip bucket={bucket} />} />
                <Area type="monotone" dataKey="events" stroke="#3cc4d8" strokeWidth={2} fill="url(#flowFill)" isAnimationActive={false} activeDot={{ r: 4, stroke: '#0f141a', strokeWidth: 2 }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-1 flex items-baseline justify-between px-2 text-[10px] text-fg-dim">
            <span className="font-semibold tracking-[0.08em] uppercase">Mean link speed, km/h</span>
            <span className="num font-mono">{speeds.length ? `range ${fmtNum(Math.min(...speeds))}–${fmtNum(Math.max(...speeds))}` : ''}</span>
          </div>
          <div className="min-h-0 flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} syncId="flow" margin={{ top: 4, right: 12, bottom: 0, left: -8 }}>
                <CartesianGrid stroke="#1e2732" vertical={false} />
                <XAxis dataKey="t" tickFormatter={(t: string) => fmtTime(t, false)} {...axis} axisLine={{ stroke: '#2a3542' }} minTickGap={28} />
                <YAxis {...axis} axisLine={false} width={44} domain={['auto', 'auto']} />
                <Tooltip cursor={{ stroke: '#7d8b9b', strokeDasharray: '3 3' }} content={() => null} />
                <Line type="monotone" dataKey="mean_speed_kmh" stroke="#d3dbe4" strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} activeDot={{ r: 4, fill: '#d3dbe4', stroke: '#0f141a', strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </Panel>
  )
}

export function AnalyticsPage() {
  const corridors = useCorridors({ limit: 40 })
  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="flex flex-col gap-3">
        <KpiRow />
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1.6fr)_minmax(340px,1fr)]">
          <VolumeChart />
          <OdHeatmap />
        </div>
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1.6fr)_minmax(340px,1fr)]">
          <FlowTrend />
          <Bottlenecks q={corridors} />
        </div>
        <CorridorTable q={corridors} />
      </div>
    </div>
  )
}
