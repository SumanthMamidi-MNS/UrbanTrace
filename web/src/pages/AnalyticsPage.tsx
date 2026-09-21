import { useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useCity, useCorridors, useOdMatrix, useSummary, useVolumes } from '../api/hooks'
import { cx, EmptyState, ErrorState, Loading, Panel, Segmented, inputCls } from '../components/ui'
import { fmtDuration, fmtNum, fmtPct, fmtTime } from '../lib/format'

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

function CorridorTable() {
  const c = useCorridors({ limit: 20 })
  const city = useCity()
  const names = useMemo(() => new Map(city.data?.cameras.map((x) => [x.camera_id, x.name]) ?? []), [city.data])
  const rows = useMemo(() => [...(c.data ?? [])].sort((a, b) => b.congestion_index - a.congestion_index), [c.data])
  const maxIdx = Math.max(2, ...rows.map((r) => r.congestion_index))
  return (
    <Panel title="Corridor congestion" actions={<span className="text-[10px] text-fg-dim">index = median / free-flow travel time · sorted worst first</span>} bodyClassName="overflow-x-auto">
      {c.isLoading && <Loading />}
      {c.isError && <ErrorState error={c.error} />}
      {c.data && rows.length === 0 && <EmptyState title="No corridors with enough trips yet" />}
      {rows.length > 0 && (
        <table className="w-full min-w-[720px] text-xs">
          <thead>
            <tr className="border-b border-ink-700 text-[10px] tracking-wider text-fg-dim uppercase">
              <th className="px-3 py-2 text-left font-semibold">Corridor</th>
              <th className="px-2 py-2 text-right font-semibold">Trips</th>
              <th className="px-2 py-2 text-right font-semibold">Free-flow</th>
              <th className="px-2 py-2 text-right font-semibold">Median</th>
              <th className="px-2 py-2 text-right font-semibold">P90</th>
              <th className="w-[240px] px-3 py-2 text-left font-semibold">Congestion index</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const heavy = r.congestion_index >= 1.5
              return (
                <tr key={`${r.from_camera}-${r.to_camera}`} className="border-b border-ink-750 hover:bg-ink-800">
                  <td className="px-3 py-1.5">
                    <span className="font-mono text-fg">{r.from_camera}</span>
                    <span className="px-1.5 text-fg-dim">→</span>
                    <span className="font-mono text-fg">{r.to_camera}</span>
                    <span className="ml-2 hidden text-[11px] text-fg-dim lg:inline">
                      {names.get(r.from_camera)} → {names.get(r.to_camera)}
                    </span>
                  </td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtNum(r.n_trips)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtDuration(r.free_flow_s)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg">{fmtDuration(r.median_travel_s)}</td>
                  <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtDuration(r.p90_travel_s)}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="relative h-2 flex-1 rounded-full bg-ink-700">
                        <div className={cx('absolute inset-y-0 left-0 rounded-full', heavy ? 'bg-caution' : 'bg-accent')} style={{ width: `${Math.min(100, (r.congestion_index / maxIdx) * 100)}%` }} />
                        <div className="absolute -inset-y-0.5 w-px bg-fg-muted" style={{ left: `${(1 / maxIdx) * 100}%` }} title="free flow (1.0)" />
                      </div>
                      <span className={cx('num w-10 text-right font-mono', heavy ? 'font-semibold text-caution' : 'text-fg')}>{r.congestion_index.toFixed(2)}</span>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </Panel>
  )
}

export function AnalyticsPage() {
  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="flex flex-col gap-3">
        <KpiRow />
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1.6fr)_minmax(340px,1fr)]">
          <VolumeChart />
          <OdHeatmap />
        </div>
        <CorridorTable />
      </div>
    </div>
  )
}
