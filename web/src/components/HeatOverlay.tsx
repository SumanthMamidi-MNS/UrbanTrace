import type { HeatMetric } from '../api/types'
import { SLOW_FAST_KMH, SLOW_SLOW_KMH, type HeatSource, type HeatView } from '../hooks/useHeat'
import { fmtNum, fmtTime } from '../lib/format'
import { HEAT_RAMP } from '../lib/heat'
import { cx, Segmented } from './ui'

const gradient = `linear-gradient(90deg, ${HEAT_RAMP.map(([s, c]) => `${c.replace(/rgba\((\d+),(\d+),(\d+),[\d.]+\)/, 'rgb($1,$2,$3)')} ${s * 100}%`).join(', ')})`

export function HeatControls({
  on,
  onToggle,
  metric,
  onMetric,
  source,
  onSource,
}: {
  on: boolean
  onToggle: (v: boolean) => void
  metric: HeatMetric
  onMetric: (m: HeatMetric) => void
  source: HeatSource
  onSource: (s: HeatSource) => void
}) {
  return (
    <div className="pointer-events-auto flex flex-col items-end gap-1.5 rounded-sm border border-ink-700 bg-ink-950/90 p-1.5 backdrop-blur-sm">
      <button
        type="button"
        role="switch"
        aria-checked={on}
        onClick={() => onToggle(!on)}
        className={cx('inline-flex h-7 items-center gap-2 rounded-sm border px-2.5 text-xs font-medium transition-colors', on ? 'border-accent-dim bg-accent-faint text-accent-strong' : 'border-ink-600 bg-ink-800 text-fg-muted hover:text-fg')}
      >
        <span className={cx('relative h-3.5 w-6 rounded-full transition-colors', on ? 'bg-accent' : 'bg-ink-600')} aria-hidden>
          <span className={cx('absolute top-0.5 h-2.5 w-2.5 rounded-full bg-ink-950 transition-[left]', on ? 'left-3' : 'left-0.5')} />
        </span>
        Heatmap
      </button>
      {on && (
        <>
          <Segmented label="Heatmap metric" value={metric} onChange={onMetric} options={[{ value: 'density', label: 'Density' }, { value: 'speed', label: 'Speed' }]} />
          <Segmented label="Heatmap source" value={source} onChange={onSource} options={[{ value: 'live', label: 'Live' }, { value: 'snapshot', label: 'Snapshot' }]} />
        </>
      )}
    </div>
  )
}

export function HeatLegend({ view }: { view: HeatView }) {
  const isSpeed = view.metric === 'speed'
  const ranked = [...view.cells].sort((a, b) => (isSpeed ? a.value - b.value : b.value - a.value)).slice(0, 3)
  const fmtVal = (v: number) => (view.valueKind === 'kmh' ? `${fmtNum(v)} km/h` : view.valueKind === 'count' ? `${fmtNum(v)} reads` : `${fmtNum(v * 100)}% of max`)
  const building = view.shown === 'live' && view.coverageMin !== null && view.coverageMin < view.windowMin - 0.05

  return (
    <div className="pointer-events-none w-[244px] rounded-sm border border-ink-700 bg-ink-950/90 px-2.5 py-2 text-[11px] text-fg-muted backdrop-blur-sm">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-semibold text-fg">{isSpeed ? 'Traffic speed' : 'Traffic density'}</span>
        <span className="num font-mono text-[10px] text-fg-dim">{view.at ? `${fmtTime(view.at, false)} UTC` : ''}</span>
      </div>
      <div className="mt-0.5 text-[10px] leading-snug text-fg-dim">
        {view.shown === 'live' ? `Live · last ${view.windowMin} min of sim time, from the feed` : `Snapshot · /api/analytics/heatmap, ${view.windowMin} min window`}
        {view.shown === 'snapshot' && view.coverageMin !== null && ' (live buffer still empty)'}
      </div>
      <div className="mt-1.5 h-2 rounded-full" style={{ background: gradient }} aria-hidden />
      <div className="mt-0.5 flex justify-between text-[10px] text-fg-dim">
        {isSpeed ? (
          <>
            <span>≥ {SLOW_FAST_KMH} km/h</span>
            <span>slower = brighter</span>
            <span>≤ {SLOW_SLOW_KMH} km/h</span>
          </>
        ) : (
          <>
            <span>few reads</span>
            <span>per camera, vs busiest</span>
            <span>most</span>
          </>
        )}
      </div>
      {building && (
        <div className="mt-1.5 flex items-center gap-2 text-[10px] text-fg-dim">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-ink-700">
            <div className="h-full bg-fg-dim" style={{ width: `${((view.coverageMin ?? 0) / view.windowMin) * 100}%` }} />
          </div>
          <span className="num font-mono">
            building {fmtNum(view.coverageMin ?? 0, 1)}/{view.windowMin} min
          </span>
        </div>
      )}
      {view.error ? (
        <div className="mt-1.5 text-[10px] text-fg">Heatmap unavailable from the API.</div>
      ) : ranked.length > 0 ? (
        <div className="mt-1.5 border-t border-ink-700 pt-1.5">
          <div className="text-[9px] font-semibold tracking-[0.1em] text-fg-dim uppercase">{isSpeed ? 'Slowest cameras' : 'Busiest cameras'}</div>
          <ol className="mt-0.5">
            {ranked.map((c) => (
              <li key={c.camera_id} className="num flex justify-between font-mono text-[11px]">
                <span className="text-fg">{c.camera_id}</span>
                <span className="text-fg-muted">{fmtVal(c.value)}</span>
              </li>
            ))}
          </ol>
        </div>
      ) : (
        <div className="mt-1.5 text-[10px] text-fg-dim">{view.loading ? 'Loading…' : isSpeed ? 'No completed links in this window yet.' : 'No reads in this window yet.'}</div>
      )}
    </div>
  )
}
