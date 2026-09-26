import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useWatchlist, useWatchlistHits } from '../api/hooks'
import type { Alert, TrajectoryDetail, WatchlistHit, WatchlistMatchedOn } from '../api/types'
import { fmtDateTime, fmtPct } from '../lib/format'
import { BLANK, closestPattern, expandPlatePattern, plateSlots } from '../lib/plate'
import { cx, Plate } from './ui'

const GAP = (i: number) => i === 2 || i === 4 || i === 6

/** Watchlist match threshold from contract v2 ("P(plate ∈ pattern | evidence) ≥ 0.5"). */
export const MATCH_THRESHOLD = 0.5

const matchedOnLabel = (m: WatchlistMatchedOn | undefined) =>
  m === 'trajectory_consensus' ? 'Trajectory consensus' : m === 'single_read' ? 'Single read' : '—'

export function MatchedOnTag({ on }: { on: WatchlistMatchedOn | undefined }) {
  if (!on) return null
  const cons = on === 'trajectory_consensus'
  return (
    <span
      className={cx('inline-flex h-[18px] items-center rounded-sm border px-1.5 text-[10px] font-semibold whitespace-nowrap', cons ? 'border-accent-dim bg-accent-faint text-accent-strong' : 'border-ink-500 text-fg-muted')}
      title={cons ? 'Matched on the fused plate across the vehicle’s cameras' : 'Matched on this camera’s own read'}
    >
      {cons ? 'Consensus' : 'Single read'}
    </span>
  )
}

/** Slot is a literal (non-wildcard) position of the pattern that the plate gets wrong. */
const slotBad = (plate: string[], pattern: string[], i: number) => pattern[i] !== '?' && plate[i] !== pattern[i]

/** A plate's characters coloured against a canonical pattern: "?" slots never count as wrong. */
export function PatternReadChars({ read, pattern, size = 'sm' }: { read: string; pattern?: string; size?: 'sm' | 'md' }) {
  const r = plateSlots(read)
  const p = pattern ? plateSlots(pattern) : r
  return (
    <span className={cx('inline-flex font-mono', size === 'md' ? 'text-[15px]' : 'text-[12px]')}>
      {r.map((ch, i) => {
        const bad = pattern ? slotBad(r, p, i) : false
        return (
          <span
            key={i}
            className={cx(size === 'md' ? 'inline-flex w-[15px] justify-center' : 'inline-flex w-[13px] justify-center', GAP(i) && 'ml-1', ch === BLANK && !bad && 'text-ink-500', bad && 'rounded-[2px] bg-caution font-bold text-ink-950', !bad && ch !== BLANK && 'text-fg')}
            title={bad ? `slot ${i + 1}: read "${ch === BLANK ? 'blank' : ch}", watchlist expects "${p[i] === BLANK ? 'blank' : p[i]}"` : undefined}
          >
            {ch === BLANK ? '·' : ch}
          </span>
        )
      })}
    </span>
  )
}

/** Pattern, camera read and fused plate aligned slot by slot. */
export function SlotCompare({ pattern, rows }: { pattern: string; rows: { label: string; plate: string | undefined; note?: string }[] }) {
  const p = plateSlots(pattern)
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-[3px] text-xs">
        <caption className="sr-only">Watchlist pattern compared slot by slot with what the camera read and the fused plate</caption>
        <tbody>
          <tr>
            <th scope="row" className="pr-3 text-left text-[11px] font-medium whitespace-nowrap text-fg-muted">
              Watchlist pattern
            </th>
            {p.map((ch, i) => (
              <td key={i} className={cx('h-7 w-6 rounded-[2px] border text-center font-mono text-[14px] font-semibold', GAP(i) && 'pl-0', ch === '?' ? 'border-dashed border-ink-500 text-fg-dim' : 'border-ink-600 bg-ink-800 text-fg-strong')} style={GAP(i) ? { borderLeftWidth: 1 } : undefined}>
                {ch === BLANK ? '·' : ch}
              </td>
            ))}
            <td className="pl-2 text-[11px] text-fg-dim">? = any</td>
          </tr>
          {rows.map((row) => {
            if (!row.plate) return null
            const r = plateSlots(row.plate)
            const wrong = r.filter((_, i) => slotBad(r, p, i)).length
            return (
              <tr key={row.label}>
                <th scope="row" className="pr-3 text-left text-[11px] font-medium whitespace-nowrap text-fg-muted">
                  {row.label}
                </th>
                {r.map((ch, i) => {
                  const bad = slotBad(r, p, i)
                  return (
                    <td key={i} className={cx('h-7 w-6 rounded-[2px] text-center font-mono text-[14px]', bad ? 'bg-caution font-bold text-ink-950' : ch === BLANK ? 'bg-ink-850 text-ink-500' : 'bg-ink-850 text-fg')}>
                      {ch === BLANK ? '·' : ch}
                    </td>
                  )
                })}
                <td className={cx('pl-2 text-[11px] whitespace-nowrap', wrong ? 'text-caution' : 'text-accent-strong')}>
                  {wrong ? `${wrong} slot${wrong > 1 ? 's differ' : ' differs'}` : 'matches'}
                  {row.note && <span className="text-fg-dim"> · {row.note}</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** Probability bar with the 50% decision threshold marked. */
export function ProbabilityMeter({ p }: { p: number | undefined }) {
  const v = p !== undefined && Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : undefined
  return (
    <div>
      <div className="num font-mono text-2xl text-fg-strong">{v === undefined ? '—' : fmtPct(v, 1)}</div>
      <div className="relative mt-1 h-2 rounded-full bg-ink-700" role="img" aria-label={`Match probability ${v === undefined ? 'unknown' : fmtPct(v, 1)}, threshold 50%`}>
        {v !== undefined && <div className="absolute inset-y-0 left-0 rounded-full bg-alert" style={{ width: `${v * 100}%` }} />}
        <div className="absolute -inset-y-1 w-px bg-fg" style={{ left: `${MATCH_THRESHOLD * 100}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-fg-dim">
        <span>0</span>
        <span>alert threshold 50%</span>
        <span>100%</span>
      </div>
    </div>
  )
}

function pickHit(hits: WatchlistHit[] | undefined, alert: Alert, triggerEventId: string | undefined): WatchlistHit | undefined {
  if (!hits?.length) return undefined
  const byEvent = triggerEventId ? hits.find((h) => h.event_id === triggerEventId) : undefined
  if (byEvent) return byEvent
  const created = Date.parse(alert.created_at)
  const pool = hits.filter((h) => (alert.trajectory_ids.length ? h.trajectory_id !== null && alert.trajectory_ids.includes(h.trajectory_id) : true) && (!alert.evidence.matched_on || h.matched_on === alert.evidence.matched_on))
  return pool.sort((a, b) => Math.abs(Date.parse(a.timestamp) - created) - Math.abs(Date.parse(b.timestamp) - created))[0]
}

/**
 * The differentiator, stated plainly: what this camera read, what the fused plate is, and that the
 * fused plate (not the read) is what matched the watchlist.
 */
export function WatchlistEvidence({ alert, traj, cameraName }: { alert: Alert; traj?: TrajectoryDetail; cameraName?: (id: string) => string | undefined }) {
  const ev = alert.evidence
  const entries = useWatchlist()
  const entry = entries.data?.find((e) => e.entry_id === ev.watchlist_entry_id)
  const hits = useWatchlistHits({ entry_id: ev.watchlist_entry_id, limit: 500 }, Boolean(ev.watchlist_entry_id), 15_000)
  const points = ev.points ?? []
  const trigger = points[points.length - 1]
  const hit = pickHit(hits.data, alert, trigger?.event_id)
  const consensus = ev.matched_on === 'trajectory_consensus'

  const triggerEventId = hit?.event_id ?? trigger?.event_id
  const plateRead = hit?.plate_read ?? traj?.events.find((e) => e.event_id === triggerEventId)?.plate_argmax ?? (consensus ? undefined : alert.plate)
  const cameraId = hit?.camera_id ?? trigger?.camera_id
  const at = hit?.timestamp ?? trigger?.timestamp ?? alert.created_at
  const fused = consensus ? (traj?.decoded_plate ?? alert.plate) : traj?.decoded_plate
  const nCams = useMemo(() => {
    if (!traj) return points.length || undefined
    const t = Date.parse(at)
    return traj.events.filter((e) => Date.parse(e.timestamp) <= t).length
  }, [traj, at, points.length])
  const pattern = ev.pattern ?? entry?.pattern ?? '?'
  const forms = entry?.canonical_patterns ?? (() => {
    const x = expandPlatePattern(pattern)
    return x.ok ? x.forms : []
  })()
  const canonical = closestPattern(fused ?? plateRead ?? alert.plate, forms)
  const prob = ev.match_probability
  const pct = prob !== undefined ? fmtPct(prob, 0) : 'above the threshold'
  const readWrong = plateRead && canonical ? plateSlots(plateRead).some((ch, i) => canonical[i] !== '?' && ch !== canonical[i]) : false

  return (
    <div className="flex flex-col gap-3 p-3">
      <div role="status" className="rounded-sm border border-alert bg-alert-faint px-3 py-2.5">
        <p className="flex flex-wrap items-center gap-x-1.5 gap-y-1.5 text-[14px] leading-relaxed text-fg-strong">
          {consensus ? (
            <>
              <span>
                This camera{cameraId && <> (<span className="font-mono">{cameraId}</span>)</>} read
              </span>
              {plateRead ? <Plate value={plateRead} size="md" tone="muted" /> : <span className="text-fg-muted">(read unavailable)</span>}
              <span>{readWrong ? ', but' : ', and'} the vehicle’s fused plate across</span>
              <b className="num">{nCams ?? 'several'} cameras</b>
              {fused && <Plate value={fused} size="md" />}
              <span>matches your watchlist entry</span>
              <Plate value={pattern} size="md" tone="alert" />
              <span>
                at <b className="num font-mono">{pct}</b>.
              </span>
            </>
          ) : (
            <>
              <span>{cameraId ? <>Camera <span className="font-mono">{cameraId}</span> read</> : 'The camera read'}</span>
              {plateRead && <Plate value={plateRead} size="md" />}
              <span>which on its own matches your watchlist entry</span>
              <Plate value={pattern} size="md" tone="alert" />
              <span>
                at <b className="num font-mono">{pct}</b>.
              </span>
            </>
          )}
        </p>
        {consensus && readWrong && (
          <p className="mt-1 text-xs text-fg-muted">An exact-match watchlist would have missed this read. UrbanTrace caught it because the plate evidence from every camera on the trajectory is fused before matching.</p>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,1fr)_220px]">
        <div className="min-w-0">
          <div className="mb-1.5 text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Slot by slot</div>
          {canonical ? (
            <SlotCompare
              pattern={canonical}
              rows={[
                { label: 'This camera read', plate: plateRead },
                ...(fused ? [{ label: `Fused plate${nCams ? ` (${nCams} cams)` : ''}`, plate: fused }] : []),
              ]}
            />
          ) : (
            <p className="text-xs text-fg-dim">Pattern details unavailable.</p>
          )}
        </div>
        <div>
          <div className="mb-1.5 text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Match probability</div>
          <ProbabilityMeter p={prob} />
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 border-t border-ink-700 pt-3 text-xs md:grid-cols-[auto_auto_minmax(0,1.4fr)_minmax(0,1.4fr)_auto]">
        <div>
          <dt className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Matched on</dt>
          <dd className="mt-0.5 text-fg">{matchedOnLabel(ev.matched_on)}</dd>
        </div>
        <div>
          <dt className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Entry</dt>
          <dd className="mt-0.5 text-fg">
            <Link to="/watchlist" className="font-mono text-accent hover:underline">
              {ev.watchlist_entry_id ?? '—'}
            </Link>
            {!entry && ev.watchlist_entry_id && entries.data ? <span className="text-fg-dim"> (removed)</span> : null}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Reason</dt>
          <dd className="mt-0.5 text-fg">{entry?.reason || '—'}</dd>
        </div>
        <div className="min-w-0">
          <dt className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Where</dt>
          <dd className="mt-0.5 truncate text-fg">
            <span className="font-mono">{cameraId ?? '—'}</span> <span className="text-fg-muted">{cameraId ? cameraName?.(cameraId) : ''}</span>
          </dd>
        </div>
        <div>
          <dt className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">When</dt>
          <dd className="num mt-0.5 font-mono whitespace-nowrap text-fg">{fmtDateTime(at)} UTC</dd>
        </div>
      </dl>
    </div>
  )
}

