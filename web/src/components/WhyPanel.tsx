import type { EventSummary, LinkEvidence } from '../api/types'
import { fmtDuration, fmtLogOdds, fmtPct, fmtTime, sigmoid } from '../lib/format'
import { mismatchSlots, plateSlots } from '../lib/plate'
import { cx, EmptyState } from './ui'

type Channel = 'prior' | 'plate' | 'appearance' | 'kinematic'

const LABEL: Record<Channel, string> = {
  prior: 'Prior (base rate)',
  plate: 'Plate',
  appearance: 'Appearance',
  kinematic: 'Travel time',
}

type Strength = 'strong' | 'moderate' | 'weak' | 'against' | 'nonfinite'

/** Per-channel bands: plate LRs naturally span a wider range than appearance or kinematics. */
const BANDS: Record<Exclude<Channel, 'prior'>, [number, number, number]> = {
  plate: [6, 2, -2],
  appearance: [3, 1, -1],
  kinematic: [2, 0.5, -1],
}

function strength(ch: Exclude<Channel, 'prior'>, v: number | null): Strength {
  if (v === null || !Number.isFinite(v)) return 'nonfinite'
  const [s, m, a] = BANDS[ch]
  if (v >= s) return 'strong'
  if (v >= m) return 'moderate'
  if (v >= a) return 'weak'
  return 'against'
}

const fin = (v: number | null | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v)

function joinNames(xs: string[]): string {
  if (xs.length <= 1) return xs[0] ?? ''
  return `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`
}

/** One plain-English sentence a judge can read in two seconds. */
export function verdict(link: LinkEvidence, fromEv?: EventSummary, toEv?: EventSummary, decoded?: string): string {
  const ch = {
    plate: strength('plate', link.plate_lr),
    appearance: strength('appearance', link.appearance_lr),
    kinematic: strength('kinematic', link.kinematic_lr),
  }
  const name = { plate: 'plate evidence', appearance: 'appearance', kinematic: 'travel time' }
  const keys = Object.keys(ch) as (keyof typeof ch)[]
  const strong = keys.filter((k) => ch[k] === 'strong')
  const weak = keys.filter((k) => ch[k] === 'weak' || ch[k] === 'against')
  const nonfinite = keys.filter((k) => ch[k] === 'nonfinite')

  let misreadHint = ''
  if (decoded && fromEv && toEv && weak.includes('plate')) {
    const bad = [fromEv, toEv].filter((e) => mismatchSlots(e.plate_argmax, decoded).length > 0)
    if (bad.length) misreadHint = ` (misread at ${bad.map((e) => e.camera_id).join(' and ')})`
  }

  const parts: string[] = []
  if (nonfinite.length) {
    parts.push(`${cap(joinNames(nonfinite.map((k) => name[k])))} returned a non-finite likelihood, so this link cannot be scored as a finite sum.`)
  }
  const cleanWeak = weak.filter((k) => !nonfinite.includes(k))
  if (strong.length === 3) {
    parts.push('All three channels strongly agree: plate, appearance and travel time each support the same vehicle.')
  } else if (cleanWeak.length && strong.length) {
    const w = cleanWeak.map((k) => `${name[k]}${k === 'plate' ? misreadHint : ''}`)
    const verb = cleanWeak.some((k) => ch[k] === 'against') ? 'argued against the link' : cleanWeak.length > 1 ? 'were weak' : 'was weak'
    parts.push(`${cap(joinNames(w))} ${verb}, but ${joinNames(strong.map((k) => name[k]))} ${strong.length > 1 ? 'were' : 'was'} strong.`)
  } else if (strong.length) {
    parts.push(`${cap(joinNames(strong.map((k) => name[k])))} ${strong.length > 1 ? 'carry' : 'carries'} the link; the other channels are supportive but moderate.`)
  } else if (!nonfinite.length) {
    parts.push('No single channel is decisive; the link rests on the combined evidence.')
  }
  if (fin(link.total_log_odds)) {
    parts.push(`Net log-odds ${fmtLogOdds(link.total_log_odds, 1)} → P(same vehicle) ${fmtProb(sigmoid(link.total_log_odds))}.`)
  }
  return parts.join(' ')
}

const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s)

export function fmtProb(p: number): string {
  if (!Number.isFinite(p)) return '—'
  if (p > 0.9999) return '>99.99%'
  if (p < 0.0001) return '<0.01%'
  return fmtPct(p, p > 0.99 || p < 0.01 ? 2 : 1)
}

// ------------------------------------------------------------------ link list

function MiniCell({ v, ch }: { v: number | null; ch: Exclude<Channel, 'prior'> }) {
  const s = strength(ch, v)
  return (
    <td className="px-1.5 py-1 text-right">
      <span
        className={cx(
          'num inline-block min-w-[46px] rounded-sm px-1 font-mono text-[11px]',
          s === 'strong' && 'bg-accent-faint text-accent-strong',
          s === 'moderate' && 'text-fg',
          s === 'weak' && 'bg-caution-faint text-caution',
          s === 'against' && 'bg-caution-faint font-semibold text-caution',
          s === 'nonfinite' && 'hatch text-fg-muted',
        )}
      >
        {fin(v) ? fmtLogOdds(v, 1) : '±∞'}
      </span>
    </td>
  )
}

// ------------------------------------------------------------------ main

type Props = {
  links: LinkEvidence[]
  events: EventSummary[]
  decodedPlate: string
  selected: number
  onSelect: (i: number) => void
  cameraName?: (id: string) => string
}

export function WhyPanel({ links, events, decodedPlate, selected, onSelect, cameraName }: Props) {
  if (!links.length) {
    return <EmptyState title="No links">A single-sighting trajectory has nothing to explain yet.</EmptyState>
  }
  const byId = new Map(events.map((e) => [e.event_id, e]))
  const idx = Math.min(Math.max(0, selected), links.length - 1)
  const link = links[idx]
  const from = byId.get(link.from_event_id)
  const to = byId.get(link.to_event_id)

  return (
    <div className="flex flex-col gap-3 p-3">
      {/* link overview: every link, every channel, clickable */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <caption className="sr-only">Links in this trajectory and their evidence channels. Select a row to explain it.</caption>
          <thead>
            <tr className="text-[10px] tracking-wider text-fg-dim uppercase">
              <th className="px-1.5 py-1 text-left font-semibold">Link</th>
              <th className="px-1.5 py-1 text-right font-semibold">Δt</th>
              <th className="px-1.5 py-1 text-right font-semibold">Plate</th>
              <th className="px-1.5 py-1 text-right font-semibold">Appear.</th>
              <th className="px-1.5 py-1 text-right font-semibold">Travel</th>
              <th className="px-1.5 py-1 text-right font-semibold">Total</th>
            </tr>
          </thead>
          <tbody>
            {links.map((l, i) => {
              const a = byId.get(l.from_event_id)
              const b = byId.get(l.to_event_id)
              const active = i === idx
              return (
                <tr
                  key={`${l.from_event_id}-${l.to_event_id}`}
                  tabIndex={0}
                  aria-selected={active}
                  onClick={() => onSelect(i)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onSelect(i)
                    }
                    if (e.key === 'ArrowDown' && i < links.length - 1) onSelect(i + 1)
                    if (e.key === 'ArrowUp' && i > 0) onSelect(i - 1)
                  }}
                  className={cx('cursor-pointer border-t border-ink-750 outline-none', active ? 'bg-ink-750 shadow-[inset_2px_0_0_var(--color-accent)]' : 'hover:bg-ink-800 focus-visible:bg-ink-800')}
                >
                  <td className="px-1.5 py-1 whitespace-nowrap">
                    <span className="num mr-1.5 text-fg-dim">{i + 1}</span>
                    <span className="font-mono text-[11px] text-fg">{a?.camera_id ?? '?'}</span>
                    <span className="px-1 text-fg-dim">→</span>
                    <span className="font-mono text-[11px] text-fg">{b?.camera_id ?? '?'}</span>
                    {l.skipped_cameras.length > 0 && <span className="ml-1.5 text-[10px] text-fg-dim">+{l.skipped_cameras.length} missed</span>}
                  </td>
                  <td className="num px-1.5 py-1 text-right font-mono text-[11px] text-fg-muted">{fmtDuration(l.delta_t_s)}</td>
                  <MiniCell v={l.plate_lr} ch="plate" />
                  <MiniCell v={l.appearance_lr} ch="appearance" />
                  <MiniCell v={l.kinematic_lr} ch="kinematic" />
                  <td className="px-1.5 py-1 text-right">
                    <span className={cx('num font-mono text-[11px] font-semibold', fin(l.total_log_odds) ? (l.total_log_odds >= 0 ? 'text-fg-strong' : 'text-caution') : 'text-fg-muted')}>
                      {fin(l.total_log_odds) ? fmtLogOdds(l.total_log_odds, 1) : '±∞'}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <LinkExplanation link={link} index={idx} from={from} to={to} decodedPlate={decodedPlate} cameraName={cameraName} />
    </div>
  )
}

// ------------------------------------------------------------------ detail

function LinkExplanation({ link, index, from, to, decodedPlate, cameraName }: { link: LinkEvidence; index: number; from?: EventSummary; to?: EventSummary; decodedPlate: string; cameraName?: (id: string) => string }) {
  const rows: { ch: Channel; v: number | null }[] = [
    { ch: 'prior', v: link.prior_log_odds },
    { ch: 'plate', v: link.plate_lr },
    { ch: 'appearance', v: link.appearance_lr },
    { ch: 'kinematic', v: link.kinematic_lr },
  ]
  // Waterfall: each bar spans running-sum-before -> running-sum-after. Non-finite channels are skipped in the running sum.
  let run = 0
  const segs = rows.map((r) => {
    const start = run
    if (fin(r.v)) run += r.v
    return { ...r, start, end: run }
  })
  const channelSum = run
  const total = link.total_log_odds
  const extent = Math.max(8, ...segs.flatMap((s) => [Math.abs(s.start), Math.abs(s.end)]), fin(total) ? Math.abs(total) : 0)
  const scale = Math.ceil(extent / 4) * 4
  const pos = (x: number) => 50 + (x / scale) * 50
  const mismatch = fin(total) && Math.abs(total - channelSum) > 0.05 && segs.every((s) => fin(s.v))

  const ratio = link.expected_t_s > 0 ? link.delta_t_s / link.expected_t_s : NaN
  const tMax = Math.max(link.delta_t_s, link.expected_t_s) * 1.1 || 1

  return (
    <div className="rounded-sm border border-ink-700 bg-ink-900/60">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 border-b border-ink-700 px-3 py-2">
        <div className="text-[11px] font-semibold tracking-[0.08em] text-fg-muted uppercase">
          Why link {index + 1} was joined
        </div>
        <div className="num font-mono text-[11px] text-fg-dim">
          {from ? `${from.camera_id} ${fmtTime(from.timestamp)}` : link.from_event_id} → {to ? `${to.camera_id} ${fmtTime(to.timestamp)}` : link.to_event_id}
        </div>
      </div>

      <p className="px-3 pt-2.5 text-[13px] leading-snug text-fg-strong" aria-live="polite">
        {verdict(link, from, to, decodedPlate)}
      </p>

      {/* waterfall of log-odds contributions */}
      <div className="px-3 pt-3 pb-2" role="img" aria-label={`Evidence breakdown: ${rows.map((r) => `${LABEL[r.ch]} ${fmtLogOdds(r.v, 2)}`).join(', ')}, total ${fmtLogOdds(total, 2)}`}>
        <div className="mb-1 grid grid-cols-[112px_1fr_70px] items-end gap-2 text-[10px] text-fg-dim">
          <span />
          <div className="relative h-3">
            <span className="absolute left-0">← different vehicles</span>
            <span className="absolute right-0">same vehicle →</span>
          </div>
          <span className="text-right">log-odds</span>
        </div>
        {segs.map((s) => {
          const nonfinite = !fin(s.v)
          const left = Math.min(pos(s.start), pos(s.end))
          const width = Math.abs(pos(s.end) - pos(s.start))
          const positive = fin(s.v) && s.v >= 0
          return (
            <div key={s.ch} className="grid grid-cols-[112px_1fr_70px] items-center gap-2 py-[3px]">
              <span className="truncate text-xs text-fg">{LABEL[s.ch]}</span>
              <div className="relative h-4 rounded-[2px] bg-ink-800">
                <div className="absolute inset-y-[-2px] left-1/2 w-px bg-ink-500" />
                {nonfinite ? (
                  <div className="hatch absolute inset-y-0.5 right-0 left-0 flex items-center justify-center rounded-[2px]">
                    <span className="rounded-sm bg-ink-900 px-1 text-[10px] text-fg-muted">non-finite — excluded from sum</span>
                  </div>
                ) : (
                  <div
                    className={cx('absolute inset-y-0.5 rounded-[2px]', s.ch === 'prior' ? 'bg-fg-dim' : positive ? 'bg-accent' : 'bg-caution')}
                    style={{ left: `${left}%`, width: `max(${width}%, 2px)` }}
                  />
                )}
              </div>
              <span className={cx('num text-right font-mono text-xs', nonfinite ? 'text-fg-muted' : s.ch === 'prior' ? 'text-fg-muted' : positive ? 'text-fg-strong' : 'text-caution')}>
                {fin(s.v) ? fmtLogOdds(s.v) : '±∞'}
              </span>
            </div>
          )
        })}
        <div className="mt-1 grid grid-cols-[112px_1fr_70px] items-center gap-2 border-t border-ink-700 pt-1.5">
          <span className="text-xs font-semibold text-fg-strong">Total</span>
          <div className="relative h-5 rounded-[2px] bg-ink-800">
            <div className="absolute inset-y-[-2px] left-1/2 w-px bg-ink-500" />
            {fin(total) ? (
              <div
                className={cx('absolute inset-y-0.5 rounded-[2px]', total >= 0 ? 'bg-accent-strong' : 'bg-caution')}
                style={{ left: `${Math.min(50, pos(total))}%`, width: `max(${Math.abs(pos(total) - 50)}%, 2px)` }}
              />
            ) : (
              <div className="hatch absolute inset-y-0.5 right-0 left-0 flex items-center justify-center rounded-[2px]">
                <span className="rounded-sm bg-ink-900 px-1 text-[10px] text-fg-muted">non-finite total (±∞ / NaN)</span>
              </div>
            )}
          </div>
          <span className={cx('num text-right font-mono text-xs font-semibold', fin(total) ? 'text-fg-strong' : 'text-fg-muted')}>{fin(total) ? fmtLogOdds(total) : '±∞'}</span>
        </div>
        <div className="mt-1 grid grid-cols-[112px_1fr_70px] gap-2 text-[10px] text-fg-dim">
          <span />
          <div className="num flex justify-between font-mono">
            <span>−{scale}</span>
            <span>0</span>
            <span>+{scale}</span>
          </div>
          <span />
        </div>
        {mismatch && (
          <p className="mt-1 text-[11px] text-fg-dim">
            Note: reported total differs from the channel sum ({fmtLogOdds(channelSum)}) by {Math.abs((total as number) - channelSum).toFixed(2)}.
          </p>
        )}
      </div>

      {/* supporting facts */}
      <div className="grid grid-cols-1 gap-3 border-t border-ink-700 px-3 py-2.5 sm:grid-cols-2">
        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Travel time</span>
            <span className={cx('num font-mono text-[11px]', Number.isFinite(ratio) && (ratio > 1.6 || ratio < 0.6) ? 'text-caution' : 'text-fg-muted')}>
              {Number.isFinite(ratio) ? `${ratio.toFixed(2)}× expected` : '—'}
            </span>
          </div>
          <TimeBar label="Observed" value={link.delta_t_s} max={tMax} tone="fill" />
          <TimeBar label="Expected" value={link.expected_t_s} max={tMax} tone="outline" />
          {link.skipped_cameras.length > 0 && (
            <p className="mt-1.5 text-[11px] text-fg-muted">
              Passed {link.skipped_cameras.length} camera{link.skipped_cameras.length > 1 ? 's' : ''} without a read:{' '}
              <span className="font-mono text-fg">{link.skipped_cameras.map((c) => (cameraName ? `${c}` : c)).join(', ')}</span>
            </p>
          )}
        </div>
        <div>
          <div className="mb-1 text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Plate reads on this link</div>
          {[from, to].map((e, i) =>
            e ? (
              <div key={e.event_id} className="flex items-center gap-2 py-0.5">
                <span className="w-9 text-[10px] text-fg-dim">{i === 0 ? 'From' : 'To'}</span>
                <ReadChars read={e.plate_argmax} truth={decodedPlate} />
                <span className="num ml-auto font-mono text-[10px] text-fg-dim">{fmtPct(e.plate_confidence, 0)}</span>
              </div>
            ) : null,
          )}
          <div className="flex items-center gap-2 py-0.5">
            <span className="w-9 text-[10px] text-fg-dim">Fused</span>
            <ReadChars read={decodedPlate} truth={decodedPlate} fused />
          </div>
        </div>
      </div>
    </div>
  )
}

function TimeBar({ label, value, max, tone }: { label: string; value: number; max: number; tone: 'fill' | 'outline' }) {
  const w = Number.isFinite(value) && max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0
  return (
    <div className="grid grid-cols-[58px_1fr_56px] items-center gap-2 py-[2px]">
      <span className="text-[11px] text-fg-muted">{label}</span>
      <div className="relative h-2.5 rounded-[2px] bg-ink-800">
        <div className={cx('absolute inset-y-0 left-0 rounded-[2px]', tone === 'fill' ? 'bg-accent' : 'border border-fg-muted')} style={{ width: `${w}%` }} />
      </div>
      <span className="num text-right font-mono text-[11px] text-fg">{fmtDuration(value)}</span>
    </div>
  )
}

/** Plate characters with slots that disagree with the decoded plate highlighted. */
export function ReadChars({ read, truth, fused }: { read: string; truth: string; fused?: boolean }) {
  const r = plateSlots(read)
  const t = plateSlots(truth)
  return (
    <span className="inline-flex font-mono text-[12px]">
      {r.map((ch, i) => {
        const bad = ch !== t[i]
        const gap = i === 2 || i === 4 || i === 6
        return (
          <span
            key={i}
            className={cx(
              'inline-flex w-[13px] justify-center',
              gap && 'ml-1',
              ch === '_' && !bad && 'text-ink-500',
              bad && 'rounded-[2px] bg-caution font-bold text-ink-950',
              !bad && ch !== '_' && (fused ? 'font-semibold text-accent-strong' : 'text-fg'),
            )}
            title={bad ? `slot ${i + 1}: read "${ch === '_' ? 'blank' : ch}", consensus "${t[i] === '_' ? 'blank' : t[i]}"` : undefined}
          >
            {ch === '_' ? '·' : ch}
          </span>
        )
      })}
    </span>
  )
}
