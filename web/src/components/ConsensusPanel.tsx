import type { EventSummary, PlateConsensus } from '../api/types'
import { fmtNum, fmtPct, fmtTime } from '../lib/format'
import { formatPlate, mismatchSlots, plateSlots } from '../lib/plate'
import { cx, EmptyState } from './ui'

const GAP_BEFORE = new Set([2, 4, 6])

/** Log-"nines" scale so 99% vs 99.99% is visible: 90% -> 1/4, 99% -> 2/4, 99.9% -> 3/4, 99.99% -> full. */
const nines = (p: number) => (Number.isFinite(p) ? Math.max(0.04, Math.min(1, -Math.log10(Math.max(1e-6, 1 - p)) / 4)) : 0)

function SlotRow({ chars, truth, variant }: { chars: string[]; truth: string[]; variant: 'read' | 'fused' }) {
  return (
    <>
      {chars.map((ch, i) => {
        const bad = variant === 'read' && ch !== truth[i]
        return (
          <div
            key={i}
            className={cx(
              'flex items-center justify-center rounded-[2px] font-mono',
              GAP_BEFORE.has(i) && 'ml-1.5',
              variant === 'read' ? 'h-6 text-[13px]' : 'h-8 text-lg font-bold',
              variant === 'read' && !bad && 'bg-ink-800 text-fg',
              variant === 'read' && bad && 'bg-caution font-bold text-ink-950 ring-1 ring-caution',
              variant === 'fused' && 'border border-accent-dim bg-accent-faint text-accent-strong',
              ch === '_' && !bad && 'text-ink-500',
            )}
            title={bad ? `Slot ${i + 1}: read "${ch === '_' ? 'blank' : ch}", consensus "${truth[i] === '_' ? 'blank' : truth[i]}"` : undefined}
          >
            {ch === '_' ? '·' : ch}
          </div>
        )
      })}
    </>
  )
}

export function ConsensusPanel({ consensus, decodedPlate, events }: { consensus: PlateConsensus; decodedPlate: string; events: EventSummary[] }) {
  const reads = consensus.single_read_plates ?? []
  if (!reads.length) return <EmptyState title="No constituent reads" />
  const truth = plateSlots(decodedPlate)
  const badReads = reads.filter((r) => mismatchSlots(r, decodedPlate).length > 0).length
  const perSlot = consensus.per_slot ?? []
  const fusedConf = perSlot.length === 10 ? perSlot.reduce((a, s) => a * (s[0]?.prob ?? 1), 1) : NaN
  const cols = 'grid-cols-[minmax(92px,120px)_repeat(10,minmax(18px,26px))_minmax(40px,1fr)]'

  return (
    <div className="flex flex-col gap-2 p-3">
      <p className="text-[13px] leading-snug text-fg-strong">
        {badReads === 0 ? (
          <>All {reads.length} reads agree on <span className="font-mono">{formatPlate(decodedPlate)}</span>.</>
        ) : (
          <>
            <span className="text-caution">{badReads} of {reads.length} single reads</span> contained OCR errors. Fusing them along the trajectory recovers{' '}
            <span className="font-mono font-semibold text-accent-strong">{formatPlate(decodedPlate)}</span>
            {Number.isFinite(fusedConf) && <> at {fmtPct(fusedConf, fusedConf > 0.99 ? 2 : 1)} confidence</>}.
          </>
        )}
      </p>

      <div className="overflow-x-auto">
        <div className={cx('grid min-w-[420px] items-center gap-x-[3px] gap-y-[3px]', cols)} role="table" aria-label="Single plate reads stacked above the fused consensus plate">
          {/* header */}
          <div className="text-[10px] font-semibold tracking-wider text-fg-dim uppercase" role="columnheader">Read</div>
          {truth.map((_, i) => (
            <div key={i} className={cx('text-center font-mono text-[9px] text-fg-dim', GAP_BEFORE.has(i) && 'ml-1.5')}>
              {i + 1}
            </div>
          ))}
          <div className="text-right text-[10px] font-semibold tracking-wider text-fg-dim uppercase">OCR conf</div>

          {reads.map((r, k) => {
            const ev = events[k]
            const nBad = mismatchSlots(r, decodedPlate).length
            return (
              <div key={k} className="contents" role="row">
                <div className="num truncate font-mono text-[11px] text-fg-muted" title={ev ? `${ev.camera_id} at ${ev.timestamp}` : undefined}>
                  {ev ? (
                    <>
                      <span className="text-fg-dim">{fmtTime(ev.timestamp, false)}</span> {ev.camera_id}
                    </>
                  ) : (
                    `read ${k + 1}`
                  )}
                </div>
                <SlotRow chars={plateSlots(r)} truth={truth} variant="read" />
                <div className={cx('num text-right font-mono text-[11px]', nBad ? 'text-caution' : 'text-fg-dim')}>
                  {ev ? fmtPct(ev.plate_confidence, 0) : nBad ? `${nBad} err` : '✓'}
                </div>
              </div>
            )
          })}

          {/* fuse divider */}
          <div className="col-span-full my-1 flex items-center gap-2">
            <div className="h-px flex-1 bg-ink-600" />
            <span className="text-[10px] font-semibold tracking-[0.1em] text-fg-dim uppercase">fused along trajectory</span>
            <div className="h-px flex-1 bg-ink-600" />
          </div>

          <div className="text-[11px] font-semibold text-accent-strong">Consensus</div>
          <SlotRow chars={truth} truth={truth} variant="fused" />
          <div className="num text-right font-mono text-[11px] text-accent-strong">{Number.isFinite(fusedConf) ? fmtPct(fusedConf, 1) : '—'}</div>

          {/* per-slot confidence */}
          <div className="self-end pb-3 text-[10px] leading-tight text-fg-dim">
            Slot confidence
            <br />
            <span className="text-ink-500">log scale</span>
          </div>
          {truth.map((ch, i) => {
            const slot = perSlot[i] ?? []
            const top = slot.find((s) => s.char === ch) ?? slot[0]
            const p = top?.prob ?? NaN
            const alt = slot.filter((s) => s.char !== ch).slice(0, 3)
            const title = [`Slot ${i + 1}: ${ch === '_' ? 'blank' : ch} ${fmtPct(p, 3)}`, ...alt.map((a) => `  ${a.char === '_' ? 'blank' : a.char} ${fmtPct(a.prob, 3)}`)].join('\n')
            const weak = Number.isFinite(p) && p < 0.9
            return (
              <div key={i} className={cx('flex flex-col items-center gap-0.5', GAP_BEFORE.has(i) && 'ml-1.5')} title={title}>
                <div className="relative h-12 w-2.5 overflow-hidden rounded-[2px] bg-ink-800" aria-hidden>
                  <div className={cx('absolute inset-x-0 bottom-0', weak ? 'bg-caution' : 'bg-accent')} style={{ height: `${nines(p) * 100}%` }} />
                </div>
                <span className={cx('num font-mono text-[9px]', weak ? 'text-caution' : 'text-fg-dim')}>
                  {Number.isFinite(p) ? (p >= 0.9995 ? '>99.9' : (p * 100).toFixed(p >= 0.99 ? 1 : 0)) : '—'}
                </span>
                <span className="sr-only">{title}</span>
              </div>
            )
          })}
          <div />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-fg-dim">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-caution" /> character disagrees with consensus
        </span>
        <span>Residual entropy {fmtNum(consensus.entropy_bits, 2)} bits</span>
        <span>Hover a slot for runner-up characters</span>
      </div>
    </div>
  )
}
