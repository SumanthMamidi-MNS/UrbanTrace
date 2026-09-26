import type { ReactNode } from 'react'
import { cx } from '../../components/ui'
import { pct } from './shared'

export function Description({ text, className }: { text: string | undefined; className?: string }) {
  return text ? <p className={cx('text-xs leading-relaxed text-fg-muted', className)}>{text}</p> : null
}

/** Long engine descriptions (paths, methodology) are useful but noisy: show on demand. */
export function Methodology({ text }: { text: string | undefined }) {
  if (!text) return null
  return (
    <details className="group px-3 pb-3">
      <summary className="cursor-pointer text-[11px] text-fg-dim select-none hover:text-fg-muted">Methodology (from the report)</summary>
      <p className="mt-1.5 text-[11px] leading-relaxed whitespace-pre-line text-fg-muted">{text}</p>
    </details>
  )
}

export function Tile({ label, value, sub, tone, big }: { label: string; value: string; sub?: ReactNode; tone?: 'accent' | 'muted'; big?: boolean }) {
  return (
    <div className="rounded-sm border border-ink-700 bg-ink-900/60 px-3 py-2">
      <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">{label}</div>
      <div className={cx('num font-mono', big ? 'text-[28px] leading-tight' : 'text-xl', tone === 'accent' ? 'text-accent-strong' : tone === 'muted' ? 'text-fg-muted' : 'text-fg-strong')}>{value}</div>
      {sub && <div className="text-[11px] text-fg-dim">{sub}</div>}
    </div>
  )
}

/** Horizontal proportion bar with the value printed beside it. */
export function PctBar({ v, tone = 'accent', target, digits = 1 }: { v: number | undefined; tone?: 'accent' | 'neutral'; target?: number; digits?: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-2 flex-1 rounded-[2px] bg-ink-700">
        {v !== undefined && <div className={cx('absolute inset-y-0 left-0 rounded-[2px]', tone === 'accent' ? 'bg-accent' : 'bg-fg-dim')} style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }} />}
        {target !== undefined && <div className="absolute -inset-y-1 w-px bg-fg" style={{ left: `${target * 100}%` }} title={`target ${pct(target, 0)}`} />}
      </div>
      <span className="num w-12 text-right font-mono text-fg">{pct(v, digits)}</span>
    </div>
  )
}

export function Missing({ what }: { what: string }) {
  return <p className="px-3 py-4 text-xs text-fg-dim">{what} not produced yet.</p>
}
