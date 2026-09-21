import type { ButtonHTMLAttributes, ReactNode } from 'react'
import type { AlertSeverity, AlertType } from '../api/types'
import { ApiError } from '../api/client'
import { formatPlate } from '../lib/plate'

export function cx(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(' ')
}

export function Panel({ title, actions, children, className, bodyClassName }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClassName?: string }) {
  return (
    <section className={cx('flex min-h-0 flex-col rounded border border-ink-700 bg-ink-850', className)}>
      {(title || actions) && (
        <header className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-ink-700 px-3">
          <h2 className="truncate text-[11px] font-semibold tracking-[0.08em] text-fg-muted uppercase">{title}</h2>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cx('min-h-0 flex-1', bodyClassName)}>{children}</div>
    </section>
  )
}

/** Monospace plate chip in spaced Indian form. */
export function Plate({ value, size = 'md', tone = 'default' }: { value: string; size?: 'sm' | 'md' | 'lg'; tone?: 'default' | 'alert' | 'muted' }) {
  return (
    <span
      className={cx(
        'inline-flex items-center rounded-sm border font-mono font-semibold tracking-wider whitespace-nowrap',
        size === 'sm' && 'px-1.5 py-px text-[11px]',
        size === 'md' && 'px-2 py-0.5 text-[13px]',
        size === 'lg' && 'px-3 py-1 text-xl',
        tone === 'default' && 'border-ink-500 bg-ink-750 text-fg-strong',
        tone === 'alert' && 'border-alert-dim bg-alert-faint text-alert',
        tone === 'muted' && 'border-ink-600 bg-ink-800 text-fg-muted',
      )}
    >
      {formatPlate(value)}
    </span>
  )
}

export function Button({ variant = 'default', className, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'default' | 'primary' | 'ghost' }) {
  return (
    <button
      type="button"
      {...rest}
      className={cx(
        'inline-flex h-7 items-center justify-center gap-1.5 rounded-sm px-2.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        variant === 'default' && 'border border-ink-600 bg-ink-800 text-fg hover:border-ink-500 hover:bg-ink-750',
        variant === 'primary' && 'border border-accent-dim bg-accent-faint text-accent-strong hover:border-accent',
        variant === 'ghost' && 'text-fg-muted hover:bg-ink-750 hover:text-fg',
        className,
      )}
    />
  )
}

export function Segmented<T extends string | number>({ options, value, onChange, label }: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex h-7 overflow-hidden rounded-sm border border-ink-600">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={cx(
            'num px-2 text-xs font-medium transition-colors not-first:border-l not-first:border-ink-600',
            o.value === value ? 'bg-accent-faint text-accent-strong' : 'bg-ink-800 text-fg-muted hover:text-fg',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Field({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <label className={cx('flex flex-col gap-1', className)}>
      <span className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">{label}</span>
      {children}
    </label>
  )
}

export const inputCls =
  'h-8 w-full rounded-sm border border-ink-600 bg-ink-900 px-2 text-[13px] text-fg placeholder:text-fg-dim hover:border-ink-500 focus:border-accent-dim focus:outline-none'

export function EmptyState({ title, children, className }: { title: string; children?: ReactNode; className?: string }) {
  return (
    <div className={cx('flex h-full flex-col items-center justify-center gap-1 p-6 text-center', className)}>
      <p className="text-[13px] font-medium text-fg-muted">{title}</p>
      {children && <div className="max-w-sm text-xs text-fg-dim">{children}</div>}
    </div>
  )
}

export function ErrorState({ error, className }: { error: unknown; className?: string }) {
  const msg = error instanceof ApiError ? `${error.status ? `HTTP ${error.status}: ` : ''}${error.message}` : error instanceof Error ? error.message : 'Unknown error'
  return (
    <div role="alert" className={cx('flex h-full flex-col items-center justify-center gap-1 p-6 text-center', className)}>
      <p className="text-[13px] font-medium text-fg">Could not load data</p>
      <p className="max-w-md font-mono text-xs text-fg-dim">{msg}</p>
    </div>
  )
}

export function Loading({ label = 'Loading', className }: { label?: string; className?: string }) {
  return (
    <div className={cx('flex h-full items-center justify-center gap-2 p-6 text-xs text-fg-dim', className)} role="status">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
      {label}…
    </div>
  )
}

/** Thin horizontal probability/proportion bar. */
export function Bar({ value, tone = 'accent', className }: { value: number; tone?: 'accent' | 'caution' | 'alert' | 'muted'; className?: string }) {
  const v = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0
  return (
    <div className={cx('h-1.5 w-full overflow-hidden rounded-full bg-ink-700', className)}>
      <div
        className={cx('h-full rounded-full', tone === 'accent' && 'bg-accent', tone === 'caution' && 'bg-caution', tone === 'alert' && 'bg-alert', tone === 'muted' && 'bg-fg-dim')}
        style={{ width: `${v * 100}%` }}
      />
    </div>
  )
}

const TYPE_LABEL: Record<AlertType, string> = { clone: 'Cloned plate', impossible_travel: 'Impossible travel', anomaly: 'Anomaly' }
export const alertTypeLabel = (t: string) => TYPE_LABEL[t as AlertType] ?? t

export function SeverityBadge({ severity }: { severity: AlertSeverity | string }) {
  return (
    <span
      className={cx(
        'inline-flex h-[18px] items-center rounded-sm px-1.5 text-[10px] font-bold tracking-wider uppercase',
        severity === 'high' && 'bg-alert text-ink-950',
        severity === 'medium' && 'border border-alert-dim text-alert',
        severity !== 'high' && severity !== 'medium' && 'border border-ink-500 text-fg-muted',
      )}
    >
      {severity}
    </span>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">{label}</div>
      <div className="num truncate text-[13px] text-fg-strong">{value}</div>
      {sub && <div className="truncate text-[11px] text-fg-dim">{sub}</div>}
    </div>
  )
}

export function Swatch({ color }: { color: string }) {
  const map: Record<string, string> = { white: '#e8ecef', black: '#1b1e22', silver: '#b8c0c8', grey: '#7b848d', red: '#b93a3a', blue: '#3a62b9', green: '#3a8a55', yellow: '#d1b13a' }
  return <span aria-hidden className="inline-block h-2.5 w-2.5 shrink-0 rounded-[2px] border border-ink-500" style={{ background: map[color] ?? '#556' }} />
}
