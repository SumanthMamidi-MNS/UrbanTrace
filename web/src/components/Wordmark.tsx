import { useId } from 'react'
import { cx } from './ui'

/**
 * Brand mark: a vehicle's path through three camera points. The two earlier sightings are
 * hollow rings, the latest one is solid with a soft halo, and the stroke brightens along the
 * direction of travel, the same visual language the map uses for tracks.
 * Keep in sync with web/public/favicon.svg.
 */
export function BrandMark({ size = 26, className }: { size?: number; className?: string }) {
  const gid = `ut-trace-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden focusable="false" className={className}>
      <defs>
        <linearGradient id={gid} x1="6" y1="24" x2="26" y2="11" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="var(--color-accent-dim)" />
          <stop offset="1" stopColor="var(--color-accent-strong)" />
        </linearGradient>
      </defs>
      <path d="M6 24 L14 12.5 L26 11" fill="none" stroke={`url(#${gid})`} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="6" cy="24" r="2.6" fill="var(--color-ink-950)" stroke="var(--color-accent-dim)" strokeWidth="1.8" />
      <circle cx="14" cy="12.5" r="2.6" fill="var(--color-ink-950)" stroke="var(--color-accent)" strokeWidth="1.8" />
      <circle cx="26" cy="11" r="5.5" fill="var(--color-accent)" opacity="0.18" />
      <circle cx="26" cy="11" r="3.1" fill="var(--color-accent-strong)" />
    </svg>
  )
}

/** Two-tone product name: "Urban" in the foreground colour, "Trace" in the track accent. */
export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cx('text-[17px] leading-none font-semibold tracking-[-0.02em] whitespace-nowrap select-none', className)}>
      <span className="text-fg-strong">Urban</span>
      <span className="text-accent">Trace</span>
    </span>
  )
}
