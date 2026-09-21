const pad = (n: number, w = 2) => String(n).padStart(w, '0')

/** Sim timestamps are rendered in UTC so they match the engine's ISO strings exactly. */
export function fmtTime(iso: string | null | undefined, withSeconds = true): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  const hm = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`
  return withSeconds ? `${hm}:${pad(d.getUTCSeconds())}` : hm
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${fmtTime(iso)}`
}

export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return '—'
  const neg = s < 0
  let t = Math.round(Math.abs(s))
  const h = Math.floor(t / 3600)
  t -= h * 3600
  const m = Math.floor(t / 60)
  const sec = t - m * 60
  const body = h > 0 ? `${h}h ${pad(m)}m` : m > 0 ? `${m}m ${pad(sec)}s` : `${sec}s`
  return neg ? `-${body}` : body
}

export function fmtDistance(m: number | null | undefined): string {
  if (m === null || m === undefined || !Number.isFinite(m)) return '—'
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  return `${(x * 100).toFixed(digits)}%`
}

export function fmtNum(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  return x.toLocaleString('en-IN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

/** Signed log-odds: "+4.21", "−1.30"; null (non-finite on the wire) renders as "non-finite". */
export function fmtLogOdds(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return 'non-finite'
  const s = Math.abs(x).toFixed(digits)
  return x >= 0 ? `+${s}` : `−${s}`
}

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

/** ISO -> value for <input type="datetime-local">, expressed in UTC (sim time). */
export function toLocalInput(iso: string): string {
  const d = new Date(iso)
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${fmtTime(iso, false)}`
}

/** Inverse of toLocalInput, treating the value as UTC. Empty -> undefined. */
export function fromLocalInput(v: string): string | undefined {
  if (!v) return undefined
  const d = new Date(`${v}:00Z`)
  return Number.isNaN(d.getTime()) ? undefined : d.toISOString()
}

export function titleCase(s: string): string {
  return s.replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}
