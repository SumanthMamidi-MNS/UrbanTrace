/**
 * Indian plate display helpers. Canonical form is 10 slots: SS DD LL DDDD with "_" blanks
 * (e.g. "MH12A_1234" for a 1-letter series, "KA05MN_123" for a 3-digit number).
 */

export const PLATE_SLOTS = 10
export const BLANK = '_'

/** Slot index ranges [start, end) of the four plate groups. */
export const PLATE_GROUPS: readonly [number, number][] = [
  [0, 2],
  [2, 4],
  [4, 6],
  [6, 10],
]

/** "MH12AB1234" -> "MH 12 AB 1234"; blanks dropped; tolerates non-canonical input. */
export function formatPlate(canonical: string | null | undefined): string {
  if (!canonical) return '—'
  if (canonical.length !== PLATE_SLOTS) return canonical.replaceAll(BLANK, '')
  const parts = PLATE_GROUPS.map(([a, b]) => canonical.slice(a, b).replaceAll(BLANK, '')).filter(Boolean)
  return parts.length ? parts.join(' ') : '—'
}

/** Per-slot chars, padded/truncated to 10 so rendering never breaks on a malformed plate. */
export function plateSlots(canonical: string): string[] {
  const out: string[] = []
  for (let i = 0; i < PLATE_SLOTS; i++) out.push(canonical[i] ?? BLANK)
  return out
}

/** Indices where `read` disagrees with `truth`. */
export function mismatchSlots(read: string, truth: string): number[] {
  const r = plateSlots(read)
  const t = plateSlots(truth)
  const out: number[] = []
  for (let i = 0; i < PLATE_SLOTS; i++) if (r[i] !== t[i]) out.push(i)
  return out
}

/** Normalise user search input: uppercase, strip spaces/dashes; keeps "?" wildcards. */
export function normalisePlateQuery(q: string): string {
  return q.toUpperCase().replace(/[\s\-.]/g, '')
}

// ---- watchlist / search pattern grammar (port of api/plate_grammar.py, same rules) ----

const WILDCARD = '?'
const isLetterOrWild = (c: string) => c === WILDCARD || /^[A-Z]$/.test(c)
const isDigitOrWild = (c: string) => c === WILDCARD || /^[0-9]$/.test(c)

export type PatternExpansion = { ok: true; forms: string[] } | { ok: false; detail: string }

/**
 * Every 10-slot canonical form consistent with the Indian plate grammar and the literal characters
 * typed: SS + RTO (1-2 digits, left-zero-padded) + series (0-2 letters, "_"-padded right) + number
 * (1-4 digits, "_"-padded left); "?" = one unknown character. Mirrors the API so the UI can preview
 * what a pattern will match before it is submitted.
 */
export function expandPlatePattern(raw: string): PatternExpansion {
  const q = raw.trim().toUpperCase().replaceAll(' ', '').replaceAll('-', '')
  if (q.length < 3) return { ok: false, detail: `"${raw}" is too short for state + RTO + number` }
  const state = q.slice(0, 2)
  if (![...state].every(isLetterOrWild)) return { ok: false, detail: `"${raw}": the first 2 characters must be state letters` }
  const rest = q.slice(2)
  const forms: string[] = []
  for (const rtoLen of [1, 2]) {
    for (const seriesLen of [0, 1, 2]) {
      const numberLen = rest.length - rtoLen - seriesLen
      if (numberLen < 1 || numberLen > 4) continue
      const rto = rest.slice(0, rtoLen)
      const series = rest.slice(rtoLen, rtoLen + seriesLen)
      const number = rest.slice(rtoLen + seriesLen)
      if (![...rto].every(isDigitOrWild) || ![...series].every(isLetterOrWild) || ![...number].every(isDigitOrWild)) continue
      const canon = state + (rto.length === 1 ? `0${rto}` : rto) + series.padEnd(2, BLANK) + number.padStart(4, BLANK)
      if (!forms.includes(canon)) forms.push(canon)
    }
  }
  if (!forms.length)
    return { ok: false, detail: `"${raw}" does not fit the Indian plate grammar SS · RTO (1-2 digits) · series (0-2 letters) · number (1-4 digits); "?" = one unknown character` }
  return { ok: true, forms }
}

/** Does a canonical plate satisfy a canonical pattern ("?" matches any non-blank)? */
export function plateMatchesPattern(plate: string, pattern: string): boolean {
  if (plate.length !== PLATE_SLOTS || pattern.length !== PLATE_SLOTS) return false
  for (let i = 0; i < PLATE_SLOTS; i++) {
    const p = pattern[i]
    if (p === WILDCARD ? plate[i] === BLANK : p !== plate[i]) return false
  }
  return true
}

/** Of several canonical patterns, the one closest (fewest literal disagreements) to a plate. */
export function closestPattern(plate: string, patterns: string[]): string | undefined {
  let best: string | undefined
  let bestD = Infinity
  for (const p of patterns) {
    let d = 0
    for (let i = 0; i < PLATE_SLOTS; i++) if (p[i] !== WILDCARD && p[i] !== plate[i]) d++
    if (d < bestD) {
      bestD = d
      best = p
    }
  }
  return best
}
