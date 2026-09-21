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
