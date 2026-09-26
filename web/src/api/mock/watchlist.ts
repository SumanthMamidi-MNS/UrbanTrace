/**
 * Mock watchlist matching (contract v2 "Semantics"): a read matches an entry when
 * P(plate ∈ pattern | evidence) ≥ 0.5, evaluated on (1) the read's own posterior and (2) the growing
 * trajectory's fused consensus. Fixture-grade maths: per-slot independence, residual mass spread
 * uniformly over the characters outside each slot's top list.
 */
import type { SlotRead } from '../types'

export const WATCHLIST_MATCH_THRESHOLD = 0.5

/** Alphabet size per canonical slot (letters, digits, "_" where the grammar allows a blank). */
const SLOT_ALPHABET_SIZE = [26, 26, 10, 10, 27, 27, 11, 11, 11, 11]

function slotProb(reads: SlotRead[] | undefined, want: string, slot: number): number {
  const top = reads ?? []
  const listed = top.reduce((a, r) => a + r.prob, 0)
  const residual = Math.max(0, 1 - listed)
  const unlisted = Math.max(1, SLOT_ALPHABET_SIZE[slot] - top.length)
  const p = (ch: string) => top.find((r) => r.char === ch)?.prob ?? residual / unlisted
  // "?" = any non-blank character
  if (want === '?') return Math.max(0, 1 - p('_'))
  return p(want)
}

/** P(plate matches any of the canonical patterns). Forms are layout-disjoint, so their mass adds. */
export function matchProbability(perSlot: SlotRead[][], forms: string[]): number {
  let total = 0
  for (const f of forms) {
    let prod = 1
    for (let s = 0; s < 10 && prod > 0; s++) prod *= slotProb(perSlot[s], f[s], s)
    total += prod
  }
  return Math.min(1, total)
}
