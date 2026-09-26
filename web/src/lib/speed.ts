/**
 * Speed definitions shared by the mock API and the Live page's client-side heatmap, so both agree with
 * contract v2: speed = road-graph shortest-path distance between consecutive cameras / observed travel
 * time; links implying more than the kinematic v_max are excluded (they are clone evidence, not speed).
 *
 * v_max: the engine's hard physical gate (engine/decode/clone_detect.py DEFAULT_V_MAX_KMH = 120).
 */
export const SPEED_V_MAX_KMH = 120

export function linkSpeedKmh(distanceM: number, dtS: number): number | null {
  if (!(dtS > 0) || !(distanceM > 0) || !Number.isFinite(distanceM)) return null
  const kmh = (distanceM / dtS) * 3.6
  return kmh > SPEED_V_MAX_KMH ? null : kmh
}

export const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN)

/** Nearest-rank-style quantile with linear interpolation. */
export function quantile(xs: number[], p: number): number {
  if (!xs.length) return NaN
  const s = [...xs].sort((a, b) => a - b)
  const i = Math.max(0, Math.min(s.length - 1, p * (s.length - 1)))
  const lo = Math.floor(i)
  const hi = Math.ceil(i)
  return s[lo] + (s[hi] - s[lo]) * (i - lo)
}
