/** Direction-of-travel helpers (contract v2: heading_deg, 0 = north, clockwise). */

const LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'] as const
const WORDS: Record<string, string> = { N: 'north', NE: 'north-east', E: 'east', SE: 'south-east', S: 'south', SW: 'south-west', W: 'west', NW: 'north-west' }

/** Initial great-circle bearing from a to b in degrees [0, 360). */
export function bearingDeg(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const r = Math.PI / 180
  const y = Math.sin((bLon - aLon) * r) * Math.cos(bLat * r)
  const x = Math.cos(aLat * r) * Math.sin(bLat * r) - Math.sin(aLat * r) * Math.cos(bLat * r) * Math.cos((bLon - aLon) * r)
  return ((Math.atan2(y, x) / r) % 360 + 360) % 360
}

export function compassLabel(deg: number | null | undefined): string | null {
  if (deg === null || deg === undefined || !Number.isFinite(deg)) return null
  return LABELS[Math.round((((deg % 360) + 360) % 360) / 45) % 8]
}

export const compassWord = (label: string | null | undefined) => (label ? (WORDS[label] ?? label) : null)
