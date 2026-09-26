// Reports are produced by the engine and may be missing or partial: every accessor is defensive.
export type Obj = Record<string, unknown>
export const isObj = (x: unknown): x is Obj => typeof x === 'object' && x !== null && !Array.isArray(x)
export const num = (x: unknown): number | undefined => (typeof x === 'number' && Number.isFinite(x) ? x : undefined)
export const str = (x: unknown): string | undefined => (typeof x === 'string' ? x : undefined)
export const objs = (x: unknown): Obj[] => (Array.isArray(x) ? x.filter(isObj) : [])
export const pct = (x: number | undefined, d = 1) => (x === undefined ? '—' : `${(x * 100).toFixed(d)}%`)
export const dec = (x: number | undefined, d = 3) => (x === undefined ? '—' : x.toFixed(d))

export const CHART = {
  grid: '#1e2732',
  axis: { stroke: '#566374', tick: { fontSize: 11, fill: '#7d8b9b' }, tickLine: false },
  tooltip: {
    contentStyle: { background: '#131920', border: '1px solid #2a3542', borderRadius: 4, fontSize: 12 },
    labelStyle: { color: '#7d8b9b' },
    itemStyle: { color: '#eef3f8' },
  },
  /** UrbanTrace / primary series */
  accent: '#3cc4d8',
  /** comparison / baseline series (drawn dashed as a second, non-colour cue) */
  neutral: '#7d8b9b',
  light: '#d3dbe4',
}

