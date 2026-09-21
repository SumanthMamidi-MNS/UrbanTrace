import type { City } from '../api/types'

export type LngLat = [number, number]

export function haversineM(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const R = 6_371_000
  const toRad = Math.PI / 180
  const dLat = (bLat - aLat) * toRad
  const dLon = (bLon - aLon) * toRad
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(aLat * toRad) * Math.cos(bLat * toRad) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(h))
}

/**
 * Road-following router over the /api/city graph (Dijkstra on length_m; edges treated as
 * undirected for display). One instance per city object, results cached per camera pair.
 */
export class RoadRouter {
  private readonly coords = new Map<string, LngLat>()
  private readonly adj = new Map<string, { to: string; w: number }[]>()
  private readonly cameraNode = new Map<string, string>()
  private readonly cameraCoord = new Map<string, LngLat>()
  private readonly cache = new Map<string, LngLat[]>()

  constructor(city: City) {
    for (const n of city.nodes) this.coords.set(n.node_id, [n.lon, n.lat])
    const add = (a: string, b: string, w: number) => {
      let l = this.adj.get(a)
      if (!l) this.adj.set(a, (l = []))
      l.push({ to: b, w })
    }
    for (const e of city.edges) {
      add(e.from_node, e.to_node, e.length_m)
      add(e.to_node, e.from_node, e.length_m)
    }
    for (const c of city.cameras) {
      this.cameraNode.set(c.camera_id, c.node_id)
      this.cameraCoord.set(c.camera_id, [c.lon, c.lat])
    }
  }

  camera(cameraId: string): LngLat | undefined {
    return this.cameraCoord.get(cameraId)
  }

  /** Coordinates between two cameras along roads; straight line if no route is known. */
  between(fromCam: string, toCam: string): LngLat[] {
    const key = `${fromCam}>${toCam}`
    const hit = this.cache.get(key)
    if (hit) return hit
    const a = this.cameraCoord.get(fromCam)
    const b = this.cameraCoord.get(toCam)
    if (!a || !b) return []
    const src = this.cameraNode.get(fromCam)
    const dst = this.cameraNode.get(toCam)
    let out: LngLat[] = [a, b]
    if (src && dst && this.adj.has(src)) {
      const nodes = this.dijkstra(src, dst)
      if (nodes) {
        const mid = nodes.map((n) => this.coords.get(n)).filter((c): c is LngLat => Boolean(c))
        out = [a, ...mid, b]
      }
    }
    this.cache.set(key, out)
    return out
  }

  /** Full road-following line through a camera sequence. */
  line(cameraSeq: string[]): LngLat[] {
    const out: LngLat[] = []
    for (let i = 0; i < cameraSeq.length; i++) {
      if (i === 0) {
        const c = this.camera(cameraSeq[0])
        if (c) out.push(c)
        continue
      }
      const seg = this.between(cameraSeq[i - 1], cameraSeq[i])
      if (out.length === 0) out.push(...seg)
      else out.push(...seg.slice(1))
    }
    return out
  }

  private dijkstra(src: string, dst: string): string[] | null {
    if (src === dst) return [src]
    const dist = new Map<string, number>([[src, 0]])
    const prev = new Map<string, string>()
    const done = new Set<string>()
    // City graphs here are small (<1k nodes): a linear frontier scan is simpler than a heap and fast enough.
    const frontier = new Set<string>([src])
    while (frontier.size) {
      let u = ''
      let best = Infinity
      for (const n of frontier) {
        const d = dist.get(n) ?? Infinity
        if (d < best) {
          best = d
          u = n
        }
      }
      frontier.delete(u)
      if (u === dst) break
      done.add(u)
      for (const { to, w } of this.adj.get(u) ?? []) {
        if (done.has(to)) continue
        const nd = best + w
        if (nd < (dist.get(to) ?? Infinity)) {
          dist.set(to, nd)
          prev.set(to, u)
          frontier.add(to)
        }
      }
    }
    if (!prev.has(dst)) return null
    const path = [dst]
    let cur = dst
    while (prev.has(cur)) {
      cur = prev.get(cur) as string
      path.push(cur)
    }
    return path.reverse()
  }
}

const routerCache = new WeakMap<City, RoadRouter>()
export function getRouter(city: City): RoadRouter {
  let r = routerCache.get(city)
  if (!r) routerCache.set(city, (r = new RoadRouter(city)))
  return r
}

/** Cumulative-length slicing so a replay head moves at constant speed along a polyline. */
export function sliceLine(coords: LngLat[], fraction: number): { line: LngLat[]; head: LngLat | null } {
  if (coords.length === 0) return { line: [], head: null }
  if (coords.length === 1 || fraction <= 0) return { line: [coords[0]], head: coords[0] }
  const seg: number[] = []
  let total = 0
  for (let i = 1; i < coords.length; i++) {
    const d = Math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1])
    seg.push(d)
    total += d
  }
  const target = Math.min(1, fraction) * total
  let acc = 0
  const line: LngLat[] = [coords[0]]
  for (let i = 0; i < seg.length; i++) {
    if (acc + seg[i] >= target) {
      const t = seg[i] === 0 ? 0 : (target - acc) / seg[i]
      const p: LngLat = [
        coords[i][0] + (coords[i + 1][0] - coords[i][0]) * t,
        coords[i][1] + (coords[i + 1][1] - coords[i][1]) * t,
      ]
      line.push(p)
      return { line, head: p }
    }
    acc += seg[i]
    line.push(coords[i + 1])
  }
  return { line, head: coords[coords.length - 1] }
}

/** Fraction of polyline length at each vertex index boundary for a list of sub-lines. */
export function lineLength(coords: LngLat[]): number {
  let total = 0
  for (let i = 1; i < coords.length; i++) {
    total += Math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1])
  }
  return total
}
