import { useEffect, useMemo, useState } from 'react'
import { useHeatmap } from '../api/hooks'
import type { City, HeatMetric } from '../api/types'
import type { MapHeat } from '../components/CityMap'
import { getRouter, type LngLat } from '../lib/geo'
import { linkSpeedKmh, mean } from '../lib/speed'
import { liveStore } from './liveStore'

export type HeatSource = 'live' | 'snapshot'

/**
 * Speed is drawn as "slowness" so congestion glows: >= SLOW_FAST_KMH is dark, <= SLOW_SLOW_KMH is
 * brightest. The underlying value is still the contract's mean km/h, shown as numbers in the legend.
 */
export const SLOW_FAST_KMH = 60
export const SLOW_SLOW_KMH = 10
const speedToWeight = (kmh: number) => Math.max(0, Math.min(1, (SLOW_FAST_KMH - kmh) / (SLOW_FAST_KMH - SLOW_SLOW_KMH)))

export type HeatCell = { camera_id: string; lngLat: LngLat; weight: number; /** reads in window, normalised density, or km/h */ value: number }

export type HeatView = {
  metric: HeatMetric
  /** what is actually drawn (live falls back to the snapshot until the buffer has data) */
  shown: HeatSource
  windowMin: number
  /** sim-time end of the window */
  at: string | null
  /** live only: minutes of the window actually covered by buffered reads */
  coverageMin: number | null
  /** live density carries raw counts; the API returns density already normalised */
  valueKind: 'count' | 'normalised' | 'kmh'
  cells: HeatCell[]
  loading: boolean
  error: unknown
}

function liveCells(city: City, metric: HeatMetric, windowMin: number) {
  const { reads, links, simMs } = liveStore.getHeatBuffer()
  if (!Number.isFinite(simMs)) return null
  const from = simMs - windowMin * 60e3
  const coord = new Map(city.cameras.map((c) => [c.camera_id, [c.lon, c.lat] as LngLat]))
  const cells: HeatCell[] = []
  let earliest = Infinity
  if (metric === 'density') {
    const counts = new Map<string, number>()
    for (const r of reads) {
      if (r.t <= from || r.t > simMs) continue
      earliest = Math.min(earliest, r.t)
      counts.set(r.cam, (counts.get(r.cam) ?? 0) + 1)
    }
    const max = Math.max(1, ...counts.values())
    counts.forEach((n, cam) => {
      const at = coord.get(cam)
      if (at) cells.push({ camera_id: cam, lngLat: at, weight: n / max, value: n })
    })
  } else {
    const router = getRouter(city)
    const per = new Map<string, number[]>()
    for (const l of links) {
      if (l.t <= from || l.t > simMs) continue
      earliest = Math.min(earliest, l.t)
      const kmh = linkSpeedKmh(router.distanceM(l.from, l.to), l.dtS)
      if (kmh === null) continue
      const xs = per.get(l.to)
      if (xs) xs.push(kmh)
      else per.set(l.to, [kmh])
    }
    per.forEach((xs, cam) => {
      const at = coord.get(cam)
      const v = mean(xs)
      if (at) cells.push({ camera_id: cam, lngLat: at, weight: speedToWeight(v), value: v })
    })
  }
  // Coverage counts from the oldest buffered item of either kind, so a sparse speed window isn't under-reported.
  const oldest = Math.min(earliest, reads[0]?.t ?? Infinity, links[0]?.t ?? Infinity)
  const coverageMin = Number.isFinite(oldest) ? Math.min(windowMin, (simMs - oldest) / 60e3) : 0
  return { cells, simMs, coverageMin }
}

export function useHeat(city: City | undefined, opts: { enabled: boolean; metric: HeatMetric; source: HeatSource; windowMin: number }): HeatView | null {
  const { enabled, metric, source, windowMin } = opts
  const [live, setLive] = useState<ReturnType<typeof liveCells>>(null)

  useEffect(() => {
    if (!enabled || source !== 'live' || !city) return
    const sample = () => setLive(liveCells(city, metric, windowMin))
    sample()
    const id = setInterval(sample, 1000)
    return () => clearInterval(id)
  }, [enabled, source, city, metric, windowMin])

  const liveUsable = source === 'live' && live !== null && live.cells.length > 0
  const snap = useHeatmap({ metric, window_minutes: windowMin }, enabled && !liveUsable)

  return useMemo(() => {
    if (!enabled || !city) return null
    if (liveUsable && live) {
      return {
        metric,
        shown: 'live',
        windowMin,
        at: new Date(live.simMs).toISOString(),
        coverageMin: live.coverageMin,
        valueKind: metric === 'density' ? 'count' : 'kmh',
        cells: live.cells,
        loading: false,
        error: null,
      }
    }
    const d = snap.data
    const cells: HeatCell[] =
      d?.points
        .filter((p) => Number.isFinite(p.weight))
        .map((p) => ({ camera_id: p.camera_id, lngLat: [p.lon, p.lat] as LngLat, weight: d.metric === 'speed' ? speedToWeight(p.weight) : p.weight, value: p.weight })) ?? []
    return {
      metric,
      shown: 'snapshot',
      windowMin: d?.window_minutes ?? windowMin,
      at: d?.at ?? null,
      coverageMin: source === 'live' ? (live?.coverageMin ?? 0) : null,
      valueKind: metric === 'density' ? 'normalised' : 'kmh',
      cells: d && d.metric === metric ? cells : [],
      loading: snap.isLoading,
      error: snap.error,
    }
  }, [enabled, city, liveUsable, live, metric, windowMin, snap.data, snap.isLoading, snap.error, source])
}

export const toMapHeat = (v: HeatView | null): MapHeat | null =>
  v ? { points: v.cells.map((c) => ({ id: c.camera_id, lngLat: c.lngLat, weight: c.weight })) } : null
