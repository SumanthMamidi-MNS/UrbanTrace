import { ApiError, type UrbanTraceApi } from '../client'
import type {
  AnalyticsSummary,
  CameraStats,
  Corridor,
  EventSummary,
  FlowBucket,
  HeatPoint,
  Page,
  SearchHit,
  TrajectorySummary,
  VolumeBucket,
  WatchlistEntry,
} from '../types'
import { getRouter } from '../../lib/geo'
import { expandPlatePattern } from '../../lib/plate'
import { linkSpeedKmh, mean, quantile } from '../../lib/speed'
import { EVAL_REPORTS } from './evalFixtures'
import type { LiveSim } from './live'
import type { MockDb } from './world'

const clampLimit = (l: number | undefined, def = 50) => Math.max(1, Math.min(500, Math.floor(l ?? def)))
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms))
const clone = <T>(x: T): T => structuredClone(x)

function summaryOf(t: TrajectorySummary): TrajectorySummary {
  return {
    trajectory_id: t.trajectory_id,
    decoded_plate: t.decoded_plate,
    plate_confidence: t.plate_confidence,
    start_time: t.start_time,
    end_time: t.end_time,
    n_events: t.n_events,
    camera_sequence: [...t.camera_sequence],
    color: t.color,
    vehicle_type: t.vehicle_type,
    has_alert: t.has_alert,
  }
}

function page<T>(all: T[], limit?: number, offset?: number): Page<T> {
  const l = clampLimit(limit)
  const o = Math.max(0, Math.floor(offset ?? 0))
  return { items: all.slice(o, o + l), total: all.length, limit: l, offset: o }
}

/** "MH12??1234" style pattern (normalised, 10 slots or display form) -> slot-aligned pattern or null. */
function toSlotPattern(q: string, plate: string): string | null {
  if (q.length === 10) return q
  // Display form without blanks: try every blank-insertion that makes the pattern align with this plate's layout.
  const compact = plate.replaceAll('_', '')
  if (q.length !== compact.length) return null
  let j = 0
  let out = ''
  for (const ch of plate) {
    if (ch === '_') out += '_'
    else out += q[j++]
  }
  return out
}

const time = (iso: string) => Date.parse(iso)

type LinkSpeed = { toCamera: string; t: number; kmh: number }

/** Every trajectory link visible at `now` with its contract-v2 speed (v_max-violating links excluded). */
function linkSpeeds(db: MockDb, now: number): LinkSpeed[] {
  const router = getRouter(db.city)
  const out: LinkSpeed[] = []
  for (const t of db.trajectories.values()) {
    const byId = new Map(t.events.map((e) => [e.event_id, e]))
    for (const l of t.links) {
      const a = byId.get(l.from_event_id)
      const b = byId.get(l.to_event_id)
      if (!a || !b) continue
      const tb = db.eventTime.get(b.event_id) ?? time(b.timestamp)
      if (tb > now) continue
      const kmh = linkSpeedKmh(router.distanceM(a.camera_id, b.camera_id), l.delta_t_s)
      if (kmh !== null) out.push({ toCamera: b.camera_id, t: tb, kmh })
    }
  }
  return out
}

export function createMockApi(db: MockDb, live: LiveSim): UrbanTraceApi {
  let watchSeq = 1
  const lat = async <T>(fn: () => T, ms = 90): Promise<T> => {
    await delay(ms + Math.random() * 90)
    return clone(fn())
  }

  const trajList = () => [...db.trajectories.values()].sort((a, b) => time(b.end_time) - time(a.end_time))

  return {
    health: () =>
      lat(() => ({ status: 'ok' as const, dataset: 'mock://pune-grid-50cam', n_events: db.events.length, n_trajectories: db.trajectories.size })),

    city: () => lat(() => db.city, 150),

    cameras: () =>
      lat(() => {
        const now = live.simTimeMs()
        const total = new Map<string, number>()
        const hour = new Map<string, number>()
        for (const e of db.events) {
          const t = db.eventTime.get(e.event_id) ?? 0
          if (t > now) continue
          total.set(e.camera_id, (total.get(e.camera_id) ?? 0) + 1)
          if (t >= now - 3600e3) hour.set(e.camera_id, (hour.get(e.camera_id) ?? 0) + 1)
        }
        return db.city.cameras.map<CameraStats>((c) => ({ ...c, events_total: total.get(c.camera_id) ?? 0, volume_last_hour: hour.get(c.camera_id) ?? 0 }))
      }),

    events: (q = {}) =>
      lat(() => {
        const from = q.from ? time(q.from) : -Infinity
        const to = q.to ? time(q.to) : Infinity
        const now = live.simTimeMs()
        const out: EventSummary[] = []
        for (let i = db.events.length - 1; i >= 0; i--) {
          const e = db.events[i]
          const t = db.eventTime.get(e.event_id) ?? 0
          if (t > now || t < from || t > to) continue
          if (q.camera_id && e.camera_id !== q.camera_id) continue
          out.push(e)
        }
        return page(out, q.limit, q.offset)
      }),

    event: async (id) => {
      await delay(60)
      const e = db.events.find((x) => x.event_id === id)
      const post = db.posterior.get(id)
      if (!e || !post) throw new ApiError(404, `event ${id} not found`)
      return clone({ ...e, plate_posterior: post })
    },

    trajectories: (q = {}) =>
      lat(() => {
        const from = q.from ? time(q.from) : -Infinity
        const to = q.to ? time(q.to) : Infinity
        const plateQ = q.plate?.toUpperCase().replace(/\s/g, '')
        const items = trajList().filter((t) => {
          if (time(t.end_time) < from || time(t.start_time) > to) return false
          if (q.camera_id && !t.camera_sequence.includes(q.camera_id)) return false
          if (q.min_len && t.n_events < q.min_len) return false
          if (q.has_alert !== undefined && t.has_alert !== q.has_alert) return false
          if (plateQ) {
            const re = new RegExp(plateQ.replace(/[^A-Z0-9?_]/g, '').replaceAll('?', '.'))
            if (!re.test(t.decoded_plate) && !re.test(t.decoded_plate.replaceAll('_', ''))) return false
          }
          return true
        })
        return page(items.map(summaryOf), q.limit, q.offset)
      }),

    trajectory: async (id) => {
      await delay(120)
      const t = db.trajectories.get(id)
      if (!t) throw new ApiError(404, `trajectory ${id} not found`)
      return clone(t)
    },

    search: (q) =>
      lat(() => {
        const norm = q.q.toUpperCase().replace(/[\s\-.]/g, '')
        if (!norm) throw new ApiError(422, 'q is required')
        const from = q.from ? time(q.from) : -Infinity
        const to = q.to ? time(q.to) : Infinity
        const hits: SearchHit[] = []
        for (const t of db.trajectories.values()) {
          if (q.color && t.color !== q.color) continue
          if (q.vehicle_type && t.vehicle_type !== q.vehicle_type) continue
          if (time(t.end_time) < from || time(t.start_time) > to) continue
          const pat = toSlotPattern(norm, t.decoded_plate)
          if (!pat) continue
          // Likelihood of the constraint under the fused per-slot posterior; unknown chars get a floor.
          let logp = 0
          let mismatches = 0
          for (let s = 0; s < 10; s++) {
            const ch = pat[s]
            if (ch === '?') continue
            const slot = t.consensus.per_slot[s] ?? []
            const p = slot.find((r) => r.char === ch)?.prob
            if (p === undefined) {
              mismatches++
              logp += Math.log(2e-4)
            } else logp += Math.log(Math.max(p, 2e-4))
          }
          if (mismatches > 1) continue
          hits.push({ trajectory: summaryOf(t), probability: Math.exp(logp), matched_plate: t.decoded_plate })
        }
        hits.sort((a, b) => b.probability - a.probability)
        const top = hits.slice(0, clampLimit(q.limit, 20))
        return top.filter((h) => h.probability > 1e-6)
      }),

    analyticsSummary: () =>
      lat(() => {
        const ts = [...db.trajectories.values()]
        const multi = ts.filter((t) => t.n_events >= 2)
        const repaired = multi.filter((t) => t.consensus.single_read_plates.some((p) => p !== t.decoded_plate))
        const s: AnalyticsSummary = {
          n_events: db.events.length,
          n_trajectories: ts.length,
          mean_trip_duration_s: ts.reduce((a, t) => a + (time(t.end_time) - time(t.start_time)) / 1000, 0) / Math.max(1, ts.length),
          mean_events_per_trajectory: ts.reduce((a, t) => a + t.n_events, 0) / Math.max(1, ts.length),
          plate_repair_rate: repaired.length / Math.max(1, multi.length),
          active_alerts: db.alerts.length,
        }
        return s
      }),

    volumes: (q = {}) =>
      lat(() => {
        const bucketMs = Math.max(1, q.bucket_minutes ?? 60) * 60e3
        const counts = new Map<string, number>()
        for (const e of db.events) {
          if (q.camera_id && e.camera_id !== q.camera_id) continue
          const t = db.eventTime.get(e.event_id) ?? 0
          const b = Math.floor(t / bucketMs) * bucketMs
          const k = `${e.camera_id}|${b}`
          counts.set(k, (counts.get(k) ?? 0) + 1)
        }
        const out: VolumeBucket[] = []
        counts.forEach((count, k) => {
          const [camera_id, b] = k.split('|')
          out.push({ camera_id, bucket_start: new Date(Number(b)).toISOString(), count })
        })
        return out.sort((a, b) => a.bucket_start.localeCompare(b.bucket_start) || a.camera_id.localeCompare(b.camera_id))
      }),

    odMatrix: () =>
      lat(() => {
        const zones = ['N', 'S', 'E', 'W', 'Central']
        const zoneOf = (cam: string) => {
          const [r, c] = db.cameraRC.get(cam) ?? [4, 4]
          const dr = r - 4
          const dc = c - 4
          if (Math.abs(dr) <= 1 && Math.abs(dc) <= 1) return 4
          if (Math.abs(dr) >= Math.abs(dc)) return dr < 0 ? 0 : 1
          return dc > 0 ? 2 : 3
        }
        const matrix = zones.map(() => zones.map(() => 0))
        for (const t of db.trajectories.values()) {
          const seq = t.camera_sequence
          if (seq.length < 1) continue
          matrix[zoneOf(seq[0])][zoneOf(seq[seq.length - 1])]++
        }
        return { zones, matrix }
      }),

    corridors: (q = {}) =>
      lat(() => {
        const router = getRouter(db.city)
        const groups = new Map<string, { tt: number[]; ff: number[] }>()
        for (const t of db.trajectories.values()) {
          for (const l of t.links) {
            if (l.skipped_cameras.length) continue
            const a = t.events.find((e) => e.event_id === l.from_event_id)
            const b = t.events.find((e) => e.event_id === l.to_event_id)
            const ff = db.linkFreeFlow.get(`${l.from_event_id}>${l.to_event_id}`)
            if (!a || !b || ff === undefined) continue
            const k = `${a.camera_id}|${b.camera_id}`
            let g = groups.get(k)
            if (!g) groups.set(k, (g = { tt: [], ff: [] }))
            g.tt.push(l.delta_t_s)
            g.ff.push(ff)
          }
        }
        const q50 = (xs: number[], p: number) => {
          const s = [...xs].sort((a, b) => a - b)
          return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))]
        }
        const out: Corridor[] = []
        groups.forEach((g, k) => {
          if (g.tt.length < 3) return
          const [from_camera, to_camera] = k.split('|')
          const median = q50(g.tt, 0.5)
          const ff = g.ff[0]
          const distance = router.distanceM(from_camera, to_camera)
          const speeds = g.tt.map((dt) => linkSpeedKmh(distance, dt)).filter((v): v is number => v !== null)
          if (!speeds.length || !(ff > 0)) return
          out.push({
            from_camera,
            to_camera,
            n_trips: g.tt.length,
            median_travel_s: +median.toFixed(1),
            p90_travel_s: +q50(g.tt, 0.9).toFixed(1),
            free_flow_s: +ff.toFixed(1),
            congestion_index: +(median / ff).toFixed(3),
            distance_m: Math.round(distance),
            avg_speed_kmh: +mean(speeds).toFixed(1),
            p85_speed_kmh: +quantile(speeds, 0.85).toFixed(1),
            free_flow_speed_kmh: +((distance / ff) * 3.6).toFixed(1),
          })
        })
        out.sort((a, b) => b.congestion_index - a.congestion_index)
        return out.slice(0, q.limit ?? 20)
      }),

    alerts: (q = {}) =>
      lat(() => {
        const now = live.simTimeMs()
        return db.alerts
          .filter((a) => (!q.type || a.type === q.type) && time(a.created_at) <= now)
          .slice(0, clampLimit(q.limit))
      }),

    evalReports: () => lat(() => ({ reports: EVAL_REPORTS }), 200),

    // ---------------------------------------------------------------- v2

    watchlist: () => lat(() => [...db.watchlist].sort((a, b) => time(b.created_at) - time(a.created_at))),

    addWatchlist: async (body) => {
      await delay(120)
      const pattern = String(body?.pattern ?? '').trim()
      if (!pattern) throw new ApiError(422, 'pattern is required')
      const exp = expandPlatePattern(pattern)
      if (!exp.ok) throw new ApiError(422, exp.detail)
      const entry: WatchlistEntry = {
        entry_id: `WL-${String(watchSeq++).padStart(4, '0')}`,
        pattern: pattern.toUpperCase(),
        canonical_patterns: exp.forms,
        reason: String(body?.reason ?? '').trim(),
        created_at: new Date(live.simTimeMs()).toISOString(),
        active: true,
        hits: 0,
      }
      db.watchlist.push(entry)
      live.plantWatchlistTargets(entry)
      return clone(entry)
    },

    deleteWatchlist: async (id) => {
      await delay(80)
      const i = db.watchlist.findIndex((e) => e.entry_id === id)
      if (i < 0) throw new ApiError(404, `watchlist entry ${id} not found`)
      db.watchlist.splice(i, 1)
      return { deleted: true as const }
    },

    watchlistHits: (q = {}) =>
      lat(() =>
        db.watchHits
          .filter((h) => !q.entry_id || h.entry_id === q.entry_id)
          .slice()
          .reverse()
          .slice(0, clampLimit(q.limit)),
      ),

    heatmap: (q = {}) =>
      lat(() => {
        const at = q.at ? time(q.at) : live.simTimeMs()
        if (!Number.isFinite(at)) throw new ApiError(422, `invalid at: ${q.at}`)
        const windowMin = q.window_minutes ?? 15
        if (!(windowMin > 0)) throw new ApiError(422, 'window_minutes must be > 0')
        const metric = q.metric ?? 'density'
        if (metric !== 'density' && metric !== 'speed') throw new ApiError(422, `unknown metric ${String(metric)}`)
        const from = at - windowMin * 60e3
        const per = new Map<string, number[]>()
        const push = (cam: string, v: number) => {
          const l = per.get(cam)
          if (l) l.push(v)
          else per.set(cam, [v])
        }
        if (metric === 'density') {
          for (const e of db.events) {
            const t = db.eventTime.get(e.event_id) ?? 0
            if (t > from && t <= at) push(e.camera_id, 1)
          }
        } else {
          for (const s of linkSpeeds(db, at)) if (s.t > from) push(s.toCamera, s.kmh)
        }
        const raw = new Map([...per].map(([cam, xs]) => [cam, metric === 'density' ? xs.length : mean(xs)] as const))
        const max = Math.max(1, ...raw.values())
        const points: HeatPoint[] = []
        raw.forEach((v, cam) => {
          const c = db.cameraById.get(cam)
          if (c) points.push({ camera_id: cam, lat: c.lat, lon: c.lon, weight: metric === 'density' ? +(v / max).toFixed(4) : +v.toFixed(1) })
        })
        return { at: new Date(at).toISOString(), window_minutes: windowMin, metric, points }
      }),

    flowTrend: (q = {}) =>
      lat(() => {
        const bucketMs = Math.max(1, q.bucket_minutes ?? 15) * 60e3
        const now = live.simTimeMs()
        const counts = new Map<number, number>()
        let lo = Infinity
        for (const e of db.events) {
          const t = db.eventTime.get(e.event_id) ?? 0
          if (t > now) continue
          const b = Math.floor(t / bucketMs) * bucketMs
          lo = Math.min(lo, b)
          counts.set(b, (counts.get(b) ?? 0) + 1)
        }
        if (!Number.isFinite(lo)) return []
        const hi = Math.floor(now / bucketMs) * bucketMs
        const speeds = new Map<number, number[]>()
        for (const s of linkSpeeds(db, now)) {
          const b = Math.floor(s.t / bucketMs) * bucketMs
          const l = speeds.get(b)
          if (l) l.push(s.kmh)
          else speeds.set(b, [s.kmh])
        }
        const spans = [...db.trajectories.values()].map((t) => [time(t.start_time), Math.min(time(t.end_time), now)] as const).filter(([s]) => s <= now)
        const out: FlowBucket[] = []
        for (let b = lo; b <= hi; b += bucketMs) {
          const sp = speeds.get(b)
          out.push({
            bucket_start: new Date(b).toISOString(),
            events: counts.get(b) ?? 0,
            active_trajectories: spans.filter(([s, e]) => s < b + bucketMs && e >= b).length,
            mean_speed_kmh: sp?.length ? +mean(sp).toFixed(1) : null,
          })
        }
        return out
      }),

    replay: async (body) => {
      await delay(80)
      if (!['start', 'pause', 'reset'].includes(body.action)) throw new ApiError(422, `unknown action ${String(body.action)}`)
      return live.control(body.action, body.speed)
    },
  }
}
