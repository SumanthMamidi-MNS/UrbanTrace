/**
 * Synthetic mock world: a 9x9 road grid around a Pune-like centre, 50 ANPR cameras,
 * a few hundred vehicle trajectories with realistic OCR misreads, and planted clone /
 * impossible-travel / anomaly alerts. Every object is built to the frozen contract shapes.
 *
 * This is demo fixture data for UI development only. It does not model the engine's maths;
 * it produces numbers in plausible ranges so every UI state (weak plate evidence, missed
 * cameras, non-finite values, clones) is exercised.
 */
import type {
  Alert,
  Camera,
  City,
  EventDetail,
  EventSummary,
  LinkEvidence,
  PathPoint,
  PlateConsensus,
  RoadEdge,
  RoadNode,
  SlotRead,
  TrajectoryDetail,
} from '../types'
import { Rng } from './rng'

// ---------------------------------------------------------------- constants

const CENTER = { lat: 18.5204, lon: 73.8567 }
const GRID = 9
const DLAT = 0.008
const DLON = 0.0084
export const V_MAX_MS = 25 // 90 km/h: the physical gate used for impossible-travel evidence

export const HIST_START = Date.parse('2026-03-09T06:00:00Z')
export const HIST_END = Date.parse('2026-03-09T12:00:00Z')

const STATE_CODES = ['MH', 'MH', 'MH', 'MH', 'MH', 'KA', 'GJ', 'DL', 'TS', 'MP', 'KA', 'RJ']
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
const DIGITS = '0123456789'.split('')
const SLOT_ALPHABET: string[][] = [LETTERS, LETTERS, DIGITS, DIGITS, [...LETTERS, '_'], [...LETTERS, '_'], [...DIGITS, '_'], [...DIGITS, '_'], [...DIGITS, '_'], [...DIGITS, '_']]

/** Visually confusable characters, kept within each slot's alphabet. */
const CONFUSE: Record<string, string> = {
  O: 'DQC', D: 'O', Q: 'O', C: 'G', G: 'C', B: 'RE', R: 'B', E: 'F', F: 'E', M: 'NH', N: 'MH', H: 'MN',
  U: 'V', V: 'UY', I: 'LT', L: 'I', T: 'I', K: 'X', X: 'K', P: 'R', S: 'B', Z: 'L', W: 'V', A: 'R', J: 'I', Y: 'V',
  '0': '86', '1': '7', '7': '1', '8': '036', '3': '8', '5': '6', '6': '58', '2': '7', '4': '1', '9': '8',
}

export const COLORS = ['white', 'black', 'silver', 'grey', 'red', 'blue', 'green', 'yellow']
const COLOR_W = [0.22, 0.16, 0.18, 0.12, 0.1, 0.1, 0.07, 0.05]
export const VEHICLE_TYPES = ['car', 'bike', 'truck', 'bus', 'auto']
const TYPE_W = [0.55, 0.22, 0.08, 0.05, 0.1]

const ROW_NAMES = ['North Ring Rd', 'Mill Rd', 'Canal Rd', 'Station Rd', 'Central Ave', 'Market Rd', 'Temple Rd', 'River Rd', 'South Ring Rd']
const COL_NAMES = ['West Bypass', '1st Cross', 'Hill Rd', 'Tank Rd', 'Arsenal Rd', 'Court Rd', 'Fort Rd', 'Lake Rd', 'East Bypass']

// ---------------------------------------------------------------- db shape

export interface MockDb {
  city: City
  cameraById: Map<string, Camera>
  cameraRC: Map<string, [number, number]>
  /** all events, ascending time */
  events: EventSummary[]
  eventTime: Map<string, number>
  posterior: Map<string, EventDetail['plate_posterior']>
  trajectories: Map<string, TrajectoryDetail>
  /** free-flow seconds per link, keyed "fromEvent>toEvent" */
  linkFreeFlow: Map<string, number>
  alerts: Alert[]
  liveTrajectoryIds: Set<string>
  liveEventIds: Set<string>
}

const pad = (n: number, w: number) => String(n).padStart(w, '0')
const iso = (ms: number) => new Date(ms).toISOString()

function haversine(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const R = 6_371_000
  const r = Math.PI / 180
  const h = Math.sin(((bLat - aLat) * r) / 2) ** 2 + Math.cos(aLat * r) * Math.cos(bLat * r) * Math.sin(((bLon - aLon) * r) / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(h))
}

const lineSpeed = (i: number) => (i % 4 === 0 ? 60 : i % 2 === 0 ? 45 : 30)

/** Time-of-day congestion multiplier (morning + evening peak). */
export function todFactor(ms: number): number {
  const d = new Date(ms)
  const h = d.getUTCHours() + d.getUTCMinutes() / 60
  return 1 + 0.45 * Math.exp(-(((h - 9) / 1.1) ** 2)) + 0.4 * Math.exp(-(((h - 18) / 1.3) ** 2))
}

// ---------------------------------------------------------------- builder

export type Route = { cams: string[]; freeFlow: number[] }

export type VehicleSpec = {
  start: number
  route?: Route
  plate?: string
  color?: string
  vehicleType?: string
  /** 0 clean .. 0.2 very noisy per-slot confusion probability */
  noise?: number
  /** force exact argmax strings for each observed read (showcase) */
  forcedReads?: string[]
  /** observe every camera on the route */
  observeAll?: boolean
  /** skip indices (showcase missed detection) */
  forceSkip?: number[]
}

export class WorldBuilder {
  readonly rng: Rng
  readonly db: MockDb
  private nodeGrid: string[][] = []
  private nodeById = new Map<string, RoadNode>()
  private edgeSpeed = new Map<string, number>()
  private removed = new Set<string>()
  private camAtNode = new Map<string, string>()
  private eventSeq = 1_000_001
  private trajSeq = 40_001
  private alertSeq = 701

  constructor(seed = 20260309) {
    this.rng = new Rng(seed)
    this.db = {
      city: { nodes: [], edges: [], cameras: [] },
      cameraById: new Map(),
      cameraRC: new Map(),
      events: [],
      eventTime: new Map(),
      posterior: new Map(),
      trajectories: new Map(),
      linkFreeFlow: new Map(),
      alerts: [],
      liveTrajectoryIds: new Set(),
      liveEventIds: new Set(),
    }
    this.buildCity()
  }

  // ------------------------------------------------ city

  private buildCity() {
    const { rng } = this
    const nodes: RoadNode[] = []
    for (let r = 0; r < GRID; r++) {
      const row: string[] = []
      for (let c = 0; c < GRID; c++) {
        const id = `N${r}${c}`
        const jitter = r % 4 === 0 || c % 4 === 0 ? 0.00025 : 0.0007
        const n: RoadNode = {
          node_id: id,
          lat: +(CENTER.lat + (4 - r) * DLAT + rng.uniform(-jitter, jitter)).toFixed(6),
          lon: +(CENTER.lon + (c - 4) * DLON + rng.uniform(-jitter, jitter)).toFixed(6),
        }
        nodes.push(n)
        this.nodeById.set(id, n)
        row.push(id)
      }
      this.nodeGrid.push(row)
    }
    // Remove a handful of minor horizontal links for an organic street pattern (verticals keep it connected).
    for (let k = 0; k < 7; k++) {
      const r = rng.pick([1, 3, 5, 7])
      const c = rng.int(0, GRID - 2)
      this.removed.add(`${this.nodeGrid[r][c]}|${this.nodeGrid[r][c + 1]}`)
    }
    const edges: RoadEdge[] = []
    const addEdge = (a: string, b: string, speed: number) => {
      if (this.removed.has(`${a}|${b}`)) return
      const na = this.nodeById.get(a) as RoadNode
      const nb = this.nodeById.get(b) as RoadNode
      const len = Math.round(haversine(na.lat, na.lon, nb.lat, nb.lon))
      edges.push({ from_node: a, to_node: b, length_m: len, speed_limit_kmh: speed })
      edges.push({ from_node: b, to_node: a, length_m: len, speed_limit_kmh: speed })
      this.edgeSpeed.set(`${a}|${b}`, speed)
      this.edgeSpeed.set(`${b}|${a}`, speed)
    }
    for (let r = 0; r < GRID; r++)
      for (let c = 0; c < GRID; c++) {
        if (c < GRID - 1) addEdge(this.nodeGrid[r][c], this.nodeGrid[r][c + 1], lineSpeed(r))
        if (r < GRID - 1) addEdge(this.nodeGrid[r][c], this.nodeGrid[r + 1][c], lineSpeed(c))
      }

    // 50 cameras: every arterial x arterial junction, then arterial nodes, then minor junctions.
    const art = (i: number) => i % 4 === 0
    const all: [number, number][] = []
    for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) all.push([r, c])
    const tier1 = all.filter(([r, c]) => art(r) && art(c))
    const tier2 = rng.shuffle(all.filter(([r, c]) => (art(r) || art(c)) && !(art(r) && art(c))))
    const tier3 = rng.shuffle(all.filter(([r, c]) => !art(r) && !art(c)))
    const chosen = [...tier1, ...tier2.slice(0, 26), ...tier3.slice(0, 15)]
    chosen.sort((a, b) => a[0] * GRID + a[1] - (b[0] * GRID + b[1]))
    const cameras: Camera[] = chosen.map(([r, c], i) => {
      const node = this.nodeById.get(this.nodeGrid[r][c]) as RoadNode
      const cam: Camera = {
        camera_id: `CAM-${pad(i + 1, 3)}`,
        name: `${ROW_NAMES[r]} × ${COL_NAMES[c]}`,
        lat: node.lat,
        lon: node.lon,
        node_id: node.node_id,
        bearing_deg: rng.pick([0, 90, 180, 270]),
        is_border: r === 0 || c === 0 || r === GRID - 1 || c === GRID - 1,
      }
      this.camAtNode.set(node.node_id, cam.camera_id)
      this.db.cameraRC.set(cam.camera_id, [r, c])
      this.db.cameraById.set(cam.camera_id, cam)
      return cam
    })
    this.db.city = { nodes, edges, cameras }
  }

  cameraIds(): string[] {
    return this.db.city.cameras.map((c) => c.camera_id)
  }

  camDistance(a: string, b: string): number {
    const ca = this.db.cameraById.get(a) as Camera
    const cb = this.db.cameraById.get(b) as Camera
    return haversine(ca.lat, ca.lon, cb.lat, cb.lon)
  }

  // ------------------------------------------------ routing

  /** Grid route that climbs onto the nearest arterial row, runs along it, then drops to the destination. */
  planRoute(fromCam: string, toCam: string): Route {
    const [r0, c0] = this.db.cameraRC.get(fromCam) as [number, number]
    const [r1, c1] = this.db.cameraRC.get(toCam) as [number, number]
    const mid = (r0 + r1) / 2
    const ra = [0, 4, 8].reduce((best, x) => (Math.abs(x - mid) < Math.abs(best - mid) ? x : best), 4)
    const cells: [number, number][] = [[r0, c0]]
    const step = (tr: number, tc: number) => {
      let [r, c] = cells[cells.length - 1]
      while (r !== tr) {
        r += Math.sign(tr - r)
        cells.push([r, c])
      }
      while (c !== tc) {
        c += Math.sign(tc - c)
        cells.push([r, c])
      }
    }
    step(ra, c0)
    step(ra, c1)
    step(r1, c1)
    const cams: string[] = []
    const freeFlow: number[] = []
    let acc = 0
    for (let i = 0; i < cells.length; i++) {
      const id = this.nodeGrid[cells[i][0]][cells[i][1]]
      if (i > 0) {
        const prev = this.nodeGrid[cells[i - 1][0]][cells[i - 1][1]]
        const a = this.nodeById.get(prev) as RoadNode
        const b = this.nodeById.get(id) as RoadNode
        const speed = this.edgeSpeed.get(`${prev}|${id}`) ?? 30
        acc += haversine(a.lat, a.lon, b.lat, b.lon) / (speed / 3.6)
      }
      const cam = this.camAtNode.get(id)
      if (cam && cams[cams.length - 1] !== cam) {
        if (cams.length > 0) freeFlow.push(acc)
        cams.push(cam)
        acc = 0
      }
    }
    return { cams, freeFlow }
  }

  randomRoute(minCams = 3): Route {
    const ids = this.cameraIds()
    for (let tries = 0; tries < 40; tries++) {
      const a = this.rng.pick(ids)
      const b = this.rng.pick(ids)
      if (a === b) continue
      const route = this.planRoute(a, b)
      if (route.cams.length >= minCams) return route
    }
    return this.planRoute(ids[0], ids[ids.length - 1])
  }

  randomPlate(): string {
    const { rng } = this
    const state = rng.pick(STATE_CODES)
    const rto = pad(state === 'MH' ? rng.pick([12, 12, 12, 14, 14, 42, 1, 4]) : rng.int(1, 60), 2)
    const series = Array.from({ length: rng.chance(0.3) ? 1 : 2 }, () => rng.pick(LETTERS.filter((l) => l !== 'I' && l !== 'O'))).join('').padEnd(2, '_')
    const nDigits = rng.weighted([1, 2, 3, 4], [0.03, 0.07, 0.15, 0.75])
    const number = Array.from({ length: nDigits }, () => String(rng.int(0, 9))).join('').padStart(4, '_')
    return state + rto + series + number
  }

  // ------------------------------------------------ OCR reads

  private confuse(ch: string, slot: number): string {
    const alpha = SLOT_ALPHABET[slot]
    const opts = (CONFUSE[ch] ?? '').split('').filter((x) => alpha.includes(x) && x !== ch)
    if (opts.length) return this.rng.pick(opts)
    let x = ch
    while (x === ch) x = this.rng.pick(alpha.filter((a) => a !== '_'))
    return x
  }

  private readPlate(truth: string, noise: number, forced?: string): { argmax: string; posterior: EventDetail['plate_posterior']; confidence: number } {
    const { rng } = this
    const occlude = !forced && rng.chance(noise * 0.9)
    const occStart = rng.int(0, 7)
    const occLen = rng.int(2, 3)
    const posterior: EventDetail['plate_posterior'] = []
    let chars = ''
    let confSum = 0
    for (let s = 0; s < 10; s++) {
      const t = truth[s]
      const alpha = SLOT_ALPHABET[s]
      if (occlude && s >= occStart && s < occStart + occLen) {
        const picks = rng.shuffle([...alpha]).slice(0, 5)
        const base = 1 / alpha.length
        const top: SlotRead[] = picks.map((c, i) => ({ char: c, prob: +(base * (1.12 - i * 0.03)).toFixed(4) }))
        posterior.push({ top, unread: true })
        chars += top[0].char
        confSum += top[0].prob
        continue
      }
      const target = forced ? forced[s] : t === '_' ? (rng.chance(noise * 0.1) ? this.confuse('0', s) : '_') : rng.chance(noise) ? this.confuse(t, s) : t
      const wrong = target !== t
      const pTop = wrong ? rng.uniform(0.4, 0.63) : rng.uniform(0.74, 0.985)
      const top: SlotRead[] = [{ char: target, prob: pTop }]
      let rest = 1 - pTop
      const second = wrong ? t : this.confuse(target === '_' ? '0' : target, s)
      const p2 = wrong ? Math.min(pTop - 0.04, rest * rng.uniform(0.55, 0.8)) : rest * rng.uniform(0.35, 0.7)
      top.push({ char: second, prob: p2 })
      rest -= p2
      const used = new Set([target, second])
      while (top.length < 5) {
        const c = rng.pick(alpha)
        if (used.has(c)) continue
        used.add(c)
        const p = rest * rng.uniform(0.2, 0.45)
        top.push({ char: c, prob: p })
        rest -= p
      }
      top.sort((a, b) => b.prob - a.prob)
      posterior.push({ top: top.map((x) => ({ char: x.char, prob: +x.prob.toFixed(4) })), unread: false })
      chars += target
      confSum += pTop
    }
    return { argmax: chars, posterior, confidence: +(confSum / 10).toFixed(3) }
  }

  private consensus(truth: string, reads: string[], unreadMasks: boolean[][]): PlateConsensus {
    const { rng } = this
    const perSlot: SlotRead[][] = []
    let entropy = 0
    for (let s = 0; s < 10; s++) {
      const alpha = SLOT_ALPHABET[s]
      let nOk = 0
      const wrongChars: string[] = []
      reads.forEach((r, i) => {
        if (unreadMasks[i][s]) return
        if (r[s] === truth[s]) nOk++
        else wrongChars.push(r[s])
      })
      const eps = Math.min(0.45, Math.max(2e-5, 0.3 ** nOk * (1 + 1.5 * wrongChars.length) * rng.uniform(0.6, 1.4)))
      const top: SlotRead[] = [{ char: truth[s], prob: 1 - eps }]
      const used = new Set([truth[s]])
      let rest = eps
      for (const c of [...wrongChars, this.confuse(truth[s] === '_' ? '0' : truth[s], s)]) {
        if (used.has(c) || top.length >= 5) continue
        used.add(c)
        const p = rest * rng.uniform(0.45, 0.7)
        top.push({ char: c, prob: p })
        rest -= p
      }
      while (top.length < 3) {
        const c = rng.pick(alpha)
        if (used.has(c)) continue
        used.add(c)
        const p = rest * 0.3
        top.push({ char: c, prob: p })
        rest -= p
      }
      top.sort((a, b) => b.prob - a.prob)
      for (const x of top) if (x.prob > 0) entropy -= x.prob * Math.log2(x.prob)
      const others = alpha.length - top.length
      if (rest > 0 && others > 0) entropy -= rest * Math.log2(rest / others)
      perSlot.push(top.map((x) => ({ char: x.char, prob: +x.prob.toFixed(5) })))
    }
    return { per_slot: perSlot, single_read_plates: reads, entropy_bits: +entropy.toFixed(3) }
  }

  // ------------------------------------------------ vehicles

  /** Build a full trajectory (not yet registered). Timestamps may be in the future (live mode). */
  buildVehicle(spec: VehicleSpec): { traj: TrajectoryDetail; posteriors: Map<string, EventDetail['plate_posterior']>; freeFlow: Map<string, number>; times: number[] } {
    const { rng } = this
    const route = spec.route ?? this.randomRoute()
    const plate = spec.plate ?? this.randomPlate()
    const color = spec.color ?? rng.weighted(COLORS, COLOR_W)
    const vehicleType = spec.vehicleType ?? rng.weighted(VEHICLE_TYPES, TYPE_W)
    const noise = spec.noise ?? rng.weighted([0.01, 0.06, 0.14], [0.6, 0.3, 0.1])
    const trajId = `T-${pad(this.trajSeq++, 6)}`

    // observation mask (missed detections)
    const n = route.cams.length
    const observed = route.cams.map((_, i) => spec.observeAll || i === 0 || i === n - 1 || !rng.chance(0.1))
    for (const i of spec.forceSkip ?? []) if (i > 0 && i < n - 1) observed[i] = false

    const vehFactor = rng.uniform(1.05, 1.4) * (vehicleType === 'truck' || vehicleType === 'bus' ? 1.15 : vehicleType === 'bike' ? 0.92 : 1)
    const camTimes: number[] = [spec.start]
    for (let i = 1; i < n; i++) {
      const prevT = camTimes[i - 1]
      camTimes.push(prevT + route.freeFlow[i - 1] * 1000 * vehFactor * todFactor(prevT) * rng.uniform(0.88, 1.18))
    }

    const idx = observed.map((o, i) => (o ? i : -1)).filter((i) => i >= 0)
    const events: EventSummary[] = []
    const path: PathPoint[] = []
    const posteriors = new Map<string, EventDetail['plate_posterior']>()
    const reads: string[] = []
    const unreadMasks: boolean[][] = []
    idx.forEach((ci, k) => {
      const camId = route.cams[ci]
      const cam = this.db.cameraById.get(camId) as Camera
      const t = Math.round(camTimes[ci])
      const eventId = `E-${pad(this.eventSeq++, 7)}`
      const read = this.readPlate(plate, noise, spec.forcedReads?.[k])
      posteriors.set(eventId, read.posterior)
      reads.push(read.argmax)
      unreadMasks.push(read.posterior.map((p) => p.unread))
      events.push({
        event_id: eventId,
        camera_id: camId,
        timestamp: iso(t),
        plate_argmax: read.argmax,
        plate_confidence: read.confidence,
        color: rng.chance(0.93) ? color : rng.weighted(COLORS, COLOR_W),
        vehicle_type: vehicleType,
        trajectory_id: trajId,
      })
      path.push({ event_id: eventId, camera_id: camId, lat: cam.lat, lon: cam.lon, timestamp: iso(t) })
    })

    const links: LinkEvidence[] = []
    const freeFlow = new Map<string, number>()
    for (let k = 1; k < idx.length; k++) {
      const a = idx[k - 1]
      const b = idx[k]
      const ff = route.freeFlow.slice(a, b).reduce((x, y) => x + y, 0)
      const dt = (camTimes[b] - camTimes[a]) / 1000
      const expected = ff * 1.22 * todFactor(camTimes[a])
      const skipped = route.cams.slice(a + 1, b)
      const mism = (s: string) => [...s].filter((ch, i) => ch !== plate[i]).length
      const unreadCount = unreadMasks[k - 1].filter(Boolean).length + unreadMasks[k].filter(Boolean).length
      const wrong = mism(reads[k - 1]) + mism(reads[k]) - unreadCount
      let plateLr = rng.uniform(10.2, 13.1) - 9.5 * Math.max(0, wrong) - 1.05 * unreadCount + rng.normal(0, 0.35)
      plateLr = Math.max(-2.2, plateLr)
      let appearance = rng.chance(0.08) ? rng.uniform(-0.6, 1.1) : rng.uniform(2.6, 5.9)
      const ratio = dt / expected
      const kin = Math.max(-4, 3.1 - 5.5 * Math.log(ratio) ** 2 - 1.25 * skipped.length + rng.normal(0, 0.2))
      const prior = -6.2 + rng.uniform(-0.25, 0.25)
      let total = plateLr + appearance + kin + prior
      if (total < 1.5) {
        const bump = 1.5 - total + rng.uniform(0.3, 1.2)
        appearance += bump
        total += bump
      }
      const eFrom = events[k - 1].event_id
      const eTo = events[k].event_id
      freeFlow.set(`${eFrom}>${eTo}`, ff)
      links.push({
        from_event_id: eFrom,
        to_event_id: eTo,
        plate_lr: +plateLr.toFixed(3),
        appearance_lr: +appearance.toFixed(3),
        kinematic_lr: +kin.toFixed(3),
        prior_log_odds: +prior.toFixed(3),
        total_log_odds: +total.toFixed(3),
        delta_t_s: +dt.toFixed(1),
        expected_t_s: +expected.toFixed(1),
        skipped_cameras: skipped,
      })
    }

    const consensus = this.consensus(plate, reads, unreadMasks)
    const plateConfidence = consensus.per_slot.reduce((acc, s) => acc * (s[0]?.prob ?? 1), 1)
    const times = idx.map((ci) => Math.round(camTimes[ci]))
    const traj: TrajectoryDetail = {
      trajectory_id: trajId,
      decoded_plate: plate,
      plate_confidence: +plateConfidence.toFixed(4),
      start_time: iso(times[0]),
      end_time: iso(times[times.length - 1]),
      n_events: events.length,
      camera_sequence: events.map((e) => e.camera_id),
      color,
      vehicle_type: vehicleType,
      has_alert: false,
      events,
      links,
      path,
      consensus,
    }
    return { traj, posteriors, freeFlow, times }
  }

  register(built: ReturnType<WorldBuilder['buildVehicle']>) {
    const { db } = this
    db.trajectories.set(built.traj.trajectory_id, built.traj)
    built.posteriors.forEach((v, k) => db.posterior.set(k, v))
    built.freeFlow.forEach((v, k) => db.linkFreeFlow.set(k, v))
    built.traj.events.forEach((e, i) => {
      db.events.push(e)
      db.eventTime.set(e.event_id, built.times[i])
    })
  }

  nextAlertId(): string {
    return `AL-${pad(this.alertSeq++, 5)}`
  }

  farCamera(from: string, minDist: number): string {
    const far = this.cameraIds().filter((c) => this.camDistance(from, c) >= minDist)
    return far.length ? this.rng.pick(far) : this.cameraIds()[this.cameraIds().length - 1]
  }

  /**
   * Plant a second vehicle carrying `source`'s plate, first sighted far away too soon after one of
   * source's sightings. Returns the clone and the alert (caller registers both).
   */
  buildClone(source: TrajectoryDetail, kind: 'clone' | 'impossible_travel', sightingIdx?: number) {
    const { rng } = this
    const ai = sightingIdx ?? Math.min(source.events.length - 1, Math.floor(source.events.length / 2))
    const a = source.path[ai]
    const startCam = this.farCamera(a.camera_id, 4800)
    const dist = this.camDistance(a.camera_id, startCam)
    const minRequired = dist / V_MAX_MS
    const dtS = minRequired * rng.uniform(0.22, 0.55)
    const endCam = this.farCamera(startCam, 2500)
    const route = this.planRoute(startCam, endCam)
    const otherColor = COLORS.filter((c) => c !== source.color)
    const built = this.buildVehicle({
      start: Date.parse(a.timestamp) + dtS * 1000,
      route: route.cams.length >= 2 ? route : this.randomRoute(),
      plate: source.decoded_plate,
      color: kind === 'clone' ? rng.pick(otherColor) : source.color,
      vehicleType: kind === 'clone' ? rng.pick(VEHICLE_TYPES.filter((t) => t !== source.vehicle_type && t !== 'bike')) : source.vehicle_type,
      noise: 0.01,
    })
    const b = built.traj.path[0]
    const appearance = kind === 'clone' ? rng.uniform(0.58, 0.81) : rng.uniform(0.16, 0.31)
    const km = (dist / 1000).toFixed(1)
    const dtRound = Math.round(dtS)
    const plateFmt = source.decoded_plate.slice(0, 2) + ' ' + source.decoded_plate.slice(2, 4) + ' ' + source.decoded_plate.slice(4, 6).replaceAll('_', '') + ' ' + source.decoded_plate.slice(6).replaceAll('_', '')
    const alert: Alert = {
      alert_id: this.nextAlertId(),
      type: kind,
      severity: kind === 'clone' ? 'high' : 'medium',
      created_at: b.timestamp,
      plate: source.decoded_plate,
      trajectory_ids: [source.trajectory_id, built.traj.trajectory_id],
      summary:
        kind === 'clone'
          ? `Plate ${plateFmt} seen ${km} km apart ${dtRound} s later (needs ≥ ${Math.round(minRequired)} s) on a visibly different ${built.traj.color} ${built.traj.vehicle_type} — likely cloned plate.`
          : `Plate ${plateFmt} seen ${km} km apart ${dtRound} s later (needs ≥ ${Math.round(minRequired)} s) with similar appearance — possible misread or timestamp fault.`,
      evidence: {
        distance_m: Math.round(dist),
        delta_t_s: +dtS.toFixed(1),
        min_required_s: +minRequired.toFixed(1),
        appearance_distance: +appearance.toFixed(3),
        points: [a, b],
      },
    }
    source.has_alert = true
    built.traj.has_alert = true
    return { built, alert }
  }
}

// ---------------------------------------------------------------- historical world

/** Showcase vehicle: plate misread at 3 of 6 cameras, one missed camera, fixed for the demo script. */
function plantShowcase(w: WorldBuilder) {
  const ids = w.cameraIds()
  // west -> east along Central Ave, then south
  const from = ids.find((id) => w.db.cameraRC.get(id)?.[0] === 4 && w.db.cameraRC.get(id)?.[1] === 0) as string
  const to = ids.find((id) => w.db.cameraRC.get(id)?.[0] === 8 && w.db.cameraRC.get(id)?.[1] === 8) as string
  const full = w.planRoute(from, to)
  // 7 cameras on the route, one missed -> exactly 6 reads, 3 of them misread.
  const hasSeven = full.cams.length >= 7
  const route = hasSeven ? { cams: full.cams.slice(0, 7), freeFlow: full.freeFlow.slice(0, 6) } : full
  const truth = 'MH12AB1234'
  const forced = ['MH12AB1234', 'NH12AB1234', 'MH12AB1234', 'MH12AR1234', 'MH12AB1284', 'MH12AB1234', 'MH12AB1234']
  const nObs = hasSeven ? 6 : route.cams.length
  const built = w.buildVehicle({
    start: Date.parse('2026-03-09T11:31:08Z'),
    route,
    plate: truth,
    color: 'white',
    vehicleType: 'car',
    noise: 0,
    observeAll: true,
    forcedReads: forced.slice(0, nObs),
    forceSkip: hasSeven ? [2] : [],
  })
  w.register(built)
}

export function buildHistoricalWorld(): WorldBuilder {
  const w = new WorldBuilder()
  const { rng } = w
  const span = HIST_END - HIST_START
  const N = 330
  for (let i = 0; i < N; i++) {
    // demand shaped toward the 09:00 peak
    let t: number
    do t = HIST_START + rng.next() * span
    while (!rng.chance(todFactor(t) / 1.5))
    w.register(w.buildVehicle({ start: t }))
  }
  plantShowcase(w)

  // Planted alerts
  const trajs = [...w.db.trajectories.values()].filter((t) => t.n_events >= 3 && !t.has_alert && t.decoded_plate !== 'MH12AB1234')
  const pickAt = (hour: number) =>
    trajs.filter((t) => !t.has_alert).reduce((best, t) => (Math.abs(Date.parse(t.start_time) - (HIST_START + hour * 3600e3)) < Math.abs(Date.parse(best.start_time) - (HIST_START + hour * 3600e3)) ? t : best))
  const planted: [number, 'clone' | 'impossible_travel'][] = [
    [3.2, 'clone'],
    [4.4, 'impossible_travel'],
    [5.6, 'clone'],
    [1.5, 'clone'],
    [2.6, 'impossible_travel'],
  ]
  for (const [hour, kind] of planted) {
    const { built, alert } = w.buildClone(pickAt(hour), kind)
    w.register(built)
    w.db.alerts.push(alert)
  }

  // Anomalies (low severity, no kinematic evidence)
  for (const [hour, text] of [
    [4.9, 'circled the Central Ave × Arsenal Rd block 4 times in 38 min'],
    [2.1, 'dwelled 47 min between consecutive cameras 900 m apart (expected ≈ 2 min)'],
  ] as const) {
    const t = pickAt(hour)
    t.has_alert = true
    const p = t.decoded_plate
    const plateFmt = `${p.slice(0, 2)} ${p.slice(2, 4)} ${p.slice(4, 6).replaceAll('_', '')} ${p.slice(6).replaceAll('_', '')}`
    w.db.alerts.push({
      alert_id: w.nextAlertId(),
      type: 'anomaly',
      severity: 'low',
      created_at: t.end_time,
      plate: p,
      trajectory_ids: [t.trajectory_id],
      summary: `Vehicle ${plateFmt} ${text}.`,
      evidence: { points: t.path.slice(0, 3) },
    })
  }

  // One link with non-finite channel values, so the UI's null handling is exercised in mock mode.
  const odd = trajs.find((t) => t.links.length >= 3 && !t.has_alert)
  if (odd) {
    // A non-finite channel makes the sum non-finite too, exactly as the engine would serialise it.
    odd.links[1] = { ...odd.links[1], appearance_lr: null, total_log_odds: null }
  }

  w.db.events.sort((a, b) => (w.db.eventTime.get(a.event_id) as number) - (w.db.eventTime.get(b.event_id) as number))
  w.db.alerts.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
  return w
}
