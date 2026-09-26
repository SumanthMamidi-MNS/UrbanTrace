/**
 * Simulated /ws/live + POST /api/replay. A sim clock advances on a real-time timer; new vehicles
 * are spawned at the clock, their sightings revealed as sim time passes, emitting exactly the
 * contract's `event` / `trajectory` / `alert` / `clock` frames.
 *
 * v2: every revealed read is checked against the watchlist twice (its own posterior, then the
 * trajectory's fused consensus). Adding an entry also plants two target vehicles so the demo always
 * produces hits: a "dirty plate" that every camera misreads (caught only by trajectory consensus) and
 * a clean one (caught on a single read).
 */
import type { Alert, LiveMessage, PathPoint, ReplayAction, ReplayState, TrajectoryDetail, WatchlistEntry, WatchlistHit, WatchlistMatchedOn } from '../types'
import { formatPlate } from '../../lib/plate'
import { matchProbability, WATCHLIST_MATCH_THRESHOLD } from './watchlist'
import { HIST_END, withHeadings, type WorldBuilder } from './world'

type Built = ReturnType<WorldBuilder['buildVehicle']>

type Pending = {
  full: TrajectoryDetail
  times: number[]
  revealed: number
  truth: string
  reads: string[]
  unreadMasks: boolean[][]
  alert?: { alert: Alert; atEventIdx: number }
}

const TICK_MS = 500
/** vehicles entering the camera network per sim-minute */
const SPAWN_PER_SIM_MIN = 3.2
const MAX_SPAWN_PER_TICK = 3
const MAX_ACTIVE = 120

const LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
const DIGITS = '0123456789'
const pad = (n: number, w: number) => String(n).padStart(w, '0')

export class LiveSim {
  private running = true
  private speed = 10
  private simMs = HIST_END
  private active: Pending[] = []
  private listeners = new Set<(m: LiveMessage) => void>()
  private timer: ReturnType<typeof setInterval> | null = null
  private tickN = 0
  private spawnN = 0
  private spawnDebt = 0
  private hitSeq = 1
  /** "entryId|fullTrajectoryId" pairs that already alerted (each pair alerts at most once) */
  private alerted = new Set<string>()
  private readonly w: WorldBuilder

  constructor(w: WorldBuilder) {
    this.w = w
  }

  simTimeMs(): number {
    return this.simMs
  }

  state(): ReplayState {
    return { running: this.running, speed: this.speed, sim_time: new Date(this.simMs).toISOString() }
  }

  subscribe(fn: (m: LiveMessage) => void): () => void {
    this.listeners.add(fn)
    if (!this.timer) this.timer = setInterval(() => this.tick(), TICK_MS)
    fn({ type: 'clock', data: this.state() })
    return () => {
      this.listeners.delete(fn)
      if (this.listeners.size === 0 && this.timer) {
        clearInterval(this.timer)
        this.timer = null
      }
    }
  }

  control(action: ReplayAction, speed?: number): ReplayState {
    if (speed !== undefined && Number.isFinite(speed) && speed > 0) this.speed = speed
    if (action === 'start') this.running = true
    if (action === 'pause') this.running = false
    if (action === 'reset') this.reset()
    this.emit({ type: 'clock', data: this.state() })
    return this.state()
  }

  /** Called by POST /api/watchlist: schedule target vehicles for this entry. */
  plantWatchlistTargets(entry: WatchlistEntry) {
    const { rng } = this.w
    const form = rng.pick(entry.canonical_patterns)
    const truth = [...form].map((c, s) => (c !== '?' ? c : s <= 1 || s === 4 || s === 5 ? rng.pick([...LETTERS]) : rng.pick([...DIGITS]))).join('')

    // A: dirty plate. Every camera misreads one (different) literal character, so no single read
    // reaches the threshold, but the fused plate does once enough cameras agree on each slot.
    let route = this.w.randomRoute(6)
    for (let i = 0; i < 20 && route.cams.length < 5; i++) route = this.w.randomRoute(6)
    const literal = rng.shuffle([...Array(10).keys()].filter((s) => form[s] !== '?' && form[s] !== '_'))
    const forcedReads = route.cams.map((_, i) => {
      const s = literal[i % literal.length]
      const chars = [...truth]
      chars[s] = this.w.confuse(truth[s], s)
      return chars.join('')
    })
    this.enqueue(this.w.buildVehicle({ start: this.simMs + rng.uniform(40, 90) * 1000, route, plate: truth, noise: 0, observeAll: true, forcedReads }))

    // B: clean, high-confidence reads later on, caught on a single read.
    const clean = this.w.buildVehicle({ start: this.simMs + rng.uniform(8, 11) * 60_000, route: this.w.randomRoute(3), plate: truth, noise: 0, observeAll: true })
    clean.posteriors.forEach((post, id) => {
      for (const slot of post) {
        const pTop = rng.uniform(0.97, 0.995)
        const restOld = slot.top.slice(1).reduce((a, r) => a + r.prob, 0) || 1
        slot.top = slot.top.map((r, i) => ({ char: r.char, prob: +(i === 0 ? pTop : ((1 - pTop) * r.prob) / restOld).toFixed(4) }))
      }
      const ev = clean.traj.events.find((e) => e.event_id === id)
      if (ev) ev.plate_confidence = +(post.reduce((a, s) => a + s.top[0].prob, 0) / post.length).toFixed(3)
    })
    this.enqueue(clean)
  }

  private enqueue(built: Built) {
    this.active.push({ full: built.traj, times: built.times, revealed: 0, truth: built.truth, reads: built.reads, unreadMasks: built.unreadMasks })
    this.stash(built)
  }

  private reset() {
    const { db } = this.w
    this.running = false
    this.simMs = HIST_END
    this.active = []
    db.liveTrajectoryIds.forEach((id) => db.trajectories.delete(id))
    db.liveTrajectoryIds.clear()
    db.events = db.events.filter((e) => !db.liveEventIds.has(e.event_id))
    db.liveEventIds.forEach((id) => {
      db.eventTime.delete(id)
      db.posterior.delete(id)
    })
    db.liveEventIds.clear()
    db.alerts = db.alerts.filter((a) => Date.parse(a.created_at) <= HIST_END)
    // Watchlist entries persist; hits from the discarded replay go, and targets are re-planted.
    db.watchHits = db.watchHits.filter((h) => Date.parse(h.timestamp) <= HIST_END)
    this.alerted.clear()
    for (const e of db.watchlist) {
      e.hits = db.watchHits.filter((h) => h.entry_id === e.entry_id).length
      if (e.active) this.plantWatchlistTargets(e)
    }
  }

  private emit(m: LiveMessage) {
    this.listeners.forEach((fn) => fn(m))
  }

  private tick() {
    this.tickN++
    if (this.running) {
      const advance = this.speed * TICK_MS
      this.simMs += advance
      this.spawn(advance)
      this.reveal()
    }
    if (this.tickN % 2 === 0) this.emit({ type: 'clock', data: this.state() })
  }

  private spawn(advanceMs: number) {
    const { rng } = this.w
    this.spawnDebt += (SPAWN_PER_SIM_MIN * advanceMs) / 60000
    let n = Math.floor(this.spawnDebt)
    this.spawnDebt -= n
    n = Math.min(n, MAX_SPAWN_PER_TICK)
    if (this.spawnDebt > 3) this.spawnDebt = 0 // drop backlog at very high replay speeds
    for (let i = 0; i < n && this.active.length < MAX_ACTIVE; i++) {
      this.spawnN++
      const start = this.simMs + rng.uniform(0, advanceMs)
      // Every 9th vehicle (starting with the 4th) is a clone of a vehicle already on the network.
      const source = this.active.find((p) => p.revealed >= 1 && p.revealed < p.full.events.length - 1 && !p.full.has_alert)
      if (this.spawnN % 9 === 4 && source) {
        const idx = source.revealed
        const { built, alert } = this.w.buildClone(source.full, 'clone', idx)
        // clone sighting B happens after source sighting idx; emit alert when clone's first read lands
        this.enqueue(built)
        this.active[this.active.length - 1].alert = { alert, atEventIdx: 0 }
        continue
      }
      this.enqueue(this.w.buildVehicle({ start }))
    }
  }

  /** Keep posteriors / free-flow so detail endpoints work as soon as a trajectory is visible. */
  private stash(built: Built) {
    const { db } = this.w
    built.posteriors.forEach((v, k) => db.posterior.set(k, v))
    built.freeFlow.forEach((v, k) => db.linkFreeFlow.set(k, v))
  }

  private reveal() {
    const { db } = this.w
    const done: Pending[] = []
    let emitted = 0
    for (const p of this.active) {
      while (p.revealed < p.full.events.length && p.times[p.revealed] <= this.simMs) {
        const k = p.revealed
        const ev = p.full.events[k]
        const t = p.times[k]
        p.revealed++
        // First sighting is not yet linked to anything.
        const evOut = { ...ev, trajectory_id: k === 0 ? null : p.full.trajectory_id }
        db.events.push(evOut)
        db.eventTime.set(ev.event_id, t)
        db.liveEventIds.add(ev.event_id)
        this.emit({ type: 'event', data: evOut })
        emitted++
        let partial: TrajectoryDetail | null = null
        if (k >= 1) {
          partial = this.truncate(p, p.revealed)
          db.trajectories.set(partial.trajectory_id, partial)
          db.liveTrajectoryIds.add(partial.trajectory_id)
          const { events: _e, links: _l, path: _p, consensus: _c, overall_heading_deg: _h, direction_label: _d, ...summary } = partial
          this.emit({ type: 'trajectory', data: summary })
        }
        if (p.alert && p.alert.atEventIdx === k) {
          db.alerts.unshift(p.alert.alert)
          this.emit({ type: 'alert', data: p.alert.alert })
        }
        this.checkWatchlist(p, k, evOut.trajectory_id, partial)
      }
      if (p.revealed >= p.full.events.length) done.push(p)
    }
    if (done.length) this.active = this.active.filter((p) => !done.includes(p))
    return emitted
  }

  private checkWatchlist(p: Pending, k: number, trajectoryId: string | null, partial: TrajectoryDetail | null) {
    const { db } = this.w
    if (!db.watchlist.length) return
    const ev = p.full.events[k]
    const post = db.posterior.get(ev.event_id)
    for (const entry of db.watchlist) {
      if (!entry.active) continue
      const pSingle = post ? matchProbability(post.map((s) => s.top), entry.canonical_patterns) : 0
      const pCons = partial ? matchProbability(partial.consensus.per_slot, entry.canonical_patterns) : 0
      let matchedOn: WatchlistMatchedOn
      let prob: number
      if (pSingle >= WATCHLIST_MATCH_THRESHOLD) {
        matchedOn = 'single_read'
        prob = pSingle
      } else if (pCons >= WATCHLIST_MATCH_THRESHOLD) {
        matchedOn = 'trajectory_consensus'
        prob = pCons
      } else continue

      const hit: WatchlistHit = {
        hit_id: `WH-${pad(this.hitSeq++, 6)}`,
        entry_id: entry.entry_id,
        pattern: entry.pattern,
        event_id: ev.event_id,
        trajectory_id: trajectoryId,
        camera_id: ev.camera_id,
        timestamp: ev.timestamp,
        probability: +prob.toFixed(4),
        matched_on: matchedOn,
        plate_read: ev.plate_argmax,
      }
      db.watchHits.push(hit)
      entry.hits++

      const key = `${entry.entry_id}|${p.full.trajectory_id}`
      if (this.alerted.has(key)) continue
      this.alerted.add(key)
      const points: PathPoint[] = partial ? partial.path : withHeadings([p.full.path[k]]).path
      const pct = `${Math.round(prob * 100)}%`
      const reason = entry.reason ? ` Reason: ${entry.reason}.` : ''
      const summary =
        matchedOn === 'trajectory_consensus' && partial
          ? `Watchlist ${entry.pattern}: ${ev.camera_id} read ${formatPlate(ev.plate_argmax)}, but the fused plate across ${partial.n_events} cameras (${formatPlate(partial.decoded_plate)}) matches at ${pct}.${reason}`
          : `Watchlist ${entry.pattern}: ${ev.camera_id} read ${formatPlate(ev.plate_argmax)}, matching at ${pct} on a single read.${reason}`
      const alert: Alert = {
        alert_id: this.w.nextAlertId(),
        type: 'watchlist',
        severity: 'high',
        created_at: ev.timestamp,
        plate: matchedOn === 'trajectory_consensus' && partial ? partial.decoded_plate : ev.plate_argmax,
        trajectory_ids: trajectoryId ? [trajectoryId] : [],
        summary,
        evidence: { points, watchlist_entry_id: entry.entry_id, pattern: entry.pattern, match_probability: hit.probability, matched_on: matchedOn },
      }
      db.alerts.unshift(alert)
      this.emit({ type: 'alert', data: alert })
    }
  }

  /** The trajectory as the engine would know it after k reads: partial path, links and consensus. */
  private truncate(p: Pending, k: number): TrajectoryDetail {
    const t = p.full
    const events = t.events.slice(0, k)
    const dir = withHeadings(t.path.slice(0, k))
    const consensus = this.w.consensus(p.truth, p.reads.slice(0, k), p.unreadMasks.slice(0, k))
    const plateConfidence = consensus.per_slot.reduce((acc, s) => acc * (s[0]?.prob ?? 1), 1)
    return {
      ...t,
      events,
      links: t.links.slice(0, k - 1),
      path: dir.path,
      overall_heading_deg: dir.overall_heading_deg,
      direction_label: dir.direction_label,
      n_events: k,
      end_time: events[k - 1].timestamp,
      camera_sequence: events.map((e) => e.camera_id),
      plate_confidence: +plateConfidence.toFixed(4),
      consensus,
    }
  }
}
