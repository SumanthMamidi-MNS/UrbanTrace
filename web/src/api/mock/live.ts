/**
 * Simulated /ws/live + POST /api/replay. A sim clock advances on a real-time timer; new vehicles
 * are spawned at the clock, their sightings revealed as sim time passes, emitting exactly the
 * contract's `event` / `trajectory` / `alert` / `clock` frames.
 */
import type { Alert, LiveMessage, ReplayAction, ReplayState, TrajectoryDetail } from '../types'
import { HIST_END, type WorldBuilder } from './world'

type Pending = {
  full: TrajectoryDetail
  times: number[]
  revealed: number
  alert?: { alert: Alert; atEventIdx: number }
}

const TICK_MS = 500
/** vehicles entering the camera network per sim-minute */
const SPAWN_PER_SIM_MIN = 3.2
const MAX_SPAWN_PER_TICK = 3
const MAX_ACTIVE = 120

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
        this.active.push({ full: built.traj, times: built.times, revealed: 0, alert: { alert, atEventIdx: 0 } })
        this.stash(built)
        continue
      }
      const built = this.w.buildVehicle({ start })
      this.active.push({ full: built.traj, times: built.times, revealed: 0 })
      this.stash(built)
    }
  }

  /** Keep posteriors / free-flow so detail endpoints work as soon as a trajectory is visible. */
  private stash(built: ReturnType<WorldBuilder['buildVehicle']>) {
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
        if (k >= 1) {
          const partial = this.truncate(p.full, p.revealed)
          db.trajectories.set(partial.trajectory_id, partial)
          db.liveTrajectoryIds.add(partial.trajectory_id)
          const { events: _e, links: _l, path: _p, consensus: _c, ...summary } = partial
          this.emit({ type: 'trajectory', data: summary })
        }
        if (p.alert && p.alert.atEventIdx === k) {
          db.alerts.unshift(p.alert.alert)
          this.emit({ type: 'alert', data: p.alert.alert })
        }
      }
      if (p.revealed >= p.full.events.length) done.push(p)
    }
    if (done.length) this.active = this.active.filter((p) => !done.includes(p))
    return emitted
  }

  private truncate(t: TrajectoryDetail, k: number): TrajectoryDetail {
    const events = t.events.slice(0, k)
    return {
      ...t,
      events,
      links: t.links.slice(0, k - 1),
      path: t.path.slice(0, k),
      n_events: k,
      end_time: events[k - 1].timestamp,
      camera_sequence: events.map((e) => e.camera_id),
      consensus: { ...t.consensus, single_read_plates: t.consensus.single_read_plates.slice(0, k) },
    }
  }
}
