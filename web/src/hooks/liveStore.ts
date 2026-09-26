import { useSyncExternalStore } from 'react'
import { subscribeLive, type LiveStatus } from '../api/client'
import type { Alert, ClockData, EventSummary, LiveMessage, TrajectorySummary } from '../api/types'

export type LiveTrajectory = { summary: TrajectorySummary; updatedAt: number }

export type LiveState = {
  status: LiveStatus
  clock: ClockData | null
  /** newest first, capped */
  events: EventSummary[]
  /** newest first, capped */
  alerts: Alert[]
  /** most recently extended trajectories, newest first, capped */
  trajectories: LiveTrajectory[]
  eventsSinceLoad: number
  trajectoryUpdatesSinceLoad: number
  /** sliding window of arrival times (ms, wall clock) for an events/min readout */
  arrivals: number[]
}

const MAX_EVENTS = 120
const MAX_ALERTS = 50
const MAX_TRAJ = 60

/** How much sim time the client-side heat buffer keeps (the heatmap window can be at most this). */
export const HEAT_BUFFER_MIN = 60

/** A read at a camera, sim time (ms). */
export type HeatRead = { cam: string; t: number }
/** A trajectory extension from -> to, arriving at sim time t after dtS seconds. */
export type HeatLink = { from: string; to: string; t: number; dtS: number }

type PulseFn = (cameraId: string) => void
type AlertFn = (a: Alert) => void

/**
 * One socket for the whole app. Frames are applied to a mutable draft and published to React at
 * most every ~200 ms, so a 300x replay doesn't re-render the console on every frame.
 */
class LiveStore {
  private state: LiveState = {
    status: 'connecting',
    clock: null,
    events: [],
    alerts: [],
    trajectories: [],
    eventsSinceLoad: 0,
    trajectoryUpdatesSinceLoad: 0,
    arrivals: [],
  }
  private listeners = new Set<() => void>()
  private pulseListeners = new Set<PulseFn>()
  private alertListeners = new Set<AlertFn>()
  private flushTimer: ReturnType<typeof setTimeout> | null = null
  private unsub: (() => void) | null = null
  private refs = 0
  // Heat buffers live outside React state: they are sampled on a timer by the Live page.
  private heatReads: HeatRead[] = []
  private heatLinks: HeatLink[] = []
  private trajLast = new Map<string, { n: number; end: number }>()
  private lastSimMs = -Infinity

  /** Rolling buffer of reads and speed links for the client-side heatmap, plus current sim time. */
  getHeatBuffer = () => ({ reads: this.heatReads, links: this.heatLinks, simMs: this.lastSimMs })

  private pruneHeat(simMs: number) {
    const cut = simMs - HEAT_BUFFER_MIN * 60e3
    if (this.heatReads.length && this.heatReads[0].t < cut) this.heatReads = this.heatReads.filter((r) => r.t >= cut)
    if (this.heatLinks.length && this.heatLinks[0].t < cut) this.heatLinks = this.heatLinks.filter((r) => r.t >= cut)
    if (this.trajLast.size > 5000) for (const [id, v] of this.trajLast) if (v.end < cut) this.trajLast.delete(id)
  }

  private noteSim(simMs: number) {
    if (!Number.isFinite(simMs)) return
    // Replay reset (clock jumped back): anything buffered is from a discarded future.
    if (simMs < this.lastSimMs - 5_000) {
      this.heatReads = []
      this.heatLinks = []
      this.trajLast.clear()
    }
    this.lastSimMs = simMs
    this.pruneHeat(simMs)
  }

  getState = () => this.state

  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    if (this.refs++ === 0) this.connect()
    return () => {
      this.listeners.delete(fn)
      if (--this.refs === 0) {
        this.unsub?.()
        this.unsub = null
      }
    }
  }

  onPulse(fn: PulseFn) {
    this.pulseListeners.add(fn)
    return () => void this.pulseListeners.delete(fn)
  }

  onAlert(fn: AlertFn) {
    this.alertListeners.add(fn)
    return () => void this.alertListeners.delete(fn)
  }

  /** Apply a replay response immediately rather than waiting for the next clock frame. */
  setClock(clock: ClockData) {
    this.state = { ...this.state, clock }
    this.publish()
  }

  private connect() {
    this.unsub = subscribeLive(
      (m) => this.apply(m),
      (status) => {
        this.state = { ...this.state, status }
        this.publish()
      },
    )
  }

  private apply(m: LiveMessage) {
    const s = this.state
    switch (m.type) {
      case 'clock':
        this.noteSim(Date.parse(m.data.sim_time))
        this.state = { ...s, clock: m.data }
        break
      case 'event': {
        const now = Date.now()
        const arrivals = s.arrivals.filter((t) => t > now - 60_000)
        arrivals.push(now)
        this.state = {
          ...s,
          events: [m.data, ...s.events].slice(0, MAX_EVENTS),
          eventsSinceLoad: s.eventsSinceLoad + 1,
          arrivals,
        }
        this.pulseListeners.forEach((fn) => fn(m.data.camera_id))
        const t = Date.parse(m.data.timestamp)
        if (Number.isFinite(t)) {
          this.heatReads.push({ cam: m.data.camera_id, t })
          if (t > this.lastSimMs) this.lastSimMs = t
        }
        break
      }
      case 'trajectory': {
        // A link's travel time needs the previous read's time: the last frame we saw for this trajectory,
        // or start_time when this is its first link. Otherwise (joined mid-way) skip it.
        const d = m.data
        const end = Date.parse(d.end_time)
        const prev = this.trajLast.get(d.trajectory_id)
        const prevT = prev && prev.n === d.n_events - 1 ? prev.end : d.n_events === 2 ? Date.parse(d.start_time) : NaN
        const seq = d.camera_sequence
        if (Number.isFinite(prevT) && Number.isFinite(end) && seq.length >= 2 && (!prev || prev.n < d.n_events)) {
          this.heatLinks.push({ from: seq[seq.length - 2], to: seq[seq.length - 1], t: end, dtS: (end - prevT) / 1000 })
        }
        if (Number.isFinite(end)) this.trajLast.set(d.trajectory_id, { n: d.n_events, end })
        const rest = s.trajectories.filter((t) => t.summary.trajectory_id !== m.data.trajectory_id)
        this.state = {
          ...s,
          trajectories: [{ summary: m.data, updatedAt: Date.now() }, ...rest].slice(0, MAX_TRAJ),
          trajectoryUpdatesSinceLoad: s.trajectoryUpdatesSinceLoad + 1,
        }
        break
      }
      case 'alert':
        this.state = { ...s, alerts: [m.data, ...s.alerts.filter((a) => a.alert_id !== m.data.alert_id)].slice(0, MAX_ALERTS) }
        this.alertListeners.forEach((fn) => fn(m.data))
        break
    }
    this.schedule()
  }

  private schedule() {
    if (this.flushTimer) return
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null
      this.publish()
    }, 200)
  }

  private publish() {
    this.listeners.forEach((fn) => fn())
  }
}

export const liveStore = new LiveStore()

export function useLive<T>(selector: (s: LiveState) => T): T {
  return useSyncExternalStore(liveStore.subscribe, () => selector(liveStore.getState()))
}
