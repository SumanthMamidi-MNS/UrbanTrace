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
        break
      }
      case 'trajectory': {
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
