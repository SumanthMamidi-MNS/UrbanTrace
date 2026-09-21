import { API_BASE, USE_MOCK, WS_URL } from '../config'
import type {
  Alert,
  AlertsQuery,
  AnalyticsSummary,
  CameraStats,
  City,
  Corridor,
  CorridorsQuery,
  EvalReports,
  EventDetail,
  EventSummary,
  EventsQuery,
  Health,
  LiveMessage,
  OdMatrix,
  Page,
  ReplayRequest,
  ReplayState,
  SearchHit,
  SearchQuery,
  TrajectoriesQuery,
  TrajectoryDetail,
  TrajectorySummary,
  VolumeBucket,
  VolumesQuery,
} from './types'

/** Every REST endpoint in docs/api-contract.md, one method each. */
export interface SutraApi {
  health(): Promise<Health>
  city(): Promise<City>
  cameras(): Promise<CameraStats[]>
  events(q?: EventsQuery): Promise<Page<EventSummary>>
  event(eventId: string): Promise<EventDetail>
  trajectories(q?: TrajectoriesQuery): Promise<Page<TrajectorySummary>>
  trajectory(trajectoryId: string): Promise<TrajectoryDetail>
  search(q: SearchQuery): Promise<SearchHit[]>
  analyticsSummary(): Promise<AnalyticsSummary>
  volumes(q?: VolumesQuery): Promise<VolumeBucket[]>
  odMatrix(): Promise<OdMatrix>
  corridors(q?: CorridorsQuery): Promise<Corridor[]>
  alerts(q?: AlertsQuery): Promise<Alert[]>
  evalReports(): Promise<EvalReports>
  replay(body: ReplayRequest): Promise<ReplayState>
}

export type LiveStatus = 'connecting' | 'open' | 'closed'
export type LiveHandler = (msg: LiveMessage) => void
export type LiveStatusHandler = (status: LiveStatus) => void

/** Error carrying the contract's `{ detail }` body. */
export class ApiError extends Error {
  readonly status: number
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.name = 'ApiError'
  }
}

type QueryValue = string | number | boolean | undefined | null

function qs(params?: Record<string, QueryValue>): string {
  if (!params) return ''
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === '') continue
    sp.set(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}) },
    })
  } catch {
    throw new ApiError(0, `Cannot reach API at ${API_BASE}`)
  }
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

const enc = encodeURIComponent

const httpApi: SutraApi = {
  health: () => request('/api/health'),
  city: () => request('/api/city'),
  cameras: () => request('/api/cameras'),
  events: (q) => request(`/api/events${qs(q)}`),
  event: (id) => request(`/api/events/${enc(id)}`),
  trajectories: (q) => request(`/api/trajectories${qs(q)}`),
  trajectory: (id) => request(`/api/trajectories/${enc(id)}`),
  search: (q) => request(`/api/search${qs(q)}`),
  analyticsSummary: () => request('/api/analytics/summary'),
  volumes: (q) => request(`/api/analytics/volumes${qs(q)}`),
  odMatrix: () => request('/api/analytics/od_matrix'),
  corridors: (q) => request(`/api/analytics/corridors${qs(q)}`),
  alerts: (q) => request(`/api/alerts${qs(q)}`),
  evalReports: () => request('/api/eval'),
  replay: (body) => request('/api/replay', { method: 'POST', body: JSON.stringify(body) }),
}

/** Real /ws/live connection with exponential-backoff reconnect. */
function httpSubscribe(onMessage: LiveHandler, onStatus?: LiveStatusHandler): () => void {
  let ws: WebSocket | null = null
  let stopped = false
  let retry = 0
  let timer: ReturnType<typeof setTimeout> | undefined

  const connect = () => {
    if (stopped) return
    onStatus?.('connecting')
    ws = new WebSocket(WS_URL)
    ws.onopen = () => {
      retry = 0
      onStatus?.('open')
    }
    ws.onmessage = (ev) => {
      try {
        onMessage(JSON.parse(String(ev.data)) as LiveMessage)
      } catch {
        /* ignore malformed frame */
      }
    }
    ws.onclose = () => {
      onStatus?.('closed')
      if (stopped) return
      const delay = Math.min(10_000, 500 * 2 ** retry++)
      timer = setTimeout(connect, delay)
    }
    ws.onerror = () => ws?.close()
  }
  connect()
  return () => {
    stopped = true
    clearTimeout(timer)
    ws?.close()
  }
}

// ---- Mock wiring (lazy-loaded so the real build never ships fixture generation on the hot path) ----

type MockModule = typeof import('./mock')
let mockModule: Promise<MockModule> | null = null
const loadMock = () => (mockModule ??= import('./mock'))

const mockApi: SutraApi = new Proxy({} as SutraApi, {
  get(_t, prop: string) {
    return async (...args: unknown[]) => {
      const m = await loadMock()
      const fn = (m.mockApi as unknown as Record<string, (...a: unknown[]) => Promise<unknown>>)[prop]
      return fn(...args)
    }
  },
})

function mockSubscribe(onMessage: LiveHandler, onStatus?: LiveStatusHandler): () => void {
  let unsub: (() => void) | null = null
  let cancelled = false
  onStatus?.('connecting')
  void loadMock().then((m) => {
    if (cancelled) return
    unsub = m.subscribeMockLive(onMessage)
    onStatus?.('open')
  })
  return () => {
    cancelled = true
    unsub?.()
  }
}

export const api: SutraApi = USE_MOCK ? mockApi : httpApi

export const subscribeLive: (onMessage: LiveHandler, onStatus?: LiveStatusHandler) => () => void = USE_MOCK
  ? mockSubscribe
  : httpSubscribe
