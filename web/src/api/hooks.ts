import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type {
  AlertsQuery,
  CorridorsQuery,
  EventsQuery,
  ReplayRequest,
  SearchQuery,
  TrajectoriesQuery,
  VolumesQuery,
} from './types'

export const qk = {
  health: ['health'] as const,
  city: ['city'] as const,
  cameras: ['cameras'] as const,
  events: (q: EventsQuery) => ['events', q] as const,
  event: (id: string) => ['event', id] as const,
  trajectories: (q: TrajectoriesQuery) => ['trajectories', q] as const,
  trajectory: (id: string) => ['trajectory', id] as const,
  search: (q: SearchQuery) => ['search', q] as const,
  summary: ['analytics', 'summary'] as const,
  volumes: (q: VolumesQuery) => ['analytics', 'volumes', q] as const,
  od: ['analytics', 'od'] as const,
  corridors: (q: CorridorsQuery) => ['analytics', 'corridors', q] as const,
  alerts: (q: AlertsQuery) => ['alerts', q] as const,
  eval: ['eval'] as const,
}

export const useHealth = () => useQuery({ queryKey: qk.health, queryFn: () => api.health(), refetchInterval: 15_000 })

/** The city graph is static for a session. */
export const useCity = () => useQuery({ queryKey: qk.city, queryFn: () => api.city(), staleTime: Infinity })

export const useCameras = (refetchMs = 10_000) =>
  useQuery({ queryKey: qk.cameras, queryFn: () => api.cameras(), refetchInterval: refetchMs })

export const useEvents = (q: EventsQuery) => useQuery({ queryKey: qk.events(q), queryFn: () => api.events(q) })

export const useEventDetail = (id: string | undefined) =>
  useQuery({ queryKey: qk.event(id ?? ''), queryFn: () => api.event(id as string), enabled: Boolean(id) })

export const useTrajectories = (q: TrajectoriesQuery) =>
  useQuery({ queryKey: qk.trajectories(q), queryFn: () => api.trajectories(q), placeholderData: keepPreviousData })

export const useTrajectory = (id: string | undefined) =>
  useQuery({ queryKey: qk.trajectory(id ?? ''), queryFn: () => api.trajectory(id as string), enabled: Boolean(id) })

export const useSearch = (q: SearchQuery | null) =>
  useQuery({
    queryKey: qk.search(q ?? { q: '' }),
    queryFn: () => api.search(q as SearchQuery),
    enabled: Boolean(q?.q),
    placeholderData: keepPreviousData,
  })

export const useSummary = () =>
  useQuery({ queryKey: qk.summary, queryFn: () => api.analyticsSummary(), refetchInterval: 15_000 })

export const useVolumes = (q: VolumesQuery) =>
  useQuery({ queryKey: qk.volumes(q), queryFn: () => api.volumes(q), placeholderData: keepPreviousData })

export const useOdMatrix = () => useQuery({ queryKey: qk.od, queryFn: () => api.odMatrix() })

export const useCorridors = (q: CorridorsQuery) => useQuery({ queryKey: qk.corridors(q), queryFn: () => api.corridors(q) })

export const useAlerts = (q: AlertsQuery = {}) => useQuery({ queryKey: qk.alerts(q), queryFn: () => api.alerts(q) })

export const useEval = () => useQuery({ queryKey: qk.eval, queryFn: () => api.evalReports() })

export function useReplay() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: ReplayRequest) => api.replay(body),
    onSuccess: (_state, body) => {
      if (body.action === 'reset') void qc.invalidateQueries()
    },
  })
}
