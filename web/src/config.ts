/** Runtime configuration. Flipping VITE_USE_MOCK is the only change needed to hit the real API. */
export const USE_MOCK: boolean = (import.meta.env.VITE_USE_MOCK ?? 'true').toLowerCase() !== 'false'

export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/$/, '')

export const WS_URL: string = API_BASE.replace(/^http/, 'ws') + '/ws/live'
