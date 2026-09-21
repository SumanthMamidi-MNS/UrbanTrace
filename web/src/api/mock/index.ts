import type { LiveMessage } from '../types'
import { LiveSim } from './live'
import { createMockApi } from './server'
import { buildHistoricalWorld } from './world'

const world = buildHistoricalWorld()
const live = new LiveSim(world)

export const mockApi = createMockApi(world.db, live)

export function subscribeMockLive(fn: (m: LiveMessage) => void): () => void {
  return live.subscribe(fn)
}
