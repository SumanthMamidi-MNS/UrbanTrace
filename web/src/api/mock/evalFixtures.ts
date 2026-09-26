/**
 * Mock GET /api/eval payload: byte-identical copies of eval/reports/*.json (copied into web/ so the UI
 * build never reaches outside its folder), keyed by file stem exactly as the API returns them.
 * Refresh with: cp ../eval/reports/*.json src/api/mock/eval/
 */
const files = import.meta.glob<unknown>('./eval/*.json', { eager: true, import: 'default' })

export const EVAL_REPORTS: Record<string, unknown> = Object.fromEntries(
  Object.entries(files)
    .map(([path, data]) => [path.replace(/^.*\/(.+)\.json$/, '$1'), data] as const)
    .sort(([a], [b]) => a.localeCompare(b)),
)
