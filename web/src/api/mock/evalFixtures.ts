/**
 * Mock GET /api/eval payload: verbatim copies of eval/reports/*.json (copied into web/ so the UI
 * build never reaches outside its folder), keyed by file stem exactly as the API returns them.
 * trajectory_metrics / baselines are not produced yet, so the Results page must show empty states.
 */
import appearance_scaling from './eval/appearance_scaling.json'
import blocking from './eval/blocking.json'
import clone_overlap_auc from './eval/clone_overlap_auc.json'
import gating from './eval/gating.json'
import stratified_auc from './eval/stratified_auc.json'

export const EVAL_REPORTS: Record<string, unknown> = {
  stratified_auc,
  clone_overlap_auc,
  gating,
  blocking,
  appearance_scaling,
}
