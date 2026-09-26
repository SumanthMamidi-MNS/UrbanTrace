import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { cx, Panel } from '../../components/ui'
import { fmtNum, titleCase } from '../../lib/format'
import { Description, Methodology, Tile } from './bits'
import { CHART, dec, isObj, num, objs, pct, str, type Obj } from './shared'

// ------------------------------------------------------------------ detector

const DETECTOR_CFG: Record<string, string> = {
  pretrained_imgsz640: 'Pretrained · 640 px',
  pretrained_imgsz1280: 'Pretrained · 1280 px',
  finetuned_imgsz640: 'Fine-tuned · 640 px',
  finetuned_imgsz1280: 'Fine-tuned · 1280 px',
}

function MetricRow({ label, m, hot }: { label: string; m: Obj; hot?: boolean }) {
  const p = num(m.precision) ?? num(m.precision_at_conf_threshold)
  const r = num(m.recall) ?? num(m.recall_at_conf_threshold)
  return (
    <tr className={cx('border-t border-ink-750', hot && 'bg-accent-faint/30')}>
      <th scope="row" className={cx('py-1.5 pr-2 text-left font-medium', hot ? 'text-fg-strong' : 'text-fg')}>
        {label}
      </th>
      <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{dec(p)}</td>
      <td className={cx('num px-2 py-1.5 text-right font-mono', hot ? 'font-semibold text-accent-strong' : 'text-fg')}>{dec(r)}</td>
      <td className="num px-2 py-1.5 text-right font-mono text-fg">{dec(num(m.map50))}</td>
      <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{m.map50_95 !== undefined ? dec(num(m.map50_95)) : ''}</td>
    </tr>
  )
}

function MetricHead({ first, with9595 }: { first: string; with9595?: boolean }) {
  return (
    <thead>
      <tr className="text-[10px] tracking-wider text-fg-dim uppercase">
        <th className="py-1 text-left font-semibold">{first}</th>
        <th className="px-2 py-1 text-right font-semibold">Precision</th>
        <th className="px-2 py-1 text-right font-semibold">Recall</th>
        <th className="px-2 py-1 text-right font-semibold">mAP50</th>
        <th className="px-2 py-1 text-right font-semibold">{with9595 ? 'mAP50-95' : ''}</th>
      </tr>
    </thead>
  )
}

export function DetectorSection({ holdout, evalR }: { holdout?: Obj; evalR?: Obj }) {
  if (!holdout && !evalR) return null
  const res = holdout && isObj(holdout.results) ? holdout.results : {}
  const pre = isObj(res.pretrained_imgsz640) ? res.pretrained_imgsz640 : undefined
  const ft = isObj(res.finetuned_imgsz640) ? res.finetuned_imgsz640 : undefined
  const train = holdout && isObj(holdout.training_time_validation_for_comparison) ? holdout.training_time_validation_for_comparison : undefined
  const steps = evalR && isObj(evalR.steps) ? Object.entries(evalR.steps).filter(([, v]) => isObj(v)) : []
  const xml = evalR && isObj(evalR.indian_vehicle_xml_rgb_fixed) ? evalR.indian_vehicle_xml_rgb_fixed : undefined
  const rgb = evalR && isObj(evalR.rgb_bug) ? evalR.rgb_bug : undefined

  return (
    <Panel title="Plate detector">
      <div className="grid grid-cols-1 gap-4 p-3 xl:grid-cols-2">
        {holdout && (
          <div className="flex flex-col gap-3">
            <div className="text-[13px] font-semibold text-fg-strong">Held-out video (never used in fine-tuning)</div>
            {pre && ft && (
              <div className="grid grid-cols-2 gap-3">
                <Tile label="Recall, pretrained" value={dec(num(pre.recall))} sub="640 px" tone="muted" />
                <Tile label="Recall, fine-tuned" value={dec(num(ft.recall))} sub="640 px" tone="accent" />
              </div>
            )}
            <table className="w-full text-xs">
              <caption className="sr-only">Detector metrics on the held-out video</caption>
              <MetricHead first="Configuration" with9595 />
              <tbody>
                {Object.entries(res)
                  .filter(([, v]) => isObj(v))
                  .map(([k, v]) => (
                    <MetricRow key={k} label={DETECTOR_CFG[k] ?? titleCase(k)} m={v as Obj} hot={k === 'finetuned_imgsz640'} />
                  ))}
              </tbody>
            </table>
            <p className="text-[11px] text-fg-dim">
              {fmtNum(num(holdout.n_frames))} frames, {fmtNum(num(holdout.n_gt_boxes))} plate boxes: a small test set, so read these as indicative.
              {train && ` Training-time validation showed recall ${dec(num(train.recall))} / mAP50 ${dec(num(train.map50))}, but on near-duplicate frames of the training videos, so it is optimistic.`}
            </p>
            <Methodology text={str(holdout.description)} />
          </div>
        )}
        {evalR && (
          <div className="flex flex-col gap-3">
            <div className="text-[13px] font-semibold text-fg-strong">Pretrained detector, by dataset and inference setting</div>
            {xml && (
              <table className="w-full text-xs">
                <caption className="sr-only">Detector on indian_vehicle_xml</caption>
                <MetricHead first={`Indian vehicle XML · ${fmtNum(num(xml.n_images))} images`} />
                <tbody>
                  <MetricRow label="After the RGB fix" m={xml} />
                </tbody>
              </table>
            )}
            {steps.length > 0 && (
              <table className="w-full text-xs">
                <caption className="sr-only">Detector evaluation steps</caption>
                <MetricHead first="Step · dataset" />
                <tbody>
                  {steps.flatMap(([step, v]) =>
                    Object.entries(v as Obj)
                      .filter(([, d]) => isObj(d))
                      .map(([ds, d]) => <MetricRow key={`${step}-${ds}`} label={`${titleCase(step.replace(/^[A-Z]_/, ''))} · ${ds === 'plate_boxes_2k_scenes' ? 'scene photos' : ds === 'video_frames_boxes' ? 'video frames' : titleCase(ds)}`} m={d as Obj} />),
                  )}
                </tbody>
              </table>
            )}
            {rgb && str(rgb.verdict) && (
              <p className="text-[11px] text-fg-muted">
                <span className="font-semibold text-fg">Colour-order bug: {str(rgb.verdict)}.</span> {str(rgb.explanation)?.split('. ')[0]}.
              </p>
            )}
            <Methodology text={str(evalR.description)} />
          </div>
        )}
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ linking vs baselines

const ABLATION_LABEL: Record<string, string> = {
  all_three: 'SUTRA · plate + appearance + travel time',
  all_three_greedy_chaining: 'Same evidence, greedy chaining',
  plate_plus_appearance: 'Plate + appearance',
  plate_plus_kinematic: 'Plate + travel time',
  plate_only: 'Plate only (probabilistic)',
  kinematic_plus_appearance_no_plate: 'No plate: appearance + travel time',
}
const BASELINE_LABEL: Record<string, string> = {
  baseline_a_exact_match: 'Baseline A · exact plate match',
  baseline_b_fuzzy_levenshtein1: 'Baseline B · fuzzy match (1 edit)',
  baseline_c_fuzzy_plus_kinematic: 'Baseline C · fuzzy + travel-time filter',
}

function FullRun({ r }: { r: Obj }) {
  const s = isObj(r.sutra) ? r.sutra : undefined
  const b = isObj(r.exact_match_baseline) ? r.exact_match_baseline : undefined
  const cfg = isObj(r.config) ? r.config : {}
  if (!s || !b) return null
  const rows: [string, string, (o: Obj) => string, boolean][] = [
    ['idf1', 'IDF1 (identity F1)', (o) => dec(num(o.idf1), 4), true],
    ['id_switches', 'ID switches', (o) => fmtNum(num(o.id_switches)), false],
    ['fragmentation', 'Fragmentation', (o) => fmtNum(num(o.fragmentation)), false],
    ['trajectory_completeness', 'Trajectory completeness', (o) => pct(num(o.trajectory_completeness), 2), true],
    ['n_predicted_trajectories', 'Trajectories built', (o) => fmtNum(num(o.n_predicted_trajectories)), false],
  ]
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">
        <Tile big label="SUTRA IDF1" value={dec(num(s.idf1), 3)} tone="accent" sub={`${fmtNum(num(s.id_switches))} ID switches`} />
        <Tile big label="Exact plate match IDF1" value={dec(num(b.idf1), 3)} tone="muted" sub={`${fmtNum(num(b.id_switches))} ID switches`} />
      </div>
      <table className="w-full text-xs">
        <caption className="sr-only">Full-dataset trajectory metrics, SUTRA against exact plate matching</caption>
        <thead>
          <tr className="text-[10px] tracking-wider text-fg-dim uppercase">
            <th className="py-1 text-left font-semibold">Metric</th>
            <th className="px-2 py-1 text-right font-semibold text-accent">SUTRA</th>
            <th className="px-2 py-1 text-right font-semibold">Exact match</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, label, f]) => (
            <tr key={k} className="border-t border-ink-750">
              <th scope="row" className="py-1 text-left font-medium text-fg-muted">
                {label}
              </th>
              <td className="num px-2 py-1 text-right font-mono font-semibold text-fg-strong">{f(s)}</td>
              <td className="num px-2 py-1 text-right font-mono text-fg-muted">{f(b)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-fg-dim">
        Simulated city: {fmtNum(num(cfg.cameras))} cameras, {fmtNum(num(cfg.vehicles))} vehicles, {fmtNum(num(cfg.hours))} h, {fmtNum(num(r.n_events))} reads
        {num(r.events_per_sec_pipeline) !== undefined && ` · pipeline ${fmtNum(num(r.events_per_sec_pipeline))} reads/s`}.
      </p>
    </div>
  )
}

export function LinkingSection({ traj, baselines, ablation }: { traj?: Obj; baselines?: Obj; ablation?: Obj }) {
  if (!traj && !baselines && !ablation) return null
  const configs = ablation && isObj(ablation.configs) ? ablation.configs : {}
  const rows: { key: string; label: string; idf1: number; switches?: number; frag?: number; completeness?: number; kind: 'sutra' | 'variant' | 'baseline'; method?: string }[] = []
  for (const [k, v] of Object.entries(configs)) {
    if (!isObj(v) || num(v.idf1) === undefined) continue
    rows.push({ key: k, label: ABLATION_LABEL[k] ?? titleCase(k), idf1: num(v.idf1) as number, switches: num(v.id_switches), frag: num(v.fragmentation), completeness: num(v.trajectory_completeness), kind: k === 'all_three' ? 'sutra' : 'variant', method: str(v.method) })
  }
  if (baselines) {
    for (const [k, v] of Object.entries(baselines)) {
      if (!k.startsWith('baseline_') || !isObj(v) || num(v.idf1) === undefined) continue
      rows.push({ key: k, label: BASELINE_LABEL[k] ?? titleCase(k), idf1: num(v.idf1) as number, switches: num(v.id_switches), frag: num(v.fragmentation), completeness: num(v.trajectory_completeness), kind: 'baseline' })
    }
  }
  rows.sort((a, b) => b.idf1 - a.idf1)
  const ds = (ablation && isObj(ablation.dataset) ? ablation.dataset : baselines && isObj(baselines.dataset) ? baselines.dataset : {}) as Obj
  const color = (k: string) => (k === 'sutra' ? CHART.accent : k === 'baseline' ? '#566374' : '#3a4e60')

  return (
    <Panel title="Trajectory linking vs baselines">
      <div className="grid grid-cols-1 gap-4 p-3 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        {traj ? (
          <div>
            <div className="mb-2 text-[13px] font-semibold text-fg-strong">Full evaluation day</div>
            <FullRun r={traj} />
          </div>
        ) : (
          <p className="text-xs text-fg-dim">trajectory_metrics not produced yet.</p>
        )}
        {rows.length > 0 && (
          <div className="min-w-0">
            <div className="text-[13px] font-semibold text-fg-strong">Channel ablation and baselines, on one shared dataset</div>
            <p className="mt-0.5 text-[11px] text-fg-dim">
              {fmtNum(num(ds.n_cameras))} cameras, {fmtNum(num(ds.n_vehicles_eval))} vehicles, {fmtNum(num(ds.hours))} h. IDF1, higher is better; bars start at 0.
            </p>
            <div style={{ height: 34 + rows.length * 30 }} className="mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 56, bottom: 0, left: 0 }} barCategoryGap={6}>
                  <CartesianGrid stroke={CHART.grid} horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} {...CHART.axis} axisLine={{ stroke: '#2a3542' }} />
                  <YAxis type="category" dataKey="label" width={236} {...CHART.axis} axisLine={false} tick={{ fontSize: 11, fill: '#d3dbe4' }} />
                  <Tooltip
                    {...CHART.tooltip}
                    cursor={{ fill: 'rgba(125,139,155,0.08)' }}
                    formatter={(v, _n, item) => {
                      const p = item?.payload as (typeof rows)[number] | undefined
                      return [`${Number(v).toFixed(4)}${p?.switches !== undefined ? ` · ${fmtNum(p.switches)} ID switches` : ''}`, 'IDF1']
                    }}
                  />
                  <Bar dataKey="idf1" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                    {rows.map((r) => (
                      <Cell key={r.key} fill={color(r.kind)} />
                    ))}
                    <LabelList dataKey="idf1" position="right" formatter={(v: unknown) => Number(v).toFixed(3)} style={{ fill: '#d3dbe4', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-fg-dim">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2 w-3 rounded-[1px]" style={{ background: CHART.accent }} /> SUTRA (all evidence, global solver)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2 w-3 rounded-[1px]" style={{ background: '#3a4e60' }} /> SUTRA with channels removed
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2 w-3 rounded-[1px]" style={{ background: '#566374' }} /> Baselines (no scoring)
              </span>
            </div>
            <details className="mt-2">
              <summary className="cursor-pointer text-[11px] text-fg-dim hover:text-fg-muted">Table view</summary>
              <table className="mt-1 w-full text-xs">
                <thead>
                  <tr className="text-[10px] tracking-wider text-fg-dim uppercase">
                    <th className="py-1 text-left font-semibold">Configuration</th>
                    <th className="px-2 py-1 text-right font-semibold">IDF1</th>
                    <th className="px-2 py-1 text-right font-semibold">ID sw.</th>
                    <th className="px-2 py-1 text-right font-semibold">Frag.</th>
                    <th className="px-2 py-1 text-right font-semibold">Complete</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.key} className="border-t border-ink-750">
                      <td className="py-1 text-fg">{r.label}</td>
                      <td className="num px-2 py-1 text-right font-mono text-fg-strong">{r.idf1.toFixed(4)}</td>
                      <td className="num px-2 py-1 text-right font-mono text-fg-muted">{fmtNum(r.switches)}</td>
                      <td className="num px-2 py-1 text-right font-mono text-fg-muted">{fmtNum(r.frag)}</td>
                      <td className="num px-2 py-1 text-right font-mono text-fg-muted">{pct(r.completeness, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {ablation && isObj(ablation.notes) && (
                <ul className="mt-2 list-disc pl-4 text-[11px] text-fg-muted">
                  {Object.values(ablation.notes)
                    .filter((n): n is string => typeof n === 'string')
                    .map((n) => (
                      <li key={n}>{n}</li>
                    ))}
                </ul>
              )}
            </details>
          </div>
        )}
      </div>
      <Methodology text={[str(traj?.description), str(baselines?.description), str(ablation?.description)].filter(Boolean).join('\n\n')} />
    </Panel>
  )
}

// ------------------------------------------------------------------ stress sweep

export function StressSection({ r }: { r?: Obj }) {
  if (!r) return null
  const ocr = objs(r.ocr_noise_sweep)
    .map((p) => ({ acc: num(p.measured_whole_plate_accuracy), sutra: num(p.sutra_idf1), base: num(p.baseline_a_idf1), gap: num(p.gap) }))
    .filter((p): p is { acc: number; sutra: number; base: number; gap: number | undefined } => p.acc !== undefined && p.sutra !== undefined && p.base !== undefined)
    .sort((a, b) => b.acc - a.acc)
  const miss = objs(r.camera_miss_sweep)
    .map((p) => ({ pMiss: num(p.p_miss), sutra: num(p.sutra_idf1), base: num(p.baseline_a_idf1), gap: num(p.gap) }))
    .filter((p): p is { pMiss: number; sutra: number; base: number; gap: number | undefined } => p.pMiss !== undefined && p.sutra !== undefined && p.base !== undefined)
  const first = ocr[0]
  const last = ocr[ocr.length - 1]
  const lo = Math.min(...ocr.map((p) => p.base), ...miss.map((p) => p.base), 1)
  const yMin = Math.max(0, Math.floor((lo - 0.05) * 10) / 10)
  const yTicks = Array.from({ length: Math.round((1 - yMin) * 10) + 1 }, (_, i) => +(yMin + i * 0.1).toFixed(1))

  const lines = (x: string) => (
    <>
      <Line name="SUTRA" dataKey="sutra" stroke={CHART.accent} strokeWidth={2.5} dot={{ r: 4, fill: CHART.accent, stroke: '#0f141a', strokeWidth: 2 }} isAnimationActive={false}>
        <LabelList dataKey="sutra" position="top" formatter={(v: unknown) => Number(v).toFixed(2)} style={{ fill: '#6fdcec', fontSize: 10, fontFamily: 'var(--font-mono)' }} />
      </Line>
      <Line name="Exact plate match" dataKey="base" stroke={CHART.neutral} strokeWidth={2} strokeDasharray="5 4" dot={{ r: 4, fill: CHART.neutral, stroke: '#0f141a', strokeWidth: 2 }} isAnimationActive={false}>
        <LabelList dataKey="base" position="bottom" formatter={(v: unknown) => Number(v).toFixed(2)} style={{ fill: '#7d8b9b', fontSize: 10, fontFamily: 'var(--font-mono)' }} />
      </Line>
      <Tooltip
        {...CHART.tooltip}
        cursor={{ stroke: '#7d8b9b', strokeDasharray: '3 3' }}
        labelFormatter={(v) => (x === 'acc' ? `plate accuracy ${pct(Number(v))}` : `camera miss rate ${pct(Number(v), 0)}`)}
        formatter={(v, n) => [Number(v).toFixed(4), n]}
      />
    </>
  )

  return (
    <Panel title="Stress test — what happens as plate reads get worse">
      <div className="px-3 pt-3">
        {first && last && first.gap !== undefined && last.gap !== undefined && (
          <p className="text-[13px] text-fg-strong">
            As measured plate accuracy falls from <b className="num font-mono">{pct(first.acc)}</b> to <b className="num font-mono">{pct(last.acc)}</b>, SUTRA’s IDF1 stays between{' '}
            <b className="num font-mono text-accent-strong">
              {Math.min(...ocr.map((p) => p.sutra)).toFixed(3)}–{Math.max(...ocr.map((p) => p.sutra)).toFixed(3)}
            </b>{' '}
            while exact matching goes from <b className="num font-mono">{first.base.toFixed(3)}</b> to <b className="num font-mono">{last.base.toFixed(3)}</b>: the gap {last.gap > first.gap ? 'widens' : 'narrows'} from{' '}
            <b className="num font-mono">{first.gap.toFixed(3)}</b> to{' '}
            <b className="num font-mono">{last.gap.toFixed(3)}</b>.
          </p>
        )}
      </div>
      <div className="grid grid-cols-1 gap-4 p-3 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        {ocr.length > 0 && (
          <div>
            <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">IDF1 vs measured whole-plate OCR accuracy (worse to the right)</div>
            <div className="h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ocr} margin={{ top: 20, right: 20, bottom: 4, left: -6 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="acc" type="number" reversed domain={['dataMin - 0.03', 'dataMax + 0.03']} ticks={ocr.map((p) => p.acc)} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} {...CHART.axis} axisLine={{ stroke: '#2a3542' }} />
                  <YAxis domain={[yMin, 1]} ticks={yTicks} tickFormatter={(v: number) => v.toFixed(1)} {...CHART.axis} axisLine={false} width={40} />
                  <Legend verticalAlign="top" height={20} wrapperStyle={{ fontSize: 11, color: '#7d8b9b' }} />
                  {lines('acc')}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        {miss.length > 0 && (
          <div>
            <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">IDF1 vs camera miss rate</div>
            <div className="h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={miss} margin={{ top: 20, right: 20, bottom: 4, left: -6 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="pMiss" type="number" domain={['dataMin - 0.03', 'dataMax + 0.03']} ticks={miss.map((p) => p.pMiss)} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} {...CHART.axis} axisLine={{ stroke: '#2a3542' }} />
                  <YAxis domain={[yMin, 1]} ticks={yTicks} tickFormatter={(v: number) => v.toFixed(1)} {...CHART.axis} axisLine={false} width={40} />
                  <Legend verticalAlign="top" height={20} wrapperStyle={{ fontSize: 11, color: '#7d8b9b' }} />
                  {lines('miss')}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
      {str(r.widen_or_not_answer) && (
        <div className="mx-3 mb-3 rounded-sm border border-ink-600 bg-ink-800 px-3 py-2 text-xs text-fg-muted">
          <span className="font-semibold text-fg">The report’s own verdict: </span>
          {str(r.widen_or_not_answer)}
        </div>
      )}
      <Methodology text={str(r.description)} />
    </Panel>
  )
}

// ------------------------------------------------------------------ calibration

export function CalibrationSection({ r }: { r?: Obj }) {
  if (!r) return null
  const sweep = objs(r.sweep)
    .map((p) => ({ beta: num(p.beta), idf1: num(p.idf1), sw: num(p.id_switches) }))
    .filter((p): p is { beta: number; idf1: number; sw: number | undefined } => p.beta !== undefined && p.idf1 !== undefined)
  const chosen = num(r.chosen_beta)
  const best = isObj(r.best_by_idf1) ? r.best_by_idf1 : undefined
  const lo = Math.min(...sweep.map((p) => p.idf1))
  const hi = Math.max(...sweep.map((p) => p.idf1))
  return (
    <Panel title="Link-bias calibration (on a training day, never the evaluation set)">
      <div className="grid grid-cols-1 gap-4 p-3 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        {sweep.length > 1 && (
          <div>
            <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">IDF1 by link bias β (y-axis zoomed: differences are small)</div>
            <div className="h-[210px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sweep} margin={{ top: 14, right: 16, bottom: 4, left: 4 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="beta" type="number" domain={['dataMin', 'dataMax']} ticks={sweep.map((p) => p.beta)} {...CHART.axis} axisLine={{ stroke: '#2a3542' }} />
                  <YAxis domain={[Math.floor(lo * 1000) / 1000, Math.ceil(hi * 1000) / 1000]} tickFormatter={(v: number) => v.toFixed(3)} {...CHART.axis} axisLine={false} width={48} />
                  {chosen !== undefined && <ReferenceLine x={chosen} stroke="#d3dbe4" strokeDasharray="4 3" label={{ value: `chosen β = ${chosen}`, position: 'insideTopLeft', fill: '#7d8b9b', fontSize: 10 }} />}
                  <Tooltip {...CHART.tooltip} labelFormatter={(v) => `β = ${v}`} formatter={(v, n) => [Number(v).toFixed(4), n]} />
                  <Line name="IDF1" dataKey="idf1" stroke={CHART.accent} strokeWidth={2} dot={{ r: 4, fill: CHART.accent, stroke: '#0f141a', strokeWidth: 2 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-3">
            <Tile label="Chosen β" value={chosen === undefined ? '—' : String(chosen)} />
            <Tile label="Best swept β" value={best ? String(num(best.beta) ?? '—') : '—'} sub={best ? `IDF1 ${dec(num(best.idf1), 4)}` : undefined} tone="muted" />
          </div>
          <Description text={str(r.decision)} />
        </div>
      </div>
      <Methodology text={str(r.description)} />
    </Panel>
  )
}
