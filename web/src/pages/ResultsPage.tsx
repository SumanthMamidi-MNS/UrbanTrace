import type { ReactNode } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useEval } from '../api/hooks'
import { cx, EmptyState, ErrorState, Loading, Panel } from '../components/ui'
import { fmtNum, fmtPct, titleCase } from '../lib/format'

// Reports are still being produced: every accessor is defensive and every section degrades to an empty state.

type Obj = Record<string, unknown>
const isObj = (x: unknown): x is Obj => typeof x === 'object' && x !== null && !Array.isArray(x)
const num = (x: unknown): number | undefined => (typeof x === 'number' && Number.isFinite(x) ? x : undefined)
const str = (x: unknown): string | undefined => (typeof x === 'string' ? x : undefined)
const isScalar = (x: unknown) => x === null || ['string', 'number', 'boolean'].includes(typeof x)

const CHANNELS: { key: string; label: string }[] = [
  { key: 'plate_auc', label: 'Plate only' },
  { key: 'appearance_auc', label: 'Appearance' },
  { key: 'kinematic_auc', label: 'Kinematic' },
  { key: 'fused_auc', label: 'Fused (SUTRA)' },
]

/** AUC cell shade: 0.5 (chance) is neutral, 1.0 is full accent. */
function aucStyle(v: number | undefined) {
  if (v === undefined) return {}
  const a = Math.max(0, Math.min(1, (v - 0.5) / 0.5))
  return { background: `rgba(60,196,216,${0.05 + 0.6 * a})`, color: a > 0.65 ? '#070a0e' : '#d3dbe4' }
}

function AucMatrix({ matrix, order, emphasise, emphasiseLabel = 'thesis case', freq, caption }: { matrix: Obj; order?: string[]; emphasise?: string; emphasiseLabel?: string; freq?: Obj; caption: string }) {
  const rows = [...(order ?? []).filter((k) => k in matrix), ...Object.keys(matrix).filter((k) => !(order ?? []).includes(k))].filter((k) => isObj(matrix[k]))
  if (!rows.length) return <EmptyState title="Matrix is empty" className="h-24" />
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-separate border-spacing-[3px] text-xs">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="text-[10px] tracking-wider text-fg-dim uppercase">
            <th className="px-2 py-1 text-left font-semibold">Stratum</th>
            {CHANNELS.map((c) => (
              <th key={c.key} className={cx('px-2 py-1 text-center font-semibold', c.key === 'fused_auc' && 'text-accent')}>
                {c.label}
              </th>
            ))}
            <th className="px-2 py-1 text-right font-semibold">Pairs</th>
            {freq && <th className="px-2 py-1 text-right font-semibold">Real-traffic freq.</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const row = matrix[r] as Obj
            const hot = r === emphasise
            return (
              <tr key={r}>
                <th scope="row" className={cx('rounded-[3px] px-2 py-1.5 text-left text-[12px] font-semibold', hot ? 'bg-ink-750 text-fg-strong shadow-[inset_2px_0_0_var(--color-accent)]' : 'text-fg')}>
                  {titleCase(r)}
                  {hot && <span className="ml-2 text-[10px] font-normal text-accent">{emphasiseLabel}</span>}
                </th>
                {CHANNELS.map((c) => {
                  const v = num(row[c.key])
                  return (
                    <td key={c.key} className={cx('num rounded-[3px] px-2 py-1.5 text-center font-mono text-[13px]', c.key === 'fused_auc' && 'font-semibold', hot && 'ring-1 ring-ink-500')} style={aucStyle(v)}>
                      {v === undefined ? '—' : v.toFixed(3)}
                    </td>
                  )
                })}
                <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtNum(num(row.n_pairs))}</td>
                {freq && <td className="num px-2 py-1.5 text-right font-mono text-fg-muted">{fmtPct(num(freq[r]), 1)}</td>}
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="mt-1 px-1 text-[11px] text-fg-dim">Cell shade runs from chance (0.5, dark) to perfect separation (1.0, bright). AUC below 0.5 is no better than a coin flip.</p>
    </div>
  )
}

function Description({ r }: { r: Obj }) {
  const d = str(r.description)
  return d ? <p className="px-3 pt-3 text-xs leading-relaxed text-fg-muted">{d}</p> : null
}

function Tile({ label, value, sub, tone }: { label: string; value: string; sub?: ReactNode; tone?: 'accent' }) {
  return (
    <div className="rounded-sm border border-ink-700 bg-ink-900/60 px-3 py-2">
      <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">{label}</div>
      <div className={cx('num font-mono text-xl', tone === 'accent' ? 'text-accent-strong' : 'text-fg-strong')}>{value}</div>
      {sub && <div className="text-[11px] text-fg-dim">{sub}</div>}
    </div>
  )
}

// ------------------------------------------------------------------ specific reports

function StratifiedSection({ r }: { r: Obj }) {
  if (!isObj(r.matrix)) return <GenericReport name="stratified_auc" r={r} />
  const fw = isObj(r.frequency_weighted) ? r.frequency_weighted : undefined
  const clone = isObj(r.matrix.clone) ? r.matrix.clone : undefined
  return (
    <Panel title="Stratified pairwise AUC — stratum × channel">
      <Description r={r} />
      {clone && (
        <div className="mx-3 mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 rounded-sm border border-accent-dim bg-accent-faint/40 px-3 py-2 text-[13px] text-fg-strong">
          <span>
            On cloned plates, plate-only matching scores <b className="num font-mono">{num(clone.plate_auc)?.toFixed(3) ?? '—'}</b> (chance), fusion scores{' '}
            <b className="num font-mono text-accent-strong">{num(clone.fused_auc)?.toFixed(3) ?? '—'}</b>.
          </span>
          {fw && (
            <span className="text-xs text-fg-muted">
              Frequency-weighted: plate {num(fw.plate_auc)?.toFixed(4) ?? '—'} · fused {num(fw.fused_auc)?.toFixed(4) ?? '—'}
            </span>
          )}
        </div>
      )}
      <div className="p-3">
        <AucMatrix matrix={r.matrix} order={['routine', 'clone', 'degraded', 'plate_similar']} emphasise="clone" freq={isObj(r.stratum_frequencies) ? r.stratum_frequencies : undefined} caption="AUC by stratum and evidence channel" />
      </div>
    </Panel>
  )
}

function CloneOverlapSection({ r }: { r: Obj }) {
  if (!isObj(r.matrix)) return <GenericReport name="clone_overlap_auc" r={r} />
  return (
    <Panel title="Clones: disjoint vs overlapping routes">
      <Description r={r} />
      <div className="flex flex-wrap gap-3 px-3 pt-3">
        {num(r.n_clone_vehicles_holdout) !== undefined && <Tile label="Clone vehicles (holdout)" value={fmtNum(num(r.n_clone_vehicles_holdout))} />}
        {num(r.n_overlap_clone_vehicles_holdout) !== undefined && <Tile label="With overlapping route" value={fmtNum(num(r.n_overlap_clone_vehicles_holdout))} />}
      </div>
      <div className="p-3">
        <AucMatrix matrix={r.matrix} order={['disjoint', 'overlap']} emphasise="overlap" emphasiseLabel="realistic hard case" caption="Clone AUC by route overlap" />
      </div>
    </Panel>
  )
}

function AppearanceScalingSection({ r }: { r: Obj }) {
  const pts = Array.isArray(r.points) ? r.points.filter(isObj) : []
  const data = pts
    .map((p) => ({ gallery: num(p.gallery_size), rank1: num(p.appearance_rank1), plate: num(p.whole_plate_accuracy) }))
    .filter((p): p is { gallery: number; rank1: number | undefined; plate: number | undefined } => p.gallery !== undefined && p.gallery > 0)
  if (!data.length) return <GenericReport name="appearance_scaling" r={r} />
  return (
    <Panel title="Appearance alone decays at city scale">
      <Description r={r} />
      <div className="h-[280px] p-3">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 24, bottom: 4, left: -4 }}>
            <CartesianGrid stroke="#1e2732" vertical={false} />
            <XAxis dataKey="gallery" type="number" scale="log" domain={['dataMin', 'dataMax']} ticks={data.map((d) => d.gallery)} tickFormatter={(v: number) => fmtNum(v)} stroke="#566374" tick={{ fontSize: 11, fill: '#7d8b9b' }} tickLine={false} axisLine={{ stroke: '#2a3542' }} label={{ value: 'distinct vehicles in gallery (log)', position: 'insideBottom', offset: -2, fill: '#566374', fontSize: 10 }} height={36} />
            <YAxis domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} stroke="#566374" tick={{ fontSize: 11, fill: '#7d8b9b' }} tickLine={false} axisLine={false} width={44} />
            <Tooltip
              contentStyle={{ background: '#131920', border: '1px solid #2a3542', borderRadius: 4, fontSize: 12 }}
              labelStyle={{ color: '#7d8b9b' }}
              labelFormatter={(v) => `gallery ${fmtNum(Number(v))} vehicles`}
              formatter={(v, name) => [fmtPct(Number(v), 1), name]}
            />
            <Legend verticalAlign="top" height={24} iconType="plainline" wrapperStyle={{ fontSize: 11, color: '#7d8b9b' }} />
            <Line name="Appearance rank-1" dataKey="rank1" stroke="#3cc4d8" strokeWidth={2} dot={{ r: 4, fill: '#3cc4d8', stroke: '#0f141a', strokeWidth: 2 }} isAnimationActive={false} />
            <Line name="Whole-plate OCR accuracy (single read)" dataKey="plate" stroke="#7d8b9b" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 4, fill: '#7d8b9b', stroke: '#0f141a', strokeWidth: 2 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function GatingSection({ gating, blocking }: { gating?: Obj; blocking?: Obj }) {
  if (!gating && !blocking) return null
  return (
    <Panel title="Candidate gating & blocking">
      {gating && <Description r={gating} />}
      <div className="grid grid-cols-2 gap-3 p-3 lg:grid-cols-4">
        {gating && (
          <>
            <Tile label="Gate recall" value={fmtPct(num(gating.recall), 2)} sub={`${fmtNum(num(gating.n_recalled))} of ${fmtNum(num(gating.n_true_predecessor_pairs))} true pairs`} tone="accent" />
            <Tile label="Pairs cut vs naive" value={fmtPct(num(gating.cut_fraction_vs_naive), 2)} />
            <Tile label="Candidates / event" value={fmtNum(num(gating.mean_candidates_per_event), 0)} sub={`naive ${fmtNum(num(gating.naive_mean_candidates_per_event), 0)}`} />
            <Tile label="Events evaluated" value={fmtNum(num(gating.n_events))} />
          </>
        )}
        {blocking && (
          <>
            <Tile label="Plate blocking engaged" value={fmtPct(num(blocking.engagement_rate), 2)} sub={`${fmtNum(num(blocking.n_events_engaged))} of ${fmtNum(num(blocking.n_events_total))} events`} />
            <Tile label="Blocking recall cost" value={fmtPct(num(blocking.recall_cost), 2)} sub={`${fmtNum(num(blocking.n_true_pairs_dropped_by_blocking))} true pairs dropped`} tone="accent" />
          </>
        )}
      </div>
      {blocking && str(blocking.description) && <p className="px-3 pb-3 text-[11px] text-fg-dim">{str(blocking.description)}</p>}
    </Panel>
  )
}

// ------------------------------------------------------------------ generic fallback

function ScalarTable({ rows }: { rows: [string, unknown][] }) {
  return (
    <table className="w-full text-xs">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} className="border-b border-ink-750 last:border-b-0">
            <th scope="row" className="px-3 py-1 text-left font-normal text-fg-muted">
              {k}
            </th>
            <td className="num px-3 py-1 text-right font-mono text-fg">{typeof v === 'number' ? (Number.isInteger(v) ? fmtNum(v) : v.toFixed(4)) : String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ArrayTable({ items }: { items: Obj[] }) {
  const cols = [...new Set(items.flatMap((i) => Object.keys(i).filter((k) => isScalar(i[k]))))]
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-ink-700 text-[10px] tracking-wider text-fg-dim uppercase">
            {cols.map((c) => (
              <th key={c} className="px-3 py-1.5 text-left font-semibold">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((it, i) => (
            <tr key={i} className="border-b border-ink-750">
              {cols.map((c) => (
                <td key={c} className="num px-3 py-1 font-mono text-fg">
                  {typeof it[c] === 'number' ? (Number.isInteger(it[c]) ? fmtNum(it[c] as number) : (it[c] as number).toFixed(4)) : String(it[c] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Renders any report shape without crashing: scalars, arrays of records, nested objects. */
function GenericReport({ name, r }: { name: string; r: unknown }) {
  if (!isObj(r)) {
    return (
      <Panel title={titleCase(name)}>
        <pre className="overflow-x-auto p-3 font-mono text-[11px] text-fg-muted">{JSON.stringify(r, null, 2)}</pre>
      </Panel>
    )
  }
  const entries = Object.entries(r).filter(([k]) => k !== 'description')
  const scalars = entries.filter(([, v]) => isScalar(v))
  const tables = entries.filter(([, v]) => Array.isArray(v) && v.length > 0 && v.every(isObj)) as [string, Obj[]][]
  const tableKeys = new Set(tables.map(([k]) => k))
  const nested = entries.filter(([k, v]) => !isScalar(v) && !tableKeys.has(k))
  const matrices = nested.filter(([, v]) => isObj(v) && Object.values(v).length > 0 && Object.values(v).every((x) => isObj(x) && Object.keys(x).some((k) => k.endsWith('_auc'))))
  const rest = nested.filter(([k]) => !matrices.some(([m]) => m === k))
  return (
    <Panel title={titleCase(name)}>
      <Description r={r} />
      {matrices.map(([k, v]) => (
        <div key={k} className="p-3">
          <AucMatrix matrix={v as Obj} caption={k} />
        </div>
      ))}
      {scalars.length > 0 && (
        <div className="p-3">
          <div className="rounded-sm border border-ink-700">
            <ScalarTable rows={scalars} />
          </div>
        </div>
      )}
      {tables.map(([k, v]) => (
        <div key={k} className="p-3">
          <div className="mb-1 text-[10px] font-semibold tracking-wider text-fg-dim uppercase">{k}</div>
          <ArrayTable items={v} />
        </div>
      ))}
      {rest.length > 0 && (
        <details className="border-t border-ink-700 px-3 py-2">
          <summary className="cursor-pointer text-[11px] text-fg-dim">Other fields ({rest.map(([k]) => k).join(', ')})</summary>
          <pre className="mt-2 overflow-x-auto font-mono text-[11px] text-fg-muted">{JSON.stringify(Object.fromEntries(rest), null, 2)}</pre>
        </details>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ page

const KNOWN = new Set(['stratified_auc', 'clone_overlap_auc', 'appearance_scaling', 'gating', 'blocking'])

export function ResultsPage() {
  const q = useEval()
  if (q.isLoading) return <Loading label="Loading evaluation reports" />
  if (q.isError) return <ErrorState error={q.error} />
  const reports = isObj(q.data) && isObj(q.data.reports) ? q.data.reports : {}
  const get = (k: string) => (isObj(reports[k]) ? (reports[k] as Obj) : undefined)
  const baselineKeys = Object.keys(reports).filter((k) => /baseline|trajectory_metrics/.test(k))
  const others = Object.keys(reports).filter((k) => !KNOWN.has(k) && !baselineKeys.includes(k))

  if (Object.keys(reports).length === 0) {
    return <EmptyState title="No evaluation reports yet">Run the eval scripts; reports in eval/reports/*.json appear here automatically.</EmptyState>
  }

  const strat = get('stratified_auc')
  const clone = get('clone_overlap_auc')
  const scaling = get('appearance_scaling')

  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="flex flex-col gap-3">
        {baselineKeys.length ? (
          baselineKeys.map((k) => <GenericReport key={k} name={k} r={reports[k]} />)
        ) : (
          <Panel title="Baselines vs SUTRA">
            <EmptyState title="Report not produced yet" className="h-24">
              Exact-match, fuzzy and fuzzy + time-gate baselines (IDF1, ID switches, fragmentation) will render here once the engine writes them.
            </EmptyState>
          </Panel>
        )}
        {strat ? <StratifiedSection r={strat} /> : <Panel title="Stratified pairwise AUC"><EmptyState title="Report not produced yet" className="h-24" /></Panel>}
        <div className="grid grid-cols-1 gap-3 2xl:grid-cols-2">
          {clone && <CloneOverlapSection r={clone} />}
          {scaling && <AppearanceScalingSection r={scaling} />}
        </div>
        <GatingSection gating={get('gating')} blocking={get('blocking')} />
        {others.map((k) => (
          <GenericReport key={k} name={k} r={reports[k]} />
        ))}
      </div>
    </div>
  )
}
