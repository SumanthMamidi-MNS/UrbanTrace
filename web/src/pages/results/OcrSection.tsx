import { Bar, BarChart, CartesianGrid, LabelList, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { cx, Panel } from '../../components/ui'
import { ReadChars } from '../../components/WhyPanel'
import { fmtNum, titleCase } from '../../lib/format'
import { Description, Methodology, Missing, PctBar, Tile } from './bits'
import { CHART, isObj, num, objs, pct, str, type Obj } from './shared'

/** PRD component 1 target: > 90% recognition accuracy. */
const TARGET = 0.9

/** The real-plate progression, in the order it happened. Labels describe the model; every number is read from the report. */
const PROGRESSION: { key: string; label: string; model: string }[] = [
  { key: 'ocr_real_synthonly', label: 'CRNN, synthetic only', model: 'Our CRNN trained on synthetic plates only' },
  { key: 'ocr_real_long', label: 'CRNN, + real fine-tune', model: 'Our CRNN fine-tuned on the real train split' },
  { key: 'ocr_real_fpo_zeroshot_postfix', label: 'fast-plate-ocr, zero-shot', model: 'Pretrained fast-plate-ocr, no fine-tuning' },
  { key: 'ocr_real_fpo_finetuned', label: 'fast-plate-ocr, fine-tuned', model: 'fast-plate-ocr fine-tuned on the real train split' },
]

const SOURCE_LABEL: Record<string, string> = { 'State-wise_OLX': 'State-wise (OLX listings)', google_images: 'Google Images', video_images: 'Video frames' }

function TargetVerdict({ whole, char }: { whole: number | undefined; char: number | undefined }) {
  if (whole === undefined || char === undefined) return null
  const charMet = char > TARGET
  const wholeMet = whole > TARGET
  return (
    <div role="status" className={cx('rounded-sm border px-3 py-2 text-[13px]', charMet && wholeMet ? 'border-accent-dim bg-accent-faint/40' : 'border-caution-dim bg-caution-faint')}>
      <span className="font-semibold text-fg-strong">PRD target &gt; 90%: </span>
      <span className={charMet ? 'text-accent-strong' : 'text-caution'}>
        {charMet ? 'met' : 'not met'} per character ({pct(char)})
      </span>
      <span className="text-fg-muted">; </span>
      <span className={wholeMet ? 'text-accent-strong' : 'text-caution'}>
        {wholeMet ? 'met' : 'not met'} for whole plates ({pct(whole)})
      </span>
      <span className="text-fg-muted">.</span>
      {!wholeMet && <span className="text-fg-muted"> A whole plate counts only if all 10 slots are right.</span>}
    </div>
  )
}

function ErrorSamples({ r }: { r: Obj }) {
  const rows = objs(r.error_sample).filter((e) => str(e.label) && str(e.prediction)).slice(0, 8)
  if (!rows.length) return null
  return (
    <div className="px-3 pb-3">
      <div className="mb-1 text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">What the misses look like (first {rows.length} errors in the report)</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-[10px] tracking-wider text-fg-dim uppercase">
            <th className="py-1 text-left font-semibold">Source</th>
            <th className="py-1 text-left font-semibold">Truth</th>
            <th className="py-1 text-left font-semibold">Read (wrong slots marked)</th>
            <th className="py-1 text-right font-semibold">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e, i) => (
            <tr key={i} className="border-t border-ink-750">
              <td className="py-1 text-fg-dim">{SOURCE_LABEL[str(e.condition) ?? ''] ?? str(e.condition)}</td>
              <td className="py-1">
                <ReadChars read={str(e.label) as string} truth={str(e.label) as string} />
              </td>
              <td className="py-1">
                <ReadChars read={str(e.prediction) as string} truth={str(e.label) as string} />
              </td>
              <td className="num py-1 text-right font-mono text-fg-muted">{pct(num(e.confidence), 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-[11px] text-fg-dim">
        {rows.filter((e) => (num(e.confidence) ?? 1) < 0.5).length} of these {rows.length} misses were read with under 50% confidence. Low-confidence reads carry little weight when a trajectory fuses its reads.
      </p>
    </div>
  )
}

function Headline({ r }: { r: Obj }) {
  const whole = num(r.whole_plate_accuracy)
  const char = num(r.char_accuracy)
  const g = isObj(r.group_by_plate) ? r.group_by_plate : {}
  const perPlate = num(g.whole_plate_accuracy_per_plate_mean)
  const cond = isObj(r.per_condition) ? r.per_condition : {}
  const ece = num(r.confidence_ece)
  return (
    <Panel title="Plate OCR on real Indian plates — held-out test set">
      <div className="flex flex-col gap-3 p-3">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Tile big label="Whole plate" value={pct(whole)} sub={`all 10 slots right · ${fmtNum(num(r.n_samples))} images`} tone="accent" />
          <Tile big label="Per character" value={pct(char)} sub="slot-level accuracy" tone="accent" />
          <Tile big label="Whole plate, per unique plate" value={pct(perPlate)} sub={`mean over ${fmtNum(num(g.n_unique_plates))} plates (no plate over-weighted)`} />
          <Tile big label="Calibration error (ECE)" value={ece === undefined ? '—' : ece.toFixed(3)} sub="confidence matches accuracy; lower is better" />
        </div>
        <TargetVerdict whole={whole} char={char} />
        <Description text={str(r.description)} />
        {Object.keys(cond).length > 0 && (
          <div>
            <div className="mb-1 text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">By image source</div>
            <table className="w-full text-xs">
              <caption className="sr-only">OCR accuracy by image source</caption>
              <thead>
                <tr className="text-[10px] tracking-wider text-fg-dim uppercase">
                  <th className="py-1 text-left font-semibold">Source</th>
                  <th className="py-1 text-right font-semibold">Images</th>
                  <th className="w-[34%] px-3 py-1 text-left font-semibold">Whole plate</th>
                  <th className="w-[34%] px-3 py-1 text-left font-semibold">Per character</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(cond)
                  .filter(([, v]) => isObj(v))
                  .map(([k, v]) => {
                    const c = v as Obj
                    return (
                      <tr key={k} className="border-t border-ink-750">
                        <th scope="row" className="py-1.5 text-left font-medium text-fg">
                          {SOURCE_LABEL[k] ?? titleCase(k)}
                        </th>
                        <td className="num py-1.5 text-right font-mono text-fg-muted">{fmtNum(num(c.n_samples))}</td>
                        <td className="px-3 py-1.5">
                          <PctBar v={num(c.whole_plate_accuracy)} target={TARGET} />
                        </td>
                        <td className="px-3 py-1.5">
                          <PctBar v={num(c.char_accuracy)} tone="neutral" target={TARGET} />
                        </td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
            <p className="mt-1 text-[10px] text-fg-dim">Vertical line = 90% target.</p>
          </div>
        )}
      </div>
      <ErrorSamples r={r} />
    </Panel>
  )
}

function Progression({ reports }: { reports: Obj }) {
  const data = PROGRESSION.map((p) => {
    const r = isObj(reports[p.key]) ? (reports[p.key] as Obj) : undefined
    return { ...p, whole: num(r?.whole_plate_accuracy), char: num(r?.char_accuracy), n: num(r?.n_samples) }
  }).filter((d) => d.whole !== undefined || d.char !== undefined)
  if (data.length < 2) return null
  const first = data[0]
  const last = data[data.length - 1]
  return (
    <Panel title="How we got there — same 510-image real test set">
      <div className="px-3 pt-3">
        <p className="text-[13px] text-fg-strong">
          Whole-plate accuracy on real plates went{' '}
          {data.map((d, i) => (
            <span key={d.key}>
              <b className={cx('num font-mono', i === data.length - 1 ? 'text-accent-strong' : 'text-fg-strong')}>{pct(d.whole)}</b>
              {i < data.length - 1 ? ' → ' : ''}
            </span>
          ))}
          .
        </p>
        <p className="mt-0.5 text-xs text-fg-muted">
          Synthetic training alone did not transfer ({first.label}: {pct(first.whole)}). Real fine-tuning and a stronger pretrained recogniser closed most of the gap ({last.label}: {pct(last.whole)}).
        </p>
      </div>
      <div className="h-[260px] p-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 18, right: 12, bottom: 0, left: -6 }} barGap={2} barCategoryGap="28%">
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="label" {...CHART.axis} axisLine={{ stroke: '#2a3542' }} interval={0} />
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 0.9, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} {...CHART.axis} axisLine={false} width={44} />
            <ReferenceLine y={TARGET} stroke="#d3dbe4" strokeDasharray="4 3" label={{ value: 'PRD target 90%', position: 'insideTopLeft', fill: '#7d8b9b', fontSize: 10 }} />
            <Tooltip {...CHART.tooltip} cursor={{ fill: 'rgba(125,139,155,0.08)' }} formatter={(v, name) => [pct(Number(v)), name]} labelFormatter={(l, p) => String(p?.[0]?.payload?.model ?? l)} />
            <Legend verticalAlign="top" height={22} iconType="square" wrapperStyle={{ fontSize: 11, color: '#7d8b9b' }} />
            <Bar name="Whole plate" dataKey="whole" fill={CHART.accent} radius={[3, 3, 0, 0]} isAnimationActive={false}>
              <LabelList dataKey="whole" position="top" formatter={(v: unknown) => pct(Number(v))} style={{ fill: '#eef3f8', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
            </Bar>
            <Bar name="Per character" dataKey="char" fill="#4a5a6c" radius={[3, 3, 0, 0]} isAnimationActive={false}>
              <LabelList dataKey="char" position="top" formatter={(v: unknown) => pct(Number(v))} style={{ fill: '#7d8b9b', fontSize: 10, fontFamily: 'var(--font-mono)' }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function Synthetic({ r, realSame }: { r: Obj; realSame: number | undefined }) {
  const cond = isObj(r.per_condition) ? Object.entries(r.per_condition).filter(([, v]) => isObj(v)) : []
  const sorted = [...cond].sort(([, a], [, b]) => (num((a as Obj).whole_plate_accuracy) ?? 0) - (num((b as Obj).whole_plate_accuracy) ?? 0))
  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          <span className="rounded-sm border border-caution-dim px-1.5 text-[10px] font-bold tracking-wider text-caution">SYNTHETIC</span>
          OCR on rendered plates — development number, not the PRD claim
        </span>
      }
    >
      <div className="grid grid-cols-1 gap-4 p-3 lg:grid-cols-[260px_minmax(0,1fr)]">
        <div className="flex flex-col gap-3">
          <Tile label="Whole plate (synthetic)" value={pct(num(r.whole_plate_accuracy))} sub={`${fmtNum(num(r.n_samples))} rendered plates`} tone="muted" />
          <Tile label="Per character (synthetic)" value={pct(num(r.char_accuracy))} tone="muted" />
          <p className="text-[11px] leading-relaxed text-fg-dim">
            The same synthetic-trained checkpoint scores {pct(num(r.whole_plate_accuracy))} here
            {realSame !== undefined ? <> but {pct(realSame)} on real plates</> : ' but much lower on real plates'}. That gap is why only the real-plate number is reported as the result.
          </p>
        </div>
        {sorted.length > 0 && (
          <div>
            <div className="mb-1 text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Whole-plate accuracy by simulated condition, hardest first</div>
            <ul className="grid grid-cols-1 gap-x-6 gap-y-1 md:grid-cols-2">
              {sorted.map(([k, v]) => (
                <li key={k} className="grid grid-cols-[110px_1fr] items-center gap-2 text-xs">
                  <span className="truncate text-fg-muted">{titleCase(k)}</span>
                  <PctBar v={num((v as Obj).whole_plate_accuracy)} tone="neutral" />
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <Methodology text={str(r.description)} />
    </Panel>
  )
}

export function OcrSection({ reports }: { reports: Obj }) {
  const fin = isObj(reports.ocr_real_fpo_finetuned) ? reports.ocr_real_fpo_finetuned : undefined
  const syn = isObj(reports.ocr_synthetic) ? reports.ocr_synthetic : undefined
  return (
    <>
      {fin ? <Headline r={fin} /> : <Panel title="Plate OCR on real Indian plates"><Missing what="The fine-tuned real-plate OCR report (ocr_real_fpo_finetuned)" /></Panel>}
      <Progression reports={reports} />
      {syn && <Synthetic r={syn} realSame={isObj(reports.ocr_real_synthonly) ? num(reports.ocr_real_synthonly.whole_plate_accuracy) : undefined} />}
    </>
  )
}
