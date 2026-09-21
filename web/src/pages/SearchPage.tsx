import { useState, type FormEvent } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useSearch } from '../api/hooks'
import type { SearchQuery } from '../api/types'
import { fmtProb } from '../components/WhyPanel'
import { Bar, Button, cx, EmptyState, ErrorState, Field, inputCls, Loading, Panel, Swatch } from '../components/ui'
import { fmtDuration, fmtTime, fromLocalInput, titleCase } from '../lib/format'
import { normalisePlateQuery, plateSlots } from '../lib/plate'

const COLORS = ['white', 'black', 'silver', 'grey', 'red', 'blue', 'green', 'yellow']
const TYPES = ['car', 'bike', 'truck', 'bus', 'auto']

/**
 * Which slots of `plate` were wildcards in the query, aligning either slot-for-slot (10-char
 * canonical query) or against the plate with blanks removed (display-form query).
 */
function wildcardSlots(query: string, plate: string): Set<number> {
  const q = normalisePlateQuery(query)
  const out = new Set<number>()
  const slots = plateSlots(plate)
  if (q.length === 10) {
    ;[...q].forEach((c, i) => c === '?' && out.add(i))
    return out
  }
  const nonBlank = slots.map((c, i) => (c === '_' ? -1 : i)).filter((i) => i >= 0)
  if (nonBlank.length === q.length) [...q].forEach((c, j) => c === '?' && out.add(nonBlank[j]))
  return out
}

function HitPlate({ plate, query }: { plate: string; query: string }) {
  const wild = wildcardSlots(query, plate)
  return (
    <span className="inline-flex rounded-sm border border-ink-500 bg-ink-750 px-2 py-0.5 font-mono text-[15px] font-semibold tracking-wider">
      {plateSlots(plate).map((ch, i) =>
        ch === '_' ? null : (
          <span key={i} className={cx((i === 2 || i === 4 || i === 6) && 'ml-2', wild.has(i) ? 'text-accent-strong underline decoration-accent decoration-2 underline-offset-4' : 'text-fg-strong')}>
            {ch}
          </span>
        ),
      )}
    </span>
  )
}

export function SearchPage() {
  const [sp, setSp] = useSearchParams()
  const navigate = useNavigate()
  const [draft, setDraft] = useState({
    q: sp.get('q') ?? '',
    color: sp.get('color') ?? '',
    vehicle_type: sp.get('vehicle_type') ?? '',
    from: sp.get('from') ?? '',
    to: sp.get('to') ?? '',
  })

  const submitted: SearchQuery | null = sp.get('q')
    ? {
        q: normalisePlateQuery(sp.get('q') ?? ''),
        color: sp.get('color') || undefined,
        vehicle_type: sp.get('vehicle_type') || undefined,
        from: fromLocalInput(sp.get('from') ?? ''),
        to: fromLocalInput(sp.get('to') ?? ''),
        limit: 50,
      }
    : null
  const res = useSearch(submitted)

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    const next = new URLSearchParams()
    Object.entries(draft).forEach(([k, v]) => v && next.set(k, v))
    setSp(next)
  }

  const valid = /^[A-Z0-9?_]{4,10}$/.test(normalisePlateQuery(draft.q))

  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="mx-auto flex max-w-5xl flex-col gap-3">
        <Panel title="Partial plate search">
          <form onSubmit={onSubmit} className="flex flex-col gap-3 p-3" role="search">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-[minmax(240px,1.4fr)_repeat(2,minmax(110px,0.6fr))]">
              <Field label="Plate · use ? for each unknown character">
                <input
                  className={cx(inputCls, 'h-10 font-mono text-lg tracking-[0.12em] uppercase')}
                  value={draft.q}
                  onChange={(e) => setDraft({ ...draft, q: e.target.value })}
                  placeholder="MH12??1234"
                  spellCheck={false}
                  autoComplete="off"
                  aria-describedby="search-help"
                  autoFocus
                />
              </Field>
              <Field label="Colour">
                <select className={cx(inputCls, 'h-10')} value={draft.color} onChange={(e) => setDraft({ ...draft, color: e.target.value })}>
                  <option value="">Any</option>
                  {COLORS.map((c) => (
                    <option key={c} value={c}>
                      {titleCase(c)}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Vehicle type">
                <select className={cx(inputCls, 'h-10')} value={draft.vehicle_type} onChange={(e) => setDraft({ ...draft, vehicle_type: e.target.value })}>
                  <option value="">Any</option>
                  {TYPES.map((c) => (
                    <option key={c} value={c}>
                      {titleCase(c)}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <div className="flex flex-wrap items-end gap-3">
              <Field label="From (UTC)" className="w-52">
                <input type="datetime-local" className={inputCls} value={draft.from} onChange={(e) => setDraft({ ...draft, from: e.target.value })} />
              </Field>
              <Field label="To (UTC)" className="w-52">
                <input type="datetime-local" className={inputCls} value={draft.to} onChange={(e) => setDraft({ ...draft, to: e.target.value })} />
              </Field>
              <Button type="submit" variant="primary" className="h-8 px-4" disabled={!valid}>
                Search
              </Button>
              <p id="search-help" className="text-[11px] text-fg-dim md:ml-auto md:max-w-sm">
                Ranked by the probability that each trajectory's fused plate posterior satisfies the pattern — not by string equality, so misread plates still surface.
              </p>
            </div>
          </form>
        </Panel>

        <Panel title={submitted ? `Results for ${submitted.q}` : 'Results'}>
          {!submitted && (
            <EmptyState title="Enter a plate pattern" className="h-40">
              Try <button type="button" className="font-mono text-accent hover:underline" onClick={() => { setDraft({ ...draft, q: 'MH12??1234' }); setSp(new URLSearchParams({ q: 'MH12??1234' })) }}>MH12??1234</button>
            </EmptyState>
          )}
          {submitted && res.isLoading && <Loading label="Searching" className="h-40" />}
          {submitted && res.isError && <ErrorState error={res.error} className="h-40" />}
          {submitted && res.data && res.data.length === 0 && <EmptyState title="No trajectory matches this pattern" className="h-40" />}
          {submitted && res.data && res.data.length > 0 && (
            <ol className={cx(res.isFetching && 'opacity-60')}>
              {res.data.map((h, i) => (
                <li key={h.trajectory.trajectory_id} className="border-b border-ink-750 last:border-b-0">
                  <button
                    type="button"
                    onClick={() => navigate(`/trajectories/${h.trajectory.trajectory_id}`)}
                    className="grid w-full grid-cols-[28px_minmax(150px,auto)_minmax(120px,1fr)_auto] items-center gap-3 px-3 py-2 text-left hover:bg-ink-800 max-md:grid-cols-[28px_1fr] max-md:gap-y-1"
                  >
                    <span className="num text-right font-mono text-xs text-fg-dim">{i + 1}</span>
                    <HitPlate plate={h.matched_plate} query={submitted.q} />
                    <div className="flex items-center gap-2 max-md:col-start-2">
                      <Bar value={h.probability} className="h-2 max-w-[260px]" />
                      <span className="num w-16 text-right font-mono text-xs text-fg-strong">{fmtProb(h.probability)}</span>
                    </div>
                    <div className="flex items-center gap-3 text-[11px] text-fg-muted max-md:col-start-2">
                      <span className="flex items-center gap-1.5">
                        <Swatch color={h.trajectory.color} /> {h.trajectory.color} {h.trajectory.vehicle_type}
                      </span>
                      <span className="num font-mono">
                        {fmtTime(h.trajectory.start_time, false)}–{fmtTime(h.trajectory.end_time, false)}
                      </span>
                      <span className="num">
                        {h.trajectory.n_events} hits · {fmtDuration((Date.parse(h.trajectory.end_time) - Date.parse(h.trajectory.start_time)) / 1000)}
                      </span>
                      {h.trajectory.has_alert && <span className="font-bold text-alert">▲ alert</span>}
                    </div>
                  </button>
                </li>
              ))}
            </ol>
          )}
          {submitted && res.data && res.data.length > 0 && (
            <p className="border-t border-ink-700 px-3 py-2 text-[11px] text-fg-dim">
              <span className="text-accent-strong underline decoration-accent decoration-2 underline-offset-4">Underlined</span> characters were unknown in the query and filled in by the trajectory's consensus plate.
            </p>
          )}
        </Panel>
      </div>
    </div>
  )
}
