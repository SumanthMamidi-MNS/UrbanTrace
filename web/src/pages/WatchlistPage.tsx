import { useMemo, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { ApiError } from '../api/client'
import { useAddWatchlist, useDeleteWatchlist, useWatchlist, useWatchlistHits } from '../api/hooks'
import type { WatchlistEntry } from '../api/types'
import { IconClose, IconWatch } from '../components/icons'
import { MatchedOnTag, PatternReadChars } from '../components/Watchlist'
import { Button, cx, EmptyState, ErrorState, Field, inputCls, Loading, Panel, Plate } from '../components/ui'
import { USE_MOCK } from '../config'
import { fmtDateTime, fmtNum, fmtPct, fmtTime } from '../lib/format'
import { closestPattern, expandPlatePattern, formatPlate, normalisePlateQuery } from '../lib/plate'

const EXAMPLES = ['MH12AB1234', 'MH12??1234', 'DL3C?456']

function AddForm({ entries }: { entries: WatchlistEntry[] }) {
  const [pattern, setPattern] = useState('')
  const [reason, setReason] = useState('')
  const [added, setAdded] = useState<WatchlistEntry | null>(null)
  const add = useAddWatchlist()
  const norm = normalisePlateQuery(pattern)
  const preview = norm ? expandPlatePattern(norm) : null
  // Same canonical layouts = same watch; guard against double submits and duplicate entries.
  const duplicate = preview?.ok ? entries.find((e) => e.canonical_patterns.length === preview.forms.length && preview.forms.every((f) => e.canonical_patterns.includes(f))) : undefined
  const canSubmit = Boolean(preview?.ok) && !duplicate && reason.trim().length > 0 && !add.isPending

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return
    add.mutate(
      { pattern: norm, reason: reason.trim() },
      {
        onSuccess: (entry) => {
          setAdded(entry)
          setPattern('')
          setReason('')
        },
      },
    )
  }

  const serverError = add.error instanceof ApiError ? add.error.message : add.error ? 'Could not add the entry.' : null

  return (
    <Panel title="Add to watchlist">
      <form onSubmit={submit} className="flex flex-col gap-3 p-3" noValidate>
        <Field label="Plate or pattern">
          <input
            className={cx(inputCls, 'h-9 font-mono text-[15px] tracking-wider uppercase')}
            value={pattern}
            onChange={(e) => {
              setPattern(e.target.value)
              setAdded(null)
              add.reset()
            }}
            placeholder="MH12AB1234"
            autoComplete="off"
            spellCheck={false}
            aria-describedby="wl-pattern-help"
            aria-invalid={preview?.ok === false}
          />
        </Field>
        <div id="wl-pattern-help" className="-mt-1.5 min-h-[34px] text-[11px] leading-snug" aria-live="polite">
          {!preview && (
            <span className="text-fg-dim">
              Use <span className="font-mono text-fg">?</span> for a character you don’t know. Try{' '}
              {EXAMPLES.map((x, i) => (
                <span key={x}>
                  <button type="button" className="font-mono text-accent hover:underline" onClick={() => setPattern(x)}>
                    {x}
                  </button>
                  {i < EXAMPLES.length - 1 ? ', ' : ''}
                </span>
              ))}
              .
            </span>
          )}
          {preview?.ok === false && <span className="text-caution">{preview.detail}</span>}
          {preview?.ok && duplicate && (
            <span className="text-caution">
              Already watching this as <span className="font-mono">{duplicate.entry_id}</span> ({duplicate.reason || 'no reason given'}).
            </span>
          )}
          {preview?.ok && !duplicate && (
            <span className="text-fg-muted">
              Watches {preview.forms.length === 1 ? 'the plate layout' : `${preview.forms.length} possible layouts`}{' '}
              {preview.forms.map((f, i) => (
                <span key={f}>
                  <span className="font-mono text-fg">{formatPlate(f)}</span>
                  {i < preview.forms.length - 1 ? ' or ' : ''}
                </span>
              ))}
            </span>
          )}
        </div>
        <Field label="Reason">
          <input className={inputCls} value={reason} onChange={(e) => setReason(e.target.value)} placeholder="e.g. stolen vehicle, FIR 1234/2026" maxLength={200} />
        </Field>
        <div className="flex items-center gap-3">
          <Button type="submit" variant="primary" disabled={!canSubmit} className="h-8 px-3">
            <IconWatch width={14} height={14} /> {add.isPending ? 'Adding…' : 'Watch this plate'}
          </Button>
          {!reason.trim() && preview?.ok && !duplicate && <span className="text-[11px] text-fg-dim">Add a reason to continue.</span>}
        </div>
        {serverError && (
          <p role="alert" className="rounded-sm border border-caution-dim bg-caution-faint px-2 py-1.5 text-xs text-caution">
            {serverError}
          </p>
        )}
        {added && (
          <p role="status" className="rounded-sm border border-accent-dim bg-accent-faint/50 px-2 py-1.5 text-xs text-fg">
            Watching <span className="font-mono">{added.pattern}</span> ({added.entry_id}). Matches on live reads will raise an alert.
            {USE_MOCK && <span className="text-fg-muted"> Mock mode plants two target vehicles, due within ~1–11 sim minutes; keep the replay running.</span>}
          </p>
        )}
      </form>
    </Panel>
  )
}

function EntryRow({ e, selected, onSelect }: { e: WatchlistEntry; selected: boolean; onSelect: () => void }) {
  const del = useDeleteWatchlist()
  const [confirm, setConfirm] = useState(false)
  return (
    <li className={cx('border-b border-ink-750', selected && 'bg-ink-800 shadow-[inset_2px_0_0_var(--color-accent)]')}>
      <div className="flex items-start gap-2 px-3 py-2">
        <button type="button" onClick={onSelect} aria-pressed={selected} className="min-w-0 flex-1 text-left" aria-label={`${e.pattern}, ${e.entry_id}: show only this entry’s hits`}>
          <div className="flex items-center gap-2">
            <Plate value={e.pattern} size="sm" tone="alert" />
            <span className="font-mono text-[10px] text-fg-dim">{e.entry_id}</span>
            {!e.active && <span className="text-[10px] text-fg-dim">inactive</span>}
          </div>
          <p className="mt-1 line-clamp-2 text-xs text-fg">{e.reason || <span className="text-fg-dim">No reason given</span>}</p>
          <p className="mt-0.5 text-[10px] text-fg-dim">
            since {fmtDateTime(e.created_at)} UTC
            {e.canonical_patterns.length > 1 && ` · ${e.canonical_patterns.length} layouts`}
          </p>
        </button>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <span className={cx('num font-mono text-[15px] leading-none', e.hits > 0 ? 'text-alert' : 'text-fg-muted')}>{fmtNum(e.hits)}</span>
          <span className="text-[10px] text-fg-dim">{e.hits === 1 ? 'hit' : 'hits'}</span>
        </div>
      </div>
      <div className="flex items-center justify-end gap-2 px-3 pb-2">
        {confirm ? (
          <>
            <span className="text-[11px] text-fg-muted">Stop watching?</span>
            <Button className="h-6 border-alert-dim px-2 text-alert hover:border-alert" disabled={del.isPending} onClick={() => del.mutate(e.entry_id)}>
              {del.isPending ? 'Removing…' : 'Remove'}
            </Button>
            <Button variant="ghost" className="h-6 px-2" onClick={() => setConfirm(false)}>
              Keep
            </Button>
          </>
        ) : (
          <Button variant="ghost" className="h-6 px-2 text-[11px]" onClick={() => setConfirm(true)} aria-label={`Remove ${e.pattern} from watchlist`}>
            <IconClose width={12} height={12} /> Remove
          </Button>
        )}
      </div>
      {del.isError && (
        <p role="alert" className="px-3 pb-2 text-[11px] text-caution">
          {del.error instanceof Error ? del.error.message : 'Could not remove.'}
        </p>
      )}
    </li>
  )
}

function HitsTable({ entryId, entries }: { entryId: string | null; entries: WatchlistEntry[] }) {
  const hits = useWatchlistHits({ entry_id: entryId ?? undefined, limit: 100 }, true, 4_000)
  const forms = useMemo(() => new Map(entries.map((e) => [e.entry_id, e.canonical_patterns])), [entries])
  const selected = entries.find((e) => e.entry_id === entryId)

  return (
    <Panel
      title={selected ? `Recent hits · ${selected.pattern}` : 'Recent hits · all entries'}
      className="min-h-[360px]"
      actions={
        <Link to="/alerts?type=watchlist" className="text-[11px] text-fg-dim hover:text-fg">
          Watchlist alerts →
        </Link>
      }
      bodyClassName="overflow-x-auto"
    >
      {hits.isLoading && <Loading />}
      {hits.isError && <ErrorState error={hits.error} />}
      {hits.data?.length === 0 && (
        <EmptyState title="No hits yet">Hits appear here as live reads are matched. A hit on the fused plate is marked “Consensus”: the camera itself may have misread the plate.</EmptyState>
      )}
      {!!hits.data?.length && (
        <table className="w-full min-w-[720px] text-xs">
          <caption className="sr-only">Watchlist hits, newest first</caption>
          <thead>
            <tr className="border-b border-ink-700 text-[10px] tracking-wider text-fg-dim uppercase">
              <th className="px-3 py-2 text-left font-semibold">Time</th>
              <th className="px-2 py-2 text-left font-semibold">Entry</th>
              <th className="px-2 py-2 text-left font-semibold">Camera</th>
              <th className="px-2 py-2 text-left font-semibold">Camera read</th>
              <th className="px-2 py-2 text-left font-semibold">Matched on</th>
              <th className="w-[150px] px-2 py-2 text-left font-semibold">Probability</th>
              <th className="px-3 py-2 text-left font-semibold">Track</th>
            </tr>
          </thead>
          <tbody>
            {hits.data.map((h) => {
              const pat = closestPattern(h.plate_read, forms.get(h.entry_id) ?? []) ?? (h.pattern.length === 10 ? h.pattern : undefined)
              return (
                <tr key={h.hit_id} className="border-b border-ink-750 hover:bg-ink-800">
                  <td className="num px-3 py-1.5 font-mono text-fg-muted">{fmtTime(h.timestamp)}</td>
                  <td className="px-2 py-1.5">
                    <span className={cx('font-mono', forms.has(h.entry_id) ? 'text-fg' : 'text-fg-dim')}>{h.pattern}</span>
                    {!forms.has(h.entry_id) && <span className="ml-1.5 text-[10px] text-fg-dim">(removed)</span>}
                  </td>
                  <td className="px-2 py-1.5 font-mono text-fg-muted">{h.camera_id}</td>
                  <td className="px-2 py-1.5">
                    <PatternReadChars read={h.plate_read} pattern={pat} />
                  </td>
                  <td className="px-2 py-1.5">
                    <MatchedOnTag on={h.matched_on} />
                  </td>
                  <td className="px-2 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-700">
                        <div className="h-full rounded-full bg-alert" style={{ width: `${Math.max(0, Math.min(1, h.probability)) * 100}%` }} />
                      </div>
                      <span className="num w-11 text-right font-mono text-fg">{fmtPct(h.probability, 0)}</span>
                    </div>
                  </td>
                  <td className="px-3 py-1.5">
                    {h.trajectory_id ? (
                      <Link to={`/trajectories/${h.trajectory_id}`} className="font-mono text-accent hover:underline">
                        {h.trajectory_id}
                      </Link>
                    ) : (
                      <span className="text-fg-dim" title="First sighting, not yet linked to a trajectory">
                        unlinked
                      </span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </Panel>
  )
}

export function WatchlistPage() {
  const list = useWatchlist(4_000)
  const [selected, setSelected] = useState<string | null>(null)
  const entries = list.data ?? []
  const sel = entries.some((e) => e.entry_id === selected) ? selected : null
  const totalHits = entries.reduce((a, e) => a + e.hits, 0)

  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[380px_minmax(0,1fr)]">
        <div className="flex flex-col gap-3">
          <AddForm entries={entries} />
          <Panel
            title={`Watching ${entries.length ? fmtNum(entries.length) : ''} ${entries.length === 1 ? 'plate' : 'plates'}`}
            actions={
              sel ? (
                <Button variant="ghost" className="h-6 px-2 text-[11px]" onClick={() => setSelected(null)}>
                  Show all hits
                </Button>
              ) : undefined
            }
          >
            {list.isLoading && <Loading />}
            {list.isError && <ErrorState error={list.error} />}
            {list.data?.length === 0 && <EmptyState title="The watchlist is empty">Add a plate above to be alerted when it is seen anywhere on the network.</EmptyState>}
            <ul>
              {entries.map((e) => (
                <EntryRow key={e.entry_id} e={e} selected={sel === e.entry_id} onSelect={() => setSelected(sel === e.entry_id ? null : e.entry_id)} />
              ))}
            </ul>
          </Panel>
        </div>
        <div className="flex min-w-0 flex-col gap-3">
          <div className="grid grid-cols-1 gap-3 rounded border border-ink-700 bg-ink-850 p-3 md:grid-cols-[1fr_auto]">
            <div className="text-xs leading-relaxed text-fg-muted">
              <p className="text-[13px] font-semibold text-fg-strong">Caught even when a camera misreads the plate</p>
              <p className="mt-1">
                Every live read is checked twice: on the camera’s own read, and on the vehicle’s <span className="text-fg">fused plate</span> across all the cameras it has passed. A match needs a probability of at
                least 50%. A dirty or damaged plate that no single camera reads correctly is still caught once the cameras together agree.
              </p>
            </div>
            <div className="flex gap-4 md:border-l md:border-ink-700 md:pl-4">
              <div>
                <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Entries</div>
                <div className="num font-mono text-[22px] text-fg-strong">{fmtNum(entries.length)}</div>
              </div>
              <div>
                <div className="text-[10px] font-semibold tracking-[0.08em] text-fg-dim uppercase">Hits</div>
                <div className={cx('num font-mono text-[22px]', totalHits ? 'text-alert' : 'text-fg-strong')}>{fmtNum(totalHits)}</div>
              </div>
            </div>
          </div>
          <HitsTable entryId={sel} entries={entries} />
        </div>
      </div>
    </div>
  )
}
