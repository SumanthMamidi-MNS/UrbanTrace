import { useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useCity, useTrajectories } from '../api/hooks'
import type { TrajectoriesQuery } from '../api/types'
import { Button, cx, EmptyState, ErrorState, Field, inputCls, Loading, Plate, Swatch } from '../components/ui'
import { useLive } from '../hooks/liveStore'
import { fmtDuration, fmtNum, fmtTime, fromLocalInput } from '../lib/format'
import { TrajectoryDetailView } from './TrajectoryDetail'

const PAGE = 50

export function TrajectoriesPage() {
  const { id } = useParams()
  const [sp, setSp] = useSearchParams()
  const navigate = useNavigate()
  const city = useCity()
  const liveUpdates = useLive((s) => s.trajectoryUpdatesSinceLoad)
  const [seenUpdates, setSeenUpdates] = useState(liveUpdates)

  const offset = Number(sp.get('offset') ?? 0) || 0
  const q: TrajectoriesQuery = {
    plate: sp.get('plate') || undefined,
    camera_id: sp.get('camera_id') || undefined,
    min_len: sp.get('min_len') ? Number(sp.get('min_len')) : undefined,
    has_alert: sp.get('has_alert') === 'true' ? true : sp.get('has_alert') === 'false' ? false : undefined,
    from: fromLocalInput(sp.get('from') ?? ''),
    to: fromLocalInput(sp.get('to') ?? ''),
    limit: PAGE,
    offset,
  }
  const list = useTrajectories(q)

  const set = (k: string, v: string) => {
    const next = new URLSearchParams(sp)
    if (v) next.set(k, v)
    else next.delete(k)
    next.delete('offset')
    setSp(next, { replace: true })
  }
  const setOffset = (o: number) => {
    const next = new URLSearchParams(sp)
    if (o > 0) next.set('offset', String(o))
    else next.delete('offset')
    setSp(next, { replace: true })
  }
  const search = sp.toString() ? `?${sp.toString()}` : ''
  const pending = liveUpdates - seenUpdates

  return (
    <div className="flex h-full min-h-0 flex-col md:flex-row">
      <aside className={cx('flex min-h-0 flex-col border-ink-700 bg-ink-850 md:w-[330px] md:shrink-0 md:border-r', id ? 'hidden md:flex' : 'flex flex-1 md:flex-none')}>
        <form className="grid grid-cols-2 gap-2 border-b border-ink-700 p-3" onSubmit={(e) => e.preventDefault()} aria-label="Filter trajectories">
          <Field label="Plate contains" className="col-span-2">
            <input className={cx(inputCls, 'font-mono uppercase')} placeholder="MH12 or MH12??1234" defaultValue={sp.get('plate') ?? ''} onChange={(e) => set('plate', e.target.value.trim())} spellCheck={false} />
          </Field>
          <Field label="Camera">
            <select className={inputCls} value={sp.get('camera_id') ?? ''} onChange={(e) => set('camera_id', e.target.value)}>
              <option value="">Any</option>
              {city.data?.cameras.map((c) => (
                <option key={c.camera_id} value={c.camera_id}>
                  {c.camera_id}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Alert">
            <select className={inputCls} value={sp.get('has_alert') ?? ''} onChange={(e) => set('has_alert', e.target.value)}>
              <option value="">Any</option>
              <option value="true">With alert</option>
              <option value="false">No alert</option>
            </select>
          </Field>
          <Field label="From (UTC)">
            <input type="datetime-local" className={inputCls} value={sp.get('from') ?? ''} onChange={(e) => set('from', e.target.value)} />
          </Field>
          <Field label="To (UTC)">
            <input type="datetime-local" className={inputCls} value={sp.get('to') ?? ''} onChange={(e) => set('to', e.target.value)} />
          </Field>
          <Field label="Min sightings">
            <input type="number" min={1} className={inputCls} value={sp.get('min_len') ?? ''} onChange={(e) => set('min_len', e.target.value)} />
          </Field>
          <div className="flex items-end">
            <Button variant="ghost" className="w-full" onClick={() => setSp(new URLSearchParams(), { replace: true })}>
              Clear filters
            </Button>
          </div>
        </form>

        <div className="flex h-8 shrink-0 items-center justify-between border-b border-ink-700 px-3 text-[11px] text-fg-dim">
          <span className="num">{list.data ? `${fmtNum(list.data.total)} trajectories` : ' '}</span>
          {pending > 0 && (
            <button
              type="button"
              className="rounded-sm border border-accent-dim px-1.5 text-accent hover:border-accent"
              onClick={() => {
                setSeenUpdates(liveUpdates)
                void list.refetch()
              }}
            >
              {pending > 99 ? '99+' : pending} live updates · refresh
            </button>
          )}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {list.isLoading && <Loading />}
          {list.isError && <ErrorState error={list.error} />}
          {list.data && list.data.items.length === 0 && <EmptyState title="No trajectories match">Loosen the filters.</EmptyState>}
          <ul>
            {list.data?.items.map((t) => {
              const active = t.trajectory_id === id
              return (
                <li key={t.trajectory_id}>
                  <Link
                    to={`/trajectories/${t.trajectory_id}${search}`}
                    aria-current={active ? 'page' : undefined}
                    className={cx('block border-b border-ink-750 px-3 py-2', active ? 'bg-ink-750 shadow-[inset_2px_0_0_var(--color-accent)]' : 'hover:bg-ink-800')}
                  >
                    <div className="flex items-center gap-2">
                      <Plate value={t.decoded_plate} size="sm" />
                      {t.has_alert && (
                        <span className="text-[10px] font-bold tracking-wider text-alert uppercase" title="Has alert">
                          ▲ alert
                        </span>
                      )}
                      <span className="num ml-auto font-mono text-[11px] text-fg-muted">{fmtTime(t.end_time, false)}</span>
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-[11px] text-fg-dim">
                      <Swatch color={t.color} />
                      <span>
                        {t.color} {t.vehicle_type}
                      </span>
                      <span className="num ml-auto">
                        {t.n_events} hits · {fmtDuration((Date.parse(t.end_time) - Date.parse(t.start_time)) / 1000)} · {(t.plate_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </Link>
                </li>
              )
            })}
          </ul>
        </div>
        {list.data && list.data.total > PAGE && (
          <div className="flex h-10 shrink-0 items-center justify-between border-t border-ink-700 px-3 text-[11px] text-fg-dim">
            <Button onClick={() => setOffset(Math.max(0, offset - PAGE))} disabled={offset === 0}>
              ← Newer
            </Button>
            <span className="num">
              {offset + 1}–{Math.min(offset + PAGE, list.data.total)} of {fmtNum(list.data.total)}
            </span>
            <Button onClick={() => setOffset(offset + PAGE)} disabled={offset + PAGE >= list.data.total}>
              Older →
            </Button>
          </div>
        )}
      </aside>

      <section className={cx('min-h-0 min-w-0 flex-1 overflow-y-auto', !id && 'hidden md:block')}>
        {id ? (
          <>
            <div className="border-b border-ink-700 px-3 py-1.5 md:hidden">
              <Button variant="ghost" onClick={() => navigate(`/trajectories${search}`)}>
                ← Back to list
              </Button>
            </div>
            <TrajectoryDetailView key={id} id={id} />
          </>
        ) : (
          <EmptyState title="Select a trajectory">
            Pick a trajectory to replay its path, see why each pair of reads was linked, and how the plate was repaired by consensus.
          </EmptyState>
        )}
      </section>
    </div>
  )
}
