import { useEffect, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { useHealth, useReplay } from '../api/hooks'
import { USE_MOCK } from '../config'
import { liveStore, useLive } from '../hooks/liveStore'
import { fmtDateTime, fmtNum } from '../lib/format'
import { IconAlert, IconChart, IconLive, IconPause, IconPlay, IconReset, IconResults, IconRoute, IconSearch, IconWatch } from './icons'
import { Button, cx, Segmented } from './ui'
import { BrandMark, Wordmark } from './Wordmark'

const NAV = [
  { to: '/live', label: 'Live', icon: IconLive },
  { to: '/trajectories', label: 'Tracks', icon: IconRoute },
  { to: '/search', label: 'Search', icon: IconSearch },
  { to: '/analytics', label: 'Analytics', icon: IconChart },
  { to: '/alerts', label: 'Alerts', icon: IconAlert },
  { to: '/watchlist', label: 'Watchlist', icon: IconWatch },
  { to: '/results', label: 'Results', icon: IconResults },
]

const SPEEDS = [1, 10, 60, 300]

function AlertNavBadge() {
  const [unseen, setUnseen] = useState(0)
  const qc = useQueryClient()
  const onAlertsPage = useLocation().pathname.startsWith('/alerts')
  const onAlertsRef = useRef(onAlertsPage)
  useEffect(() => {
    onAlertsRef.current = onAlertsPage
    if (onAlertsPage) setUnseen(0)
  }, [onAlertsPage])
  useEffect(
    () =>
      liveStore.onAlert((a) => {
        if (!onAlertsRef.current) setUnseen((n) => n + 1)
        void qc.invalidateQueries({ queryKey: ['alerts'] })
        if (a.type === 'watchlist') void qc.invalidateQueries({ queryKey: ['watchlist'] })
        void qc.invalidateQueries({ queryKey: ['analytics', 'summary'] })
      }),
    [qc],
  )
  if (!unseen) return null
  return (
    <span className="num pointer-events-none absolute top-1 right-1.5 min-w-4 rounded-full bg-alert px-1 text-center text-[10px] leading-4 font-bold text-ink-950">
      <span className="sr-only">new alerts: </span>
      {unseen > 99 ? '99+' : unseen}
    </span>
  )
}

function ReplayBar() {
  const clock = useLive((s) => s.clock)
  const status = useLive((s) => s.status)
  const replay = useReplay()
  const running = clock?.running ?? false
  const speed = clock?.speed ?? 1

  const send = (action: 'start' | 'pause' | 'reset', sp?: number) =>
    replay.mutate(sp === undefined ? { action } : { action, speed: sp }, {
      onSuccess: (state) => liveStore.setClock(state),
    })

  return (
    <div className="flex items-center gap-2">
      <div className="hidden items-center gap-2 pr-1 md:flex">
        <span className={cx('h-2 w-2 rounded-full', status === 'open' ? (running ? 'bg-accent' : 'bg-fg-muted') : status === 'connecting' ? 'animate-pulse bg-caution' : 'bg-alert')} aria-hidden />
        <span className="text-[10px] font-semibold tracking-[0.1em] text-fg-dim uppercase">{status !== 'open' ? (status === 'connecting' ? 'Connecting' : 'Feed down') : running ? 'Replaying' : 'Paused'}</span>
      </div>
      <div className="num rounded-sm border border-ink-700 bg-ink-950 px-2.5 py-1 font-mono text-[13px] text-fg-strong" aria-label="Simulation clock" aria-live="off">
        {clock ? fmtDateTime(clock.sim_time) : '—— --:--:--'}
        <span className="ml-1.5 text-[10px] text-fg-dim">UTC</span>
      </div>
      <Button
        variant={running ? 'default' : 'primary'}
        onClick={() => send(running ? 'pause' : 'start', speed)}
        disabled={replay.isPending || status !== 'open'}
        aria-label={running ? 'Pause replay' : 'Start replay'}
        className="w-8 px-0"
      >
        {running ? <IconPause width={14} height={14} /> : <IconPlay width={14} height={14} />}
      </Button>
      <Button onClick={() => send('reset')} disabled={replay.isPending || status !== 'open'} aria-label="Reset replay" className="w-8 px-0">
        <IconReset width={14} height={14} />
      </Button>
      <Segmented
        label="Replay speed"
        value={SPEEDS.includes(speed) ? speed : 1}
        options={SPEEDS.map((s) => ({ value: s, label: `${s}×` }))}
        onChange={(s) => send(running ? 'start' : 'pause', s)}
      />
    </div>
  )
}

export function Layout() {
  const health = useHealth()
  const eventsPerMin = useLive((s) => s.arrivals.length)

  return (
    <div className="flex h-full min-h-0 bg-ink-900">
      <nav aria-label="Primary" className="flex w-[68px] shrink-0 flex-col items-stretch border-r border-ink-700 bg-ink-950">
        <div className="flex h-12 items-center justify-center border-b border-ink-700" title="UrbanTrace">
          <BrandMark size={28} />
        </div>
        <ul className="flex flex-col py-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <li key={to} className="relative">
              <NavLink
                to={to}
                className={({ isActive }) =>
                  cx(
                    'flex flex-col items-center gap-1 py-2.5 text-[10px] font-medium tracking-wide transition-colors',
                    isActive ? 'bg-ink-800 text-accent-strong shadow-[inset_2px_0_0_var(--color-accent)]' : 'text-fg-muted hover:bg-ink-850 hover:text-fg',
                  )
                }
              >
                <Icon />
                {label}
              </NavLink>
              {to === '/alerts' && <AlertNavBadge />}
            </li>
          ))}
        </ul>
      </nav>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-ink-700 bg-ink-950 px-3">
          <div className="flex min-w-0 items-center gap-3">
            <Wordmark />
            <span className="hidden h-4 w-px shrink-0 bg-ink-600 lg:block" aria-hidden />
            <span className="hidden min-w-0 truncate text-xs text-fg-dim lg:block" title="City-scale vehicle tracking that reasons in probabilities, not string matches.">
              City-scale vehicle tracking that reasons in probabilities, not string matches.
            </span>
            {USE_MOCK && (
              <span className="rounded-sm border border-caution-dim px-1.5 text-[10px] font-semibold tracking-wider text-caution uppercase" title="VITE_USE_MOCK=true: in-browser fixture data, no backend">
                Mock data
              </span>
            )}
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden items-center gap-4 text-[11px] text-fg-dim xl:flex">
              <span className="num">
                <span className="font-mono text-fg">{fmtNum(eventsPerMin)}</span> reads/min
              </span>
              {health.data && (
                <span className="num">
                  <span className="font-mono text-fg">{fmtNum(health.data.n_trajectories)}</span> tracks ·{' '}
                  <span className="font-mono text-fg">{fmtNum(health.data.n_events)}</span> reads
                </span>
              )}
              {health.isError && <span className="text-alert">API unreachable</span>}
            </div>
            <ReplayBar />
          </div>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
