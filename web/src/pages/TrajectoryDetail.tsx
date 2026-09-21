import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAlerts, useCity, useTrajectory } from '../api/hooks'
import type { City, LinkEvidence, TrajectoryDetail as TDetail } from '../api/types'
import { CityMap, type MapLabel, type MapLine, type MapPoint } from '../components/CityMap'
import { ConsensusPanel } from '../components/ConsensusPanel'
import { IconPause, IconPlay, IconReset } from '../components/icons'
import { Button, cx, EmptyState, ErrorState, Loading, Panel, Plate, SeverityBadge, Stat, Swatch, alertTypeLabel } from '../components/ui'
import { ReadChars, WhyPanel } from '../components/WhyPanel'
import { fmtDateTime, fmtDuration, fmtLogOdds, fmtPct, fmtTime } from '../lib/format'
import { getRouter, sliceLine, type LngLat } from '../lib/geo'
import { mismatchSlots } from '../lib/plate'

const REPLAY_MS = 7000
const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/** The link a judge should see first: weakest finite plate evidence (the "linking repaired OCR" case). */
function defaultLink(links: LinkEvidence[]): number {
  let best = 0
  let bestV = Infinity
  links.forEach((l, i) => {
    const v = l.plate_lr === null ? -Infinity : l.plate_lr
    if (v < bestV) {
      bestV = v
      best = i
    }
  })
  return best
}

function useReplayProgress(key: string) {
  const [progress, setProgress] = useState(1)
  const [playing, setPlaying] = useState(false)
  const startRef = useRef(0)

  useEffect(() => {
    if (reducedMotion()) return
    startRef.current = performance.now()
    setProgress(0)
    setPlaying(true)
  }, [key])

  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = 0
    const tick = (now: number) => {
      if (now - last > 30) {
        last = now
        const p = Math.min(1, (now - startRef.current) / REPLAY_MS)
        setProgress(p)
        if (p >= 1) {
          setPlaying(false)
          return
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing])

  return {
    progress,
    playing,
    play: () => {
      const from = progress >= 1 ? 0 : progress
      startRef.current = performance.now() - from * REPLAY_MS
      setPlaying(true)
    },
    pause: () => setPlaying(false),
    restart: () => {
      startRef.current = performance.now()
      setProgress(0)
      setPlaying(true)
    },
  }
}

/** Road-following segments between consecutive hits, and time-proportional replay position. */
function useReplayGeometry(city: City | undefined, t: TDetail | undefined, progress: number) {
  const segments = useMemo(() => {
    if (!city || !t) return [] as LngLat[][]
    const r = getRouter(city)
    const seq = t.path.map((p) => p.camera_id)
    const segs: LngLat[][] = []
    for (let i = 1; i < seq.length; i++) {
      const s = r.between(seq[i - 1], seq[i])
      segs.push(s.length ? s : [[t.path[i - 1].lon, t.path[i - 1].lat], [t.path[i].lon, t.path[i].lat]])
    }
    return segs
  }, [city, t])

  return useMemo(() => {
    if (!t || t.path.length === 0) return { revealed: [] as LngLat[], head: null as LngLat | null, hitIndex: -1, segments }
    const times = t.path.map((p) => Date.parse(p.timestamp))
    const t0 = times[0]
    const span = Math.max(1, times[times.length - 1] - t0)
    const now = t0 + progress * span
    let k = 0
    while (k < times.length - 1 && times[k + 1] <= now) k++
    const revealed: LngLat[] = []
    for (let i = 0; i < k && i < segments.length; i++) revealed.push(...(i === 0 ? segments[i] : segments[i].slice(1)))
    let head: LngLat | null = [t.path[k].lon, t.path[k].lat]
    if (k < segments.length) {
      const f = (now - times[k]) / Math.max(1, times[k + 1] - times[k])
      const part = sliceLine(segments[k], f)
      revealed.push(...(revealed.length ? part.line.slice(1) : part.line))
      head = part.head
    }
    if (revealed.length === 0 && head) revealed.push(head)
    return { revealed, head, hitIndex: k, segments }
  }, [t, segments, progress])
}

export function TrajectoryDetailView({ id }: { id: string }) {
  const q = useTrajectory(id)
  const city = useCity()
  const alerts = useAlerts({ limit: 200 })
  const t = q.data
  const [selected, setSelected] = useState(0)
  const whyRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (t) setSelected(defaultLink(t.links))
  }, [t?.trajectory_id]) // eslint-disable-line react-hooks/exhaustive-deps

  const replay = useReplayProgress(id)
  const geo = useReplayGeometry(city.data, t, replay.progress)

  const camName = useMemo(() => new Map(city.data?.cameras.map((c) => [c.camera_id, c.name]) ?? []), [city.data])

  const mapLayers = useMemo(() => {
    const lines: MapLine[] = []
    const points: MapPoint[] = []
    const labels: MapLabel[] = []
    if (!t || !geo.segments) return { lines, points, labels }
    geo.segments.forEach((s, i) => lines.push({ id: `seg-${i}`, coords: s, color: '#1d6f7c', width: 3, opacity: 0.7 }))
    lines.push({ id: 'revealed', coords: geo.revealed, color: '#3cc4d8', width: 4, opacity: 1 })
    const sel = geo.segments[selected]
    if (sel && replay.progress >= 1) lines.push({ id: 'selected', coords: sel, color: '#eef3f8', width: 5, opacity: 0.9 })
    t.path.forEach((p, i) => {
      const passed = i <= geo.hitIndex
      points.push({ id: p.event_id, lngLat: [p.lon, p.lat], color: passed ? '#3cc4d8' : '#182029', stroke: passed ? '#070a0e' : '#3cc4d8', radius: 5.5, strokeWidth: 2 })
    })
    if (geo.head) points.push({ id: 'head', lngLat: geo.head, color: '#eef3f8', radius: 5, stroke: '#3cc4d8', strokeWidth: 2.5 })
    const first = t.path[0]
    const last = t.path[t.path.length - 1]
    if (first) labels.push({ id: 'start', lngLat: [first.lon, first.lat], text: `START ${fmtTime(first.timestamp, false)}`, tone: 'neutral' })
    if (last && t.path.length > 1) labels.push({ id: 'end', lngLat: [last.lon, last.lat], text: `END ${fmtTime(last.timestamp, false)}`, tone: 'accent' })
    return { lines, points, labels }
  }, [t, geo, selected, replay.progress])

  const fitCoords = useMemo(() => (t ? t.path.map((p) => [p.lon, p.lat] as LngLat) : undefined), [t])

  if (q.isLoading) return <Loading label="Loading trajectory" />
  if (q.isError) return <ErrorState error={q.error} />
  if (!t) return <EmptyState title="Trajectory not found" />

  const related = (alerts.data ?? []).filter((a) => a.trajectory_ids.includes(t.trajectory_id))
  const durationS = (Date.parse(t.end_time) - Date.parse(t.start_time)) / 1000
  const repaired = t.consensus.single_read_plates.filter((r) => mismatchSlots(r, t.decoded_plate).length > 0).length
  const selectLink = (i: number) => {
    setSelected(i)
    whyRef.current?.scrollIntoView({ block: 'nearest', behavior: reducedMotion() ? 'auto' : 'smooth' })
  }

  return (
    <div className="flex flex-col gap-3 p-3">
      {/* header */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded border border-ink-700 bg-ink-850 px-3 py-2.5">
        <div className="flex items-center gap-3">
          <Plate value={t.decoded_plate} size="lg" />
          <div>
            <div className="font-mono text-[11px] text-fg-dim">{t.trajectory_id}</div>
            <div className="flex items-center gap-1.5 text-xs text-fg-muted">
              <Swatch color={t.color} /> {t.color} {t.vehicle_type}
            </div>
          </div>
        </div>
        <Stat label="Plate confidence" value={fmtPct(t.plate_confidence, 2)} />
        <Stat label="Window" value={`${fmtTime(t.start_time)} – ${fmtTime(t.end_time)}`} sub={fmtDateTime(t.start_time).slice(0, 10)} />
        <Stat label="Duration" value={fmtDuration(durationS)} />
        <Stat label="Sightings" value={`${t.n_events} reads · ${t.links.length} links`} />
        <Stat label="Reads repaired" value={<span className={repaired ? 'text-caution' : undefined}>{repaired} of {t.consensus.single_read_plates.length}</span>} />
        {related.length > 0 && (
          <div className="ml-auto flex flex-col gap-1">
            {related.map((a) => (
              <Link key={a.alert_id} to={`/alerts/${a.alert_id}`} className="flex items-center gap-2 rounded-sm border border-alert-dim bg-alert-faint px-2 py-1 text-xs text-alert hover:border-alert">
                <SeverityBadge severity={a.severity} /> {alertTypeLabel(a.type)} →
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* map + timeline */}
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1.35fr)_minmax(300px,1fr)]">
        <Panel
          title="Path replay"
          className="h-[340px]"
          actions={
            <>
              <span className="num font-mono text-[11px] text-fg-dim">
                {geo.hitIndex >= 0 ? `${geo.hitIndex + 1}/${t.path.length} · ${fmtTime(t.path[geo.hitIndex]?.timestamp)}` : ''}
              </span>
              <Button className="w-7 px-0" onClick={replay.playing ? replay.pause : replay.play} aria-label={replay.playing ? 'Pause path replay' : 'Play path replay'}>
                {replay.playing ? <IconPause width={12} height={12} /> : <IconPlay width={12} height={12} />}
              </Button>
              <Button className="w-7 px-0" onClick={replay.restart} aria-label="Restart path replay">
                <IconReset width={12} height={12} />
              </Button>
            </>
          }
        >
          {city.data ? (
            <CityMap
              city={city.data}
              highlightCameras={t.camera_sequence}
              lines={mapLayers.lines}
              points={mapLayers.points}
              labels={mapLayers.labels}
              fit={{ key: t.trajectory_id, coords: fitCoords, padding: 50, maxZoom: 15 }}
              ariaLabel={`Path of ${t.decoded_plate} across ${t.n_events} cameras`}
            />
          ) : city.isError ? (
            <ErrorState error={city.error} />
          ) : (
            <Loading />
          )}
        </Panel>

        <Panel title="Camera hits" className="h-[340px]" bodyClassName="overflow-y-auto">
          <ol className="py-1">
            {t.events.map((e, i) => {
              const link = t.links[i]
              const passed = i <= geo.hitIndex
              const bad = mismatchSlots(e.plate_argmax, t.decoded_plate).length > 0
              return (
                <li key={e.event_id}>
                  <div className={cx('grid grid-cols-[18px_1fr_auto] items-center gap-2 px-3 py-1', i === geo.hitIndex && replay.playing && 'bg-ink-750')}>
                    <span className={cx('flex h-[18px] w-[18px] items-center justify-center rounded-full border text-[9px] font-bold', passed ? 'border-accent bg-accent text-ink-950' : 'border-ink-500 text-fg-dim')}>{i + 1}</span>
                    <div className="min-w-0">
                      <div className="flex items-baseline gap-2">
                        <span className="num font-mono text-[11px] text-fg-muted">{fmtTime(e.timestamp)}</span>
                        <span className="font-mono text-[11px] text-fg">{e.camera_id}</span>
                        <span className="truncate text-[11px] text-fg-dim">{camName.get(e.camera_id)}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <ReadChars read={e.plate_argmax} truth={t.decoded_plate} />
                        {bad && <span className="text-[10px] font-semibold text-caution">misread</span>}
                      </div>
                    </div>
                    <span className={cx('num font-mono text-[11px]', e.plate_confidence < 0.7 ? 'text-caution' : 'text-fg-dim')}>{fmtPct(e.plate_confidence, 0)}</span>
                  </div>
                  {link && (
                    <button
                      type="button"
                      onClick={() => selectLink(i)}
                      aria-pressed={selected === i}
                      className={cx('ml-[20px] flex w-[calc(100%-20px)] items-center gap-2 border-l-2 py-1 pr-3 pl-4 text-left text-[11px] transition-colors', selected === i ? 'border-accent bg-accent-faint/60 text-fg-strong' : 'border-ink-600 text-fg-dim hover:bg-ink-800 hover:text-fg')}
                    >
                      <span className="num font-mono">+{fmtDuration(link.delta_t_s)}</span>
                      {link.skipped_cameras.length > 0 && <span>{link.skipped_cameras.length} missed</span>}
                      <span className="ml-auto">
                        log-odds <span className="num font-mono text-fg">{fmtLogOdds(link.total_log_odds, 1)}</span>
                      </span>
                      <span className="text-accent">why →</span>
                    </button>
                  )}
                </li>
              )
            })}
          </ol>
        </Panel>
      </div>

      {/* WHY + consensus */}
      <div className="grid grid-cols-1 gap-3 2xl:grid-cols-2">
        <div ref={whyRef} className="scroll-mt-3">
          <Panel title="Why these reads were linked" actions={<span className="text-[10px] text-fg-dim">click any link</span>}>
            <WhyPanel links={t.links} events={t.events} decodedPlate={t.decoded_plate} selected={selected} onSelect={setSelected} />
          </Panel>
        </div>
        <Panel title="Plate consensus — linking repairs OCR">
          <ConsensusPanel consensus={t.consensus} decodedPlate={t.decoded_plate} events={t.events} />
        </Panel>
      </div>
    </div>
  )
}
