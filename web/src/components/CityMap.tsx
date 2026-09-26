import { LngLatBounds, Map as MLMap, Marker, Popup, type ExpressionSpecification, type GeoJSONSource, type StyleSpecification } from 'maplibre-gl'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Feature, FeatureCollection, Point } from 'geojson'
import type { CameraStats, City } from '../api/types'
import { liveStore } from '../hooks/liveStore'
import type { LngLat } from '../lib/geo'
import { HEAT_RAMP } from '../lib/heat'
import { cx } from './ui'

export type MapLine = { id: string; coords: LngLat[]; color?: string; width?: number; opacity?: number; dashed?: boolean }
export type MapPoint = { id: string; lngLat: LngLat; color?: string; radius?: number; stroke?: string; strokeWidth?: number }
export type MapLabel = { id: string; lngLat: LngLat; text: string; tone?: 'accent' | 'alert' | 'neutral' }
/** Direction-of-travel arrow drawn just ahead of a point, rotated to a compass heading (0 = north). */
export type MapArrow = { id: string; lngLat: LngLat; heading: number }
/** Heat points with weight already normalised to [0, 1]. */
export type MapHeat = { points: { id: string; lngLat: LngLat; weight: number }[] }

type Props = {
  city: City
  cameraStats?: CameraStats[]
  /** cameras to emphasise (e.g. those on the selected trajectory) */
  highlightCameras?: string[]
  lines?: MapLine[]
  points?: MapPoint[]
  labels?: MapLabel[]
  arrows?: MapArrow[]
  /** heatmap layer drawn over the roads, under cameras and tracks; null/undefined hides it */
  heat?: MapHeat | null
  /** refit when this key changes; coords to fit (defaults to the whole city) */
  fit?: { key: string; coords?: LngLat[]; padding?: number; maxZoom?: number }
  /** animate a ring at a camera whenever a live event arrives */
  livePulses?: boolean
  onCameraClick?: (cameraId: string) => void
  className?: string
  ariaLabel?: string
}

const C = {
  bg: '#080b10',
  road: '#2b3643',
  roadArterial: '#3b4a5a',
  cam: '#3cc4d8',
  camFill: '#0f3940',
  camDim: '#56708a',
}

const STYLE: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [{ id: 'bg', type: 'background', paint: { 'background-color': C.bg } }],
}

const EMPTY: FeatureCollection = { type: 'FeatureCollection', features: [] }

const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

function roadsGeoJSON(city: City): FeatureCollection {
  const nodes = new Map(city.nodes.map((n) => [n.node_id, n]))
  const seen = new Set<string>()
  const features: Feature[] = []
  for (const e of city.edges) {
    const key = e.from_node < e.to_node ? `${e.from_node}|${e.to_node}` : `${e.to_node}|${e.from_node}`
    if (seen.has(key)) continue
    seen.add(key)
    const a = nodes.get(e.from_node)
    const b = nodes.get(e.to_node)
    if (!a || !b) continue
    features.push({
      type: 'Feature',
      properties: { speed: e.speed_limit_kmh },
      geometry: { type: 'LineString', coordinates: [[a.lon, a.lat], [b.lon, b.lat]] },
    })
  }
  return { type: 'FeatureCollection', features }
}

/** Arrow glyph drawn on a canvas, so no sprite server is needed offline. */
function arrowImage(): { width: number; height: number; data: Uint8Array } {
  const size = 48
  const c = document.createElement('canvas')
  c.width = c.height = size
  const ctx = c.getContext('2d') as CanvasRenderingContext2D
  ctx.beginPath()
  ctx.moveTo(24, 5)
  ctx.lineTo(40, 40)
  ctx.lineTo(24, 31)
  ctx.lineTo(8, 40)
  ctx.closePath()
  ctx.lineJoin = 'round'
  ctx.lineWidth = 5
  ctx.strokeStyle = '#070a0e'
  ctx.stroke()
  ctx.fillStyle = '#eef3f8'
  ctx.fill()
  return { width: size, height: size, data: new Uint8Array(ctx.getImageData(0, 0, size, size).data.buffer) }
}

export function CityMap({ city, cameraStats, highlightCameras, lines, points, labels, arrows, heat, fit, livePulses, onCameraClick, className, ariaLabel = 'City road network map' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MLMap | null>(null)
  const [ready, setReady] = useState(false)
  const clickRef = useRef(onCameraClick)
  const statsRef = useRef<Map<string, CameraStats>>(new Map())

  useEffect(() => {
    clickRef.current = onCameraClick
  }, [onCameraClick])

  const cityBounds = useMemo(() => {
    const b = new LngLatBounds()
    city.nodes.forEach((n) => b.extend([n.lon, n.lat]))
    return b
  }, [city])

  // ---- create map once per city
  useEffect(() => {
    if (!containerRef.current) return
    const map = new MLMap({
      container: containerRef.current,
      style: STYLE,
      bounds: cityBounds,
      fitBoundsOptions: { padding: 30 },
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
      maxZoom: 18,
      minZoom: 10,
      // Dev only: lets screenshot tooling capture the WebGL canvas. Off in production for performance.
      canvasContextAttributes: { preserveDrawingBuffer: import.meta.env.DEV },
    })
    map.touchZoomRotate.disableRotation()
    mapRef.current = map
    if (import.meta.env.DEV) (window as unknown as { __sutraMap?: MLMap }).__sutraMap = map

    map.on('load', () => {
      map.addSource('roads', { type: 'geojson', data: roadsGeoJSON(city) })
      map.addLayer({
        id: 'roads',
        type: 'line',
        source: 'roads',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': ['step', ['get', 'speed'], C.road, 50, C.roadArterial],
          'line-width': [
            'interpolate', ['linear'], ['zoom'],
            11, ['interpolate', ['linear'], ['get', 'speed'], 30, 0.6, 60, 2.4],
            15, ['interpolate', ['linear'], ['get', 'speed'], 30, 2.5, 60, 9],
          ],
        },
      })

      // Heat sits over the roads but under tracks and cameras. Radius grows geometrically with zoom so a
      // camera's kernel covers roughly the same ground (~1.2 km, about one block to each neighbour) at every zoom level.
      map.addSource('heat', { type: 'geojson', data: EMPTY })
      map.addLayer({
        id: 'heat',
        type: 'heatmap',
        source: 'heat',
        paint: {
          'heatmap-weight': ['get', 'w'],
          'heatmap-radius': ['interpolate', ['exponential', 2], ['zoom'], 10, 8.5, 16, 530],
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 10, 1.3, 16, 2.2],
          'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'], ...HEAT_RAMP.flat()] as unknown as ExpressionSpecification,
          'heatmap-opacity': 0.85,
        },
      })

      map.addSource('traj', { type: 'geojson', data: EMPTY })
      map.addLayer({
        id: 'traj',
        type: 'line',
        source: 'traj',
        filter: ['!=', ['get', 'dashed'], true],
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': ['get', 'color'], 'line-width': ['get', 'width'], 'line-opacity': ['get', 'opacity'] },
      })
      map.addLayer({
        id: 'traj-dashed',
        type: 'line',
        source: 'traj',
        filter: ['==', ['get', 'dashed'], true],
        paint: { 'line-color': ['get', 'color'], 'line-width': ['get', 'width'], 'line-opacity': ['get', 'opacity'], 'line-dasharray': [2, 2] },
      })

      map.addSource('cams', { type: 'geojson', data: EMPTY })
      map.addLayer({
        id: 'cams',
        type: 'circle',
        source: 'cams',
        paint: {
          'circle-radius': ['get', 'r'],
          'circle-color': ['case', ['get', 'hl'], C.cam, C.camFill],
          'circle-opacity': ['case', ['get', 'dim'], 0.35, 0.9],
          'circle-stroke-color': ['case', ['get', 'hl'], '#e8fbff', C.cam],
          'circle-stroke-width': ['case', ['get', 'hl'], 1.5, 1],
          'circle-stroke-opacity': ['case', ['get', 'dim'], 0.4, 0.9],
        },
      })

      map.addSource('pts', { type: 'geojson', data: EMPTY })
      map.addLayer({
        id: 'pts',
        type: 'circle',
        source: 'pts',
        paint: {
          'circle-radius': ['get', 'radius'],
          'circle-color': ['get', 'color'],
          'circle-stroke-color': ['get', 'stroke'],
          'circle-stroke-width': ['get', 'sw'],
        },
      })

      map.addImage('dir-arrow', arrowImage(), { pixelRatio: 2 })
      map.addSource('arrows', { type: 'geojson', data: EMPTY })
      map.addLayer({
        id: 'arrows',
        type: 'symbol',
        source: 'arrows',
        layout: {
          'icon-image': 'dir-arrow',
          'icon-rotate': ['get', 'heading'],
          'icon-rotation-alignment': 'map',
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
          'icon-size': ['interpolate', ['linear'], ['zoom'], 11, 0.6, 15, 0.9],
          // offset is applied in the rotated frame: "up" = along the heading, so the arrow sits ahead of the camera
          'icon-offset': [0, -30],
        },
      })

      map.addSource('pulse', { type: 'geojson', data: EMPTY })
      map.addLayer({
        id: 'pulse',
        type: 'circle',
        source: 'pulse',
        paint: {
          'circle-radius': ['get', 'r'],
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-color': C.cam,
          'circle-stroke-width': 1.5,
          'circle-stroke-opacity': ['get', 'o'],
        },
      })

      const popup = new Popup({ closeButton: false, closeOnClick: false, offset: 10 })
      map.on('mouseenter', 'cams', (e) => {
        map.getCanvas().style.cursor = clickRef.current ? 'pointer' : 'default'
        const f = e.features?.[0]
        if (!f) return
        const id = String(f.properties?.id)
        const cam = city.cameras.find((c) => c.camera_id === id)
        const st = statsRef.current.get(id)
        const el = document.createElement('div')
        const title = document.createElement('div')
        title.style.fontWeight = '600'
        title.textContent = `${id}${cam?.is_border ? ' · border' : ''}`
        const name = document.createElement('div')
        name.style.color = '#7d8b9b'
        name.textContent = cam?.name ?? ''
        el.append(title, name)
        if (st) {
          const v = document.createElement('div')
          v.style.fontVariantNumeric = 'tabular-nums'
          v.textContent = `${st.volume_last_hour} reads last hour · ${st.events_total} total`
          el.append(v)
        }
        popup.setLngLat((f.geometry as Point).coordinates as [number, number]).setDOMContent(el).addTo(map)
      })
      map.on('mouseleave', 'cams', () => {
        map.getCanvas().style.cursor = ''
        popup.remove()
      })
      map.on('click', 'cams', (e) => {
        const id = e.features?.[0]?.properties?.id
        if (id && clickRef.current) clickRef.current(String(id))
      })
      setReady(true)
    })

    const ro = new ResizeObserver(() => map.resize())
    ro.observe(containerRef.current)
    return () => {
      ro.disconnect()
      map.remove()
      mapRef.current = null
      setReady(false)
    }
  }, [city, cityBounds])

  // ---- cameras
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const stats = new Map((cameraStats ?? []).map((s) => [s.camera_id, s]))
    statsRef.current = stats
    const max = Math.max(1, ...(cameraStats ?? []).map((s) => s.volume_last_hour))
    const hl = new Set(highlightCameras ?? [])
    const anyHl = hl.size > 0
    const data: FeatureCollection = {
      type: 'FeatureCollection',
      features: city.cameras.map((c) => {
        const vol = stats.get(c.camera_id)?.volume_last_hour
        const r = vol === undefined ? 4 : 3 + 8 * Math.sqrt(vol / max)
        return {
          type: 'Feature',
          properties: { id: c.camera_id, r: anyHl && hl.has(c.camera_id) ? Math.max(r, 5.5) : r, hl: hl.has(c.camera_id), dim: anyHl && !hl.has(c.camera_id) },
          geometry: { type: 'Point', coordinates: [c.lon, c.lat] },
        }
      }),
    }
    ;(map.getSource('cams') as GeoJSONSource | undefined)?.setData(data)
  }, [ready, city, cameraStats, highlightCameras])

  // ---- lines
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const data: FeatureCollection = {
      type: 'FeatureCollection',
      features: (lines ?? [])
        .filter((l) => l.coords.length >= 2)
        .map((l) => ({
          type: 'Feature',
          properties: { id: l.id, color: l.color ?? C.cam, width: l.width ?? 3, opacity: l.opacity ?? 0.9, dashed: Boolean(l.dashed) },
          geometry: { type: 'LineString', coordinates: l.coords },
        })),
    }
    ;(map.getSource('traj') as GeoJSONSource | undefined)?.setData(data)
  }, [ready, lines])

  // ---- points
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const data: FeatureCollection = {
      type: 'FeatureCollection',
      features: (points ?? []).map((p) => ({
        type: 'Feature',
        properties: { id: p.id, color: p.color ?? C.cam, radius: p.radius ?? 5, stroke: p.stroke ?? '#080b10', sw: p.strokeWidth ?? 2 },
        geometry: { type: 'Point', coordinates: p.lngLat },
      })),
    }
    ;(map.getSource('pts') as GeoJSONSource | undefined)?.setData(data)
  }, [ready, points])

  // ---- direction arrows
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const data: FeatureCollection = {
      type: 'FeatureCollection',
      features: (arrows ?? [])
        .filter((a) => Number.isFinite(a.heading))
        .map((a) => ({ type: 'Feature', properties: { id: a.id, heading: a.heading }, geometry: { type: 'Point', coordinates: a.lngLat } })),
    }
    ;(map.getSource('arrows') as GeoJSONSource | undefined)?.setData(data)
  }, [ready, arrows])

  // ---- heatmap
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const data: FeatureCollection = {
      type: 'FeatureCollection',
      features: (heat?.points ?? []).map((p) => ({
        type: 'Feature',
        properties: { id: p.id, w: Math.max(0, Math.min(1, p.weight)) },
        geometry: { type: 'Point', coordinates: p.lngLat },
      })),
    }
    ;(map.getSource('heat') as GeoJSONSource | undefined)?.setData(data)
    map.setLayoutProperty('heat', 'visibility', heat ? 'visible' : 'none')
  }, [ready, heat])

  // ---- HTML labels (no glyph server needed offline)
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map || !labels?.length) return
    const markers = labels.map((l) => {
      const el = document.createElement('div')
      el.textContent = l.text
      const tone = l.tone ?? 'neutral'
      el.style.cssText = [
        'font: 600 11px/1.2 var(--font-mono)',
        'padding: 2px 5px',
        'border-radius: 2px',
        'white-space: nowrap',
        'pointer-events: none',
        tone === 'alert' ? 'background:#ef4444;color:#070a0e' : tone === 'accent' ? 'background:#3cc4d8;color:#070a0e' : 'background:#1e2732;color:#d3dbe4;border:1px solid #3a4757',
      ].join(';')
      return new Marker({ element: el, anchor: 'bottom', offset: [0, -9] }).setLngLat(l.lngLat).addTo(map)
    })
    return () => markers.forEach((m) => m.remove())
  }, [ready, labels])

  // ---- fit
  const fitKey = fit?.key
  const fitRef = useRef(fit)
  useEffect(() => {
    fitRef.current = fit
  })
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map || fitKey === undefined) return
    const f = fitRef.current
    const coords = f?.coords
    const bounds = new LngLatBounds()
    if (coords && coords.length) coords.forEach((c) => bounds.extend(c))
    else city.nodes.forEach((n) => bounds.extend([n.lon, n.lat]))
    map.fitBounds(bounds, { padding: f?.padding ?? 50, maxZoom: f?.maxZoom ?? 15, duration: reducedMotion() ? 0 : 600 })
  }, [ready, fitKey, city])

  // ---- live pulses
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map || !livePulses || reducedMotion()) return
    const camCoord = new Map(city.cameras.map((c) => [c.camera_id, [c.lon, c.lat] as LngLat]))
    let pulses: { at: LngLat; t0: number }[] = []
    let raf = 0
    let last = 0
    const frame = (now: number) => {
      raf = 0
      // The map can be removed (remount / hot reload) with a frame already queued.
      if (mapRef.current !== map) return
      if (now - last < 33) {
        raf = requestAnimationFrame(frame)
        return
      }
      last = now
      pulses = pulses.filter((p) => now - p.t0 < 1100)
      const src = map.getSource('pulse') as GeoJSONSource | undefined
      src?.setData({
        type: 'FeatureCollection',
        features: pulses.map((p) => {
          const a = (now - p.t0) / 1100
          return { type: 'Feature', properties: { r: 5 + 20 * a, o: 0.9 * (1 - a) }, geometry: { type: 'Point', coordinates: p.at } }
        }),
      })
      if (pulses.length) raf = requestAnimationFrame(frame)
    }
    const off = liveStore.onPulse((cam) => {
      const at = camCoord.get(cam)
      if (!at || pulses.length > 60) return
      pulses.push({ at, t0: performance.now() })
      if (!raf) raf = requestAnimationFrame(frame)
    })
    return () => {
      off()
      if (raf) cancelAnimationFrame(raf)
    }
  }, [ready, livePulses, city])

  return <div ref={containerRef} role="region" aria-label={ariaLabel} className={cx('h-full w-full', className)} />
}
