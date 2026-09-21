/** Small seeded PRNG (mulberry32) so fixture data is identical on every reload. */
export class Rng {
  private s: number
  constructor(seed: number) {
    this.s = seed >>> 0
  }
  next(): number {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  uniform(a: number, b: number): number {
    return a + (b - a) * this.next()
  }
  int(a: number, b: number): number {
    return Math.floor(this.uniform(a, b + 1))
  }
  chance(p: number): boolean {
    return this.next() < p
  }
  pick<T>(xs: readonly T[]): T {
    return xs[Math.floor(this.next() * xs.length)]
  }
  weighted<T>(xs: readonly T[], ws: readonly number[]): T {
    const total = ws.reduce((a, b) => a + b, 0)
    let r = this.next() * total
    for (let i = 0; i < xs.length; i++) {
      r -= ws[i]
      if (r <= 0) return xs[i]
    }
    return xs[xs.length - 1]
  }
  normal(mu = 0, sigma = 1): number {
    const u = Math.max(1e-12, this.next())
    const v = this.next()
    return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  shuffle<T>(xs: T[]): T[] {
    for (let i = xs.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1))
      ;[xs[i], xs[j]] = [xs[j], xs[i]]
    }
    return xs
  }
}
