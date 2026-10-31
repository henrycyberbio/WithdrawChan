/**
 * @fileoverview Provides WithdrawChan's mixer withdrawal amount recommendation.
 *
 * Given the need `x`, the algorithm first rounds it up to the nearest multiple of
 * the rounding unit `d`, then walks up a ladder of candidate amounts with probability `q`
 * at each step. Finally, with probability `p`, it adds a random decimal `u` uniform in `[0, d)`,
 * provided the result stays within the maximum `m`.
 *
 * All amounts are integers in the asset's smallest unit, and `p` and `q` are exact fractions.
 * The Python implementation follows the same procedure and is validated against identical test
 * vectors.
 */

import UNITS_JSON from "../units.json" with { type: "json" };

// ==== fractions ====

export interface Fraction {
  readonly num: bigint;
  readonly den: bigint;
}

function gcd(a: bigint, b: bigint): bigint {
  while (b) [a, b] = [b, a % b];
  return a < 0n ? -a : a;
}

function frac(num: bigint, den: bigint = 1n): Fraction {
  const g = gcd(num, den) || 1n;
  return { num: num / g, den: den / g };
}

function add(a: Fraction, b: Fraction): Fraction {
  return frac(a.num * b.den + b.num * a.den, a.den * b.den);
}

function mul(a: Fraction, b: Fraction): Fraction {
  return frac(a.num * b.num, a.den * b.den);
}

function complement(a: Fraction): Fraction {
  return frac(a.den - a.num, a.den);
}

// ==== defaults ====

/** Probability of climbing one rung. */
export const Q: Fraction = frac(1n, 2n);
/** Probability of adding the random decimal below one rounding unit. */
export const P: Fraction = frac(0n);

// ==== units ====

/** Parses a non-negative decimal string such as `"0.001"` into an exact fraction. */
function parseDecimal(s: string): Fraction {
  const [whole, part = ""] = s.split(".");
  const den = 10n ** BigInt(part.length);
  return frac(BigInt(whole) * den + BigInt(part || "0"), den);
}

/** Rounding unit per asset class, in whole tokens, read from `../units.json`. */
export const UNITS: Readonly<Record<string, Fraction>> = Object.fromEntries(
  Object.entries(UNITS_JSON).map(([code, value]) => [code, parseDecimal(value)]),
);

/**
 * Return the rounding unit of an asset class in the asset's smallest unit.
 *
 * @param code The asset class, a key of {@link UNITS}.
 * @param decimals The number of decimals in the asset's smallest unit.
 * @returns `UNITS[code]` scaled to the smallest unit, as an integer.
 * @throws {RangeError} If `code` is unknown or `decimals` is too small for a whole number.
 */
export function unit(code: string, decimals: number): bigint {
  if (!Object.hasOwn(UNITS, code)) throw new RangeError(`unknown unit code: ${code}`);
  const u = UNITS[code];
  const scaled = u.num * 10n ** BigInt(decimals);
  if (scaled % u.den !== 0n) throw new RangeError(`unit(${code}, ${decimals}) is not a whole number of smallest units`);
  return scaled / u.den;
}

/**
 * A callable returning a uniform integer in `[0, n)`.
 *
 * @param n The exclusive upper bound.
 * @returns A uniform integer in `[0, n)`.
 */
export type Rng = (n: bigint) => bigint;

/**
 * Return a uniform integer in `[0, n)` using `crypto.getRandomValues` and rejection sampling.
 *
 * @param n The exclusive upper bound.
 * @returns A uniform integer in `[0, n)`.
 */
export function secureRng(n: bigint): bigint {
  if (n <= 0n) throw new RangeError(`n=${n} must be positive`);
  const bits = n.toString(2).length;
  const mask = (1n << BigInt(bits)) - 1n;
  const buf = new Uint8Array(Math.ceil(bits / 8));
  for (;;) {
    crypto.getRandomValues(buf);
    let v = 0n;
    for (const b of buf) v = (v << 8n) | BigInt(b);
    v &= mask;
    if (v < n) return v;
  }
}

// ==== ladder ====

function ceilTo(v: bigint, unit: bigint): bigint {
  return ((v + unit - 1n) / unit) * unit;
}

/**
 * Return the default maximum amount.
 *
 * The default maximum is `2 * x` rounded up to the rounding unit `d`.
 *
 * @param x The need, the least amount to withdraw.
 * @param d The rounding unit.
 * @returns The smallest multiple of `d` not below `2 * x`.
 */
export function defaultM(x: bigint, d: bigint): bigint {
  return ceilTo(2n * x, d);
}

/**
 * Build the decimal rung ladder for the walk.
 *
 * Rungs start at `x` rounded up to `d`, then that amount rounded up to one significant digit,
 * then each higher power of ten, keeping those not above `m`.
 *
 * @param x The need, the least amount to withdraw.
 * @param d The rounding unit.
 * @param m The maximum, the largest amount to withdraw.
 * @returns Increasing candidate amounts not above `m`, starting at `x` rounded up to `d`.
 */
export function decimalRungs(x: bigint, d: bigint, m: bigint): bigint[] {
  const c0 = ceilTo(x, d);
  const e = c0.toString().length - 1;
  const candidates = [ceilTo(c0, 10n ** BigInt(e))];
  for (let k = e + 1; k <= m.toString().length; k++) candidates.push(10n ** BigInt(k));
  const rungs = [c0];
  for (const c of candidates) if (rungs[rungs.length - 1] < c && c <= m) rungs.push(c);
  return rungs;
}

// ==== walk ====

export interface Options {
  /** The rounding unit. */
  readonly d: bigint;
  /** The probability of climbing one rung. */
  readonly q?: Fraction;
  /** The probability of adding the random decimal. */
  readonly p?: Fraction;
  /** A callable returning a uniform integer in `[0, n)`. */
  readonly rng?: Rng;
}

/** The options of the exact distributions, which draw nothing. */
export type DistributionOptions = Omit<Options, "rng">;

/**
 * Return whether a Bernoulli trial succeeds.
 *
 * Certain outcomes draw nothing from `rng`.
 *
 * @param raw The success probability.
 * @param rng A callable returning a uniform integer in `[0, n)`.
 * @returns `true` with probability `prob`; otherwise `false`.
 */
function coin(raw: Fraction, rng: Rng): boolean {
  const prob = frac(raw.num, raw.den);
  if (prob.num === 0n || prob.num === prob.den) return prob.num === prob.den;
  return rng(prob.den) < prob.num;
}

/**
 * Draw one recommended amount from the rung ladder.
 *
 * The walk climbs with probability `q` at each rung, then adds the random decimal with
 * probability `p` if the result stays within `m`. `rng` is called once per uncertain coin
 * and once for the random decimal, in walk order.
 *
 * @param rungs Increasing candidate amounts not above `m`.
 * @param m The maximum, the largest amount to withdraw.
 * @param options The rounding unit, probabilities, and random number generator.
 * @returns The recommended amount.
 * @throws {AmountError} If `q` or `p` is not a probability, or `rungs` is empty.
 */
export function walk(rungs: readonly bigint[], m: bigint, options: Options): bigint {
  const { d, q = Q, p = P, rng = secureRng } = options;
  checkProbs({ q, p });
  if (rungs.length === 0) throw new AmountError("rungs must not be empty");
  let j = 0;
  while (j + 1 < rungs.length && coin(q, rng)) j++;
  let y = rungs[j];
  if (coin(p, rng)) {
    const u = rng(d);
    if (y + u <= m) y += u;
  }
  return y;
}

/**
 * Return the exact output distribution of {@link walk}.
 *
 * Zero-probability outputs are omitted.
 *
 * @param rungs Increasing candidate amounts not above `m`.
 * @param m The maximum, the largest amount to withdraw.
 * @param options The rounding unit and probabilities.
 * @returns A mapping from output amount to its exact probability.
 * @throws {AmountError} If `q` or `p` is not a probability.
 */
export function walkDistribution(rungs: readonly bigint[], m: bigint, options: DistributionOptions): Map<bigint, Fraction> {
  const { d, q = Q, p = P } = options;
  checkProbs({ q, p });
  const out = new Map<bigint, Fraction>();
  const put = (y: bigint, f: Fraction): void => {
    out.set(y, add(out.get(y) ?? frac(0n), f));
  };
  let reach = frac(1n);
  rungs.forEach((v, j) => {
    const stop = j === rungs.length - 1 ? reach : mul(reach, complement(q));
    reach = mul(reach, q);
    if (stop.num === 0n) return;
    const top = p.num === 0n ? v : (v + d - 1n < m ? v + d - 1n : m);
    put(v, mul(stop, add(complement(p), mul(p, frac(d - (top - v), d)))));
    const tail = mul(stop, mul(p, frac(1n, d)));
    for (let y = v + 1n; y <= top; y++) put(y, tail);
  });
  return out;
}

// ==== recommendation ====

/** Error raised when the need, the maximum, the rounding unit, or a probability is invalid. */
export class AmountError extends RangeError {
  /**
   * Create an error for an invalid input.
   *
   * @param message Which input is invalid and what it must satisfy.
   */
  constructor(message: string) {
    super(message);
    this.name = "AmountError";
  }
}

function isPow10(d: bigint): boolean {
  return d > 0n && 10n ** BigInt(d.toString().length - 1) === d;
}

function limit(x: bigint, m: bigint | undefined, d: bigint): bigint {
  if (x <= 0n || !isPow10(d) || (m !== undefined && (m % d !== 0n || m < x))) {
    throw new AmountError(`x=${x} must be positive, d=${d} a power of ten, and m=${m} a multiple of d at least x`);
  }
  return m ?? defaultM(x, d);
}

/** Throw {@link AmountError} unless `q` and `p` are probabilities. */
function checkProbs({ q = Q, p = P }: Pick<Options, "q" | "p">): void {
  for (const [name, prob] of [["q", q], ["p", p]] as const) {
    if (prob.num < 0n || prob.num > prob.den) {
      throw new AmountError(`${name}=${prob.num}/${prob.den} must satisfy 0 <= ${name} <= 1`);
    }
  }
}

/**
 * Recommend a withdrawal amount.
 *
 * @param x The need, the least amount to withdraw.
 * @param m The maximum, the largest amount to withdraw. If omitted, {@link defaultM} is used.
 * @param options The rounding unit, probabilities, and random number generator.
 * @returns One recommended amount.
 * @throws {AmountError} If `x <= 0n`, `d` is not a positive power of ten, `m` is not a
 * multiple of `d` at least `x`, or `q` or `p` is not a probability.
 */
export function recommend(x: bigint, m: bigint | undefined, options: Options): bigint {
  const maximum = limit(x, m, options.d);
  return walk(decimalRungs(x, options.d, maximum), maximum, options);
}

/**
 * Return the exact output distribution of {@link recommend}.
 *
 * @param x The need, the least amount to withdraw.
 * @param m The maximum, the largest amount to withdraw. If omitted, {@link defaultM} is used.
 * @param options The rounding unit and probabilities.
 * @returns A mapping from output amount to its exact probability.
 * @throws {AmountError} If `x <= 0n`, `d` is not a positive power of ten, `m` is not a
 * multiple of `d` at least `x`, or `q` or `p` is not a probability.
 */
export function distribution(x: bigint, m: bigint | undefined, options: DistributionOptions): Map<bigint, Fraction> {
  const maximum = limit(x, m, options.d);
  return walkDistribution(decimalRungs(x, options.d, maximum), maximum, options);
}
