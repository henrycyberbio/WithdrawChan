import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  AmountError,
  distribution,
  type Fraction,
  P,
  Q,
  recommend,
  secureRng,
  unit,
  walk,
  walkDistribution,
} from "./index.ts";

// Test names and behaviour match `../python/tests/test_withdrawchan.py`; keep both suites in step,
// apart from the tests that only make sense in one language: fraction reduction here, vector
// regeneration there.

// ==== helpers ====

interface Vector {
  x: string;
  m: string | null;
  d: string;
  q: string;
  p: string;
  distribution: Record<string, string>;
  rng: [string, string][];
  y: string;
}

interface Examples {
  decimals: number;
  d: string;
  q: string;
  p: string;
  examples: { x: string; m: string | null; distribution: Record<string, string> }[];
}

function load<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`../${name}`, import.meta.url), "utf-8"));
}

const vectors = load<{ vectors: Vector[] }>("vectors.json").vectors;
const EXAMPLES = load<Examples>("examples.json");

function parse(s: string): Fraction {
  const [num, den] = s.split("/");
  return { num: BigInt(num), den: BigInt(den) };
}

function inputs(v: Vector): [bigint, bigint | undefined, { d: bigint; q: Fraction; p: Fraction }] {
  return [BigInt(v.x), v.m === null ? undefined : BigInt(v.m), { d: BigInt(v.d), q: parse(v.q), p: parse(v.p) }];
}

/** Returns a worked-example amount in the asset's smallest unit. */
function exampleAmount(amount: string): bigint {
  const [whole, part = ""] = amount.split(".");
  return BigInt(whole) * 10n ** BigInt(EXAMPLES.decimals) + BigInt(part.padEnd(EXAMPLES.decimals, "0"));
}

/** Returns `{y: "num/den"}` sorted by `y`, the vector format. */
function asRecord(dist: Map<bigint, Fraction>): Record<string, string> {
  const sorted = [...dist].sort(([a], [b]) => (a < b ? -1 : 1));
  return Object.fromEntries(sorted.map(([y, f]) => [`${y}`, `${f.num}/${f.den}`]));
}

function compare(a: Fraction, b: Fraction): number {
  const diff = a.num * b.den - b.num * a.den;
  return diff < 0n ? -1 : diff > 0n ? 1 : 0;
}

const ceilTo = (v: number, unit: number): number => Math.ceil(v / unit) * unit;

// ==== vectors ====

test("closed form distribution equals the vectors", () => {
  // GIVEN: Vectors whose distributions come from running `walk` on every `rng` outcome.
  for (const v of vectors) {
    const [x, m, options] = inputs(v);
    // WHEN: Computing the closed-form distribution.
    const dist = distribution(x, m, options);
    // THEN: It matches outcome by outcome, so the audit formula describes the sampler.
    assert.deepEqual(asRecord(dist), v.distribution);
  }
});

test("scripted rng reproduces the recorded draw", () => {
  // GIVEN: Each vector's recorded `rng` calls (`n`, `value`) and output.
  for (const v of vectors) {
    const [x, m, options] = inputs(v);
    const script = v.rng.map(([n, value]) => [BigInt(n), BigInt(value)]);
    const calls: bigint[][] = [];
    const rng = (n: bigint): bigint => {
      calls.push([n, script[calls.length][1]]);
      return calls[calls.length - 1][1];
    };
    // WHEN: Replaying the recorded calls.
    const y = recommend(x, m, { ...options, rng });
    // THEN: The same calls happen in the same order and give the same amount.
    assert.deepEqual([calls, y], [script, BigInt(v.y)]);
  }
});

// ==== worked examples ====

for (const example of EXAMPLES.examples) {
  test(`worked example has the tabulated distribution [${example.x}-${example.m}]`, () => {
    // GIVEN: A hand-computed row of `../examples.json` (USDT, `d` = 1 USDT,
    // `q` = 1/2, `p` = 0).
    const options = { d: exampleAmount(EXAMPLES.d), q: parse(EXAMPLES.q), p: parse(EXAMPLES.p) };
    const m = example.m === null ? undefined : exampleAmount(example.m);
    // WHEN: Computing its distribution.
    const dist = distribution(exampleAmount(example.x), m, options);
    // THEN: It matches the row, an oracle written independently of the code.
    const expected = new Map(Object.entries(example.distribution).map(([y, f]) => [exampleAmount(y), parse(f)]));
    assert.deepEqual(asRecord(dist), asRecord(expected));
  });
}

// ==== privacy properties ====

test("any two feasible maximums change an output probability at most twofold", () => {
  // GIVEN: Needs from 1 to 300 and every maximum from `ceil(x)` to `10^4` in rounding
  // unit 10, `q` = 1/2.
  const d = 10;
  let worst: Fraction = { num: 0n, den: 1n };
  for (let x = 1; x <= 300; x += 7) {
    const dists = new Map<number, Map<bigint, Fraction>>();
    for (let m = ceilTo(x, d); m <= 10 ** 4; m += d) dists.set(m, distribution(BigInt(x), BigInt(m), { d: BigInt(d) }));
    // WHEN: Comparing `P(y | m)` over all maximums `m >= y`.
    const outputs = new Set([...dists.values()].flatMap((dist) => [...dist.keys()]));
    for (const y of outputs) {
      const probs = [...dists].filter(([m]) => BigInt(m) >= y).map(([, dist]) => dist.get(y) ?? { num: 0n, den: 1n });
      const max = probs.reduce((a, b) => (compare(a, b) >= 0 ? a : b));
      const min = probs.reduce((a, b) => (compare(a, b) <= 0 ? a : b));
      const ratio = { num: max.num * min.den, den: max.den * min.num };
      if (compare(ratio, worst) > 0) worst = ratio;
    }
  }
  // THEN: The ratio never exceeds `1 / (1 - q) = 2`; a maximum tied to the balance
  // leaks at most 1 bit.
  assert.equal(compare(worst, { num: 2n, den: 1n }), 0);
});

test("output never falls below the need rounded up", () => {
  // GIVEN: A need below one rounding unit (1, 37, 99), on a rung (100, 1200), and a maximum
  // from the rung up to ten times `c0`.
  const cases: [bigint, bigint][] = [
    [1n, 100n],
    [37n, 1000n],
    [99n, 100n],
    [100n, 1000n],
    [1200n, 12000n],
  ];
  for (const [x, m] of cases) {
    // WHEN: Listing possible outputs.
    const outputs = [...distribution(x, m, { d: 100n, p: { num: 1n, den: 2n } }).keys()];
    const low = outputs.reduce((a, b) => (a < b ? a : b));
    // THEN: The raw need with its part below the rounding unit is never sent, which is why `m`
    // must be whole units.
    assert.ok(low >= BigInt(ceilTo(Number(x), 100)));
  }
});

// ==== defaults ====

test("default secure rng draws an amount the distribution allows", () => {
  // GIVEN: `x` = 520 USDT with the default maximum and the system's secure randomness.
  const usdt = 10n ** 6n;
  const support = distribution(520n * usdt, undefined, { d: usdt });
  // WHEN: Drawing many times.
  const seen = new Set(Array.from({ length: 200 }, () => recommend(520n * usdt, undefined, { d: usdt })));
  // THEN: Every draw is a possible output and all three rungs (520, 600, 1000) appear.
  assert.deepEqual([...seen].sort(), [...support.keys()].sort());
});

test("secure rng rejects a non-positive bound", () => {
  // GIVEN: Bounds at or below zero, which no integer in `[0, n)` can satisfy.
  // WHEN: Asking for a uniform integer.
  // THEN: It throws instead of looping forever.
  for (const n of [0n, -1n]) assert.throws(() => secureRng(n), RangeError);
});

// ==== units ====

test("unit scales whole tokens to the smallest unit", () => {
  // GIVEN: Whole-token units of 1 (USD stablecoins) and 0.001 (ETH).
  // WHEN: Converting them to an asset's smallest unit.
  // THEN: The scale multiplies the unit, and an impossible scale or an unknown code throws.
  assert.deepEqual(
    [unit("USD-stable", 6), unit("USD-stable", 18), unit("ETH", 18)],
    [10n ** 6n, 10n ** 18n, 10n ** 15n],
  );
  assert.throws(() => unit("ETH-cents", 6), RangeError);
  assert.throws(() => unit("ETH", 2), RangeError);
  assert.throws(() => unit("toString", 6), RangeError);
});

// ==== input contract ====

test("invalid need, maximum or rounding unit is rejected", () => {
  // GIVEN: A non-positive need, a maximum that is not a multiple of the rounding
  // unit, a maximum below the need, or a rounding unit that is not a power of ten.
  const cases = [
    [0n, undefined, 100n],
    [150n, 250n, 100n],
    [150n, 100n, 100n],
    [150n, undefined, 25n],
  ] as const;
  for (const [x, m, d] of cases) {
    // WHEN: Asking for a recommendation.
    // THEN: It throws instead of silently correcting the input.
    assert.throws(() => recommend(x, m, { d }), AmountError);
  }
});

test("probabilities outside the unit interval are rejected", () => {
  // GIVEN: A climb probability above 1 and a decimal probability below 0.
  const cases: [Fraction, Fraction][] = [
    [{ num: 5n, den: 4n }, P],
    [Q, { num: -1n, den: 4n }],
  ];
  for (const [q, p] of cases) {
    // WHEN: Asking for a recommendation, its distribution, or a walk on given rungs.
    // THEN: Each throws instead of returning negative probabilities.
    assert.throws(() => recommend(520n, 1040n, { d: 10n, q, p }), AmountError);
    assert.throws(() => distribution(520n, 1040n, { d: 10n, q, p }), AmountError);
    assert.throws(() => walk([10n, 20n], 20n, { d: 10n, q, p }), AmountError);
    assert.throws(() => walkDistribution([10n, 20n], 20n, { d: 10n, q, p }), AmountError);
  }
});

test("empty ladder is rejected", () => {
  // GIVEN: A walk with no rungs.
  // WHEN: Drawing an amount.
  // THEN: It throws the package error instead of returning undefined.
  assert.throws(() => walk([], 10n, { d: 10n }), { name: "AmountError", message: "rungs must not be empty" });
});

test("unreduced probabilities draw the same rng bounds", () => {
  // GIVEN: A recording rng and walks over the same rungs.
  const bounds = (rungs: bigint[], q: Fraction, p: Fraction): bigint[] => {
    const calls: bigint[] = [];
    const rng = (n: bigint): bigint => {
      calls.push(n);
      return 0n;
    };
    walk(rungs, 1000n, { d: 10n, q, p, rng });
    return calls;
  };

  // WHEN: Writing the probabilities as 2/4 and as 1/2.
  // THEN: The same bounds are requested, so one random stream gives the same amount either way.
  assert.deepEqual(bounds([100n, 200n, 400n], { num: 2n, den: 4n }, P), bounds([100n, 200n, 400n], { num: 1n, den: 2n }, P));
  assert.deepEqual(bounds([100n, 200n, 400n], { num: 2n, den: 4n }, P), [2n, 2n]);
  assert.deepEqual(bounds([100n], Q, { num: 2n, den: 4n }), [2n, 10n]);
});
