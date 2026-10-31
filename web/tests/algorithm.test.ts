import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const adapter = await import('../src/algorithm.ts');
const upstream = await import('@withdrawchan/engine');
test('exact decimal input is converted to the asset smallest unit, not floating point', () => {
  assert.equal(typeof adapter.parseAmount, 'function', 'local engine adapter is missing');
  assert.equal(adapter.parseAmount('43.21', 'USD-stable'), 43210000n);
  assert.equal(adapter.parseAmount('0.0011', 'ETH'), 1100000000000000n);
});

test('recommends by the upstream algorithm rather than synthetic-deposit ranking', () => {
  assert.equal(typeof adapter.recommendAmount, 'function', 'upstream recommendation is not wired');
  const result = adapter.recommendAmount('43.21', '60', 'USD-stable', n => n - 1n);
  assert.equal(result.recommended, '44');
});

test('blank maximum uses upstream rounding, including ETH precision', () => {
  assert.equal(adapter.defaultMaximum('43.21', 'USD-stable'), '87');
  assert.equal(adapter.defaultMaximum('0.0011', 'ETH'), '0.003');
  assert.equal(adapter.resolveRange('499.99', '', 'USD-stable').maximum, '1000');
});
test('an explicit fractional maximum is rejected rather than silently rounded', () => {
  assert.throws(() => adapter.resolveRange('43.21', '60.42', 'USD-stable'), { code: 'maximum-unit' });
  assert.throws(() => adapter.resolveRange('0.0011', '0.0022', 'ETH'), { code: 'maximum-unit' });
  assert.equal(adapter.resolveRange('43.21', '60', 'USD-stable').maximum, '60');
});
test('default maximum is not constrained by the former demo limits', () => {
  for (const [minimum, asset] of [['600', 'USD-stable'], ['1001.25', 'USD-stable'], ['0.6', 'ETH'], ['2.0001', 'ETH']] as const) {
    const expected = upstream.defaultM(adapter.parseAmount(minimum, asset), adapter.config(asset).roundingUnit);
    assert.equal(adapter.resolveRange(minimum, '', asset).m, expected);
  }
});
test('explicit ranges above the former demo limits use the unchanged upstream algorithm', () => {
  for (const [minimum, maximum, asset] of [['1001.25', '10000', 'USD-stable'], ['2.0001', '10', 'ETH']] as const) {
    const { x, m } = adapter.resolveRange(minimum, maximum, asset);
    const rng = (n: bigint) => n - 1n;
    assert.equal(adapter.parseAmount(adapter.recommendAmount(minimum, maximum, asset, rng).recommended, asset), upstream.recommend(x, m, { d: adapter.config(asset).roundingUnit, rng }));
  }
});
test('amount strings have no fixed digit cap and remain exact beyond Number range', () => {
  const amount = `1${'0'.repeat(320)}`;
  for (const asset of ['USD-stable', 'ETH'] as const) {
    assert.equal(adapter.formatAmount(adapter.parseAmount(amount, asset), asset), amount);
    assert.equal(adapter.recommendAmount(amount, amount, asset).recommended, amount);
    const { x, m } = adapter.resolveRange(amount, '', asset);
    assert.equal(m, upstream.defaultM(x, adapter.config(asset).roundingUnit));
  }
});
test('precision, exponent notation, malformed and out of range input is rejected', () => {
  for (const raw of ['', '1e2', 'Infinity', 'NaN', '-1', '1,000', '+1', '.5', '1.', '1.001']) {
    assert.throws(() => adapter.parseAmount(raw, 'USD-stable'), { code: 'invalid' }, raw);
  }
  for (const [minimum, maximum, asset] of [['0', '10', 'USD-stable'], ['1', '0', 'USD-stable'], ['.0001', '1', 'ETH'], ['0.00101', '1', 'ETH']] as const) {
    assert.throws(() => adapter.resolveRange(minimum, maximum, asset), { code: 'invalid' });
  }
  assert.throws(() => adapter.resolveRange('20', '10', 'USD-stable'), { code: 'order' });
  assert.throws(() => adapter.config('BTC' as never), /Invalid pool/);
});
test('formatting preserves exact smallest units beyond Number safe integer range', () => {
  assert.equal(adapter.formatAmount(1_100_000_000_000_000n, 'ETH'), '0.0011');
  assert.equal(adapter.formatAmount(999_999_999_999_999_999n, 'ETH'), '0.999999999999999999');
  assert.equal(adapter.formatAmount(1_000_000n, 'USD-stable'), '1');
});
test('asset precision and rounding units match the repository manifests', () => {
  // There is no ppv1-usd-stable manifest: the engine's USD-stable class covers both stablecoins,
  // and the usdt and usdc manifests agree on 6 decimals and a 10^6 rounding unit. ETH is eth.
  for (const [asset, tokens] of [['USD-stable', ['usdt', 'usdc']], ['ETH', ['eth']]] as const) {
    for (const token of tokens) {
      const manifest = JSON.parse(readFileSync(new URL(`../../lab/manifests/ppv1-${token}-26078315.json`, import.meta.url), 'utf8'));
      assert.equal(adapter.config(asset).assetDecimals, manifest.decimals, `${token}: decimals`);
      assert.equal(adapter.config(asset).roundingUnit, BigInt(manifest.d), `${token}: rounding unit`);
    }
  }
  // The manifests only ever encode these two figures: 10^6 for a whole USD-stable and 10^15
  // for 0.001 ETH. Nothing else may be introduced without a matching manifest.
  assert.equal(adapter.config('USD-stable').roundingUnit, 1_000_000n);
  assert.equal(adapter.config('ETH').roundingUnit, 1_000_000_000_000_000n);
});
test('the FAQ coin-flip example matches the engine: 43.21 up to 120 gives 44, 50 or 100', () => {
  const d = adapter.config('USD-stable').roundingUnit;
  const dist = upstream.distribution(adapter.parseAmount('43.21', 'USD-stable'), adapter.parseAmount('120', 'USD-stable'), { d });
  // Tails at 44 stops (1/2); heads then tails at 50 stops (1/4); heads twice reaches 100, the last candidate (1/4).
  assert.deepEqual([...dist].map(([amount, prob]) => [adapter.formatAmount(amount, 'USD-stable'), `${prob.num}/${prob.den}`]), [['44', '1/2'], ['50', '1/4'], ['100', '1/4']]);
  const faq = readFileSync(new URL('../templates/partials/faq.html', import.meta.url), 'utf8');
  for (const lang of ['en', 'zh']) {
    const blocks = [...faq.matchAll(new RegExp(`<div data-lang="${lang}"[^>]*>(.*?)</div>`, 'gs'))].map(match => match[1]).join('\n');
    const table = blocks.match(/<table class="faq-table">[\s\S]*?<\/table>/)?.[0] ?? '';
    assert.ok(table, `${lang}: the example table is present`);
    for (const value of ['44', '50', '100']) assert.ok(table.includes(value), `${lang}: the table walks through ${value}`);
    for (const value of ['43.21', '120']) assert.ok(blocks.includes(value), `${lang}: the example states ${value}`);
  }
});
for (const asset of ['USD-stable', 'ETH'] as const) {
  test(`${asset}: deterministic draws match the unchanged upstream engine`, () => {
    const unit = adapter.config(asset).roundingUnit;
    const x = unit + unit / 10n, m = 13n * unit;
    const min = adapter.formatAmount(x, asset), max = adapter.formatAmount(m, asset);
    // Two cases per asset: the first rung (choose 0) and the second (choose 1).
    for (const choose of [0n, 1n]) {
      const rng = (n: bigint) => choose % n;
      const expected = upstream.recommend(x, m, { d: unit, rng });
      const actual = adapter.recommendAmount(min, max, asset, rng);
      assert.equal(adapter.parseAmount(actual.recommended, asset), expected);
      assert.ok(expected >= x && expected <= m);
      assert.doesNotThrow(() => JSON.stringify(actual));
    }
  });
}

test('the committed engine vectors replay through the browser adapter input/output contract', () => {
  const vectors = JSON.parse(readFileSync(new URL('../../src/vectors.json', import.meta.url), 'utf8')).vectors as {
    x: string; m: string | null; d: string; q: string; p: string; distribution: Record<string, string>; rng: [string, string][]; y: string;
  }[];
  const d = adapter.config('USD-stable').roundingUnit;
  let replayed = 0;
  for (const vector of vectors) {
    if (BigInt(vector.d) !== d) continue;
    // The page fixes q = 1/2 and p = 0, so only vectors under those parameters describe it.
    if (vector.q !== '1/2' || vector.p !== '0/1') continue;
    // The frontend only accepts amounts at or above the asset minimum (1 USD-stable here).
    if (BigInt(vector.x) < adapter.parseAmount('1', 'USD-stable')) continue;
    const maximumUnits = vector.m === null ? null : BigInt(vector.m);
    if (maximumUnits !== null && maximumUnits % d !== 0n) continue;
    const minimum = adapter.formatAmount(BigInt(vector.x), 'USD-stable');
    const maximum = maximumUnits === null ? '' : adapter.formatAmount(maximumUnits, 'USD-stable');
    // Only replay inputs the UI's own precision can represent exactly.
    if (adapter.parseAmount(minimum, 'USD-stable') !== BigInt(vector.x)) continue;
    if (maximum !== '' && adapter.parseAmount(maximum, 'USD-stable') !== maximumUnits) continue;
    // The UI adapter must not change the engine's candidate support for these inputs.
    const { x, m } = adapter.resolveRange(minimum, maximum, 'USD-stable');
    const expected = upstream.distribution(x, m, { d });
    assert.deepEqual(
      [...expected]
        .sort(([a], [b]) => (a < b ? -1 : 1))
        .map(([amount, prob]) => [`${amount}`, `${prob.num}/${prob.den}`]),
      Object.entries(vector.distribution).sort(([a], [b]) => (BigInt(a) < BigInt(b) ? -1 : 1)),
      `vector ${vector.x}-${vector.m ?? 'default'}`,
    );
    // Replaying the recorded RNG calls must produce the recorded amount through the adapter.
    const script = vector.rng.map(([n, value]) => [BigInt(n), BigInt(value)] as const);
    const calls: (readonly [bigint, bigint])[] = [];
    const rng = (n: bigint): bigint => {
      const step = script[calls.length];
      assert.ok(step, `unexpected extra RNG call ${n}`);
      calls.push([n, step[1]]);
      return step[1];
    };
    const result = adapter.recommendAmount(minimum, maximum, 'USD-stable', rng);
    assert.deepEqual(calls, script);
    assert.equal(result.recommended, adapter.formatAmount(BigInt(vector.y), 'USD-stable'));
    replayed++;
  }
  // Exactly the vectors under the page's fixed q/p and precision filters replay; a
  // filter or adapter change that narrows the set fails here instead of passing on 3.
  assert.equal(replayed, 84, `expected the 84 frontend-contract vectors, replayed ${replayed}`);
});

test('unmocked secure randomness only produces amounts in the upstream support', () => {
  // The page passes no rng, so the recommendation has to draw from the real
  // crypto.getRandomValues. It is wrapped only to count the calls, never to
  // substitute values, and the wrapper is removed before the assertions.
  type Draw = (buffer: Uint8Array) => Uint8Array;
  const secure = globalThis.crypto as unknown as { getRandomValues: Draw };
  const original = secure.getRandomValues;
  assert.equal(typeof original, 'function', 'the real crypto.getRandomValues must be available');
  let calls = 0;
  secure.getRandomValues = buffer => {
    calls++;
    return original.call(secure, buffer);
  };
  const values: string[] = [];
  try {
    for (let index = 0; index < 200; index++) values.push(adapter.recommendAmount('520', '1000', 'USD-stable').recommended);
  } finally {
    secure.getRandomValues = original;
  }
  const { x, m } = adapter.resolveRange('520', '1000', 'USD-stable');
  const d = adapter.config('USD-stable').roundingUnit;
  const support = [...upstream.distribution(x, m, { d }).keys()].map(amount => adapter.formatAmount(amount, 'USD-stable')).sort();
  // The page default range 520-1000 has exactly these three upstream candidates.
  assert.deepEqual(support, ['1000', '520', '600']);
  assert.ok(calls > 0, 'recommendations must draw from crypto.getRandomValues');
  assert.equal(values.length, 200);
  for (const value of values) assert.ok(support.includes(value), `${value} falls outside the upstream support ${support.join(', ')}`);
});
