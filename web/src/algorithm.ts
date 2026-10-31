/** UI adapter only. The recommendation implementation lives unchanged in src/js. */
// Resolved through the engine package's own `exports` entry (a local `file:` dependency),
// so the frontend follows engine build-layout changes instead of freezing a dist path.
import { defaultM, recommend, unit } from '@withdrawchan/engine';
import type { Rng } from '@withdrawchan/engine';
export type { Rng };
export interface Recommendation {
  asset: Asset;
  minimum: string;
  maximum: string;
  recommended: string;
}
/**
 * Build one asset class entry.
 *
 * The token decimals and the minimum are the frontend's presentation facts; the rounding unit is
 * read from the engine's `src/units.json` through the engine's own `unit()`.
 */
function assetClass(code: 'USD-stable' | 'ETH', minimum: string, decimals: number, assetDecimals: number) {
  return { minimum, decimals, assetDecimals, roundingUnit: unit(code, assetDecimals) } as const;
}
/**
 * The engine's two asset classes, in the engine's order (the first entry is the UI default).
 */
export const assets = {
  'USD-stable': assetClass('USD-stable', '1', 2, 6),
  ETH: assetClass('ETH', '0.001', 4, 18),
} as const;
export type Asset = keyof typeof assets;
export type ErrorCode = 'invalid' | 'order' | 'maximum-unit';
export class InputError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode) { super(code); this.name = 'InputError'; this.code = code; }
}
export function config(asset: Asset) {
  // The message text is shared with the copy task; only the identifiers moved to asset naming.
  if (!Object.hasOwn(assets, asset)) throw new Error('Invalid pool');
  return assets[asset];
}
/** Reject exponent notation/extra precision; never pass token values through Number. */
export function parseAmount(raw: string, asset: Asset): bigint {
  const c = config(asset), text = raw.trim();
  if (!/^\d+(?:\.\d+)?$/.test(text)) throw new InputError('invalid');
  const [whole = '', fraction = ''] = text.split('.');
  if (fraction.length > c.decimals) throw new InputError('invalid');
  return BigInt(whole) * 10n ** BigInt(c.assetDecimals) + BigInt(fraction.padEnd(c.assetDecimals, '0'));
}
/** Canonical plain token amount. Safe to copy or serialize without losing ETH wei. */
export function formatAmount(units: bigint, asset: Asset): string {
  const decimals = config(asset).assetDecimals, factor = 10n ** BigInt(decimals);
  const whole = units / factor, fraction = (units % factor).toString().padStart(decimals, '0').replace(/0+$/, '');
  return `${whole}${fraction ? `.${fraction}` : ''}`;
}
export function defaultMaximum(minimum: string, asset: Asset): string {
  return formatAmount(defaultM(parseAmount(minimum, asset), config(asset).roundingUnit), asset);
}
export function resolveRange(minimum: string, maximum: string, asset: Asset) {
  const c = config(asset), lower = parseAmount(c.minimum, asset);
  const x = parseAmount(minimum, asset);
  if (x < lower) throw new InputError('invalid');
  const m = maximum.trim() === '' ? defaultM(x, c.roundingUnit) : parseAmount(maximum, asset);
  if (m < lower) throw new InputError('invalid');
  if (m < x) throw new InputError('order');
  if (m % c.roundingUnit !== 0n) throw new InputError('maximum-unit');
  return { x, m, minimum: formatAmount(x, asset), maximum: formatAmount(m, asset) };
}
export function recommendAmount(minimum: string, maximum: string, asset: Asset, rng?: Rng): Recommendation {
  const { x, m, minimum: min, maximum: max } = resolveRange(minimum, maximum, asset);
  const amount = recommend(x, m, { d: config(asset).roundingUnit, ...(rng ? { rng } : {}) });
  return { asset, minimum: min, maximum: max, recommended: formatAmount(amount, asset) };
}
