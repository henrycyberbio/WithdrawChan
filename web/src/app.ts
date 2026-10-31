import { config as assetConfig, defaultMaximum, formatAmount, InputError, parseAmount, assets as assetTable, recommendAmount, resolveRange } from './algorithm.ts';
import type { Recommendation, Asset, ErrorCode as InputErrorCode } from './algorithm.ts';
import { assetLabel, copy, messages, nextLanguage } from './i18n.ts';
import type { Language, MessageKey, Theme } from './i18n.ts';

type ErrorCode = '' | InputErrorCode;
type CopyFeedback = '' | 'copied' | 'failed';
interface AssetState { min: string; max: string; last: Recommendation | null }
/**
 * The selectable assets are exactly the engine's asset keys, in the engine's order:
 * the first entry is the default. The set is never duplicated in the UI.
 */
const ASSETS = Object.keys(assetTable) as Asset[];
// The asset table is a fixed internal non-empty list, so its first key is the default.
const DEFAULT_ASSET = ASSETS[0]!;
interface Elements {
  asset: HTMLSelectElement;
  minimum: HTMLInputElement;
  maximum: HTMLInputElement;
  form: HTMLFormElement;
  recommend: HTMLButtonElement;
  'copy-amount': HTMLButtonElement;
  language: HTMLButtonElement;
  theme: HTMLButtonElement;
  'copy-status': HTMLElement;
  value: HTMLElement;
  'range-note': HTMLElement;
  error: HTMLElement;
  'result-label': HTMLElement;
  'result-hint': HTMLElement;
}
function $<K extends keyof Elements>(id: K): Elements[K] {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Required element missing: ${id}`);
  return element as Elements[K];
}
const assetSelect = $('asset');
let language: Language = 'en';
let theme: Theme = 'playful';
let last: Recommendation | null = null;
let activeAsset: Asset = DEFAULT_ASSET;
let errorCode: ErrorCode = '';
let submitted = false;
let lastInputKey = '';
let copyFeedback: CopyFeedback = '';
let copyTimer: ReturnType<typeof setTimeout> | null = null;
const assetStates: Partial<Record<Asset, AssetState>> = {};
const config = (asset: Asset = activeAsset) => assetConfig(asset);
/** All UI sentences come from the single typed copy source. */
const t = () => messages[language];
const fmt = (amount: string) => {
  const [whole = '0', fraction] = amount.split('.');
  const grouped = new Intl.NumberFormat(t().locale).format(BigInt(whole));
  return `${grouped}${fraction ? `.${fraction}` : ''}`;
};

function paintAssetOptions(): void {
  const values = [...assetSelect.options].map(option => option.value);
  if (values.length === ASSETS.length && values.every((value, index) => value === ASSETS[index])) return;
  assetSelect.replaceChildren(...ASSETS.map(key => {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = key;
    return option;
  }));
  assetSelect.value = activeAsset;
}
function paintAsset(): void {
  const c = config(), unit = assetLabel(activeAsset);
  paintAssetOptions();
  document.querySelectorAll('[data-asset-unit]').forEach(n => { n.textContent = unit; });
  assetSelect.value = activeAsset;
  for (const id of ['minimum', 'maximum'] as const) {
    $(id).setAttribute('aria-describedby', 'range-limit');
  }
  const limit = t().limitLine({ minimum: fmt(c.minimum), unit });
  document.querySelectorAll('[data-i18n="limit"]').forEach(n => { n.textContent = limit; });
}
function readRange() {
  if ($('minimum').validity.badInput || $('maximum').validity.badInput) throw new InputError('invalid');
  return resolveRange($('minimum').value, $('maximum').value, activeAsset);
}
function inputKey(): string { return [activeAsset, $('minimum').value, $('maximum').value, $('minimum').validity.badInput, $('maximum').validity.badInput].join('|'); }
/**
 * Whether the shown result still describes the current input range.
 *
 * Compares the canonical resolved range on both sides, so an empty optional maximum equals
 * its default, alternative spellings of the same amount compare equal, illegal input yields
 * no copyable result, and a result only describes the asset it was computed for.
 */
function resultMatchesRange(): boolean {
  if (!last || last.asset !== activeAsset) return false;
  try {
    const { minimum, maximum } = readRange();
    return minimum === last.minimum && maximum === last.maximum;
  } catch { return false; }
}
function paintMaximumHint(): void {
  try {
    $('maximum').placeholder = parseAmount($('minimum').value, activeAsset) > 0n ? defaultMaximum($('minimum').value, activeAsset) : '0';
  } catch { $('maximum').placeholder = '0'; }
  const unit = `${formatAmount(config().roundingUnit, activeAsset)} ${assetLabel(activeAsset)}`;
  $('maximum').title = t().defaultMaximumTitle({ unit });
}
function paintButton(): void {
  const button = $('recommend');
  button.classList.toggle('is-calculated', submitted);
  button.disabled = submitted;
  button.setAttribute('aria-disabled', String(submitted));
  button.title = submitted ? t().resubmitTitle() : '';
}
function paintCopyButton(): void {
  const button = $('copy-amount');
  button.disabled = !resultMatchesRange();
  button.classList.toggle('copied', copyFeedback === 'copied');
  button.classList.toggle('copy-failed', copyFeedback === 'failed');
  const label = copyFeedback === 'copied' ? t().copyAmountCopied : copyFeedback === 'failed' ? t().copyAmountFailed : t().copyAmountLabel;
  button.setAttribute('aria-label', label);
  button.title = copyFeedback === 'failed' ? t().copyAmountFailed : label;
  $('copy-status').textContent = copyFeedback === 'copied' ? t().copyStatusCopied : copyFeedback === 'failed' ? t().copyStatusFailed : '';
}
function resetCopyFeedback(): void {
  if (copyTimer !== null) clearTimeout(copyTimer);
  copyTimer = null;
  copyFeedback = '';
  paintCopyButton();
}
async function writeAmount(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(text); return; } catch { /* Fallback below. */ }
  }
  const focused = document.activeElement, field = document.createElement('textarea');
  field.value = text;
  field.setAttribute('readonly', '');
  field.setAttribute('aria-hidden', 'true');
  field.tabIndex = -1;
  field.style.position = 'fixed';
  field.style.left = '-9999px';
  document.body.appendChild(field);
  let success = false;
  try { field.select(); success = !!document.execCommand?.('copy'); }
  finally { field.remove(); if (focused instanceof HTMLElement) focused.focus({ preventScroll: true }); }
  if (!success) throw new Error('Clipboard unavailable');
}
async function copyAmount(): Promise<void> {
  if (!last || !resultMatchesRange()) return;
  const result = last;
  const text = result.recommended;
  if (copyTimer !== null) clearTimeout(copyTimer);
  try {
    await writeAmount(text);
    if (last !== result) return;
    copyFeedback = 'copied';
    paintCopyButton();
    copyTimer = setTimeout(() => { copyFeedback = ''; copyTimer = null; paintCopyButton(); }, 1800);
  } catch {
    if (last !== result) return;
    copyFeedback = 'failed';
    paintCopyButton();
  }
}
function markSubmitted(): void { submitted = true; lastInputKey = inputKey(); paintButton(); }
function paintResult(): void {
  if (!last) {
    $('value').textContent = '—';
    $('range-note').textContent = '';
    return;
  }
  $('value').textContent = fmt(last.recommended);
  $('range-note').textContent = t().resultRange({ minimum: fmt(last.minimum), maximum: fmt(last.maximum), unit: assetLabel(activeAsset) });
}
function run(minimum: string, maximum: string): Recommendation {
  last = recommendAmount(minimum, maximum, activeAsset);
  errorCode = '';
  resetCopyFeedback(); paintError(); paintResult(); markSubmitted();
  return last;
}
function paintError(): void {
  const c = config(), unit = `${formatAmount(c.roundingUnit, activeAsset)} ${assetLabel(activeAsset)}`;
  let message = '';
  if (errorCode === 'order') message = t().order;
  else if (errorCode === 'maximum-unit') message = t().maximumUnitError({ unit });
  else if (errorCode) message = t().invalidAmountError({ minimum: fmt(c.minimum), unit: assetLabel(activeAsset), decimals: c.decimals });
  $('error').textContent = message;
}
function paintLabels(): void {
  document.documentElement.lang = t().htmlLang;
  document.title = t().tabTitle;
  document.querySelectorAll<HTMLElement>('[data-i18n]').forEach(n => {
    const key = n.dataset.i18n;
    if (!key || !Object.hasOwn(copy[language], key)) throw new Error(`Missing translation: ${key}`);
    n.textContent = copy[language][key as MessageKey];
  });
  // Rich FAQ prose ships in both languages as static HTML; only visibility changes.
  document.querySelectorAll<HTMLElement>('[data-lang]').forEach(n => { n.hidden = n.dataset.lang !== language; });
  $('language').textContent = t().languageButtonLabel;
  $('language').setAttribute('aria-label', t().languageSwitch);
  $('result-label').textContent = copy[language][theme];
  $('result-hint').textContent = theme === 'zen' ? t().hintZen : t().hintPlayful;
  $('theme').setAttribute('aria-label', theme === 'zen' ? t().themeSwitchPlayful : t().themeSwitchZen);
  paintAsset(); paintMaximumHint(); paintError(); paintResult(); checkDirty(); paintButton(); paintCopyButton();
}
function checkDirty(): void {
  if (last && !resultMatchesRange()) $('range-note').textContent = t().stale;
}
function setAsset(next: Asset): void {
  if (next === activeAsset) return;
  assetStates[activeAsset] = { min: $('minimum').value, max: $('maximum').value, last };
  activeAsset = next;
  errorCode = '';
  const state = assetStates[next];
  $('minimum').value = state ? state.min : '';
  $('maximum').value = state ? state.max : '';
  last = state ? state.last : null;
  resetCopyFeedback();
  submitted = false;
  lastInputKey = inputKey();
  paintLabels();
}
function requireAsset(value: unknown): Asset {
  if (typeof value !== 'string' || !Object.hasOwn(assetTable, value)) throw new Error('Invalid asset');
  return value as Asset;
}
$('copy-amount').addEventListener('click', copyAmount);
assetSelect.addEventListener('change', () => setAsset(requireAsset(assetSelect.value)));
$('form').addEventListener('submit', e => {
  e.preventDefault();
  if (submitted) return;
  try { const { minimum, maximum } = readRange(); run(minimum, maximum); }
  catch (error: unknown) { errorCode = error instanceof InputError ? error.code : 'invalid'; paintError(); }
  finally { markSubmitted(); paintCopyButton(); }
});
(['minimum', 'maximum'] as const).forEach(id => $(id).addEventListener('input', () => {
  const key = inputKey();
  if (key !== lastInputKey) { submitted = false; lastInputKey = key; }
  errorCode = '';
  paintError(); paintMaximumHint(); paintResult(); checkDirty(); paintButton(); paintCopyButton();
}));
$('language').onclick = () => { language = nextLanguage(language); paintLabels(); };
$('theme').onclick = () => {
  theme = theme === 'playful' ? 'zen' : 'playful';
  document.body.classList.toggle('zen', theme === 'zen');
  $('theme').setAttribute('aria-checked', String(theme === 'zen'));
  paintLabels();
};
lastInputKey = inputKey();
paintLabels();

// Progressive enhancement: the chart stays visible without animation support.
const comparisonSection = document.getElementById('privacy-comparison');
if (comparisonSection && 'IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  if (comparisonSection.getBoundingClientRect().top > window.innerHeight) comparisonSection.classList.add('reveal-pending');
  const revealObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) { entry.target.classList.remove('reveal-pending'); entry.target.classList.add('revealed'); revealObserver.unobserve(entry.target); }
    }
  }, { threshold: .12 });
  revealObserver.observe(comparisonSection);
  document.querySelector('.scroll-cue')?.addEventListener('click', () => comparisonSection.classList.remove('reveal-pending'));
}
