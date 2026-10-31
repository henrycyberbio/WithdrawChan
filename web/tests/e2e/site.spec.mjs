import { test, expect } from '@playwright/test';
import { copy } from '../../src/i18n.ts';
import { assets } from '../../src/algorithm.ts';
import { deterministic, submit } from './harness.mjs';

// The engine exposes two asset classes, not three pools. Only their integer scale is taken
// from the adapter; the conversions and the oracle below stay independent of the page's
// arithmetic (they never call defaultMaximum/parseAmount/formatAmount).
const assetSelect = page => page.locator('#asset');
const selectAsset = (page, asset) => assetSelect(page).selectOption(asset);
function toUnits(amount, asset) {
  const decimals = assets[asset].assetDecimals;
  const [whole, fraction = ''] = amount.split('.');
  return BigInt(whole) * 10n ** BigInt(decimals) + BigInt(fraction.padEnd(decimals, '0'));
}
function fromUnits(units, asset) {
  const decimals = assets[asset].assetDecimals;
  const factor = 10n ** BigInt(decimals);
  const fraction = (units % factor).toString().padStart(decimals, '0').replace(/0+$/, '');
  return `${units / factor}${fraction ? `.${fraction}` : ''}`;
}
/** Independent ceil(2x, d) oracle for the documented default maximum. */
function expectedDefaultMaximum(amount, asset) {
  const unit = assets[asset].roundingUnit;
  const doubled = toUnits(amount, asset) * 2n;
  return fromUnits(((doubled + unit - 1n) / unit) * unit, asset);
}

test('@webkit heading, browser tab title and eyebrow follow the language copy, not the theme', async ({ page }) => {
  await page.goto('/ipfs/test-cid/');
  await expect(page.locator('#input-title')).toHaveText(copy.en.title);
  await expect(page.locator('.eyebrow').first()).toHaveText(copy.en.eyebrow);
  await expect(page).toHaveTitle(copy.en.tabTitle);
  await page.locator('#language').click();
  await expect(page.locator('#input-title')).toHaveText(copy.zh.title);
  await expect(page.locator('#input-title')).not.toHaveText(copy.en.title);
  await expect(page.locator('.eyebrow').first()).toHaveText(copy.zh.eyebrow);
  await expect(page).toHaveTitle(copy.zh.tabTitle);
  await page.locator('#theme').click();
  await expect(page.locator('#input-title')).toHaveText(copy.zh.title);
  await expect(page).toHaveTitle(copy.zh.tabTitle);
  await page.locator('#language').click();
  await expect(page.locator('#input-title')).toHaveText(copy.en.title);
  await expect(page).toHaveTitle(copy.en.tabTitle);
});

// Regression: the heading used a fixed block-size, so the two-line English slogan spilled
// into the form. Measure the text's own line boxes: a heading element's box can look
// correct while its glyphs hang below it.
const heroLayout = page => page.evaluate(() => {
  const heading = document.querySelector('#input-title');
  const form = document.querySelector('#form');
  const range = document.createRange();
  range.selectNodeContents(heading);
  const lines = [...range.getClientRects()].filter(rect => rect.width > 1 && rect.height > 1);
  const style = getComputedStyle(heading);
  const box = heading.getBoundingClientRect();
  return {
    lines: lines.length,
    textTop: Math.min(...lines.map(rect => rect.top)),
    textBottom: Math.max(...lines.map(rect => rect.bottom)),
    headingTop: box.top,
    headingBottom: box.bottom,
    formTop: form.getBoundingClientRect().top,
    clientHeight: heading.clientHeight,
    lineHeight: Number.parseFloat(style.lineHeight),
    lineLeft: Math.min(...lines.map(rect => rect.left)),
    lineRight: Math.max(...lines.map(rect => rect.right)),
    viewportWidth: document.documentElement.clientWidth,
    marginBottom: style.marginBottom,
    fontSize: style.fontSize,
  };
});

// Two frames: the toggle changes layout synchronously, but a frame settles it before measuring.
const settled = page => page.evaluate(
  () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))),
);

// One case per viewport. A WebKit actionability check on a slow runner can cost
// seconds, so the 16 state blocks of a single combined loop spent the whole 120 s
// budget before the third width was reached. Registering the loop keeps all four
// widths, both languages and both themes, each case inside the per-test budget.
const heroViewports = [
  { width: 1440, height: 1000 },
  { width: 1366, height: 768 },
  { width: 390, height: 844 },
  { width: 320, height: 640 },
];
const heroStates = [
  { language: 'en', theme: 'playful' },
  { language: 'en', theme: 'zen' },
  { language: 'zh', theme: 'playful' },
  { language: 'zh', theme: 'zen' },
];
for (const viewport of heroViewports) {
  test(`@webkit every hero heading line stays above the input form in both themes and languages at ${viewport.width}px`, async ({ page }) => {
    await page.goto('/');
    await submit(page, '43.21', '60');
    const recommended = await page.locator('#value').textContent();
    const desktop = viewport.width > 650;
    await page.setViewportSize(viewport);
    let current = { language: 'en', theme: 'playful' };
    for (const state of heroStates) {
      if (state.theme !== current.theme) await page.locator('#theme').click();
      if (state.language !== current.language) await page.locator('#language').click();
      current = state;
      await settled(page);
      const where = `${state.language}/${state.theme} at ${viewport.width}px`;
      await expect(page.locator('#input-title')).toHaveText(copy[state.language].title);
      const layout = await heroLayout(page);
      expect(layout.lines, `${where}: the slogan renders`).toBeGreaterThanOrEqual(1);
      expect(layout.textBottom, `${where}: no heading line reaches the form`).toBeLessThan(layout.formTop);
      // The heading box must contain its own text lines. `scrollHeight` is not usable here:
      // the playful theme's decorative ::before glow sits outside the box and inflates it.
      expect(layout.textTop, `${where}: text starts inside the heading box`).toBeGreaterThanOrEqual(layout.headingTop - 0.5);
      expect(layout.textBottom, `${where}: text ends inside the heading box`).toBeLessThanOrEqual(layout.headingBottom + 0.5);
      expect(layout.clientHeight, `${where}: the heading is tall enough for every line`).toBeGreaterThanOrEqual(layout.lines * layout.lineHeight - 0.5);
      expect(layout.lineLeft, `${where}: no line spills left`).toBeGreaterThanOrEqual(-0.5);
      expect(layout.lineRight, `${where}: no line spills right`).toBeLessThanOrEqual(layout.viewportWidth + 0.5);
      // The overlap must not be bought by shrinking the heading or stealing its margin.
      expect(layout.marginBottom, `${where}: heading keeps its margin`).toBe(desktop ? '24px' : '22px');
      expect(layout.fontSize, `${where}: heading keeps its type size`).toBe(desktop ? '32px' : '24px');
      // Neither preference redraws the current suggestion.
      await expect(page.locator('#value')).toHaveText(recommended);
    }
  });
}

test('the input panel keeps only its labels, the minimum hint and the action', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('label[for="minimum"]')).toHaveText('Amount you need now');
  await expect(page.locator('label[for="maximum"]')).toHaveText('Maximum amount (optional)');
  await expect(page.locator('#minimum')).toHaveAttribute('aria-describedby', 'range-limit');
  await expect(page.locator('#range-limit')).toHaveText('At least 1 USD');
  // The result hint is theme-owned copy, not a separate translation branch.
  await expect(page.locator('#result-hint')).toHaveAttribute('data-i18n', 'hintPlayful');
  // The removed hygiene paragraph, gross note, local-algorithm badge and disclaimer
  // are gone from the form; the habit advice now lives in one themed line.
  for (const selector of ['#hygiene-note', '#gross-note', '#result-hygiene', '.demo', '.disclaimer']) {
    await expect(page.locator(selector)).toHaveCount(0);
  }
  await expect(page.locator('.result-hint')).toHaveCount(1);
  await expect(page.locator('#result-hint')).toHaveText(copy.en.hintPlayful);
  // Only the select keeps the engine key; the unit next to each amount is USD.
  await expect(page.locator('[data-asset-unit]')).toHaveText(['USD', 'USD', 'USD']);
  await selectAsset(page, 'ETH');
  await expect(page.locator('#range-limit')).toHaveText('At least 0.001 ETH');
  await expect(page.locator('[data-asset-unit]')).toHaveText(['ETH', 'ETH', 'ETH']);
  await page.locator('#language').click();
  await expect(page.locator('label[for="minimum"]')).toHaveText('当前所需金额');
  await expect(page.locator('label[for="maximum"]')).toHaveText('可选最大金额');
  await expect(page.locator('#range-limit')).toHaveText('至少 0.001 ETH');
  await expect(page.locator('#result-hint')).toHaveText(copy.zh.hintPlayful);
  await selectAsset(page, 'USD-stable');
  await expect(page.locator('#range-limit')).toHaveText('至少 1 USD');
  // The hints are text, not a balance field or a fixed retention threshold.
  await expect(page.locator('input')).toHaveCount(2);
  await expect(page.locator('body')).not.toContainText('10 USD');
});

test('static English fallback keeps the full page useful without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  try {
    const page = await context.newPage();
    await page.goto('/');
    await expect(page).toHaveTitle('WithdrawChan — Withdrawal Amount Suggestions');
    await expect(page.locator('#input-title')).toHaveText(copy.en.title);
    await expect(page.locator('.eyebrow').first()).toHaveText('WITHDRAWAL HYGIENE · BETTER HABITS');
    await expect(page.locator('label[for="minimum"]')).toHaveText('Amount you need now');
    // No JS repaints these, so the template's own options and unit labels are what ships.
    await expect(assetSelect(page).locator('option')).toHaveText(['USD-stable', 'ETH']);
    await expect(assetSelect(page)).toHaveValue('USD-stable');
    await expect(page.locator('[data-asset-unit]')).toHaveText(['USD', 'USD', 'USD']);
    await expect(page.locator('#range-limit')).toHaveText('At least 1 USD');
    await expect(page.locator('#result-label')).toHaveText('This looks good today ♡');
    await expect(page.locator('#result-hint')).toHaveText(copy.en.hintPlayful);
    await expect(page.locator('#gross-note')).toHaveCount(0);
    await expect(page.locator('#value')).toHaveText('—');
    for (const [key, text] of Object.entries(copy.en).filter(([key]) => key.startsWith('faq'))) {
      await expect(page.locator(`[data-i18n="${key}"]`)).toHaveText(text);
    }
  } finally {
    await context.close();
  }
});

test('FAQ shows exactly one language block, follows the language switch and keeps its open state', async ({ page }) => {
  await page.goto('/ipfs/test-cid/');
  const faq = page.locator('.faq');
  const item = page.locator('.faq-item').nth(2);
  await item.locator('summary').click();
  await expect(item.locator('table.faq-table').filter({ visible: true })).toHaveCount(1);
  for (const [step, language] of ['en', 'zh', 'en'].entries()) {
    if (step > 0) await page.locator('#language').click();
    const other = language === 'en' ? 'zh' : 'en';
    await expect(faq.locator('[data-i18n="faqTitle"]')).toHaveText(copy[language].faqTitle);
    await expect(faq.locator(`[data-lang="${language}"]`)).toHaveCount(12);
    await expect(faq.locator(`.faq-question [data-lang="${language}"]`).filter({ visible: true })).toHaveCount(6);
    await expect(faq.locator(`.faq-question [data-lang="${other}"]`).filter({ visible: true })).toHaveCount(0);
    await expect(item).toHaveAttribute('open', '');
    await expect(item.locator(`.faq-answer > [data-lang="${language}"]`)).toBeVisible();
    await expect(item.locator(`.faq-answer > [data-lang="${other}"]`)).toBeHidden();
    await expect(item.locator('table.faq-table').filter({ visible: true })).toContainText(language === 'zh' ? '抛到正面' : 'Heads');
    await page.locator('#theme').click();
    await expect(item.locator(`.faq-answer > [data-lang="${language}"]`)).toBeVisible();
  }
});

test('FAQ formulas render as MathML in the browser', async ({ page }) => {
  await page.goto('/');
  const item = page.locator('.faq-item').nth(3);
  await item.locator('summary').click();
  const formulas = item.locator('.faq-answer > [data-lang="en"] math[display="block"]');
  await expect(formulas).toHaveCount(2);
  for (const box of await formulas.evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height))) expect(box).toBeGreaterThan(10);
  await expect(item.locator('.faq-answer > [data-lang="en"] math').first()).toBeVisible();
});

test('@webkit local algorithm result, rounded default, stale state and asset memory', async ({ context, page }) => {
  await deterministic(context);
  await page.goto('/');
  await expect(assetSelect(page).locator('option')).toHaveText(['USD-stable', 'ETH']);
  await expect(assetSelect(page)).toHaveValue('USD-stable');
  await expect(page.locator('#copy-amount')).toBeDisabled();
  await submit(page, '43.21', '60');
  await expect(page.locator('#value')).toHaveText('44');
  await expect(page.locator('#range-note')).toHaveText('Within your 43.21–60 USD range');
  await expect(page.locator('#range-note')).not.toContainText('balance');
  // The range note is screen-reader only: the visible line is the theme hint.
  await expect(page.locator('#range-note')).toHaveClass(/sr-only/);
  await expect(page.locator('#recommend')).toBeDisabled();
  await page.locator('#language').click();
  await expect(page.locator('#result-hint')).toHaveText(copy.zh.hintPlayful);
  await page.locator('#theme').click();
  // Zen in Chinese must stay Chinese: the two lines come from the same i18n record.
  await expect(page.locator('#result-label')).toHaveText(copy.zh.zen);
  await expect(page.locator('#result-hint')).toHaveText(copy.zh.hintZen);
  await expect(page.locator('#result-hint')).not.toHaveText(copy.en.hintZen);
  await expect(page.locator('#value')).toHaveText('44');
  await page.locator('#maximum').fill('');
  await expect(page.locator('#maximum')).toHaveAttribute('placeholder', '87');
  await expect(page.locator('#range-note')).toContainText('重新计算');
  await expect(page.locator('#recommend')).toBeEnabled();
  await page.locator('#recommend').click();
  await expect(page.locator('#range-note')).toContainText('43.21–87');
  await selectAsset(page, 'ETH');
  await submit(page, '0.0011');
  await expect(page.locator('#maximum')).toHaveAttribute('placeholder', '0.003');
  await expect(page.locator('#value')).toHaveText('0.002');
  await selectAsset(page, 'USD-stable');
  await expect(page.locator('#minimum')).toHaveValue('43.21');
  await expect(page.locator('#maximum')).toHaveValue('');
  await expect(page.locator('#value')).toHaveText('44');
});

// One case per asset class: the stable class keeps the 1-unit step, ETH keeps 0.001.
for (const [asset, min, max, amount, step] of [
  ['USD-stable', '43.21', '60', '44', '1'],
  ['ETH', '0.0011', '0.003', '0.002', '0.001'],
]) {
  test(`${asset}: upstream recommendation and displayed unit`, async ({ context, page }) => {
    await deterministic(context);
    await page.goto('/ipfs/test-cid/');
    await selectAsset(page, asset);
    await expect(page.locator('#range-limit')).toHaveText(`At least ${step} ${asset === 'USD-stable' ? 'USD' : 'ETH'}`);
    await submit(page, min, max);
    await expect(page.locator('#value')).toHaveText(amount);
    await expect(page.locator('#error')).toBeEmpty();
  });
}

test('invalid inputs are rejected without silently changing the entered maximum', async ({ page }) => {
  await page.goto('/');
  for (const [min, max, message] of [
    ['', '', 'valid amounts'], ['1.001', '2', 'valid amounts'], ['1e2', '200', 'exponent'],
    ['20', '10', 'at least'], ['43.21', '60.42', 'multiple of 1 USD'],
    ['0', '1001', 'valid amounts'], ['1', '0', 'valid amounts'],
  ]) {
    await submit(page, min, max);
    await expect(page.locator('#error')).toContainText(message);
    await expect(page.locator('#maximum')).toHaveValue(max);
    await expect(page.locator('#recommend')).toBeDisabled();
  }
  await selectAsset(page, 'ETH');
  await submit(page, '0.0011', '0.0022');
  await expect(page.locator('#error')).toContainText('multiple of 0.001 ETH');
  await page.locator('#language').click();
  await expect(page.locator('#error')).toContainText('不会自动修改');
});

for (const [asset, minimum, maximum, defaultMin, defaultMax] of [
  ['USD-stable', '1001.25', '10000', '600', '1200'],
  ['ETH', '2.0001', '10', '0.6', '1.2'],
]) {
  test(`${asset}: no fixed amount ceiling and exact copy of the recommended string`, async ({ context, page }) => {
    await deterministic(context);
    await context.addInitScript(() => {
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async text => { window.__copied = text; } } });
    });
    await page.goto('/ipfs/test-cid/');
    await selectAsset(page, asset);
    for (const id of ['minimum', 'maximum']) {
      await expect(page.locator(`#${id}`)).toHaveAttribute('type', 'text');
      await expect(page.locator(`#${id}`)).toHaveAttribute('inputmode', 'decimal');
      await expect(page.locator(`#${id}`)).not.toHaveAttribute('max');
      await expect(page.locator(`#${id}`)).not.toHaveAttribute('maxlength');
    }
    await submit(page, minimum, maximum);
    await expect(page.locator('#error')).toBeEmpty();
    await expect(page.locator('#value')).not.toHaveText('—');
    await submit(page, defaultMin);
    await expect(page.locator('#error')).toBeEmpty();
    await expect(page.locator('#maximum')).toHaveValue('');
    await expect(page.locator('#maximum')).toHaveAttribute('placeholder', defaultMax);
    const amount = `1${'0'.repeat(320)}`;
    await submit(page, amount, amount);
    await expect(page.locator('#error')).toBeEmpty();
    await expect(page.locator('#maximum')).toHaveValue(amount);
    await expect(page.locator('#value')).toHaveText(new Intl.NumberFormat('en-US').format(BigInt(amount)));
    await page.locator('#copy-amount').click();
    expect(await page.evaluate(() => window.__copied)).toBe(amount);
    await page.locator('#language').click();
    await expect(page.locator('#range-limit')).toContainText('至少');
    await expect(page.locator('.faq-item').nth(2).locator('.faq-answer > [data-lang="zh"]')).not.toContainText('1,000');
  });
}

test('long exact decimals keep every low-order digit in the default maximum and result', async ({ context, page }) => {
  await deterministic(context);
  await page.goto('/');
  for (const [asset, minimum] of [
    ['USD-stable', '12345678901234567890123456789012345678901.43'],
    ['ETH', '123456789012345678901234567890.0013'],
  ]) {
    await selectAsset(page, asset);
    const expected = expectedDefaultMaximum(minimum, asset);
    const unit = assets[asset].roundingUnit;
    const grouped = new Intl.NumberFormat('en-US');
    const group = value => {
      const [whole = '0', fraction] = value.split('.');
      return `${grouped.format(BigInt(whole))}${fraction ? `.${fraction}` : ''}`;
    };
    await page.locator('#minimum').fill(minimum);
    await page.locator('#maximum').fill('');
    await expect(page.locator('#maximum')).toHaveAttribute('placeholder', expected);
    await page.locator('#recommend').click();
    await expect(page.locator('#error')).toBeEmpty();
    await expect(page.locator('#range-note')).toContainText(`${group(minimum)}–${group(expected)}`);
    const shown = await page.locator('#value').textContent();
    const plain = (shown ?? '').replace(/,/g, '');
    expect(plain).toMatch(/^\d+(?:\.\d+)?$/);
    const shownUnits = toUnits(plain, asset);
    expect(shownUnits % unit).toBe(0n);
    expect(shownUnits).toBeGreaterThanOrEqual(toUnits(minimum, asset));
    expect(shownUnits).toBeLessThanOrEqual(toUnits(expected, asset));
  }
});

test('fixed bounds and copy keep exact ungrouped token strings', async ({ context, page }) => {
  await context.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async text => { window.__copied = text; } } });
  });
  await page.goto('/');
  await submit(page, '1000', '1000');
  await expect(page.locator('#value')).toHaveText('1,000');
  await page.locator('#copy-amount').click();
  expect(await page.evaluate(() => window.__copied)).toBe('1000');
  await selectAsset(page, 'ETH');
  await submit(page, '0.001', '0.001');
  await page.locator('#copy-amount').click();
  expect(await page.evaluate(() => window.__copied)).toBe('0.001');
  await expect(page.locator('#copy-status')).toHaveText('Suggested amount copied.');
});

test('@webkit clipboard fallback and failure feedback remain functional', async ({ context, page }) => {
  await context.addInitScript(() => {
    window.__fallbackSuccess = true;
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async () => { throw new Error('Denied'); } } });
    document.execCommand = () => window.__fallbackSuccess;
  });
  await page.goto('/');
  await submit(page, '50', '50');
  await page.locator('#copy-amount').click();
  await expect(page.locator('#copy-status')).toContainText('copied');
  await page.evaluate(() => { window.__fallbackSuccess = false; });
  await page.locator('#copy-amount').click();
  await expect(page.locator('#copy-status')).toContainText('Copy failed');
  await expect(page.locator('textarea')).toHaveCount(0);
});

test('incomplete decimal maximum is rejected, not treated as a blank default', async ({ context, page }) => {
  await deterministic(context);
  await page.goto('/');
  await page.locator('#minimum').fill('43.21');
  await page.locator('#maximum').pressSequentially('1e');
  await expect(page.locator('#maximum')).toHaveValue('1e');
  await page.locator('#recommend').click();
  await expect(page.locator('#error')).not.toBeEmpty();
  await expect(page.locator('#value')).toHaveText('—');
  await page.locator('#maximum').fill('');
  await expect(page.locator('#recommend')).toBeEnabled();
  await page.locator('#recommend').click();
  await expect(page.locator('#value')).toHaveText('44');
  await expect(page.locator('#range-note')).toContainText('43.21–87');
});

test('@webkit unmocked secure randomness keeps real UI draws inside the upstream support', async ({ context, page }) => {
  await context.addInitScript(() => {
    const native = crypto.getRandomValues.bind(crypto);
    window.__rngCalls = 0;
    crypto.getRandomValues = buffer => {
      window.__rngCalls++;
      return native(buffer);
    };
  });
  await page.goto('/');
  const values = [];
  // A few real clicks only prove the button-to-page wiring. The repeated sampling
  // (200 unmocked draws over the same 520-1000 support) runs without a browser in
  // web/tests/algorithm.test.ts, so this test stays inside its per-test budget.
  for (let index = 0; index < 3; index++) {
    // Clearing first re-enables the button without changing the drawn support.
    await page.locator('#minimum').fill('');
    await page.locator('#minimum').fill('520');
    await page.locator('#maximum').fill('1000');
    await page.locator('#recommend').click();
    await expect(page.locator('#error')).toBeEmpty();
    values.push((await page.locator('#value').textContent() ?? '').replace(/,/g, ''));
  }
  expect(await page.evaluate(() => window.__rngCalls)).toBeGreaterThan(0);
  expect(values.every(value => ['520', '600', '1000'].includes(value))).toBe(true);
});

test('the fixed chart does not change with input and its bars agree with the table and data', async ({ context, page }) => {
  await deterministic(context);
  await page.goto('/ipfs/test-cid/');
  await page.waitForLoadState('networkidle');
  const before = await page.locator('#privacy-comparison').innerHTML();
  await submit(page, '43.21', '60');
  await page.locator('#minimum').fill('44.21');
  await page.locator('#recommend').click();
  expect(await page.locator('#privacy-comparison').innerHTML()).toBe(before);
  await expect(page.locator('#privacy-note')).toHaveText(copy.en.privacyNote);
  await expect(page.locator('#privacy-note')).toContainText('50% adoption');
  await expect(page.locator('.chart-unit')).toContainText(copy.en.privacyIndex);
  // Each footnote starts a real line (one string plus `white-space: pre-line`): its first
  // character sits at the block's left edge instead of mid-sentence. The removed -log2
  // sentence must not come back.
  const note = await page.locator('#privacy-note').evaluate(node => {
    const text = node.textContent;
    const blockLeft = Math.round(node.getBoundingClientRect().left);
    const charBox = index => {
      const range = document.createRange();
      range.setStart(node.firstChild, index);
      range.setEnd(node.firstChild, index + 1);
      return Math.round(range.getBoundingClientRect().left);
    };
    return {
      whiteSpace: getComputedStyle(node).whiteSpace,
      singleTextNode: node.childNodes.length === 1,
      markerLeft: [...text.matchAll(/\n\*/g)].map(match => charBox(match.index + 1)),
      blockLeft,
      text,
    };
  });
  expect(note.whiteSpace).toBe('pre-line');
  expect(note.singleTextNode).toBe(true);
  expect(note.markerLeft.length).toBe(2);
  for (const left of note.markerLeft) expect(Math.abs(left - note.blockLeft)).toBeLessThanOrEqual(1);
  expect(note.text).not.toMatch(/log2|log₂/);
  await expect(page.locator('.chart-legend li')).toHaveText(['Non-WithdrawChan', 'Drain', 'Keep']);
  // The chart must agree with itself: each bar's visible value is its data-bits and the
  // accessible table cell for the same pool. No hand-copied value list here.
  const chart = await page.evaluate(() => ({
    groups: [...document.querySelectorAll('.pool-group')].map(group => ({
      pool: group.querySelector('.chart-pool-name')?.textContent?.trim(),
      bars: [...group.querySelectorAll('.entropy-bar')].map(bar => ({
        bits: bar.dataset.bits,
        value: bar.querySelector('.bar-value')?.textContent?.trim(),
      })),
    })),
    rows: [...document.querySelectorAll('.chart-accessible tbody tr')].map(row => ({
      pool: row.querySelector('th')?.textContent?.trim(),
      cells: [...row.querySelectorAll('td')].map(cell => cell.textContent?.trim()),
    })),
  }));
  expect(chart.groups.map(group => group.pool)).toEqual(chart.rows.map(row => row.pool));
  expect(chart.groups.length).toBeGreaterThan(0);
  for (const [index, group] of chart.groups.entries()) {
    expect(group.bars.map(bar => bar.value)).toEqual(group.bars.map(bar => bar.bits));
    expect(chart.rows[index].cells).toEqual(group.bars.map(bar => `${bar.bits} bits`));
  }
});

test.describe('normal motion', () => {
  test.use({ reducedMotion: 'no-preference' });
  test('chart reveal still works', async ({ page }) => {
    await page.goto('/');
    await page.locator('#privacy-comparison').scrollIntoViewIfNeeded();
    await expect(page.locator('#privacy-comparison')).toHaveClass(/revealed/);
    await expect(page.locator('#privacy-comparison')).not.toHaveClass(/reveal-pending/);
  });
});
