import { createServer } from 'node:http';
import { test, expect } from '@playwright/test';
import { allowedOrigin, audit, isApproved, probePolicy } from './harness.mjs';

test('a CSP violation on the first page still fails the audit after a clean navigation', async ({ browser }) => {
  const context = await browser.newContext();
  try {
    await probePolicy(context);
    const page = await context.newPage();
    const log = audit(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // script-src 'self' blocks the inline script and form-action 'none' blocks the
    // POST; both are stopped before a network request exists, so only the
    // securitypolicyviolation event records the attempt.
    await page.evaluate(() => {
      const script = document.createElement('script');
      script.textContent = 'window.__injected = true';
      document.body.append(script);
      const form = document.createElement('form');
      form.method = 'POST';
      form.action = 'https://blocked.invalid/withdraw';
      document.body.append(form);
      form.requestSubmit();
    });
    await expect.poll(async () => (await log.violations()).length).toBeGreaterThan(1);
    expect(await page.evaluate(() => window.__cspViolations.length)).toBeGreaterThan(1);
    expect(log.requests.filter(url => url.includes('blocked.invalid'))).toEqual([]);
    expect(await page.evaluate(() => window.__injected)).toBeUndefined();

    // The violation belonged to the previous document; the next one is clean.
    await page.goto('/ipfs/test-cid/');
    await page.waitForLoadState('networkidle');
    expect(await page.evaluate(() => window.__cspViolations.length)).toBe(0);

    await expect(log.assertClean('after clean navigation')).rejects.toThrow(/CSP violations/);
  } finally {
    await context.close();
  }
});

test('the audit treats another origin, including a longer port, as external', async ({ browser }) => {
  // Pure predicate first: a port that merely starts with the allowed port is a
  // different origin, as are scheme and host/userinfo lookalikes.
  expect(isApproved(`${allowedOrigin}/`)).toBe(true);
  expect(isApproved(`${allowedOrigin}/ipfs/test-cid/app.js`)).toBe(true);
  expect(isApproved('data:image/png;base64,AAAA')).toBe(true);
  // A longer port that merely has the allowed port as a string prefix.
  expect(isApproved('http://127.0.0.1:43210/', 'http://127.0.0.1:4321')).toBe(false);
  expect(isApproved('http://127.0.0.1:4321/', 'http://127.0.0.1:4321')).toBe(true);
  expect(isApproved('http://127.0.0.1:4321@evil.example/', 'http://127.0.0.1:4321')).toBe(false);
  expect(isApproved('http://127.0.0.1:4321.evil.example/', 'http://127.0.0.1:4321')).toBe(false);
  expect(isApproved('https://127.0.0.1:4321/', 'http://127.0.0.1:4321')).toBe(false);
  expect(isApproved('http://localhost:4321/', 'http://127.0.0.1:4321')).toBe(false);
  // The specs pass the predicate straight to Array#every; argument 2 is an index there.
  expect([`${allowedOrigin}/`, 'data:text/plain,x'].every(isApproved)).toBe(true);
  expect([`${allowedOrigin}/`, 'http://127.0.0.1:43210/'].every(isApproved)).toBe(false);

  // And end to end: navigating to a second server on this machine is an
  // unapproved origin even though it shares the host name.
  const server = createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }).end('<!doctype html><title>other</title>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const other = `http://127.0.0.1:${server.address().port}`;
  const context = await browser.newContext();
  try {
    await probePolicy(context);
    const page = await context.newPage();
    const log = audit(page);
    await page.goto(`${other}/`);
    expect(log.external()).toEqual([`${other}/`]);
    await expect(log.assertClean('other origin')).rejects.toThrow(/unapproved external requests/);
  } finally {
    await context.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test('an audit on a context without the CSP probe fails closed instead of reporting zero violations', async ({ browser }) => {
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    // The check is Node-side only: no probe means the audit cannot be armed at
    // all, so it must throw at creation, before any navigation.
    expect(() => audit(page)).toThrow(/probePolicy/);
    // Proof the page really is unprobed: there is no in-page array to read, and
    // treating that absence as "no violations" is exactly the bug being fixed.
    expect(await page.evaluate(() => window.__cspViolations)).toBeUndefined();
  } finally {
    await context.close();
  }
});

test('an audit whose probe cannot be read fails instead of reporting zero violations', async ({ browser }) => {
  const context = await browser.newContext();
  try {
    await probePolicy(context);
    const page = await context.newPage();
    // Creating the audit before the first navigation must stay allowed: the
    // context collector exists even though about:blank has no marker yet.
    const log = audit(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    expect(await page.evaluate(() => Array.isArray(window.__cspViolations))).toBe(true);

    // Simulate a document the init script never reached: the array is gone even
    // though the context collector is still registered and empty.
    await page.evaluate(() => {
      delete window.__cspViolations;
    });
    expect(await page.evaluate(() => window.__cspViolations)).toBeUndefined();
    await expect(log.violations()).rejects.toThrow(/__cspViolations/);
    await expect(log.assertClean('marker deleted')).rejects.toThrow(/CSP probe/);

    // The same refusal applies once the page itself is gone.
    await page.close();
    await expect(log.violations()).rejects.toThrow(/CSP probe unreadable/);
    await expect(log.assertClean('page closed')).rejects.toThrow(/CSP probe unreadable/);
  } finally {
    await context.close();
  }
});

test('a 404 under normal browsing fails the audit', async ({ browser }) => {
  const context = await browser.newContext();
  try {
    await probePolicy(context);
    const page = await context.newPage();
    const log = audit(page);
    const response = await page.goto('/missing-page-for-audit');
    expect(response.status()).toBe(404);
    await expect(log.assertClean('missing page')).rejects.toThrow(/HTTP >= 400/);
  } finally {
    await context.close();
  }
});
