// Shared browser-test helpers. No request is ever blocked here: every attempt is
// recorded so an unapproved third-party request fails the test instead of being hidden.
export const allowedOrigin = `http://127.0.0.1:${process.env.PORT ?? '4173'}`;

// Schemes that never leave the process; everything else must match the audited origin.
const LOCAL_SCHEMES = new Set(['data:', 'blob:', 'about:']);

/** True only for the audited origin itself — `startsWith` would accept a longer port. */
export function isApproved(url, base = allowedOrigin) {
  let target;
  let site;
  try {
    target = new URL(url);
    // Also safe when passed straight to Array#every, where arg 2 is an index.
    site = new URL(typeof base === 'string' ? base : allowedOrigin);
  } catch {
    return false;
  }
  if (LOCAL_SCHEMES.has(target.protocol)) return true;
  if (target.protocol === 'http:' || target.protocol === 'https:') return target.origin === site.origin;
  if (target.protocol === 'ws:' || target.protocol === 'wss:') {
    const secure = site.protocol === 'https:';
    return target.host === site.host && (target.protocol === 'ws:') !== secure;
  }
  return false;
}

// One collector per browser context: violations must survive navigations, which
// replace the window object and therefore any in-page array.
const collectors = new WeakMap();

/** Fail closed: an audit that could not measure must never look like a clean page. */
function probeFailure(message) {
  return new Error(`CSP probe fail-closed: ${message}`);
}

/**
 * Records every request, response status, failure, error and policy violation from page creation onwards.
 * Requires `probePolicy(context)` first: without it the audit refuses to exist, because
 * "not measuring violations" must not be reported as "zero violations".
 */
export function audit(page) {
  const collector = collectors.get(page.context());
  if (!collector) {
    throw new Error(
      'audit(): this browser context has no CSP collector. Call probePolicy(context) before audit(page); '
      + 'an unprobed page cannot be reported as violation-free.',
    );
  }
  const requests = [];
  const failures = [];
  const errors = [];
  const badResponses = [];
  page.on('request', request => requests.push(request.url()));
  page.on('requestfailed', request => failures.push(`${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    // A plain 404 is a response and never a requestfailed, so it must be caught here.
    if (response.status() >= 400) badResponses.push(`${response.status()} ${response.request().method()} ${response.url()}`);
  });
  return {
    requests,
    failures,
    errors,
    badResponses,
    external: () => requests.filter(url => !isApproved(url)),
    async violations() {
      if (!collectors.get(page.context())) throw probeFailure('the context collector is missing; call probePolicy(context) before audit(page).');
      // Read the document's own copy too: until this returns an array, the page
      // was never probed and its silence proves nothing.
      let url = '(unknown page)';
      try {
        url = page.url();
        // No navigation happens between here and the evaluate, so a destroyed
        // context is reported as a broken probe rather than retried away.
        const inPage = await page.evaluate(() => window.__cspViolations);
        if (!Array.isArray(inPage)) {
          const actual = inPage === undefined ? 'undefined' : typeof inPage;
          throw probeFailure(`window.__cspViolations is ${actual} on ${url}; the probe did not run in this document, so violations were not measured.`);
        }
        const all = [...collector.violations(), ...inPage];
        const seen = new Set();
        return all.filter(violation => {
          const key = JSON.stringify(violation);
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
      } catch (error) {
        if (/^CSP probe fail-closed:/.test(error.message)) throw error;
        throw probeFailure(`CSP probe unreadable on ${url}: ${error.message}. Refusing to report zero violations.`);
      }
    },
    async assertClean(label) {
      const violations = await this.violations();
      if (this.external().length) throw new Error(`${label}: unapproved external requests\n${this.external().join('\n')}`);
      if (this.badResponses.length) throw new Error(`${label}: HTTP >= 400 responses\n${this.badResponses.join('\n')}`);
      if (this.failures.length) throw new Error(`${label}: failed requests\n${this.failures.join('\n')}`);
      if (violations.length) throw new Error(`${label}: CSP violations\n${JSON.stringify(violations, null, 2)}`);
      if (this.errors.length) throw new Error(`${label}: page errors\n${this.errors.join('\n')}`);
    },
  };
}

/**
 * Installs the CSP probe before any navigation of this context and returns the
 * context's collector, so violations from every document accumulate in Node.
 */
export async function probePolicy(context) {
  const collected = [];
  const collector = { violations: () => [...collected] };
  collectors.set(context, collector);
  await context.exposeBinding('__reportCspViolation', (_source, record) => {
    collected.push(record);
  });
  await context.addInitScript(() => {
    const seen = [];
    Object.defineProperty(window, '__cspViolations', { value: seen, writable: false, configurable: true });
    document.addEventListener('securitypolicyviolation', event => {
      const record = {
        directive: event.effectiveDirective,
        blockedURI: event.blockedURI,
        sample: event.sample,
        disposition: event.disposition,
      };
      seen.push(record);
      const report = window.__reportCspViolation;
      if (typeof report === 'function') {
        try {
          const pending = report(record);
          if (pending && typeof pending.catch === 'function') pending.catch(() => {});
        } catch {
          // The context may already be closing; the in-page copy still has it.
        }
      }
    });
  });
  return collector;
}

/** Test-only RNG: filling the buffer with 1 makes the q = 1/2 coin stop at its first rung. */
export async function deterministic(context) {
  await context.addInitScript(() => {
    crypto.getRandomValues = buffer => {
      buffer.fill(1);
      return buffer;
    };
  });
}

export async function submit(page, minimum, maximum = '') {
  await page.locator('#minimum').fill(minimum);
  await page.locator('#maximum').fill(maximum);
  await page.locator('#recommend').click();
}
