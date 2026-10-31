import { defineConfig, devices } from '@playwright/test';
// These tests only visit loopback. Do not proxy the readiness check or browser
// traffic: inherited proxies can turn local requests into false HTTP 502 errors.
// This changes this test process and its children only, never system settings.
for (const key of Object.keys(process.env)) {
  if (/^(https?|all)_proxy$/i.test(key)) delete process.env[key];
}
const port = process.env.PORT ?? '4173';
const baseURL = `http://127.0.0.1:${port}`;
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: { timeout: 8_000 },
  fullyParallel: true,
  workers: 2,
  retries: 0,
  reporter: [['list'], ['html', { open: 'never' }], ['json', { outputFile: 'test-results/results.json' }]],
  use: {
    baseURL,
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    // Every test runs here. A new test needs no annotation to be covered.
    { name: 'chromium-desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } },
    // WebKit is scoped by the `@webkit` tag in a test title, matched against
    // "project describe title tags" by Playwright's project-level grep. A test
    // without the tag therefore runs on Chromium only by default: adding a test
    // never silently lengthens the slow WebKit pass, and opting in is one word.
    // WebKit's actionability checks run about an order of magnitude slower on CI than
    // locally, and what fired was the teardown budget, not an assertion. This project only.
    { name: 'webkit-mobile', grep: /@webkit/, timeout: 120_000, use: { ...devices['iPhone 13'], defaultBrowserType: 'webkit' } },
  ],
  webServer: {
    command: 'node scripts/serve.mjs',
    url: baseURL,
    reuseExistingServer: false,
    timeout: 20_000,
  },
});
