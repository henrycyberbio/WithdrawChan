---
name: withdrawchan-web-dev
description: Use when building, previewing or testing the website in web/. Skip for the recommendation rule in src/ and for lab/.
---

# Build and test the website

## Setup and build

Run every command from the repository root with Node.js 22.18 or newer.

- `npm run setup` installs the engine and website dependencies with `npm ci` and downloads Zola 0.23.6, checking its SHA-256. It supports Windows x64, Linux x64/arm64 and macOS x64/arm64.
- `npm run build` compiles the engine in `src/js`, type-checks and bundles the website, and builds the site into `web/public/`. Deploy only that directory. A failed build removes it.
- Do not run `zola build` directly. The templates read `web/data/messages.json`, which the build generates from `web/src/i18n.ts`.
- `npm run dev` builds and serves the site on `http://127.0.0.1:4173/`; set `PORT` to change the port. `npm run preview` serves an existing build. There is no hot reload. On Windows, `RUN-WINDOWS.bat` runs setup and dev.

## Tests

- `npm test` builds, then runs the engine tests and the website's tooling, unit and browser tests.
- The browser tests need Playwright's Chromium and WebKit: `npm exec --prefix web -- playwright install chromium webkit` (add `--with-deps` on Linux). Without them every browser test fails at launch. A browser test that did not run is unverified, not passed.

## Changing the site

- Rounding units come from `src/units.json` through the engine's `unit()`. Never copy them into `web/`.
- Interface text lives in `web/src/i18n.ts`, with the same keys in `en` and `zh`. The FAQ is written in both languages in `web/templates/partials/faq.html`; change both blocks together.
- The page loads no external resources, and the CSP in `web/templates/base.html` blocks network requests. `web/tests/e2e/network.spec.mjs` fails on any unexpected request.
