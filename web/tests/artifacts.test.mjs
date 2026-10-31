import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = path.join(root, 'public');
const repo = path.resolve(root, '..');

test('the frontend resolves the engine through the published package entry, not a frozen dist path', () => {
  const pkg = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
  assert.equal(pkg.devDependencies['@withdrawchan/engine'], 'file:../src/js');
  for (const name of ['src/algorithm.ts', 'tests/algorithm.test.ts']) {
    const source = readFileSync(path.join(root, name), 'utf8');
    assert.doesNotMatch(source, /src\/js\/dist/, `${name} must not freeze the engine dist layout`);
    assert.match(source, /(['"])@withdrawchan\/engine\1/, `${name} must import the engine package entry`);
  }
  const engine = JSON.parse(readFileSync(path.join(repo, 'src/js/package.json'), 'utf8'));
  assert.ok(engine.exports?.['.']?.default, 'the engine package must expose its compiled entry');
});

// A missing build must fail the suite, not silently skip every generated-page check.
assert.ok(existsSync(path.join(output, 'index.html')), 'missing build output: run npm run build before npm run test:unit');

const html = () => readFileSync(path.join(output, 'index.html'), 'utf8');

test('the generated page loads no external stylesheet, script, image or font', () => {
  const page = html();
  for (const match of page.matchAll(/<(script|img|link)\b[^>]*?(?:src|href)="([^"]+)"/g)) {
    const [, tag, url] = match;
    assert.ok(!/^(https?:)?\/\//.test(url), `${tag} must not load a third party resource: ${url}`);
  }
  assert.doesNotMatch(page, /@import/, 'no CSS import may add a runtime request');
  for (const name of readdirSync(output).filter(n => n.endsWith('.css'))) {
    const css = readFileSync(path.join(output, name), 'utf8');
    assert.doesNotMatch(css, /@import/, name);
    for (const match of css.matchAll(/url\(\s*['"]?([^)'"\s]+)/g)) {
      assert.ok(!/^(https?:)?\/\//.test(match[1]), `${name} must not load a remote resource: ${match[1]}`);
    }
    assert.doesNotMatch(css, /fonts\.googleapis|fonts\.gstatic/, name);
  }
});

test('the static page ships a meta CSP that limits resources, connections and form targets', () => {
  const page = html();
  const csp = page.match(/<meta http-equiv="Content-Security-Policy" content="([^"]+)">/i);
  assert.ok(csp, 'a Content-Security-Policy meta element is required');
  const directives = new Map(csp[1].split(';').map(part => part.trim()).filter(Boolean).map(part => {
    const [name, ...values] = part.split(/\s+/);
    return [name, values];
  }));
  assert.deepEqual(directives.get('default-src'), ["'none'"]);
  assert.deepEqual(directives.get('script-src'), ["'self'"]);
  assert.deepEqual(directives.get('connect-src'), ["'none'"]);
  assert.deepEqual(directives.get('form-action'), ["'none'"]);
  assert.deepEqual(directives.get('object-src'), ["'none'"]);
  assert.deepEqual(directives.get('base-uri'), ["'none'"]);
  assert.deepEqual(directives.get('font-src'), ["'self'"]);
  assert.ok(directives.get('style-src').includes("'unsafe-inline'"), 'existing inline styles need an explicit allowance');
  assert.ok(directives.get('style-src').includes("'self'"));
  assert.ok(directives.get('img-src').includes("'self'"));
  assert.ok(!directives.has('frame-ancestors'), 'meta CSP cannot enforce frame-ancestors');
  const policyIndex = page.indexOf('Content-Security-Policy');
  const scriptIndex = page.search(/<script\b/);
  assert.ok(policyIndex > 0 && policyIndex < 1024, 'CSP must sit early in the document head');
  if (scriptIndex >= 0) assert.ok(policyIndex < scriptIndex, 'CSP must precede scripts');
  assert.ok(policyIndex < page.indexOf('<link'), 'CSP must precede stylesheets');
});

test('all local HTML/CSS assets resolve without a domain-root dependency', () => {
  const page = html();
  const css = readFileSync(path.join(output, 'style.css'), 'utf8');
  const refs = [
    ...[...page.matchAll(/<(?:script|img|link)\b[^>]*?(?:src|href)="([^"]+)"/g)].map(m => m[1]),
    ...[...css.matchAll(/url\(['"]?([^)'"\s]+)['"]?\)/g)].map(m => m[1]),
  ];
  for (const ref of refs) {
    if (/^(data:|#|mailto:)/.test(ref)) continue;
    assert.ok(!/^(https?:)?\/\//.test(ref), `remote asset reference: ${ref}`);
    assert.ok(!ref.startsWith('/'), `root-absolute asset breaks path gateways: ${ref}`);
    const clean = ref.split(/[?#]/)[0];
    assert.ok(existsSync(path.resolve(output, clean)), ref);
  }
});

test('generated pages do not embed the build domain', () => {
  const domain = readFileSync(path.join(root, 'config.toml'), 'utf8').match(/^base_url = "([^"]+)"/m)[1];
  for (const name of readdirSync(output).filter(n => /\.(html|xml|txt)$/.test(n))) {
    assert.ok(!readFileSync(path.join(output, name), 'utf8').includes(domain), name);
  }
});

test('deployment output contains neither TS source nor test/tool files', () => {
  const files = [];
  const walk = dir => { for (const n of readdirSync(dir)) { const p = path.join(dir, n); statSync(p).isDirectory() ? walk(p) : files.push(path.relative(output, p)); } };
  walk(output);
  assert.ok(files.includes('app.js'));
  assert.ok(!files.some(n => /(^|[\\/])(node_modules|tests|\.tools|src)([\\/]|$)|\.(ts|map)$/.test(n)), files.join('\n'));
});
