import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { platformAsset } from './setup.mjs';
import { createPreviewServer, resolveRequest } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
test('installer pins supported archives and rejects unknown targets', () => {
  assert.equal(platformAsset('linux', 'x64').triple, 'x86_64-unknown-linux-gnu');
  assert.throws(() => platformAsset('win32', 'arm64'), /Unsupported/);
});
test('the preview serves the built site at the root and under the IPFS mount only', () => {
  assert.equal(resolveRequest('/ipfs/test-cid/style.css?v=1', root).file, path.join(root, 'public', 'style.css'));
  assert.equal(resolveRequest('/', root).file, path.join(root, 'public'));
  // The retired one-off snapshot mount must not come back: it resolves inside public/ and 404s.
  assert.equal(resolveRequest('/baseline/app.js', root).file, path.join(root, 'public', 'baseline', 'app.js'));
  for (const url of ['/../README.md', '/%2e%2e/README.md', '/%2e%2e%5cREADME.md', '/bad%00', '/bad%']) {
    assert.throws(() => resolveRequest(url, root));
  }
});
test('HTTP serves the IPFS mount, HEAD, redirects, missing files and rejects other methods', async () => {
  const server = createPreviewServer(root);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const redirect = await fetch(`${base}/ipfs/test-cid?test=1`, { redirect: 'manual' });
    assert.equal(redirect.status, 308);
    assert.equal(redirect.headers.get('location'), '/ipfs/test-cid/?test=1');
    const page = await fetch(`${base}/ipfs/test-cid/`);
    assert.equal(page.status, 200);
    assert.match(page.headers.get('content-type'), /text\/html/);
    assert.match(await page.text(), /<!doctype html>/i);
    const head = await fetch(`${base}/ipfs/test-cid/style.css`, { method: 'HEAD' });
    assert.equal(head.status, 200);
    assert.equal(await head.text(), '');
    assert.equal((await fetch(`${base}/ipfs/test-cid/no-such-file`)).status, 404);
    assert.equal((await fetch(`${base}/baseline/`)).status, 404);
    assert.equal((await fetch(`${base}/ipfs/test-cid/`, { method: 'POST' })).status, 405);
    assert.equal((await fetch(`${base}/%2e%2e%5cREADME.md`)).status, 400);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});
test('the build log reports repository-relative paths only', () => {
  const result = spawnSync(process.execPath, [path.join(root, 'scripts', 'build.mjs')], {
    cwd: root,
    encoding: 'utf8',
    env: process.env,
  });
  const output = `${result.stdout ?? ''}${result.stderr ?? ''}`;
  assert.equal(result.error, undefined, output);
  assert.equal(result.status, 0, output);
  const lines = output.split(/\r?\n/).filter(Boolean);
  const verified = lines.filter(line => line.startsWith('Verified Zola '));
  const ready = lines.filter(line => line.startsWith('Build ready: '));
  assert.equal(verified.length, 1, `expected one Zola verification line in:\n${output}`);
  assert.equal(ready.length, 1, `expected one build ready line in:\n${output}`);
  // Both lines must name a path relative to the project directory, POSIX-style.
  assert.match(verified[0], /^Verified Zola \d+\.\d+\.\d+: \.tools\/zola(\.exe)?$/, output);
  assert.equal(ready[0], 'Build ready: public', output);
  for (const line of [verified[0], ready[0]]) {
    assert.ok(!path.isAbsolute(line.slice(line.indexOf(': ') + 2)), `absolute path in log line: ${line}`);
  }
  // The published log must never reveal where the checkout lives on this machine.
  const secrets = new Set([root, path.resolve(root, '..'), os.homedir()]);
  for (const secret of secrets) {
    for (const form of [secret, secret.replaceAll('\\', '/')]) {
      assert.ok(!output.includes(form), `build log leaks the local checkout path (${form}):\n${output}`);
    }
  }
  assert.doesNotMatch(output, /(^|[\s"'(])(?:[A-Za-z]:[\\/]|\\\\)/, `build log contains an absolute Windows path:\n${output}`);
});
