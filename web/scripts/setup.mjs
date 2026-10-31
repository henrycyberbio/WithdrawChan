import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { chmod, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const ZOLA_VERSION = '0.23.6';
const ASSETS = {
  'win32-x64': ['x86_64-pc-windows-msvc', 'zip', '79179284f816febc281adbedb27189806cf22f83e8be4d0fca6835fbeeb09587'],
  'linux-x64': ['x86_64-unknown-linux-gnu', 'tar.gz', '8f5132b3522412d04e395e0b25f6d68613ad272a873e54a2b3ebf664873024a4'],
  'linux-arm64': ['aarch64-unknown-linux-gnu', 'tar.gz', '266448fffbf7c7004ca399d0e76dd699541771096d8a42aede98cebe2a029d02'],
  'darwin-arm64': ['aarch64-apple-darwin', 'tar.gz', 'cbffbd29b3f59c3f52633507c8cb945a7a02d8b1399b43b235f5932912297aa3'],
  'darwin-x64': ['x86_64-apple-darwin', 'tar.gz', '79a4d0ab51a4d863c068e6e594c6fce36f0aa17429a414ea63066f5910d14460'],
};
export function platformAsset(platform = process.platform, arch = process.arch) {
  const asset = ASSETS[`${platform}-${arch}`];
  if (!asset) throw new Error(`Unsupported Zola target: ${platform}-${arch}. Supported: ${Object.keys(ASSETS).join(', ')}`);
  const [triple, extension, sha256] = asset;
  return { triple, extension, sha256, filename: `zola-v${ZOLA_VERSION}-${triple}.${extension}` };
}
export function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: ROOT, stdio: 'inherit', ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (exit ${result.status ?? result.signal})`);
  return result;
}
export function isMain(metaUrl) {
  return Boolean(process.argv[1]) && path.resolve(process.argv[1]) === fileURLToPath(metaUrl);
}
// Logs are read and copied by users; they must never reveal where the checkout lives.
export function projectPath(absolute) {
  return path.relative(ROOT, absolute).split(path.sep).join('/') || '.';
}
export async function ensureZola() {
  const asset = platformAsset();
  const tools = path.join(ROOT, '.tools');
  await mkdir(tools, { recursive: true });
  const archive = path.join(tools, asset.filename);
  let data;
  try {
    data = await readFile(archive);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const url = `https://github.com/getzola/zola/releases/download/v${ZOLA_VERSION}/${asset.filename}`;
    console.log(`Downloading ${url}`);
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(30_000), redirect: 'follow' });
      if (!response.ok) throw new Error(`Zola download failed: HTTP ${response.status}`);
      data = Buffer.from(await response.arrayBuffer());
    } catch (error) {
      // curl uses the OS certificate store / proxy settings on Windows.
      console.warn(`Node download failed (${error.message}); trying system curl with TLS verification enabled.`);
      const downloadDirectory = await mkdtemp(path.join(tools, 'download-'));
      const partial = path.join(downloadDirectory, asset.filename);
      try {
        run(process.platform === 'win32' ? 'curl.exe' : 'curl', [
          '--fail', '--location', '--show-error', '--silent', '--retry', '2',
          '--connect-timeout', '30', '--max-time', '240',
          '--proto', '=https', '--proto-redir', '=https', '--output', partial, url,
        ]);
        data = await readFile(partial);
      } finally {
        await rm(downloadDirectory, { recursive: true, force: true });
      }
    }
  }
  const digest = createHash('sha256').update(data).digest('hex');
  if (digest !== asset.sha256) {
    throw new Error(`SHA-256 mismatch for ${asset.filename}: expected ${asset.sha256}, got ${digest}. Delete the cached archive in .tools and retry; do not bypass verification.`);
  }
  // Persist only a verified archive; extraction is repeated to avoid trusting a changed executable.
  await writeFile(archive, data);
  const staging = await mkdtemp(path.join(tools, 'extract-'));
  const executable = process.platform === 'win32' ? 'zola.exe' : 'zola';
  const destination = path.join(tools, executable);
  try {
    if (process.platform === 'win32') {
      run('powershell.exe', ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', 'Expand-Archive -LiteralPath $env:ZOLA_ARCHIVE -DestinationPath $env:ZOLA_STAGE -Force'], {
        env: { ...process.env, ZOLA_ARCHIVE: archive, ZOLA_STAGE: staging },
      });
    } else {
      run('tar', ['-xzf', archive, '-C', staging]);
    }
    const extracted = path.join(staging, executable);
    await chmod(extracted, 0o755);
    run(extracted, ['--version']);
    await rm(destination, { force: true });
    await rename(extracted, destination);
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
  console.log(`Verified Zola ${ZOLA_VERSION}: ${projectPath(destination)}`);
  return destination;
}
if (isMain(import.meta.url)) {
  ensureZola().catch(error => { console.error(error.message); process.exitCode = 1; });
}
