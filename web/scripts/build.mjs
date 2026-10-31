import { copyFile, mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { ROOT, ensureZola, isMain, projectPath, run } from './setup.mjs';
// Node runs this TypeScript module directly (type stripping); `copy` is the single
// source of UI strings, so the templates never keep a second hand-written copy.
import { copy } from '../src/i18n.ts';

export async function buildSite() {
  const zola = await ensureZola();
  const output = path.join(ROOT, 'public');
  const generated = path.join(ROOT, '.generated');
  await rm(output, { recursive: true, force: true });
  await rm(generated, { recursive: true, force: true });
  await mkdir(generated, { recursive: true });
  try {
    // Zola reads this generated data file for the static, no-JavaScript English copy.
    const data = path.join(ROOT, 'data');
    await mkdir(data, { recursive: true });
    await writeFile(path.join(data, 'messages.json'), `${JSON.stringify({ en: copy.en }, null, 2)}\n`, 'utf8');
    const { build } = await import('esbuild');
    await build({
      absWorkingDir: ROOT,
      entryPoints: ['src/app.ts'],
      outfile: path.join(generated, 'app.js'),
      bundle: true,
      format: 'iife',
      platform: 'browser',
      target: 'es2022',
      charset: 'utf8',
      minify: false,
      sourcemap: false,
      logLevel: 'info',
    });
    // Zola owns and recreates public/. Never put the bundle there before Zola.
    run(zola, ['--config', 'config.toml', 'build', '--force']);
    await copyFile(path.join(generated, 'app.js'), path.join(output, 'app.js'));
    console.log(`Build ready: ${projectPath(output)}`);
  } catch (error) {
    // A failed build must not leave an apparently deployable partial site.
    await rm(output, { recursive: true, force: true });
    throw error;
  }
}
if (isMain(import.meta.url)) {
  buildSite().catch(error => { console.error(error.message); process.exitCode = 1; });
}
