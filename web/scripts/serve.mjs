import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { ROOT, isMain } from './setup.mjs';

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
  '.woff': 'font/woff', '.woff2': 'font/woff2', '.txt': 'text/plain; charset=utf-8',
};
function inside(root, file) {
  const relative = path.relative(root, file);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}
export function resolveRequest(requestUrl, root = ROOT) {
  const question = requestUrl.indexOf('?');
  const pathname = decodeURIComponent(question < 0 ? requestUrl : requestUrl.slice(0, question));
  if (!pathname.startsWith('/') || pathname.includes('\0') || pathname.includes('\\') || pathname.split('/').some(part => part === '..')) {
    throw new Error('Invalid request path');
  }
  let mount = '';
  let directory = 'public';
  if (pathname === '/ipfs/test-cid' || pathname.startsWith('/ipfs/test-cid/')) {
    mount = '/ipfs/test-cid';
  }
  const mountRoot = path.resolve(root, directory);
  const file = path.resolve(mountRoot, `.${pathname.slice(mount.length) || '/'}`);
  if (!inside(mountRoot, file)) throw new Error('Invalid request path');
  return { file, mountRoot, pathname, search: question < 0 ? '' : requestUrl.slice(question) };
}
export function createPreviewServer(root = ROOT) {
  return createServer(async (request, response) => {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, { Allow: 'GET, HEAD' }).end('Method not allowed'); return;
    }
    let target;
    try { target = resolveRequest(request.url ?? '/', root); }
    catch { response.writeHead(400).end('Bad request'); return; }
    try {
      let filename = target.file;
      let info = await stat(filename);
      if (info.isDirectory()) {
        if (!target.pathname.endsWith('/')) {
          response.writeHead(308, { Location: `${encodeURI(target.pathname)}/${target.search}` }).end(); return;
        }
        filename = path.join(filename, 'index.html');
        info = await stat(filename);
      }
      if (!info.isFile()) { response.writeHead(404).end('Not found'); return; }
      // Resolve symlinks as well as lexical traversal before reading.
      const [canonicalRoot, canonicalFile] = await Promise.all([realpath(target.mountRoot), realpath(filename)]);
      if (!inside(canonicalRoot, canonicalFile)) { response.writeHead(403).end('Forbidden'); return; }
      response.writeHead(200, {
        'Content-Type': TYPES[path.extname(filename).toLowerCase()] ?? 'application/octet-stream',
        'Content-Length': info.size,
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      });
      if (request.method === 'HEAD') { response.end(); return; }
      createReadStream(canonicalFile).on('error', () => response.destroy()).pipe(response);
    } catch (error) {
      const status = error.code === 'ENOENT' || error.code === 'ENOTDIR' ? 404 : 500;
      response.writeHead(status).end(status === 404 ? 'Not found' : 'Internal server error');
    }
  });
}
if (isMain(import.meta.url)) {
  const portFlag = process.argv.indexOf('--port');
  const value = portFlag >= 0 ? process.argv[portFlag + 1] : process.env.PORT ?? '4173';
  const port = Number(value);
  if (!value || !Number.isInteger(port) || port < 1 || port > 65535) {
    console.error('PORT / --port must be an integer from 1 to 65535'); process.exitCode = 1;
  } else {
    const server = createPreviewServer();
    server.on('error', error => { console.error(error.message); process.exitCode = 1; });
    server.listen(port, '127.0.0.1', () => console.log(`Preview: http://127.0.0.1:${port}/ (Ctrl+C to stop)`));
    for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close());
  }
}
