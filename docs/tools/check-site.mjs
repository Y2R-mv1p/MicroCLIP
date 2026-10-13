import assert from 'node:assert/strict';
import {readFile, stat} from 'node:fs/promises';
import {dirname, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const html = await readFile(resolve(root, 'index.html'), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
assert.equal(new Set(ids).size, ids.length, 'Duplicate HTML IDs');
for (const m of html.matchAll(/\b(?:src|href|data-image|data-figure)="([^"]+)"/g)) {
  const url = m[1];
  if (url.startsWith('#')) {
    if (url.length > 1) assert(ids.includes(url.slice(1)), 'Missing anchor: ' + url);
    continue;
  }
  if (/^https?:/.test(url)) continue;
  assert(!url.startsWith('/'), 'Root-relative URL breaks project Pages: ' + url);
  const target = resolve(root, url.split('#')[0]);
  assert(target.startsWith(root + sep), 'Resource outside published docs: ' + url);
  await stat(target);
}
const citation = html.match(/<pre id="citation-text">([\s\S]*?)<\/pre>/);
assert(citation, 'Missing citation');
assert.equal(citation[1].trim(), (await readFile(resolve(root, 'resources/microclip.bib'), 'utf8')).trim(), 'Citation differs from download');
console.log('PASS: project Pages links, local resources, anchors, unique IDs, and citation.');
