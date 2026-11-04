import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { assetLabel, copy, messages } from '../src/i18n.ts';
import { assets } from '../src/algorithm.ts';

const root = fileURLToPath(new URL('../', import.meta.url));
const read = file => readFileSync(path.join(root, file), 'utf8');
const templateFiles = (dir = path.join(root, 'templates')) =>
  readdirSync(dir, { withFileTypes: true }).flatMap(entry =>
    entry.isDirectory() ? templateFiles(path.join(dir, entry.name))
      : entry.name.endsWith('.html') ? [path.join(dir, entry.name)] : []);

test('both languages expose the same static message keys with non-empty strings', () => {
  const en = Object.keys(copy.en).sort();
  const zh = Object.keys(copy.zh).sort();
  assert.deepEqual(zh, en);
  for (const language of ['en', 'zh']) {
    for (const [key, value] of Object.entries(copy[language])) {
      assert.equal(typeof value, 'string', `${language}.${key} must be a string`);
      assert.notEqual(value.trim(), '', `${language}.${key} must not be empty`);
    }
  }
});

test('every message key used by a Zola template exists in the single i18n source', () => {
  const used = new Set();
  for (const file of templateFiles()) {
    for (const match of readFileSync(file, 'utf8').matchAll(/data-i18n="([A-Za-z0-9]+)"/g)) used.add(match[1]);
  }
  // The FAQ answers are rich bilingual HTML inside the template, so they no longer
  // contribute data-i18n keys; only the page chrome does (19 keys today). The guard
  // stays just under that so a template losing its keys still fails.
  assert.ok(used.size > 18, `expected the templates to declare their copy keys, saw ${used.size}`);
  for (const key of used) {
    assert.ok(Object.hasOwn(copy.en, key), `template key missing from i18n: ${key}`);
    assert.ok(Object.hasOwn(copy.zh, key), `template key missing from zh i18n: ${key}`);
  }
  for (const language of ['en', 'zh']) {
    for (const [key, value] of Object.entries(messages[language])) {
      if (typeof value === 'string') assert.ok(value.length > 0, `${language}.${key}`);
    }
  }
});

test('templates render static English from the build-time i18n data, not retyped literals', () => {
  const generated = path.join(root, 'data/messages.json');
  assert.ok(existsSync(generated), 'run npm run build first: data/messages.json is generated from src/i18n.ts');
  const data = JSON.parse(readFileSync(generated, 'utf8'));
  assert.deepEqual(Object.keys(data), ['en'], 'the template data file must only ship the static English fallback');
  assert.deepEqual(data.en, { ...copy.en });
  for (const file of templateFiles()) {
    const source = readFileSync(file, 'utf8');
    if (source.includes('data-i18n=')) {
      assert.match(source, /messages\.en\./, `${path.relative(root, file)} must render copy from data/messages.json`);
    }
  }
});

test('each theme carries a short non-empty hint under the result', () => {
  // Language switching keeps the same drawn amount; the browser suite checks that
  // against the live control. Here only the two themes must differ.
  for (const language of ['en', 'zh']) {
    for (const key of ['hintPlayful', 'hintZen']) {
      assert.equal(typeof copy[language][key], 'string');
      assert.notEqual(copy[language][key].trim(), '', `${language}.${key} must carry hint text`);
    }
    assert.notEqual(copy[language].hintPlayful, copy[language].hintZen, `${language} hint must change with the theme`);
  }
});

test('the asset display label drops the stablecoin qualifier without renaming the engine key', () => {
  // Both stablecoin pools share one engine asset class: users see the unit they
  // recognise and type, while units.json and the adapter keep `USD-stable`.
  assert.equal(assetLabel('USD-stable'), 'USD');
  assert.equal(assetLabel('ETH'), 'ETH');
  assert.ok(Object.hasOwn(assets, 'USD-stable'));
  assert.ok(Object.hasOwn(assets, 'ETH'));
});

test('every user-visible unit reads USD while only the select option keeps the engine key', () => {
  for (const language of ['en', 'zh']) {
    for (const [key, value] of Object.entries(copy[language])) {
      assert.doesNotMatch(value, /USD-stable/, `${language}.${key} must show USD, not the engine key`);
    }
  }
  assert.equal(copy.en.limit, 'At least 1 USD');
  assert.equal(copy.zh.limit, '至少 1 USD');
  assert.equal(messages.en.limitLine({ minimum: '1', unit: assetLabel('USD-stable') }), 'At least 1 USD');
  assert.equal(messages.en.limitLine({ minimum: '0.001', unit: assetLabel('ETH') }), 'At least 0.001 ETH');
  assert.equal(messages.zh.limitLine({ minimum: '1', unit: assetLabel('USD-stable') }), '至少 1 USD');
  assert.equal(messages.zh.limitLine({ minimum: '0.001', unit: assetLabel('ETH') }), '至少 0.001 ETH');

  const option = '<option value="USD-stable" selected>USD-stable</option>';
  const tool = read('templates/partials/tool.html');
  assert.ok(tool.includes(option), 'the select keeps the engine key as its value and option text');
  assert.doesNotMatch(tool.replace(option, ''), /USD-stable/, 'no other markup may print the engine key');
});

test('every Chinese message is Chinese, except keys that are intentionally not', () => {
  // eyebrow is a fixed English kicker by design. languageSwitch/languageButtonLabel
  // name the other language in its own script. locale/htmlLang hold BCP 47 tags.
  // privacyBaseline/Drain/Keep are the chart's English method labels in both languages.
  const untranslated = new Set([
    'eyebrow', 'languageSwitch', 'languageButtonLabel', 'locale', 'htmlLang',
    'privacyBaseline', 'privacyDrain', 'privacyPartial',
  ]);
  for (const key of untranslated) assert.ok(Object.hasOwn(copy.zh, key), `allowlisted key ${key} no longer exists`);
  for (const [key, value] of Object.entries(copy.zh)) {
    if (untranslated.has(key)) continue;
    assert.match(value, /[\u4e00-\u9fff]/, `zh.${key} must be Chinese, not copied English`);
  }
});

test('chart copy states the simulated experiment\'s 50% adoption rate in both languages', () => {
  for (const language of ['en', 'zh']) {
    assert.match(copy[language].privacyNote, /50\s*%/, `${language}.privacyNote states the adoption rate`);
  }
});

test('the FAQ ships every answer in both languages as static HTML, English visible by default', () => {
  const faq = read('templates/partials/faq.html');
  const items = faq.split('<details class="faq-item">').slice(1);
  assert.equal(items.length, 6, 'six questions');
  for (const [index, item] of items.entries()) {
    const blocks = [...item.matchAll(/<(span|div) data-lang="(en|zh)" lang="([^"]+)"( hidden)?>/g)].map(match => match.slice(1));
    // Question span, then answer block, for each language.
    assert.deepEqual(blocks.map(([tag, lang]) => `${tag}:${lang}`), ['span:en', 'span:zh', 'div:en', 'div:zh'], `item ${index + 1} pairs both languages`);
    for (const [, lang, tag, hidden] of blocks) {
      assert.equal(tag, lang === 'zh' ? 'zh-CN' : 'en', `item ${index + 1}: ${lang} block declares its lang`);
      assert.equal(Boolean(hidden), lang === 'zh', `item ${index + 1}: only the zh block starts hidden`);
    }
    const zh = item.slice(item.indexOf('<div data-lang="zh"'));
    assert.match(zh, /[\u4e00-\u9fff]/, `item ${index + 1}: the zh answer is Chinese`);
  }
  // Both languages keep the same rich structure.
  const count = (lang, pattern) => [...faq.matchAll(new RegExp(`<div data-lang="${lang}"[^>]*>(.*?)</div>`, 'gs'))]
    .map(match => (match[1].match(pattern) ?? []).length).reduce((a, b) => a + b, 0);
  for (const pattern of [/<table class="faq-table">/g, /<ul>/g, /<blockquote>/g, /<a href=/g, /<code>/g]) {
    assert.equal(count('zh', pattern), count('en', pattern), `zh and en carry the same number of ${pattern.source}`);
  }
  // The zh sentence inlines its second formula, so only the presence of a display
  // block and the overall MathML coverage are compared, not the display count.
  for (const lang of ['en', 'zh']) {
    assert.ok(count(lang, /<math display="block">/g) >= 1, `${lang} carries at least one display formula`);
    assert.ok(count(lang, /<math[ >]/g) >= 4, `${lang} marks up its formulas as MathML`);
  }
});

test('the FAQ has no stray Markdown, LaTeX, private notes or unsafe links', () => {
  const faq = read('templates/partials/faq.html');
  const text = faq.replace(/\{#[\s\S]*?#\}/, '');
  assert.doesNotMatch(text, /\$[^$]+[^$]*\$/, 'no unrendered LaTeX delimiters');
  // Formulas are MathML now, so `log`/`max` appear as real text; only a literal
  // backslash command signals leftover LaTeX.
  assert.doesNotMatch(text, /\\[a-zA-Z]+/, 'no unrendered LaTeX commands');
  assert.doesNotMatch(text, /\*\*|\]\(/, 'no unrendered Markdown emphasis or links');
  assert.doesNotMatch(text, /\((?:H|h|p_i|i)\)/, 'no formula symbols left as plain parentheses');
  // The FAQ links into the public repo, and those file names end in `.md`; scan the
  // prose without those URLs, then prove each linked path really exists in this repo.
  const repoLinks = [...text.matchAll(/https:\/\/github\.com\/henrycyberbio\/WithdrawChan\/blob\/master\/([\w./-]+)/g)];
  const prose = text.replace(/https:\/\/github\.com\/henrycyberbio\/WithdrawChan\/blob\/master\/[\w./-]+/g, '');
  assert.doesNotMatch(prose, /chatgpt\.com|Projects\/WithdrawChan|\.md\b|obsidian/i, 'no chat sessions or vault paths');
  for (const [, file] of repoLinks) {
    assert.ok(existsSync(path.join(root, '..', file)), `FAQ links to a repo path that does not exist: ${file}`);
  }
  assert.doesNotMatch(text, /均抛硬币|Privacy Pool v1/, 'fixed typos stay fixed');
  for (const [anchor] of text.matchAll(/<a\b[^>]*>/g)) {
    assert.match(anchor, /href="https:\/\//, `${anchor} is an absolute https link`);
    assert.match(anchor, /target="_blank"/, `${anchor} opens in a new tab`);
    assert.match(anchor, /rel="noopener noreferrer"/, `${anchor} sends no opener or referrer`);
  }
  assert.doesNotMatch(text, /data-i18n="faq(?!Title)/, 'answers are no longer plain-text i18n keys');
  for (const language of ['en', 'zh']) {
    assert.deepEqual(Object.keys(copy[language]).filter(key => key.startsWith('faq')), ['faqTitle'], `${language} keeps only the FAQ heading key`);
  }
});
