import type { Asset } from './algorithm.ts';

/**
 * Single source of UI copy.
 *
 * `copy` holds the static strings: Zola renders the English set at build time from
 * `data/messages.json`, which `scripts/build.mjs` generates from this file, and app.ts
 * reads the same object at runtime. `messages` adds the parameterised sentences so no
 * module ever retypes a translation for one language.
 *
 * The FAQ answers are the one exception: they are rich bilingual prose (lists, a
 * table, MathML formulas, links) written as static HTML in
 * `templates/partials/faq.html`, and app.ts only shows the `data-lang` block that
 * matches the current language.
 */
export type Language = 'zh' | 'en';
export type Theme = 'playful' | 'zen';

/**
 * Visible unit label; the engine keys stay `USD-stable` and `ETH`.
 */
const assetDisplay: Record<Asset, string> = { 'USD-stable': 'USD', ETH: 'ETH' };

export function assetLabel(asset: Asset): string {
  return assetDisplay[asset];
}

const en = {
  tabTitle: 'WithdrawChan — Withdrawal Amount Suggestions',
  metaDescription: 'Enter the amount you need now and an optional maximum to get a local withdrawal amount suggestion. Two styles, the same result.',
  // ASCII apostrophe kept as the user wrote it; double quotes avoid an escape.
  title: "WithdrawChan: Withdraw like nobody's watching",
  eyebrow: 'WITHDRAWAL HYGIENE · BETTER HABITS',
  needed: 'Amount you need now',
  maximum: 'Maximum amount (optional)',
  limit: 'At least 1 USD',
  calculate: 'Find my amount',
  playful: 'This looks good today ♡',
  zen: 'Suggested withdrawal:',
  hintPlayful: 'Taking it all is too obvious. Leave a little behind.',
  hintZen: 'Blend in. Leave less behind.',
  order: 'Maximum must be at least the amount you need.',
  stale: 'Range changed. Calculate again.',
  copyAmountLabel: 'Copy suggested amount',
  copyAmountCopied: 'Amount copied',
  copyAmountFailed: 'Copy failed. Select the number to copy it manually.',
  copyStatusCopied: 'Suggested amount copied.',
  copyStatusFailed: 'Copy failed. Select the number to copy it manually.',
  languageSwitch: '切换至中文',
  languageButtonLabel: '中文',
  themeSwitchZen: 'Switch to Zen style',
  themeSwitchPlayful: 'Switch to Playful style',
  homeLabel: 'WithdrawChan home',
  socialLabel: 'WithdrawChan social links',
  preferencesLabel: 'Preferences',
  locale: 'en-US',
  htmlLang: 'en',
  faqTitle: 'About WithdrawChan',
  privacyHeading: 'Three ways to withdraw',
  privacyIndex: 'Conditional min-entropy upper bound',
  privacyBaseline: 'Non-WithdrawChan',
  privacyDrain: 'Drain',
  privacyPartial: 'Keep',
  // Newlines are real: the figcaption renders them with `white-space: pre-line`, so
  // each footnote starts its own line in both the static and the runtime copy.
  // The caption states the experiment only; the default-maximum rule lives in the
  // maximum field's hint (`defaultMaximumTitle`).
  privacyNote: 'Simulation of WithdrawChan use: 50% adoption.\n* USDT/USDC: highest hit rate among four observers.\n** ETH: highest hit rate among three public observers.',
  scrollCompare: 'Explore three ways to withdraw',
  poolLabel: 'Select asset',
  chartPool: 'Pool',
  chartTableCaption: 'Simulated conditional min-entropy upper bound in bits by pool and withdrawal method',
} as const;

export type StaticKey = keyof typeof en;
export type MessageKey = StaticKey;
export type StaticCopy = { readonly [K in StaticKey]: string };

const zh: StaticCopy = {
  tabTitle: 'WithdrawChan — 提现金额推荐',
  metaDescription: '输入当前所需金额与可选的最大金额，在浏览器本地获得提现金额建议。两种主题，同一结果。',
  title: 'WithdrawChan：提在人声鼎沸时',
  eyebrow: 'WITHDRAWAL HYGIENE · BETTER HABITS',
  needed: '当前所需金额',
  maximum: '可选最大金额',
  limit: '至少 1 USD',
  calculate: '推荐一个金额',
  playful: '♡ 今天建议你提',
  zen: '建议提现金额：',
  hintPlayful: '全部提走太显眼啦，留下一点比较好',
  hintZen: '融入人群，少留痕迹。',
  order: '最大金额不能小于当前所需金额。',
  stale: '区间已更改，请重新计算。',
  copyAmountLabel: '复制推荐金额',
  copyAmountCopied: '已复制金额',
  copyAmountFailed: '复制失败，请选择数字手动复制。',
  copyStatusCopied: '推荐金额已复制。',
  copyStatusFailed: '复制失败，请选择数字手动复制。',
  languageSwitch: 'Switch to English',
  languageButtonLabel: 'EN',
  themeSwitchZen: '切换至禅模式',
  themeSwitchPlayful: '切换至酱模式',
  homeLabel: 'WithdrawChan 首页',
  socialLabel: 'WithdrawChan 社交链接',
  preferencesLabel: '偏好设置',
  locale: 'zh-CN',
  htmlLang: 'zh-CN',
  faqTitle: '关于 WithdrawChan',
  privacyHeading: '三种提现方式',
  privacyIndex: '条件最小熵上界',
  privacyBaseline: 'Non-WithdrawChan',
  privacyDrain: 'Drain',
  privacyPartial: 'Keep',
  // Same single string as `en`: `\n` plus `white-space: pre-line` gives each footnote
  // its own line without innerHTML or a translation branch.
  privacyNote: 'WithdrawChan 使用模拟实验：采用率 50%。\n* USDT/USDC 取四观察者最高命中率。\n** ETH 取三公开观察者最高命中率。',
  scrollCompare: '向下探索三种提现方式',
  poolLabel: '选择资产',
  chartPool: '池子',
  chartTableCaption: '各池与提现方式的模拟条件最小熵上界（bits）',
};

export interface DynamicCopy {
  limitLine(parts: { minimum: string; unit: string }): string;
  defaultMaximumTitle(parts: { unit: string }): string;
  resubmitTitle(): string;
  resultRange(parts: { minimum: string; maximum: string; unit: string }): string;
  maximumUnitError(parts: { unit: string }): string;
  invalidAmountError(parts: { minimum: string; unit: string; decimals: number }): string;
}

const dynamic: Record<Language, DynamicCopy> = {
  en: {
    limitLine: ({ minimum, unit }) => `At least ${minimum} ${unit}`,
    defaultMaximumTitle: ({ unit }) => `Leave blank to round twice the amount you need up to a multiple of ${unit}.`,
    resubmitTitle: () => 'Change an amount to enable this button again.',
    resultRange: ({ minimum, maximum, unit }) => `Within your ${minimum}–${maximum} ${unit} range`,
    maximumUnitError: ({ unit }) => `Maximum must be a multiple of ${unit}; your entered cap is not rounded automatically.`,
    invalidAmountError: ({ minimum, unit, decimals }) => `Enter valid amounts of at least ${minimum} ${unit}, with up to ${decimals} decimals and no exponent notation.`,
  },
  zh: {
    limitLine: ({ minimum, unit }) => `至少 ${minimum} ${unit}`,
    defaultMaximumTitle: ({ unit }) => `留空时将当前所需金额的两倍向上取整至 ${unit} 的倍数。`,
    resubmitTitle: () => '修改金额后可重新推荐。',
    resultRange: ({ minimum, maximum, unit }) => `位于 ${minimum}–${maximum} ${unit} 区间内`,
    maximumUnitError: ({ unit }) => `最大金额须为 ${unit} 的倍数；不会自动修改你填写的上限。`,
    invalidAmountError: ({ minimum, unit, decimals }) => `请输入不小于 ${minimum} ${unit} 的有效金额，最多 ${decimals} 位小数，不使用科学计数法。`,
  },
};

export type Messages = StaticCopy & DynamicCopy;

/** Language toggle lives with the copy so no module retypes the language names. */
export function nextLanguage(language: Language): Language {
  return language === 'zh' ? 'en' : 'zh';
}

/** Static strings, also used to generate the Zola data file. */
export const copy: Record<Language, StaticCopy> = { zh, en };

/** Everything app.ts needs, per language. */
export const messages: Record<Language, Messages> = {
  zh: { ...zh, ...dynamic.zh },
  en: { ...en, ...dynamic.en },
};
