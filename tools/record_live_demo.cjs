#!/usr/bin/env node
// Optional M5 browser recording. Install Playwright outside this C++ project.
// See https://playwright.dev/docs/videos for recording lifecycle requirements.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');

const argv = process.argv.slice(2);
function option(name, fallback) {
  const i = argv.indexOf(name);
  return i < 0 ? fallback : argv[i + 1];
}
const url = option('--url', 'http://127.0.0.1:8770/');
const out = path.resolve(option('--out', 'docs/results/m5/video'));
const seconds = Number(option('--seconds', '20'));
const maxWait = Number(option('--max-wait', '360')) * 1000;
if (!(seconds >= 10 && seconds <= 120 && maxWait >= 1000)) throw new Error('invalid recording duration');
const manifest = option('--manifest', null);
const cases = manifest ? JSON.parse(fs.readFileSync(manifest, 'utf8')).cases.map(c =>
  [c.name, `${(c.freq/1e6).toFixed(3)} MHz`, `× ${c.rx.length}`]) : [
  ['ism24', '2440.000 MHz', '× 1'], ['wlan5', '5770.000 MHz', '× 1'],
  ['cell900', '939.000 MHz', '× 1'], ['adsb', '1100.000 MHz', '× 1'],
  ['dual_ism24', '2440.000 MHz', '× 2'],
];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const errors = [];
const report = { url, started: new Date().toISOString(), cases: [], errors };
let browser;

async function state(page) {
  return page.evaluate(() => ({
    source: document.querySelector('#src')?.textContent,
    freq: document.querySelector('#fc')?.textContent,
    rate: document.querySelector('#rate')?.textContent,
    time: document.querySelector('#t')?.textContent,
    horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1,
  }));
}

async function main() {
  fs.mkdirSync(out, { recursive: true });
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.browser = await browser.version();
  const observer = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  observer.on('pageerror', error => errors.push(String(error)));
  const start = Date.now();
  while (true) {
    try { await observer.goto(url+'?view=demo', { waitUntil: 'domcontentloaded', timeout: 5000 }); break; }
    catch (e) { if (Date.now()-start > maxWait) throw e; await sleep(1000); }
  }
  for (const [name, freq, rate] of cases) {
    const waiting = Date.now();
    while (true) {
      const s = await state(observer);
      if (s.source === 'SDR 实采' && s.freq === freq && s.rate?.endsWith(rate)) break;
      if (Date.now()-waiting > maxWait) throw new Error(`did not observe live case ${name}`);
      await sleep(1000);
    }
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 },
      recordVideo: { dir: path.join(out, 'recording'), size: { width: 1280, height: 900 } } });
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(url+'?view=demo');
    await page.waitForFunction(({freq, rate}) => document.querySelector('#src')?.textContent === 'SDR 实采' &&
      document.querySelector('#fc')?.textContent === freq && document.querySelector('#rate')?.textContent.endsWith(rate) &&
      parseFloat(document.querySelector('#t')?.textContent) >= 1, {freq, rate});
    const began = await state(page);
    console.log(`recording ${name}`);
    await sleep(seconds*500);
    await page.screenshot({ path: path.join(out, `${name}-demo.png`), fullPage: true });
    await page.locator('#mode').selectOption('engineering');
    const inspect = page.getByRole('button', { name: '检查', exact: false }).first();
    if (await inspect.count()) await inspect.click();
    await sleep(seconds*500-500);
    const fresh = page.locator('tr').filter({hasText:'0.0 s'}).getByRole('button',{name:'检查',exact:false}).first();
    if (await fresh.count()) await fresh.click();
    await sleep(500);
    await page.screenshot({ path: path.join(out, `${name}-engineering.png`), fullPage: true });
    const ended = await state(page);
    if (ended.source !== began.source || ended.freq !== began.freq || ended.rate !== began.rate)
      throw new Error(`receiver changed during ${name} recording`);
    const video = page.video();
    await context.close();
    await video.saveAs(path.join(out, `${name}.webm`));
    await video.delete();
    report.cases.push({ name, began, ended, file: `${name}.webm` });
    fs.writeFileSync(path.join(out, 'recording.json'), JSON.stringify(report, null, 2));
    if (name === 'ism24') {
      const narrow = await browser.newPage({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce' });
      await narrow.goto(url+'?view=demo');
      await narrow.waitForFunction(() => document.querySelector('#src')?.textContent === 'SDR 实采');
      await sleep(1000);
      report.narrow = await state(narrow);
      await narrow.screenshot({ path: path.join(out, 'narrow.png'), fullPage: true });
      await narrow.close();
    }
  }
  report.finished = new Date().toISOString();
  if (errors.length) throw new Error('browser errors; see recording.json');
}

main().catch(error => { report.failure = String(error); process.exitCode = 1; }).finally(async () => {
  fs.writeFileSync(path.join(out, 'recording.json'), JSON.stringify(report, null, 2));
  if (browser) await browser.close();
});
