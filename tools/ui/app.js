"use strict";
// Distinct hues (Tableau-like) for the signal classes; index 0 is "noise", 10 is "wide" (slice of a signal wider
// than one channel).
const PALETTE = ["#5c6672", "#4e9de6", "#f28e2b", "#59d0c4", "#edc948", "#e15759", "#b07aa1", "#8cd17d", "#ff9da7", "#d9d9d9",
                 "#9c755f", "#70b3bb", "#909ba8"];
// A narrowband class fits in one 120 kHz channel (two at a boundary, three for broadcast FM); a run of this many
// adjacent tracked channels is one wideband signal whatever its per-channel labels say.
const WIDE_RUN = 4;
const ROWS = 250, CROWS = 80;     // waterfall / class-waterfall history in rows
const SPAN_DB = 50;
// Track hysteresis on the presence EMA (fraction of recent rows with a detection)
const ON = 0.35, OFF = 0.12;

let classes = [], channelClasses = [], WIDE = -1, meta = null, panels = [], counts = [], hidden = new Set();
let policyStats = null, dirty = false, es = null;
let lastEnd = null;
let lastDataAt = performance.now(), feedFailed = false;
const plotObserver = new ResizeObserver(() => { if (meta) { redrawAll(); airDirty = true; } });
let rowDt = 0.034, rateT0 = performance.now(), rateRows = 0, lastList = 0;
let view = [0, 512];               // displayed column range (fft-shifted columns)

const LUT = (() => {
  const stops = [[0, 6, 8, 14], [0.25, 20, 40, 110], [0.5, 30, 140, 150], [0.75, 235, 200, 70], [1, 255, 255, 240]];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255; let k = 0;
    while (k < stops.length - 2 && x > stops[k + 1][0]) k++;
    const [x0, r0, g0, b0] = stops[k], [x1, r1, g1, b1] = stops[k + 1], u = (x - x0) / (x1 - x0);
    lut[3 * i] = r0 + (r1 - r0) * u; lut[3 * i + 1] = g0 + (g1 - g0) * u; lut[3 * i + 2] = b0 + (b1 - b0) * u;
  }
  return lut;
})();
const RGB = PALETTE.map(c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)));
const color = c => PALETTE[c % PALETTE.length];
const name = c => classes[c] || "c" + c;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
const engineering = () => document.getElementById("mode").value === "engineering";
const title = g => meta.stage2 ? g.title : name(g.cls);
const reasons = {allowed_mass:"允许类别的原始概率总量不足", raw_confidence:"原始模型置信度不足", low_confidence:"先验内置信度不足",
  small_margin:"前两名差距不足", confirming:"等待连续一致判断", confirmed:"连续判断已确认", hysteresis:"门限滞回保持",
  missing_probabilities:"缺少完整有效概率，保持未识别", calibrated_spur:"匹配同一接收设置的杂散校准表", decoder_only:"此频段只显示 CRC 通过的 ADS-B 解码", invalid_record:"信号记录无效"};
const col = (k, M) => (k + M / 2) % M;                       // channel -> display column (left edge = -fs/2)
const colHz = x => (x - meta.channels / 2) * meta.rate / meta.channels; // display column -> offset Hz
const fmtMHz = x => ((meta.freq + colHz(x)) / 1e6).toFixed(2);

function newTrack(M, C) {
  return { act: new Float32Array(M), on: new Uint8Array(M), since: new Float64Array(M),
           score: new Float32Array(M * C), pw: new Float32Array(M).fill(-200), cls: new Int16Array(M).fill(-1),
           conf: new Float32Array(M) };
}

function build() {
  const root = document.getElementById("streams");
  plotObserver.disconnect();
  root.innerHTML = "";
  panels = [];
  const M = meta.channels;
  view = [0, M];
  rowDt = meta.every * meta.window * M / (2 * meta.rate); // 2x-oversampled bank: channel rate 2 fs / M
  for (let s = 0; s < meta.streams; s++) {
    const el = document.createElement("section");
    el.className = "stream";
    el.innerHTML = `<div class="plots"><h2><span>RX${(meta.rx || [])[s] || s + 1}</span><span class="info"></span></h2>
        <canvas class="axis" aria-label="频率刻度"></canvas><canvas class="labels" aria-label="信号频带标注"></canvas><canvas class="wf" role="img" aria-label="RX${s+1} 功率瀑布图，可用工具栏按钮或信号频率按钮放大"></canvas><canvas class="cls" role="img" aria-label="RX${s+1} 信号类别随时间的变化"></canvas>
        <div class="sel"></div></div>
      <div class="list"><h3><span>信号列表</span><span class="n"></span></h3><div class="rows"><p class="empty">等待信号级结果</p></div></div>
      <div class="engineering engine-grid"><details><summary>信号判决与原始模型概率</summary><pre class="inspection">点击信号的“检查”按钮查看原始概率、先验和判决原因。</pre></details>
      <details class="channel-detail"><summary>逐信道检测 / 分类</summary><div class="rows channel-rows"><p class="empty">等待逐信道数据</p></div></details></div>`;
    root.appendChild(el);
    const [axis, labels, wf, cls] = el.querySelectorAll("canvas");
    const off = document.createElement("canvas"); off.width = M; off.height = ROWS;
    const coff = document.createElement("canvas"); coff.width = M; coff.height = CROWS;
    const p = { s, axis, labels, wf, cls, off, coff, info: el.querySelector(".info"), rows: el.querySelector(".rows"),
                n: el.querySelector(".n"), sel: el.querySelector(".sel"), plots: el.querySelector(".plots"),
                octx: off.getContext("2d"), cctx: coff.getContext("2d"), row: new ImageData(M, 1), crow: new ImageData(M, 1),
                trk: newTrack(M, Math.max(channelClasses.length, 16)), floor: null, last: null, groups: [], signals: new Map(),
                inspection: el.querySelector(".inspection"), channels: el.querySelector(".channel-rows"),
                channelDetail: el.querySelector(".channel-detail"), selected: null, now: 0 };
    p.octx.fillStyle = "#06080e"; p.octx.fillRect(0, 0, M, ROWS);
    p.cctx.fillStyle = "#0b0e13"; p.cctx.fillRect(0, 0, M, CROWS);
    for (const c of [wf, cls]) hookMouse(p, c);
    p.channelDetail.addEventListener("toggle", () => drawChannels(p));
    panels.push(p);
    plotObserver.observe(p.plots);
  }
  redrawAll();
}

function fit(cv) {
  const r = devicePixelRatio || 1, w = Math.round(cv.clientWidth * r), h = Math.round(cv.clientHeight * r);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(r, 0, 0, r, 0, 0);
  return [ctx, cv.clientWidth, cv.clientHeight];
}
const xOf = (x, w) => (x - view[0]) / (view[1] - view[0]) * w;       // column -> CSS px
const colAt = (px, w) => Math.floor(view[0] + px / w * (view[1] - view[0]));

function drawAxis(p) {
  const [ctx, w, h] = fit(p.axis);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#7d8896"; ctx.strokeStyle = "#39424e"; ctx.font = "11px ui-monospace, Menlo, monospace"; ctx.textBaseline = "top";
  const ticks = Math.max(2, Math.min(8, Math.floor(w / 90)));
  for (let i = 0; i <= ticks; i++) {
    const x = view[0] + (view[1] - view[0]) * i / ticks, px = i / ticks * w;
    ctx.beginPath(); ctx.moveTo(px + 0.5, h - 4); ctx.lineTo(px + 0.5, h); ctx.stroke();
    ctx.textAlign = i === 0 ? "left" : i === ticks ? "right" : "center";
    ctx.fillText(meta.freq ? fmtMHz(x) : (colHz(x) >= 0 ? "+" : "") + (colHz(x) / 1e6).toFixed(2), px, 1);
  }
}

function blit(p) {
  for (const [cv, src, rows] of [[p.wf, p.off, ROWS], [p.cls, p.coff, CROWS]]) {
    const [ctx, w, h] = fit(cv);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(src, view[0], 0, view[1] - view[0], rows, 0, 0, w, h);
  }
}

function redrawAll() { for (const p of panels) { drawAxis(p); blit(p); drawLabels(p); } }

// One feed row: update tracks, push one waterfall row, redraw.
function onRow(row) {
  if (!Array.isArray(row.p) || !Array.isArray(row.f) || !Array.isArray(row.r)) return;
  const M = meta.channels, C = Math.max(channelClasses.length, 16);
  const tau = parseFloat(document.getElementById("tau").value);
  const d = Math.exp(-rowDt / tau);
  const det = panels.map(() => ({ cls: new Int16Array(M).fill(-1), prob: new Uint8Array(M) }));
  // Detected (energy above the floor) but noise-like: a slice of a wideband noise-like signal (OFDM, fast PSK/QAM),
  // which the model cannot tell from noise inside one channel.
  for (const [s, k, c, pr] of row.r) { if (!det[s] || k < 0 || k >= M || c < 0 || c >= C) continue; const x = col(k, M); det[s].cls[x] = c === 0 && WIDE > 0 ? WIDE : c; det[s].prob[x] = pr; }
  const now = row.t;
  counts = new Array(C).fill(0);
  for (const p of panels) {
    const s = p.s, T = p.trk, D = det[s];
    const f = row.f[s];
    p.floor = p.floor === null ? f : p.floor + 0.05 * (f - p.floor);
    const lo = p.floor - 6, scale = 255 / (SPAN_DB + 6), pd = p.row.data, cd = p.crow.data;
    for (let k = 0; k < M; k++) {
      const x = col(k, M), pw = row.p[s * M + k];
      const v = Math.max(0, Math.min(255, (pw - lo) * scale)) | 0;
      pd[4 * x] = LUT[3 * v]; pd[4 * x + 1] = LUT[3 * v + 1]; pd[4 * x + 2] = LUT[3 * v + 2]; pd[4 * x + 3] = 255;
      T.pw[x] = pw;
    }
    for (let x = 0; x < M; x++) {
      const c = D.cls[x], hit = c >= 0;
      T.act[x] = d * T.act[x] + (hit ? 1 - d : 0);
      for (let j = 0; j < C; j++) T.score[x * C + j] *= d;
      if (hit) T.score[x * C + c] += D.prob[x] / 100;
      if (!T.on[x] && T.act[x] > ON) { T.on[x] = 1; T.since[x] = now; }
      else if (T.on[x] && T.act[x] < OFF) T.on[x] = 0;
      let best = -1, bs = 0, tot = 0;
      for (let j = 0; j < C; j++) { const v = T.score[x * C + j]; tot += v; if (v > bs) { bs = v; best = j; } }
      T.cls[x] = best; T.conf[x] = tot > 0 ? bs / tot : 0;
      // class waterfall: tracked channels in their (smoothed) class colour, raw detections faint
      if (T.on[x] && best > 0 && !hidden.has(best)) {
        const [r, g, b] = RGB[best % RGB.length], a = 0.35 + 0.65 * T.conf[x];
        cd[4 * x] = r * a; cd[4 * x + 1] = g * a; cd[4 * x + 2] = b * a;
      } else if (hit) { cd[4 * x] = 40; cd[4 * x + 1] = 46; cd[4 * x + 2] = 54; }
      else { cd[4 * x] = 11; cd[4 * x + 1] = 14; cd[4 * x + 2] = 19; }
      cd[4 * x + 3] = 255;
    }
    p.now = Math.max(p.now, now);
    p.last = row;
    p.groups = meta.stage2 ? signalGroups(p) : group(p, now);
    if (meta.stage2) {
      for (let x = 0; x < M; ++x) { cd[4*x] = 11; cd[4*x+1] = 14; cd[4*x+2] = 19; }
      for (const g of visible(p)) {
        const rgb = RGB[g.cls % RGB.length];
        for (let x = Math.max(0, Math.floor(g.x0)); x < Math.min(M, Math.ceil(g.x1)); ++x)
          for (let i = 0; i < 3; ++i) cd[4*x+i] = rgb[i] * (g.bursty ? .65 : 1);
      }
    }
    p.octx.drawImage(p.off, 0, 0, M, ROWS - 1, 0, 1, M, ROWS - 1); p.octx.putImageData(p.row, 0, 0);
    p.cctx.drawImage(p.coff, 0, 0, M, CROWS - 1, 0, 1, M, CROWS - 1); p.cctx.putImageData(p.crow, 0, 0);
    for (const g of p.groups) if (!(document.getElementById("steady").checked && g.bursty)) counts[g.cls]++;
    blit(p); drawLabels(p);
    p.info.textContent = `本底 ${f.toFixed(1)} dBFS · ${meta.stage2 ? "信号级" : "逐信道工程数据"} · ${visible(p).length} 个${p.hint || ""}`;
  }
  document.getElementById("t").textContent = row.t.toFixed(2) + " s";
  rateRows++;
  const t = performance.now();
  if (t - rateT0 > 1000) { document.getElementById("rps").textContent = (rateRows * 1000 / (t - rateT0)).toFixed(1); rateT0 = t; rateRows = 0; }
  if (t - lastList > 250) { lastList = t; refresh(); } // lists at 4 Hz
}

function onSignal(message) {
  const d = message.display;
  if (!d || d.uid === undefined || !panels[d.s]) return;
  // Old recorded stage-2 feeds did not identify their schema in metadata.
  if (!meta.stage2) { meta.stage2 = true; classes = meta.signal_classes || []; hidden.clear(); document.querySelector('#mode option[value="demo"]').disabled = false; updatePolicy(); }
  const p = panels[d.s];
  p.now = Math.max(p.now, d.t);
  p.signals.set(String(d.uid), {d, raw: message.sig || null});
  if (p.signals.size > 512) p.signals.delete(p.signals.keys().next().value);
  dirty = true;
}

function signalGroups(p) {
  const out = [], M = meta.channels;
  for (const [key, entry] of p.signals) {
    const {d, raw} = entry;
    if (p.now - d.t > d.ttl_s) { p.signals.delete(key); continue; }
    if (!engineering() && !d.visible) continue;
    const xc = M/2 + d.f * M/meta.rate, width = d.bw*M/meta.rate;
    const c = engineering() && raw ? raw.c : classes.indexOf(d.family);
    const cls = c >= 0 ? c : classes.indexOf("unknown");
    out.push({x0:xc-width/2, x1:xc+width/2, xc, cls:Math.max(0, cls), conf:d.confidence ?? 0,
      title: engineering() && raw && classes[raw.c] !== d.family ? (classes[raw.c] || "未识别").toUpperCase() + " → " + d.label : d.label,
      age:d.age_s || 0, bursty:d.activity_fraction !== null && d.activity_fraction < .65, d, raw});
  }
  return out.sort((a,b) => a.xc-b.xc || String(a.d.uid).localeCompare(String(b.d.uid)));
}

function refresh() {
  if (!meta) return;
  counts = new Array(Math.max(classes.length, 16)).fill(0);
  for (const p of panels) {
    if (meta.stage2) p.groups = signalGroups(p);
    for (const g of visible(p)) counts[g.cls]++;
    drawList(p); drawLabels(p); drawChannels(p);
    if (p.selected !== null) inspect(p, p.selected, false);
    p.info.textContent = `本底 ${p.floor === null ? '—' : p.floor.toFixed(1)} dBFS · ${meta.stage2 ? '信号级' : '逐信道工程数据'} · ${visible(p).length} 个${p.hint || ''}`;
  }
  drawLegend(); dirty = false;
}

function updatePolicy(stats) {
  if (stats) policyStats = stats;
  const p = meta?.policy;
  const spur = {not_loaded:"杂散表未加载", active:"已使用匹配接收设置的杂散表", context_mismatch:"杂散表与接收设置不符，未使用"}[p?.spur_status] || "杂散表未加载";
  document.getElementById("policy-note").textContent = meta?.stage2
    ? `识别结合频段规划 · 协议名称为推测 · ${spur}` : "逐信道工程数据；演示视图需要 --stage2 信号级 feed。";
  if (policyStats) document.getElementById("policy-counts").innerHTML = `累计记录：已命名 <b>${policyStats.named}</b> · 未识别 <b>${policyStats.unknown}</b>（${policyStats.unknown_fraction === null ? "—" : (100*policyStats.unknown_fraction).toFixed(1)+"%"}）`;
  document.getElementById("policy-detail").textContent = JSON.stringify({policy:p || null, counts:policyStats}, null, 2);
}

// Adjacent tracked columns form one signal. With a model that has the "wide" class, a whole run of adjacent
// tracked channels is one signal: >= WIDE_RUN channels is a wideband signal; a shorter run is one narrowband
// signal plus its skirts in the neighbouring channels (2x oversampled channelizer), named by its strongest channel
// - the skirts look like the edge of a wide signal to the model and would otherwise show up as extra "wide" ones.
// Without it (older models), adjacent columns of the same class form one signal.
function group(p, now) {
  const T = p.trk, M = meta.channels, out = [];
  const tracked = x => T.on[x] && T.cls[x] > 0;
  for (let x = 0; x < M;) {
    if (!tracked(x)) { x++; continue; }
    let r = x, peak = x;
    while (r < M && tracked(r)) { if (T.pw[r] > T.pw[peak]) peak = r; r++; }
    const run = WIDE > 0;
    const c = !run ? T.cls[x] : r - x >= WIDE_RUN ? WIDE : T.cls[peak];
    let e = x, pmax = -200, conf = 0, since = now, wsum = 0, xw = 0, act = 0;
    while (e < M && tracked(e) && (run || T.cls[e] === c)) {
      const lin = Math.pow(10, T.pw[e] / 10);
      pmax = Math.max(pmax, T.pw[e]); conf += T.conf[e]; since = Math.min(since, T.since[e]); wsum += lin; xw += lin * e;
      act = Math.max(act, T.act[e]); e++;
    }
    out.push({ x0: x, x1: e, xc: xw / wsum, cls: c, conf: run && c !== WIDE ? T.conf[peak] : conf / (e - x),
               snr: pmax - p.floor, age: now - since,
               bursty: act < 0.8 }); // present in < 80 % of recent rows: a burst signal (e.g. WiFi), not a steady one
    x = e;
  }
  return out;
}

// Label strip: every signal gets a colour bar (class colour; bursty signals half transparent). The class name
// is written where the gap to the neighbouring signals holds it - at full band ~100 signals share ~800 px,
// so names appear once you zoom in. No greedy "first label that fits" choice: that made a different signal win the
// space every row and the labels flicker.
function drawLabels(p) {
  const [ctx, w, h] = fit(p.labels);
  ctx.clearRect(0, 0, w, h);
  ctx.textBaseline = "middle"; ctx.textAlign = "center";
  let named = 0, shown = 0;
  const gs = visible(p), cx = gs.map(g => (xOf(g.x0, w) + xOf(g.x1, w)) / 2), lineEnd = [-Infinity, -Infinity];
  gs.forEach((g, i) => {
    const x0 = xOf(g.x0, w), x1 = xOf(g.x1, w);
    ctx.globalAlpha = g.bursty ? 0.45 : 1;
    ctx.fillStyle = color(g.cls);
    ctx.fillRect(x0 + 0.5, h - 10, Math.max(2, x1 - x0 - 1), 10);
    ctx.globalAlpha = 1;
    shown++;
    // Names go on the first of two text lines where they do not overlap the previous name on that line.
    // With the steady signals only (the default), the set of signals - and so the layout - does not change
    // while the tracks are steady.
    ctx.font = "600 11px system-ui, sans-serif";
    const t1 = title(g), tw = ctx.measureText(t1).width + 4, a = cx[i] - tw / 2, b = cx[i] + tw / 2;
    if (a >= 0 && b <= w)
      for (let ln = 0; ln < 2; ln++)
        if (a >= lineEnd[ln]) { ctx.fillText(t1, cx[i], ln ? 19 : 7); lineEnd[ln] = b; named++; break; }
  });
  p.hint = !shown || named === shown ? "" : named ? ` · 已标注 ${named}/${shown}，继续放大可看全部` : " · 颜色即类别，拖动放大显示类别名";
}

// Signals shown: not hidden by the legend, inside the view, and (optionally) not bursty.
function visible(p) {
  const steady = document.getElementById("steady").checked;
  return p.groups.filter(g => !hidden.has(g.cls) && g.x1 > view[0] && g.x0 < view[1] &&
    !(steady && (g.d ? g.d.status !== 'decoded' && (g.d.activity_fraction == null || g.d.activity_fraction < .8) : g.bursty)));
}

function drawList(p) {
  const gs = visible(p);
  p.n.textContent = `${gs.length} 个`;
  const focused = p.rows.contains(document.activeElement) ? document.activeElement.dataset.key : null;
  const scroll = p.rows.scrollTop;
  if (meta.stage2) {
    p.rows.innerHTML = gs.length ? `<table aria-label="RX${p.s+1} 信号列表"><thead><tr><th scope="col">频率 MHz</th><th scope="col">带宽 kHz</th><th scope="col">信号 / 协议推测</th><th scope="col">置信/校验</th><th scope="col">最近</th>${engineering() ? '<th scope="col">检查</th>' : ''}</tr></thead><tbody>` +
      gs.map(g => `<tr><td><button class="freq-action" data-x="${g.xc}" data-key="f${g.d.uid}" aria-label="放大 ${fmtMHz(g.xc)} MHz 信号">${fmtMHz(g.xc)}</button></td><td>${(g.d.bw/1e3).toFixed(g.d.bw < 1e5 ? 1 : 0)}</td>` +
        `<td><i style="background:${color(g.cls)}"></i>${esc(title(g))}${g.bursty ? ' · 突发' : ''}${g.d.protocol && g.d.status !== 'decoded' ? `<div class="signal-sub">${esc(g.d.protocol)}</div>` : ''}</td>` +
        `<td>${g.d.status === 'decoded' ? 'CRC 通过' : g.d.status === 'spur' ? '已校准' : g.d.family === 'unknown' ? (g.d.reason === 'confirming' ? '待确认' : '未识别') : Math.round(g.conf*100)+'%'}</td>` +
        `<td>${Math.max(0,p.now-g.d.t).toFixed(1)} s</td>${engineering() ? `<td><button class="inspect-action" data-id="${g.d.uid}" data-key="i${g.d.uid}">检查</button></td>` : ''}</tr>`).join("") + '</tbody></table>'
      : '<p class="empty">当前没有符合显示条件的信号。可取消类别或“只看持续信号”筛选；工程视图可检查被抑制的候选。</p>';
    for (const button of p.rows.querySelectorAll("button[data-id]")) button.onclick = () => inspect(p, button.dataset.id, true);
  } else {
  const bw = meta.rate / meta.channels;
  p.rows.innerHTML = `<table aria-label="逐信道工程合并结果"><tr><th>频率 MHz</th><th>带宽 kHz</th><th>类别</th><th>置信</th><th>SNR</th><th>持续</th></tr>` +
    gs.map(g => `<tr><td><button class="freq-action" data-x="${g.xc}" data-key="f${g.xc}">${fmtMHz(g.xc)}</button></td><td>${Math.round((g.x1 - g.x0 + 1) * bw / 1e3)}</td>` +
      `<td><i style="background:${color(g.cls)};opacity:${g.bursty ? 0.45 : 1}"></i>${esc(name(g.cls))}${g.bursty ? " 突发" : ""}</td><td>${Math.round(g.conf * 100)}%</td>` +
      `<td>${g.snr.toFixed(0)} dB</td><td>${g.age < 60 ? g.age.toFixed(1) + " s" : (g.age / 60).toFixed(1) + " min"}</td></tr>`).join("") +
    `</table>`;
  }
  for (const button of p.rows.querySelectorAll("button[data-x]"))
    button.onclick = () => {
      const x = parseFloat(button.dataset.x), g = gs.find(g => g.xc === x), half = Math.max(8, g ? (g.x1-g.x0)*.8 : 24);
      setView(x-half,x+half);
      if (g?.d?.status === "decoded") {
        airPick = g.d.icao; airDirty = true; drawAir(); document.getElementById("air").scrollIntoView({block:"nearest"});
      }
    };
  p.rows.scrollTop = scroll;
  if (focused) [...p.rows.querySelectorAll("button[data-key]")].find(b => b.dataset.key === focused)?.focus({preventScroll:true});
}

function inspect(p, key, open) {
  p.selected = String(key);
  const entry = p.signals.get(p.selected);
  if (!entry) { p.inspection.textContent = "该信号已超时；请选择当前信号。"; return; }
  const {d, raw} = entry;
  p.inspection.textContent = `${reasons[d.reason] || d.label}\n` + JSON.stringify({decision:d, raw_model:raw}, null, 2);
  if (open) p.inspection.closest("details").open = true;
}

function drawChannels(p) {
  if (!engineering() || !p.channelDetail.open || !p.last) return;
  const rs = p.last.r.filter(r => r[0] === p.s).sort((a,b) => col(a[1],meta.channels)-col(b[1],meta.channels));
  p.channels.innerHTML = rs.length ? `<table aria-label="RX${p.s+1} 逐信道结果"><thead><tr><th scope="col">频率 MHz</th><th scope="col">功率 dBFS</th><th scope="col">逐信道模型</th><th scope="col">投票份额</th></tr></thead><tbody>` +
    rs.map(([,k,c,prob]) => `<tr><td>${fmtMHz(col(k,meta.channels))}</td><td>${p.last.p[p.s*meta.channels+k]}</td><td>${meta.channel_classes === false ? '仅检测' : esc(channelClasses[c] || 'c'+c)}</td><td>${meta.channel_classes === false ? '—' : prob+'%'}</td></tr>`).join('') + '</tbody></table>'
    : '<p class="empty">当前没有超过检测门限的信道。</p>';
}

function drawLegend() {
  const el = document.getElementById("legend");
  const focused = el.contains(document.activeElement) ? document.activeElement.dataset.c : null;
  const band = meta?.policy?.config.bands.find(b => b.lo_mhz*1e6 <= meta.freq && meta.freq < b.hi_mhz*1e6);
  el.innerHTML = classes.map((n, c) => c === 0 || (!engineering() && band && !band.allow.includes(n) && n !== 'unknown' && !counts[c]) ? "" :
    `<button class="chip${hidden.has(c) ? " off" : ""}" data-c="${c}" aria-pressed="${!hidden.has(c)}"><i style="background:${color(c)}"></i>${n === 'unknown' ? '未识别' : esc(n.toUpperCase())} <b>${counts[c] || 0}</b></button>`).join("");
  for (const ch of el.querySelectorAll(".chip"))
    ch.onclick = () => { const c = +ch.dataset.c; hidden.has(c) ? hidden.delete(c) : hidden.add(c); refresh(); };
  if (focused !== null) [...el.querySelectorAll("button")].find(b => b.dataset.c === focused)?.focus({preventScroll:true});
}

function setView(a, b) {
  const M = meta.channels;
  a = Math.max(0, Math.floor(a)); b = Math.min(M, Math.ceil(b));
  if (b - a < 8) { const c = (a + b) / 2; a = Math.max(0, Math.floor(c - 4)); b = Math.min(M, a + 8); }
  view = [a, b];
  redrawAll();
  for (const p of panels) drawList(p);
}

function hookMouse(p, cv) {
  const tip = document.getElementById("tip");
  let drag = null;
  cv.addEventListener("mousedown", e => { drag = e.offsetX; });
  cv.addEventListener("mousemove", e => {
    const w = cv.clientWidth, x = colAt(e.offsetX, w);
    if (drag !== null) {
      const a = Math.min(drag, e.offsetX), b = Math.max(drag, e.offsetX);
      Object.assign(p.sel.style, { display: "block", left: a + "px", width: (b - a) + "px", top: cv.offsetTop + "px", height: cv.offsetHeight + "px" });
    }
    const T = p.trk, M = meta.channels;
    if (x < 0 || x >= M) return;
    const k = (x + M / 2) % M;
    tip.style.display = "block";
    tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 12) + "px";
    const g = visible(p).find(g => g.x0 <= x && x < g.x1);
    tip.textContent = `${fmtMHz(x)} MHz  (信道 ${k})\n功率 ${T.pw[x].toFixed(0)} dBFS · 高于本底 ${(T.pw[x] - p.floor).toFixed(0)} dB\n` +
      (g ? `${title(g)} · ${(g.d?.bw/1e3 || (g.x1-g.x0)*meta.rate/meta.channels/1e3).toFixed(1)} kHz\n${g.d?.protocol || ''}` : "当前无信号级标注");
  });
  cv.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  addEventListener("mouseup", e => {
    if (drag === null) return;
    const w = cv.clientWidth, a = Math.min(drag, e.offsetX), b = Math.max(drag, e.offsetX);
    p.sel.style.display = "none";
    if (e.target === cv && b - a > 6) setView(colAt(a, w), colAt(b, w) + 1);
    drag = null;
  });
  cv.addEventListener("dblclick", () => setView(0, meta.channels));
}

// ADS-B: one entry per ICAO address from the {"adsb": {...}} lines (sample time t in seconds).
const air = new Map();
let airT = 0, airMsgs = 0, airPick = null, airDirty = false;
function onAdsb(a) {
  if (a.crc_ok !== true || ![17,18].includes(a.df)) return;
  airT = Math.max(airT, a.t); airMsgs++;
  let e = air.get(a.icao);
  if (!e) air.set(a.icao, e = { icao: a.icao, n: 0, trail: [], first: a.t });
  e.n++; e.last = a.t; e.sig = a.sig;
  e.stream = a.s; e.freq = a.freq;
  for (const k of ["call", "alt", "gs", "trk", "vr", "sq"]) if (a[k] !== undefined) e[k] = a[k];
  if (a.lat !== undefined) { e.lat = a.lat; e.lon = a.lon; e.trail.push([a.lat, a.lon]); if (e.trail.length > 400) e.trail.shift(); }
  document.getElementById("air").style.display = "grid";
  while (air.size > 128) air.delete(air.keys().next().value);
  airDirty = true;
}
function drawAir() {
  if (!airDirty) return;
  airDirty = false;
  const live = [...air.values()].filter(e => airT - e.last < 120).sort((a, b) => b.last - a.last);
  document.getElementById("air-n").textContent = `${live.length} 架（2 分钟内） · 共 ${air.size}`;
  document.getElementById("air-info").textContent = `${airMsgs} 条消息`;
  const fmt = (v, d = 0) => v === undefined ? "" : (+v).toFixed(d);
  const el = document.getElementById("air-rows");
  const focused = el.contains(document.activeElement) ? document.activeElement.dataset.i : null;
  const scroll = el.scrollTop;
  el.innerHTML = `<table aria-label="CRC 通过的飞机"><tr><th>ICAO</th><th>呼号</th><th>高度 ft</th><th>速度 kt</th><th>航向</th><th>升降 ft/min</th><th>应答码</th><th>位置</th><th>本页消息</th><th>信号</th><th>最近</th></tr>` +
    live.map(e => `<tr class="${e.icao === airPick ? "sel" : ""}"><td><button class="freq-action" data-i="${esc(e.icao)}" aria-pressed="${e.icao === airPick}">${esc(e.icao)}</button></td><td>${esc(e.call)}</td><td>${fmt(e.alt)}</td><td>${fmt(e.gs)}</td>` +
      `<td>${fmt(e.trk)}</td><td>${fmt(e.vr)}</td><td>${esc(e.sq)}</td><td>${e.lat !== undefined ? e.lat.toFixed(3) + ", " + e.lon.toFixed(3) : ""}</td>` +
      `<td>${e.n}</td><td>${fmt(e.sig)} dB</td><td>${(airT - e.last).toFixed(0)} s</td></tr>`).join("") + `</table>`;
  for (const button of el.querySelectorAll("button[data-i]")) button.onclick = () => { airPick = airPick === button.dataset.i ? null : button.dataset.i; airDirty = true; drawAir(); };
  el.scrollTop = scroll;
  if (focused) [...el.querySelectorAll("button")].find(b => b.dataset.i === focused)?.focus({preventScroll:true});

  // map: equirectangular around the positions seen, cos(lat) scaled, 1-degree grid
  const [ctx, w, h] = fit(document.getElementById("air-map"));
  ctx.clearRect(0, 0, w, h);
  const pos = live.filter(e => e.trail.length);
  ctx.font = "12px system-ui, sans-serif"; ctx.textBaseline = "middle";
  if (!pos.length) { ctx.fillStyle = "#7d8896"; ctx.textAlign = "center"; ctx.fillText("还没有位置报文（需要同一架飞机的奇偶 CPR 帧对）", w / 2, h / 2); return; }
  let la0 = 90, la1 = -90, lo0 = 180, lo1 = -180;
  for (const e of pos) for (const [la, lo] of e.trail) { la0 = Math.min(la0, la); la1 = Math.max(la1, la); lo0 = Math.min(lo0, lo); lo1 = Math.max(lo1, lo); }
  const k = Math.cos((la0 + la1) / 2 * Math.PI / 180), pad = 0.15;
  let spanLa = Math.max(la1 - la0, 0.2), spanLo = Math.max((lo1 - lo0) * k, 0.2);
  const sc = Math.min(w / spanLo, h / spanLa) / (1 + 2 * pad), cla = (la0 + la1) / 2, clo = (lo0 + lo1) / 2;
  const X = lo => w / 2 + (lo - clo) * k * sc, Y = la => h / 2 - (la - cla) * sc;
  ctx.strokeStyle = "#262e38"; ctx.fillStyle = "#4d5866"; ctx.lineWidth = 1; ctx.textAlign = "left";
  const step = spanLa > 4 ? 1 : spanLa > 1 ? 0.5 : 0.1;
  for (let la = Math.ceil((cla - h / 2 / sc) / step) * step; la < cla + h / 2 / sc; la += step) {
    ctx.beginPath(); ctx.moveTo(0, Y(la)); ctx.lineTo(w, Y(la)); ctx.stroke(); ctx.fillText(la.toFixed(1) + "°N", 4, Y(la) - 7);
  }
  for (let lo = Math.ceil((clo - w / 2 / sc / k) / step) * step; lo < clo + w / 2 / sc / k; lo += step) {
    ctx.beginPath(); ctx.moveTo(X(lo), 0); ctx.lineTo(X(lo), h); ctx.stroke(); ctx.fillText(lo.toFixed(1) + "°E", X(lo) + 3, h - 8);
  }
  for (const e of pos) {
    const fresh = airT - e.last < 60, pick = e.icao === airPick;
    ctx.globalAlpha = fresh ? 1 : 0.35;
    ctx.strokeStyle = pick ? "#ffd166" : "#4e79a7"; ctx.lineWidth = pick ? 2 : 1.2;
    ctx.beginPath(); e.trail.forEach(([la, lo], i) => i ? ctx.lineTo(X(lo), Y(la)) : ctx.moveTo(X(lo), Y(la))); ctx.stroke();
    const x = X(e.lon), y = Y(e.lat), a = (e.trk || 0) * Math.PI / 180;
    ctx.save(); ctx.translate(x, y); ctx.rotate(a); ctx.fillStyle = pick ? "#ffd166" : "#e8edf3";
    ctx.beginPath(); ctx.moveTo(0, -8); ctx.lineTo(5, 6); ctx.lineTo(0, 3); ctx.lineTo(-5, 6); ctx.closePath(); ctx.fill(); ctx.restore();
    ctx.fillStyle = "#d8dee6"; ctx.fillText(`${e.call || e.icao}${e.alt !== undefined ? "  " + e.alt + " ft" : ""}`, x + 10, y);
  }
  ctx.globalAlpha = 1;
}
setInterval(drawAir, 500);

function setState(cls, text) {
  const el = document.getElementById("state");
  if (el.dataset.stateText === text && el.className === "kv " + cls) return;
  el.dataset.stateText = text;
  el.className = "kv " + cls; el.innerHTML = `状态 <b>${text}</b>`;
}

function onEnd(st) {
  lastEnd = st;
  refresh();
  if (st.policy) updatePolicy(st.policy);
  setState("ended", "已结束");
  drawEnd();
}

function drawEnd() {
  const st = lastEnd;
  if (!st) return;
  const el = document.getElementById("end");
  el.style.display = "block";
  const keys = ["wall_s", "x_realtime", "windows", "classified", "dropped_frames", "gap_frames", "latency_ms", "stage2"];
  el.textContent = engineering() ? "运行统计  " + keys.filter(k => k in st).map(k => `${k}=${JSON.stringify(st[k])}`).join("  ") :
    `运行结束 · 丢样 ${st.dropped_frames ?? '—'} · 时间缺口 ${st.gap_frames ?? '—'}` +
    (st.stage2 ? ` · 宽带处理 p99 ${st.stage2.wide_latency_ms.p99.toFixed(2)} ms（不含观察窗）` : '');
}

async function start() {
  try { channelClasses = await (await fetch("/classes")).json(); } catch { channelClasses = []; }
  if (!Array.isArray(channelClasses)) channelClasses = [];
  classes = channelClasses;
  WIDE = channelClasses.indexOf("wide");
  document.getElementById("mode").value = new URLSearchParams(location.search).get("view") === "engineering" ? "engineering" : "demo";
  document.getElementById("mode").onchange = () => {
    document.body.dataset.view = document.getElementById("mode").value;
    const url = new URL(location.href); url.searchParams.set("view", document.body.dataset.view); history.replaceState(null,"",url);
    document.title = `cuRF-Stream · ${engineering() ? "工程视图" : "演示视图"}`;
    refresh();
    drawEnd();
  };
  document.getElementById("mode").onchange();
  drawLegend();
  document.getElementById("reset").onclick = () => meta && setView(0, meta.channels);
  for (const [id, factor] of [["zoom-in",.5],["zoom-out",2]]) document.getElementById(id).onclick = () => {
    if (!meta) return; const center = (view[0]+view[1])/2, half = (view[1]-view[0])*factor/2; setView(center-half,center+half);
  };
  document.getElementById("steady").onchange = refresh;
  document.getElementById("air-focus").onclick = () => {
    if (!meta) return;
    const x = meta.channels/2+(1090e6-meta.freq)*meta.channels/meta.rate;
    setView(x-24,x+24);
    document.getElementById("streams").scrollIntoView({block:"start"});
  };
  document.getElementById("reconnect").onclick = connect;
  document.getElementById("reconnect").disabled = false;
  connect();
  addEventListener("resize", () => { if (meta) { redrawAll(); airDirty = true; } });
  setInterval(() => { if (dirty) refresh(); },250);
  setInterval(() => {
    if (meta && !lastEnd && !feedFailed && performance.now()-lastDataAt > 3000) {
      setState("wait", "数据暂未更新，保留上次数据");
      showError("超过 3 秒未收到新样点；请检查采集程序，也可重新连接。");
    }
  },1000);
}

function showError(text) {
  const el = document.getElementById("feed-error");
  el.hidden = false; el.querySelector("span").textContent = text;
}

function connect() {
  if (es) es.close();
  es = new EventSource("/feed");
  setState("wait", "连接中，等待数据");
  es.onmessage = ev => {
    let m;
    try { m = JSON.parse(ev.data); } catch { showError("收到无法解析的数据；请检查 feed 格式后重新连接。"); return; }
    if (m.meta) {
      meta = m.meta;
      classes = meta.stage2 ? meta.signal_classes || [] : channelClasses;
      hidden.clear(); counts = []; policyStats = null; lastEnd = null;
      lastDataAt = performance.now(); feedFailed = false;
      air.clear(); airMsgs = airT = 0; airPick = null; airDirty = true;
      document.getElementById("air").style.display = meta.adsb ? "grid" : "none";
      document.getElementById("air-focus").disabled = !meta.freq || Math.abs(1090e6-meta.freq) > meta.rate/2;
      document.getElementById("feed-error").hidden = true;
      document.querySelector('#mode option[value="demo"]').disabled = !meta.stage2;
      if (!meta.stage2) document.getElementById("mode").value = "engineering";
      document.getElementById("mode").onchange();
      document.getElementById("src").textContent = meta.source;
      document.getElementById("rate").textContent = (meta.rate / 1e6).toFixed(2) + " MSPS × " + meta.streams;
      document.getElementById("fc").textContent = meta.freq ? (meta.freq / 1e6).toFixed(3) + " MHz" : "基带（相对频率）";
      document.getElementById("end").style.display = "none";
      for (const id of ["reset", "zoom-in", "zoom-out"]) document.getElementById(id).disabled = false;
      document.getElementById("tau").disabled = meta.channel_classes === false;
      build();
      updatePolicy(); refresh();
      setState("live", "运行中");
    } else if (m.end) onEnd(m.end);
    else if (m.error) { feedFailed = true; setState("wait", "数据流异常"); showError(m.error.message); }
    else if (meta) {
      if (m.policy_stats) updatePolicy(m.policy_stats);
      if (m.display) onSignal(m);
      if (m.adsb && !m.adsb_rejected) onAdsb(m.adsb);
      else if (m.p) onRow(m);
      if (m.p || m.sig || m.adsb) {
        lastDataAt = performance.now();
        if (!lastEnd && !feedFailed) { setState("live", "运行中"); document.getElementById("feed-error").hidden = true; }
      }
      const sourceTime = m.t ?? m.display?.t;
      if (sourceTime !== undefined && sourceTime > airT) { airT = sourceTime; airDirty = true; }
    }
  };
  es.onerror = () => { setState("wait", "连接中断，保留上次数据"); showError("显示为上次收到的数据，正在尝试重新连接。也可手动重新连接。"); };
}
start();
