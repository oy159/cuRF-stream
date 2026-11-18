#!/usr/bin/env python3
"""Generate the README figures (SVG) in docs/img/.

Numbers are copied from README.md; rerun after updating them:
    python3 docs/img/make_figures.py
"""
from pathlib import Path
import math
import random

OUT = Path(__file__).resolve().parent

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', "
        "'Hiragino Sans GB', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif")
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Dark card palette: reads well on both GitHub light and dark themes.
BG = "#0d1117"
CARD = "#161b22"
LINE = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#76b900"      # GPU / NVIDIA green
BLUE = "#58a6ff"
ORANGE = "#f0883e"
PURPLE = "#bc8cff"


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{title}">\n'
            f'<title>{title}</title>\n'
            f'<style>text{{font-family:{FONT};}} .mono{{font-family:{MONO};}}</style>\n'
            f'<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>\n'
            f'{body}\n</svg>\n')


def t(x, y, s, size=14, fill=TEXT, anchor="start", weight=400, cls=""):
    c = f' class="{cls}"' if cls else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{c}>{s}</text>')


# ---------------------------------------------------------------- banner
def banner():
    w, h = 1200, 340
    rnd = random.Random(7)
    parts = []
    parts.append(
        '<defs>'
        '<linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{BG}" stop-opacity="1"/>'
        f'<stop offset="0.42" stop-color="{BG}" stop-opacity="0.92"/>'
        f'<stop offset="0.7" stop-color="{BG}" stop-opacity="0.15"/>'
        f'<stop offset="1" stop-color="{BG}" stop-opacity="0"/>'
        '</linearGradient>'
        '<clipPath id="r"><rect width="1200" height="340" rx="14"/></clipPath>'
        '</defs><g clip-path="url(#r)">')

    # Synthetic waterfall: columns = channels, rows = time.
    cols, rows = 128, 34
    cw, rh = w / cols, (h - 70) / rows
    # signals: (center col, half width, colour, burst probability)
    sigs = [(18, 6, BLUE, 1.0), (40, 1, ORANGE, 0.35), (47, 1, ORANGE, 0.35),
            (60, 9, BLUE, 1.0), (79, 2, PURPLE, 0.7), (92, 1, ACCENT, 1.0),
            (104, 7, BLUE, 0.85), (118, 1, ORANGE, 0.5)]
    on = {i: True for i in range(len(sigs))}
    for r in range(rows):
        y = 70 + r * rh
        for i, (_, _, _, prob) in enumerate(sigs):
            if prob < 1.0 and rnd.random() < 0.35:
                on[i] = rnd.random() < prob
        for c in range(cols):
            v = rnd.random() * 0.16
            col = "#1f6feb"
            for i, (cc, hw, colour, _) in enumerate(sigs):
                if on[i] and abs(c - cc) <= hw:
                    edge = 1 - abs(c - cc) / (hw + 1.5)
                    v = 0.5 + 0.4 * edge * (0.85 + 0.15 * rnd.random())
                    col = colour
            if v > 0.07:
                parts.append(f'<rect x="{c*cw:.1f}" y="{y:.1f}" width="{cw+0.3:.1f}" '
                             f'height="{rh+0.3:.1f}" fill="{col}" opacity="{v:.2f}"/>')
    # Spectrum trace on top.
    pts = []
    for i in range(0, w + 1, 4):
        c = i / cw
        v = 0.08 + 0.05 * rnd.random()
        for (cc, hw, _, _) in sigs:
            d = abs(c - cc)
            if d <= hw + 1:
                v = max(v, 0.55 + 0.35 * (1 - d / (hw + 1)) + 0.05 * rnd.random())
        pts.append(f"{i},{70 - v * 46:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ACCENT}" '
                 f'stroke-width="1.6" opacity="0.9"/>')
    # Label chips on the right half.
    for (cc, hw, colour, _), name in zip(
            [sigs[3], sigs[4], sigs[5], sigs[6]], ["OFDM", "DSSS", "FSK", "OFDM"]):
        x = cc * cw
        parts.append(f'<rect x="{x-28:.0f}" y="{h-42}" width="56" height="22" rx="11" '
                     f'fill="{CARD}" stroke="{colour}"/>')
        parts.append(t(x, h - 26, name, 12, colour, "middle", 600, "mono"))
    parts.append(f'<rect width="{w}" height="{h}" fill="url(#fade)"/>')

    parts.append(t(56, 140, "cuRF-Stream", 58, TEXT, weight=700))
    parts.append(t(58, 184, "USB3 SDR + 笔记本 GPU 上的实时宽带信号识别", 22, TEXT, weight=500))
    parts.append(t(58, 216, "2 × 61.44 MHz · 1024 信道 · 自研 INT8 推理引擎 · 不依赖 TensorRT",
                   16, MUTED))
    chips = [("CUDA", ACCENT), ("Tensor Core", ACCENT), ("AD9361", BLUE), ("USB3", BLUE)]
    x = 58
    for name, colour in chips:
        wd = 14 + 8.6 * len(name)
        parts.append(f'<rect x="{x}" y="244" width="{wd:.0f}" height="26" rx="13" '
                     f'fill="none" stroke="{colour}" stroke-opacity="0.8"/>')
        parts.append(t(x + wd / 2, 262, name, 13, colour, "middle", 600, "mono"))
        x += wd + 10
    parts.append("</g>")
    return svg(w, h, "\n".join(parts), "cuRF-Stream")


# ---------------------------------------------------------------- architecture
def architecture():
    w, h = 1200, 560
    p = []
    p.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/>'
             '</marker></defs>')

    def box(x, y, bw, bh, title, sub, colour, idx=None):
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{CARD}" '
                 f'stroke="{colour}" stroke-opacity="0.7"/>')
        tx = x + 16
        if idx is not None:
            p.append(f'<circle cx="{x+22}" cy="{y+bh/2}" r="12" fill="{colour}" opacity="0.18"/>')
            p.append(t(x + 22, y + bh / 2 + 5, idx, 13, colour, "middle", 700, "mono"))
            tx = x + 44
        if sub:
            p.append(t(tx, y + bh / 2 - 3, title, 15, TEXT, weight=600))
            p.append(t(tx, y + bh / 2 + 17, sub, 12, MUTED))
        else:
            p.append(t(tx, y + bh / 2 + 5, title, 15, TEXT, weight=600))

    def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" '
                 f'stroke-width="1.6" marker-end="url(#a)"/>')
        if label:
            p.append(t(lx, ly, label, 12, MUTED, "middle", cls="mono"))

    # Lanes
    p.append(f'<rect x="24" y="24" width="400" height="512" rx="12" fill="none" stroke="{LINE}" '
             'stroke-dasharray="4 4"/>')
    p.append(t(44, 54, "CPU · C++", 14, BLUE, weight=700, cls="mono"))
    p.append(f'<rect x="520" y="24" width="656" height="512" rx="12" fill="none" '
             f'stroke="{ACCENT}" stroke-opacity="0.5" stroke-dasharray="4 4"/>')
    p.append(t(540, 54, "GPU · CUDA  (RTX 3050 Laptop, 60 W)", 14, ACCENT, weight=700, cls="mono"))

    # CPU side
    box(48, 76, 352, 64, "OmniCore SDR · AD9361 2T2R", "2 × 61.44 MSPS · 12 bit · USB3 368.6 MB/s", BLUE)
    arrow(224, 140, 224, 168)
    box(48, 168, 352, 64, "USB 异步收包 → HostRing", "SPSC 无锁环 · 2 MB 大页 + cudaHostRegister", BLUE)
    box(48, 290, 352, 64, "调度 / 背压 / 丢样计量", "GPU 决定 batch · 样点时间戳查缺口", BLUE)
    box(48, 452, 352, 64, "浏览器界面（SSE）", "瀑布图 · 信号跟踪 · 识别标注", BLUE)
    arrow(224, 354, 224, 452)

    # GPU side
    steps = [
        ("多相信道化", "12 bit 解包融进 FIR fold + 批量 cuFFT · 512 × 120 kHz / 路"),
        ("功率 · 噪声底 · 白化", "25% 分位噪声底 · 前端噪声形状校准"),
        ("检测 · 泄漏抑制 · 压缩", "warp ballot 压缩活跃信道"),
        ("gather", "转置 + 归一化，直接写 fp16"),
        ("推理：自研 INT8 引擎", "6 个 kernel · CUDA Graph · 34.3 万窗口/秒"),
        ("softmax / argmax", "每信道每 4.27 ms 一次识别"),
    ]
    y = 76
    for i, (a, b) in enumerate(steps):
        colour = ORANGE if i == 4 else ACCENT
        box(544, y, 608, 60, a, b, colour, str(i + 1))
        if i < len(steps) - 1:
            arrow(848, y + 60, 848, y + 76)
        y += 76

    # H2D / D2H
    arrow(400, 200, 544, 106, "H2D", 470, 140)
    p.append(f'<path d="M544,486 C500,486 500,322 400,322" fill="none" stroke="{MUTED}" '
             'stroke-width="1.6" marker-end="url(#a)"/>')
    p.append(t(440, 470, "D2H", 12, MUTED, "middle", cls="mono"))
    p.append(t(440, 488, "计数 · 结果 · 功率", 11, MUTED, "middle"))
    return svg(w, h, "\n".join(p), "cuRF-Stream 系统架构")


# ---------------------------------------------------------------- ladder
def ladder():
    rows = [
        ("1. naive", 1277, MUTED),
        ("2. shared memory 分块", 16503, MUTED),
        ("3. WMMA Tensor Core", 79939, BLUE),
        ("4. + epilogue 融合", 140848, BLUE),
        ("5. cp.async + ldmatrix + mma.sync", 177472, BLUE),
        ("6–8. + GAP / stem / 整块融合", 224044, ACCENT),
        ("8'. + swizzle · fragment 双缓冲", 230200, ACCENT),
        ("9. INT8 + 融合块", 315000, ORANGE),
        ("10. + stem / shortcut / head 融合", 343000, ORANGE),
    ]
    w, rh, top, left = 1200, 40, 92, 330
    h = top + rh * len(rows) + 56
    right = w - 150
    lo, hi = math.log10(1000), math.log10(400000)

    def sx(v):
        return left + (math.log10(v) - lo) / (hi - lo) * (right - left)

    p = [t(32, 46, "推理引擎优化阶梯", 22, TEXT, weight=700),
         t(32, 72, "batch 512 · CUDA Graph · 冷机 · 窗口/秒（对数轴）", 13, MUTED)]
    for v, lab in [(1e3, "1k"), (1e4, "10k"), (1e5, "100k")]:
        x = sx(v)
        p.append(f'<line x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{top+rh*len(rows)}" '
                 f'stroke="{LINE}"/>')
        p.append(t(x, top + rh * len(rows) + 20, lab, 12, MUTED, "middle", cls="mono"))
    for i, (name, v, colour) in enumerate(rows):
        y = top + i * rh
        p.append(t(left - 14, y + 25, name, 14, TEXT, "end"))
        x2 = sx(v)
        p.append(f'<rect x="{left}" y="{y+8}" width="{x2-left:.1f}" height="{rh-16}" rx="4" '
                 f'fill="{colour}" opacity="0.85"/>')
        label = "300,000–315,000" if v == 315000 else f"{v:,}"
        p.append(t(x2 + 10, y + 25, label, 13, colour if colour != MUTED else TEXT,
                   weight=600, cls="mono"))
    y = top + rh * len(rows) + 42
    legend = [(MUTED, "CUDA core"), (BLUE, "Tensor Core FP16"), (ACCENT, "融合 FP16"),
              (ORANGE, "INT8")]
    x = left
    for colour, name in legend:
        p.append(f'<rect x="{x}" y="{y-11}" width="12" height="12" rx="2" fill="{colour}"/>')
        p.append(t(x + 18, y, name, 12, MUTED))
        x += 36 + sum(14 if ord(ch) > 127 else 7.5 for ch in name)
    p.append(t(w - 32, y, "naive → INT8 最终版 ≈ 270×", 14, ORANGE, "end", 700))
    return svg(w, h, "\n".join(p), "推理引擎优化阶梯")


# ---------------------------------------------------------------- vs libs
def vs_libs():
    rows = [
        ("自研 · 完整融合", 204.4, ACCENT, ""),
        ("CUTLASS 4.5", 178.5, MUTED, "1.15×"),
        ("自研 · 逐层不融合", 176.9, MUTED, "1.16×"),
        ("TensorRT 11.4", 154.5, MUTED, "1.32×"),
        ("cuDNN 9.21", 113.5, MUTED, "1.80×"),
    ]
    w, rh, top, left = 1200, 48, 96, 260
    h = top + rh * len(rows) + 40
    right = w - 220
    vmax = 220.0
    p = [t(32, 46, "与库的对比（FP16）", 22, TEXT, weight=700),
         t(32, 72, "同一 ONNX · 同一轮背靠背测量 · batch 512 · 千窗口/秒", 13, MUTED)]
    for v in (0, 50, 100, 150, 200):
        x = left + v / vmax * (right - left)
        p.append(f'<line x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{top+rh*len(rows)}" '
                 f'stroke="{LINE}"/>')
        p.append(t(x, top + rh * len(rows) + 20, f"{v}k", 12, MUTED, "middle", cls="mono"))
    p.append(t(w - 40, top - 12, "自研领先", 12, MUTED, "end"))
    for i, (name, v, colour, lead) in enumerate(rows):
        y = top + i * rh
        x2 = left + v / vmax * (right - left)
        p.append(t(left - 14, y + 29, name, 15, TEXT, "end", 600 if i == 0 else 400))
        p.append(f'<rect x="{left}" y="{y+10}" width="{x2-left:.1f}" height="{rh-20}" rx="4" '
                 f'fill="{colour}" opacity="{0.95 if i == 0 else 0.55}"/>')
        p.append(t(x2 + 10, y + 29, f"{v}k", 14, ACCENT if i == 0 else TEXT, weight=600,
                   cls="mono"))
        if lead:
            p.append(t(w - 40, y + 29, lead, 16, ACCENT, "end", 700, "mono"))
    return svg(w, h, "\n".join(p), "与 CUTLASS / TensorRT / cuDNN 的对比")


# ---------------------------------------------------------------- stats strip
def stats():
    items = [
        ("1024", "信道同时识别", "2 路 × 512 × 120 kHz", ACCENT),
        ("300 s", "实采零丢样", "2 × 61.44 MSPS 连续接收", BLUE),
        ("2.47 ms", "端到端 p99 延迟", "实采 · 平均 318 个活跃信道", BLUE),
        ("343k", "INT8 窗口/秒", "6 个 kernel · CUDA Graph", ORANGE),
        ("1.32×", "领先 TensorRT 11.4", "FP16 · batch 512", ACCENT),
    ]
    w, h = 1200, 150
    cw = (w - 48) / len(items)
    p = []
    for i, (big, label, sub, colour) in enumerate(items):
        x = 24 + i * cw
        p.append(f'<rect x="{x+6:.0f}" y="20" width="{cw-12:.0f}" height="{h-40}" rx="10" '
                 f'fill="{CARD}" stroke="{LINE}"/>')
        cx = x + cw / 2
        p.append(t(cx, 66, big, 32, colour, "middle", 700, "mono"))
        p.append(t(cx, 94, label, 14, TEXT, "middle", 600))
        p.append(t(cx, 114, sub, 11, MUTED, "middle"))
    return svg(w, h, "\n".join(p), "关键指标")


if __name__ == "__main__":
    for name, fn in [("banner", banner), ("architecture", architecture),
                     ("ladder", ladder), ("vs-libs", vs_libs), ("stats", stats)]:
        (OUT / f"{name}.svg").write_text(fn(), encoding="utf-8")
        print("wrote", OUT / f"{name}.svg")
