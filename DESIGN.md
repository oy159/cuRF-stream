---
version: alpha
name: cuRF-Stream instrument display
description: A Chinese RF instrument display linking measured spectrum, one label per carrier, and inspectable evidence.
colors:
  background: "#0d1014"
  surface: "#151a20"
  border: "#262e38"
  text: "#d8dee6"
  muted: "#7d8896"
  focus: "#7aa7ff"
  primary: "#7aa7ff"
typography:
  sans:
    fontFamily: 'system-ui, -apple-system, "PingFang SC", sans-serif'
    fontSize: "13px"
    lineHeight: "1.4"
  mono:
    fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace'
rounded:
  DEFAULT: "6px"
  control: "4px"
spacing:
  page: "16px"
  panels: "14px"
components:
  instrumentPanel:
    rounded: "6px"
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
  control:
    rounded: "4px"
    textColor: "{colors.primary}"
---

# cuRF-Stream display

## Overview

The reference is the existing RF waterfall instrument in `tools/ui/index.html`: spectrum occupies the main
surface, carrier annotations share its frequency axis, and a compact signal list provides exact values.
The audience is the project's developer and a Chinese-speaking hardware-demo audience (PLAN.md, P6).
Desktop is the primary use; a narrow browser stacks spectrum and tables. This is a product instrument,
with no marketing shell. The signature is the direct link between each complete carrier, its frequency
bar, and the model/policy evidence. Preserve that link rather than adding decorative status cards.

The existing runtime CSS is canonical (token ownership model B). This document mirrors its accepted
values. `:root` variables map background→`--bg`, surface→`--panel`, border→`--line`, text→`--text`,
muted→`--dim`, focus→`--focus`; typography maps to `--sans` and `--mono`. There is no theme adapter.

## Colors

Preserve the graphite instrument palette above. Signal-class colors encode modulation families, not
certainty. Unknown and spur labels use neutral gray and explicit text. Confidence, calibration status,
CRC verification and errors must have text alternatives. The waterfall retains its measured-power LUT.
Only the dark theme is maintained; forced-colors mode keeps controls and scrollbars system-operable.

## Typography

Use the existing system sans stack for Chinese labels and restrained 600-weight headings. Technical
values, frequencies, probabilities and timings use the monospace stack and tabular numbers. Never
download fonts for this instrument. Long signal/protocol labels wrap or scroll within their own panel.

## Layout

Keep 16 px page padding, 14 px panel gaps and the existing 1000 px stacking breakpoint. Signal tables
own bounded internal scrolling; the page itself scrolls normally. Engineering details appear below the
plots without resizing the demo controls. Controls remain in the same toolbar when data is absent.

## Elevation & Depth

Flat tonal panels and single-pixel borders provide hierarchy. Only the existing pointer tooltip floats.
No decorative shadows, gradients, cards or page-load animation are introduced.

## Shapes

Panels use 6 px radius and controls 4 px. Class-filter chips retain their pill shape. Frequency bars
remain square because they encode occupied intervals on a continuous frequency axis.

## Components

Canonical capability owners for this single-page instrument:

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | Native HTML select in the common toolbar | DESIGN.md | Platform popup geometry is accepted | Keyboard and opened popup in browser |
| Scrollbar | Global stylesheet in `tools/ui/index.html` | DESIGN.md | Standards properties, WebKit fallback, forced-colors override | Computed style and narrow viewport |
| Feedback | `setState` and the persistent feed-error region | PLAN.md P6 and this contract | Waiting, live, ended, reconnecting, error | Browser state checks |
| Tables | `drawList`, channel details and `drawAir` | PLAN.md P6 and this contract | Native tables, frequency/ICAO buttons, internal scrolling | Keyboard and responsive inspection |

No editable forms, destructive actions or CRUD workflows exist. Demo/engineering view and class filters
are transient inspection state; the view is shareable via `?view=demo` or `?view=engineering`. A signal
selection zooms the existing frequency axis; toolbar zoom buttons provide a non-drag alternative.

Demo reads signal-level policy decisions only. Unknown signals remain visible. Engineering adds the raw
13 probabilities, allowed-set posterior and mass, decision reason, raw track ID, processing/observation
timing, suppressed candidates, and optional channel output. It never rewrites the measured spectrum.
The signal list is bounded by the policy's carrier cap and source-time TTL; aircraft have a separate
two-minute lifetime. Empty states explain whether data is awaited, no carriers are present, or filters
exclude results. Reconnect uses a bounded server snapshot. Every new run or replay loop resets state.

Controls use native buttons, visible focus, hover/pressed states, and accessible names. Updating tables
preserves focused actions and scroll position. The tooltip is supplementary; tables expose its data.
Source-relative times use seconds, RF frequencies MHz, bandwidths kHz/MHz, and powers dBFS. Protocol
names always carry “推测”; ADS-B is marked “CRC 通过”, never an invented probability. Continuous counters
are not live announcements. Motion is limited to new measured rows; reduced motion adds no animation.

## Do's and Don'ts

- Do preserve the spectrum-first layout and one annotation per carrier.
- Do expose why a label was rejected and report the unknown proportion separately.
- Don't label channel slices as separate recognized transmitters in the demo view.
- Don't present a normalized one-class prior as strong model evidence or a protocol guess as decoding.
