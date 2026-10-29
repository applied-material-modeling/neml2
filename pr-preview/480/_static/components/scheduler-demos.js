// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Self-contained, dependency-free animations for the work scheduler /
// dispatcher documentation page. A `<div class="scheduler-demo" data-demo="...">`
// is replaced with an animated SVG schematic of how a batch is chunked and
// dispatched across devices.
//
// Loaded as a classic <script> (NOT type="module"): the doc-render validator
// opens the built pages over file://, where browsers block ES modules under
// the CORS "null origin" rule. A classic script loads fine there. Everything
// is wrapped in an IIFE so the generic helper names don't leak to the global
// scope, and the bootstrap self-defers until the DOM is ready.
//
// These are *illustrative schematics*, not literal traces of the C++ runtime.
// The one behaviour they model faithfully is device selection: the hybrid
// scheduler is greedy by priority among devices with spare capacity, matching
// neml2::aoti::StaticHybridScheduler::schedule_work_impl.

(() => {
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

// Named demo configurations referenced from the page via `data-demo`. Numbers
// are illustrative (chosen so the animation reads clearly), not benchmarks.
const PRESETS = {
  // SimpleScheduler: the whole batch goes to a single device, chunked.
  simple: {
    workSize: 520,
    devices: [{ label: "GPU", capacity: 240, batchSize: 80, speed: 0.05, hue: 205 }],
  },
  // StaticHybridScheduler: one batch spread across CPU + GPUs concurrently.
  // Priority ranks faster devices higher, the documented "keep faster devices
  // filled" configuration; selection is greedy by priority with spare capacity.
  hybrid: {
    workSize: 600,
    devices: [
      { label: "cuda:1", capacity: 120, batchSize: 60, speed: 0.03, priority: 3, hue: 192 },
      { label: "cuda:0", capacity: 100, batchSize: 50, speed: 0.025, priority: 2, hue: 205 },
      { label: "cpu", capacity: 50, batchSize: 25, speed: 0.012, priority: 1, hue: 110 },
    ],
  },
};

// Animation timing (ms of animation time, not wall clock).
const DISPATCH_MS = 500; // time to move a chunk to its device row
const RETRIEVE_MS = 500; // time to gather a finished chunk back

// SVG layout (user units).
const LAYOUT = {
  width: 820,
  pad: 16,
  labelWidth: 150,
  rowHeight: 34,
  rowSpacing: 16,
  fontSize: 19,
};

// Discrete-event simulation of dispatch. Mirrors the v3 scheduler's greedy
// selection: among devices that can take another chunk
// (load + batchSize <= capacity), pick the highest priority; ties resolve to
// the earliest device in config order.
//
// A device row is a packed FIFO queue: when its head chunk completes, every
// chunk behind it squeezes left by the freed width (recorded per chunk as a
// `shifts` event), so the row stays left-packed and a freshly dispatched chunk
// always lands at the open right edge (`load`) instead of overlapping.
function simulate(cfg) {
  const devs = cfg.devices;
  const load = devs.map(() => 0);
  const queues = devs.map(() => []); // in-flight chunks per device, FIFO
  const chunks = [];
  let remaining = cfg.workSize;
  let time = 0;
  let cumulative = 0; // total work dispatched so far (source x position)

  const pick = () => {
    let best = -1;
    for (let i = 0; i < devs.length; i++) {
      const size = Math.min(devs[i].batchSize, remaining);
      if (size <= 0) continue;
      if (load[i] + size <= devs[i].capacity) {
        const pri = devs[i].priority ?? 1;
        if (best < 0 || pri > (devs[best].priority ?? 1)) best = i;
      }
    }
    return best;
  };

  // Retire every chunk finished by `now`, squeezing trailing chunks left.
  const freeCompleted = (now) => {
    for (let d = 0; d < devs.length; d++) {
      while (queues[d].length && queues[d][0].retrievedAt <= now) {
        const done = queues[d].shift();
        load[d] -= done.size;
        for (const later of queues[d]) later.shifts.push({ at: done.retrievedAt, by: done.size });
      }
    }
  };

  let guard = 0;
  while ((remaining > 0 || queues.some((q) => q.length)) && guard++ < 100000) {
    freeCompleted(time);
    const dev = remaining > 0 ? pick() : -1;
    if (dev >= 0) {
      const size = Math.min(devs[dev].batchSize, remaining);
      const chunk = {
        dev,
        size,
        sourceOffset: cumulative, // x slot in the work row
        deviceSlot: load[dev], // x slot in the device row (packed right edge)
        dispatchedAt: time,
        processStart: time + DISPATCH_MS,
        processDuration: Math.ceil(size / devs[dev].speed),
        shifts: [], // {at, by} left-squeezes as chunks ahead complete
      };
      chunk.retrievedAt = chunk.processStart + chunk.processDuration;
      chunks.push(chunk);
      queues[dev].push(chunk);
      load[dev] += size;
      remaining -= size;
      cumulative += size;
      time += DISPATCH_MS;
    } else {
      // Nothing dispatchable now: jump to the next completion.
      let next = Infinity;
      for (let d = 0; d < devs.length; d++)
        if (queues[d].length) next = Math.min(next, queues[d][0].retrievedAt);
      if (!isFinite(next)) break;
      time = Math.max(time + 1, next);
    }
  }

  let total = 0;
  for (const c of chunks) total = Math.max(total, c.retrievedAt + RETRIEVE_MS);
  return { chunks, total };
}

function el(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function rowY(row) {
  return (LAYOUT.rowHeight + LAYOUT.rowSpacing) * row;
}

// Build the static SVG scaffold (work row + one row per device) and a <g> per
// chunk. Returns the svg plus the per-chunk groups for the timeline builder.
function buildScene(cfg, sim) {
  const devs = cfg.devices;
  const nRows = 1 + devs.length;
  const maxUnits = Math.max(cfg.workSize, ...devs.map((d) => d.capacity));
  const unit = (LAYOUT.width - LAYOUT.labelWidth) / maxUnits;
  const height = (LAYOUT.rowHeight + LAYOUT.rowSpacing) * nRows - LAYOUT.rowSpacing;

  const svg = el("svg", {
    viewBox: `${-LAYOUT.pad} ${-LAYOUT.pad} ${LAYOUT.width + 2 * LAYOUT.pad} ${height + 2 * LAYOUT.pad}`,
    role: "img",
    "aria-label":
      devs.length > 1
        ? "Animation: a batch is split into chunks and dispatched across multiple devices concurrently, then gathered back."
        : "Animation: a batch is split into fixed-size chunks dispatched to a single device, then gathered back.",
  });

  const rowOutline = (row, units, label, accent) => {
    const y = rowY(row);
    const text = el("text", {
      x: LAYOUT.labelWidth - 12,
      y: y + LAYOUT.rowHeight - 9,
      "text-anchor": "end",
      "font-size": LAYOUT.fontSize,
      "font-family": "var(--sd-font, sans-serif)",
      fill: "currentColor",
    });
    text.textContent = label;
    svg.appendChild(text);
    svg.appendChild(
      el("rect", {
        x: LAYOUT.labelWidth,
        y,
        width: units * unit,
        height: LAYOUT.rowHeight,
        rx: 3,
        fill: "none",
        stroke: accent ?? "var(--sd-outline)",
        "stroke-width": 1.5,
        "stroke-dasharray": accent ? "none" : "4 3",
      })
    );
  };

  rowOutline(0, cfg.workSize, "batch", "var(--sd-outline)");
  devs.forEach((d, i) => rowOutline(i + 1, d.capacity, d.label, deviceColor(d)));

  // Chunks tile exactly edge-to-edge (chunk B starts where A ends). Drawing
  // them a hair wider (SEAM) makes neighbours overlap by < 1px so their shared
  // edge fully covers and the un-dispatched batch reads as one solid bar with
  // no anti-aliased hairlines. The overlap is same-colour within the batch and
  // within a device row, so it is imperceptible.
  const SEAM = 1;

  const groups = sim.chunks.map((chunk) => {
    const d = devs[chunk.dev];
    const w = chunk.size * unit;
    const srcX = LAYOUT.labelWidth + chunk.sourceOffset * unit;
    // Opaque so overlapping (SEAM) neighbours don't darken at the shared edge;
    // the batch reads as one solid bar. Dispatch/gather are conveyed by motion,
    // not opacity.
    const g = el("g");
    // Chunk body (neutral) sitting in the work row at its source slot.
    g.appendChild(
      el("rect", {
        x: srcX,
        y: rowY(0),
        width: w + SEAM,
        height: LAYOUT.rowHeight,
        fill: "var(--sd-work)",
      })
    );
    // Progress fill (device-coloured), grows left-to-right during compute.
    // `transform-box: fill-box` makes scaleX pivot on the element's own left
    // edge regardless of the SVG viewBox coordinate system (the portable way
    // to scale an SVG shape via CSS/WAAPI).
    const fill = el("rect", {
      x: srcX,
      y: rowY(0),
      width: w + SEAM,
      height: LAYOUT.rowHeight,
      fill: deviceColor(d),
    });
    fill.style.transformBox = "fill-box";
    fill.style.transformOrigin = "left center";
    fill.style.transform = "scaleX(0)";
    g.appendChild(fill);
    svg.appendChild(g);

    const destX = LAYOUT.labelWidth + chunk.deviceSlot * unit;
    return {
      g,
      fill,
      tx: destX - srcX,
      ty: rowY(chunk.dev + 1) - rowY(0),
      shifts: chunk.shifts.map((s) => ({ at: s.at, by: s.by * unit })),
      chunk,
    };
  });

  return { svg, groups, height };
}

function deviceColor(d) {
  return `hsl(${d.hue}, 70%, 50%)`;
}

// Build a paused Web Animations API timeline from the simulation. Each chunk
// group gets a single keyframe animation spanning the whole timeline, with
// keyframes placed at offset = absoluteTime / total. A small controller plays /
// pauses / restarts all of them together.
function buildTimeline(sim, groups) {
  const T = sim.total || 1;
  const at = (t) => Math.max(0, Math.min(1, t / T));
  const animations = [];

  const EASE = "cubic-bezier(0.7,0,0.3,1)";
  const SHIFT_MS = 250; // squeeze-left duration when a chunk ahead completes

  for (const { g, fill, tx, ty, shifts, chunk } of groups) {
    const dispatchEnd = chunk.dispatchedAt + DISPATCH_MS;
    const retrieveEnd = chunk.retrievedAt + RETRIEVE_MS;

    // Group: opacity + translate. Work row -> device slot, then squeeze left as
    // chunks ahead retire, then back to the work row (gathered).
    let x = tx;
    const frames = [
      { offset: 0, transform: "translate(0px,0px)" },
      { offset: at(chunk.dispatchedAt), transform: "translate(0px,0px)", easing: EASE },
      { offset: at(dispatchEnd), transform: `translate(${x}px,${ty}px)` },
    ];
    for (const s of shifts) {
      const start = Math.max(s.at, dispatchEnd);
      frames.push({ offset: at(start), transform: `translate(${x}px,${ty}px)`, easing: EASE });
      x -= s.by;
      frames.push({ offset: at(start + SHIFT_MS), transform: `translate(${x}px,${ty}px)` });
    }
    frames.push({ offset: at(chunk.retrievedAt), transform: `translate(${x}px,${ty}px)`, easing: EASE });
    frames.push({ offset: at(retrieveEnd), transform: "translate(0px,0px)" });
    frames.push({ offset: 1, transform: "translate(0px,0px)" });
    animations.push(g.animate(dedupeOffsets(frames), { duration: T, fill: "both" }));

    // Progress fill: scaleX 0 -> 1 over the compute window.
    const fillFrames = [
      { offset: 0, transform: "scaleX(0)" },
      { offset: at(chunk.processStart), transform: "scaleX(0)", easing: "linear" },
      { offset: at(chunk.processStart + chunk.processDuration), transform: "scaleX(1)" },
      { offset: 1, transform: "scaleX(1)" },
    ];
    animations.push(fill.animate(dedupeOffsets(fillFrames), { duration: T, fill: "both" }));
  }

  animations.forEach((a) => a.pause());
  return new Timeline(animations, T);
}

// WAAPI requires strictly non-decreasing offsets; nudge any that collide.
function dedupeOffsets(frames) {
  let prev = -1;
  for (const f of frames) {
    if (f.offset <= prev) f.offset = Math.min(1, prev + 1e-4);
    prev = f.offset;
  }
  return frames;
}

class Timeline {
  constructor(animations, duration) {
    this.animations = animations;
    this.duration = duration;
  }
  play() {
    this.animations.forEach((a) => a.play());
  }
  pause() {
    this.animations.forEach((a) => a.pause());
  }
  restart() {
    this.animations.forEach((a) => {
      a.currentTime = 0;
      a.play();
    });
  }
  seekEnd() {
    this.animations.forEach((a) => {
      a.currentTime = this.duration;
      a.pause();
    });
  }
}

function button(cls, label, svgPath) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `sd-btn sd-${cls}`;
  b.setAttribute("aria-label", label);
  b.title = label;
  b.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="${svgPath}" fill="currentColor"/></svg>`;
  return b;
}

function buildControls(timeline) {
  const bar = document.createElement("div");
  bar.className = "sd-controls";
  const play = button("play", "Play", "M8 5v14l11-7z");
  const pause = button("pause", "Pause", "M6 5h4v14H6zM14 5h4v14h-4z");
  const restart = button(
    "restart",
    "Restart",
    "M12 5V2L7 7l5 5V8a4 4 0 1 1-4 4H6a6 6 0 1 0 6-7z"
  );
  play.onclick = () => timeline.play();
  pause.onclick = () => timeline.pause();
  restart.onclick = () => timeline.restart();
  bar.append(play, pause, restart);
  return bar;
}

// Render one container: scene + timeline + controls + autoplay policy.
function render(container) {
  const preset = PRESETS[container.dataset.demo];
  if (!preset) {
    console.warn(`scheduler-demos: unknown data-demo="${container.dataset.demo}"`);
    return;
  }
  container.textContent = "";
  const sim = simulate(preset);
  const scene = buildScene(preset, sim);
  const timeline = buildTimeline(sim, scene.groups);

  const figure = document.createElement("div");
  figure.className = "sd-figure";
  figure.appendChild(scene.svg);
  figure.appendChild(buildControls(timeline));
  container.appendChild(figure);

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) {
    // Respect the user's reduced-motion preference: show the completed state.
    timeline.seekEnd();
    return;
  }

  // Autoplay once when scrolled into view.
  let started = false;
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !started) {
          started = true;
          timeline.play();
          io.disconnect();
        }
      }
    },
    { threshold: 0.4 }
  );
  io.observe(container);
}

function bootstrap() {
  document.querySelectorAll(".scheduler-demo").forEach(render);
}

// Classic scripts may run before the DOM is parsed, so self-defer.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}
})();
