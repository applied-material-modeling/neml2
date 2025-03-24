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

class WorkDispatcher {
  constructor(
    works,
    work_sizes,
    devices,
    capacities,
    batch_sizes,
    speeds,
    colors
  ) {
    this.works = works;
    this.work_sizes = work_sizes;
    this.devices = devices;
    this.capacities = capacities;
    this.batch_sizes = batch_sizes;
    this.speeds = speeds;
    this.colors = colors;

    // The work dispatcher will be visualized as rows, with each row
    // representing a different group:
    //   - Works (each work is a separate group)
    //   - Devices (each device is a separate group)
    this.rows = [...works, ...devices];
    this.row_batches = [...work_sizes, ...capacities];
  }

  apply_defaults(options) {
    options.canvas_width ||= 800;
    options.canvas_padding ||= 20;

    options.row_height ||= 35;
    options.row_spacing ||= 15;

    options.label_width ||= 200;
    options.label_padding_right ||= 15;
    options.label_pardding_bottom ||= 5;
    options.label_font_size ||= 24;
    options.label_font_family ||= "sans-serif";

    options.box_width ||= 2;
    options.work_color ||= "rgb(211, 211, 211)";

    options.work_opacity ||= 0.3;
    options.highlight_opacity ||= 1.0;
  }

  animate(scheduler, options = {}) {
    let canvas = this.render(options);

    // Simulate the work dispatching process
    let all_work_events = scheduler.simulate(options);
    var tl = anime.timeline({
      easing: "easeInOutExpo",
      autoplay: false,
    });

    // Render discrete work batches
    this.works.forEach((work, i) => {
      let work_events = all_work_events[i];
      let x0 = options.label_width;
      let y0 = (options.row_height + options.row_spacing) * i;
      for (let [id, work_event] of Object.entries(work_events)) {
        let dx = this.batch_width * work_event.size;
        let doms = this.draw_work(canvas, x0, y0, dx, work_event, options);
        this.animate_work(tl, doms, i, work_event, x0, y0, dx, options);
        x0 += dx;
      }
    });

    return [canvas, tl];
  }

  animate_work(tl, doms, work_i, work_event, x0, y0, dx, options) {
    // Dispatch event
    const tx = options.label_width + this.batch_width * work_event.pos0 - x0;
    const ty =
      (this.works.length - work_i + work_event.dispatched_to) *
      (options.row_height + options.row_spacing);
    tl.add(
      {
        targets: doms,
        opacity: options.highlight_opacity,
        duration: options.dispatch_speed,
        translateX: tx,
        translateY: ty,
      },
      `${work_event.dispatched_at}`
    );
    // Shift events
    work_event.shifted_at.forEach((shifted_at, j) => {
      const tx = this.batch_width * work_event.shifted_by[j];
      tl.add(
        {
          targets: doms,
          translateX: `-=${tx}`,
          duration: options.retrieve_speed,
        },
        `${shifted_at}`
      );
    });
    // Retrieve event
    tl.add(
      {
        targets: doms,
        duration: options.retrieve_speed,
        translateX: 0,
        translateY: 0,
      },
      `${work_event.retrieved_at}`
    );
    // Process event
    const x1 = x0 + dx;
    const y1 = y0 + options.row_height;
    tl.add(
      {
        targets: doms[1],
        points: `${x0},${y0} ${x1},${y0} ${x1},${y1} ${x0},${y1}`,
        duration: work_event.process_duration,
        easing: "linear",
      },
      `${work_event.processed_at}`
    );
  }

  render(options = {}) {
    // Default options
    this.apply_defaults(options);

    // Initialize the canvas
    let canvas = this.draw_canvas(options);

    // Calculate the width of each batch
    const max_batch_size = Math.max(...this.row_batches);
    this.batch_width =
      (options.canvas_width - options.label_width) / max_batch_size;

    // Draw the rows
    this.row_colors = [
      ...Array.from({ length: this.works.length }, () => options.work_color),
      ...this.colors,
    ];
    this.rows.forEach((row, i) => {
      this.draw_row(canvas, row, i, options);
    });

    return canvas;
  }

  draw_canvas(options) {
    const height =
      options.row_height * this.rows.length +
      options.row_spacing * this.rows.length;
    const xmin = -options.canvas_padding;
    const ymin = -options.canvas_padding;
    const xmax = options.canvas_width + options.canvas_padding;
    const ymax = height + options.canvas_padding;
    const view_box = `${xmin} ${ymin} ${xmax} ${ymax}`;

    let canvas = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    canvas.setAttribute("viewBox", view_box);
    canvas.setAttribute("width", "95%");
    canvas.setAttribute("height", "95%");
    return canvas;
  }

  draw_row(canvas, label, i, options) {
    let text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const label_x = options.label_width - options.label_padding_right;
    const label_y =
      options.row_height * (i + 1) +
      options.row_spacing * i -
      options.label_pardding_bottom;
    text.setAttribute("text-anchor", "end");
    text.setAttribute("x", label_x);
    text.setAttribute("y", label_y);
    text.setAttribute("font-size", options.label_font_size);
    text.setAttribute("font-family", options.label_font_family);
    text.textContent = label;
    canvas.appendChild(text);

    let box = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const x0 = options.label_width;
    const y0 = (options.row_height + options.row_spacing) * i;
    const x1 = x0 + this.batch_width * this.row_batches[i];
    const y1 = y0 + options.row_height;
    box.setAttribute("d", `M ${x0} ${y0} H ${x1} V ${y1} H ${x0} Z`);
    box.setAttribute("fill", "transparent");
    box.setAttribute("stroke", this.row_colors[i]);
    // box.setAttribute("stroke-dasharray", "4 1");
    box.setAttribute("stroke-width", options.box_width);
    canvas.appendChild(box);
  }

  draw_work(canvas, x0, y0, dx, work_event, options) {
    const x1 = x0 + dx;
    const y1 = y0 + options.row_height;

    let work = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    work.setAttribute("x", x0);
    work.setAttribute("y", y0);
    work.setAttribute("width", dx);
    work.setAttribute("height", options.row_height);
    work.setAttribute("fill", options.work_color);
    work.setAttribute("opacity", options.work_opacity);
    canvas.appendChild(work);

    let completed = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "polygon"
    );
    completed.setAttribute(
      "points",
      `${x0},${y1} ${x1},${y1} ${x1},${y1} ${x0},${y1}`
    );
    completed.setAttribute("fill", this.colors[work_event.dispatched_to]);
    canvas.appendChild(completed);

    return [work, completed];
  }
}
