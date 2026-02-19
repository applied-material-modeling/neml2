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

class StaticHybridSchedulerDemo {
  static init(
    containername = "static-hybrid-scheduler-demo",
    controlsname = "static-hybrid-scheduler-controls"
  ) {
    $(function () {
      $(document).ready(function () {
        let container = document.querySelector(`.${containername}`);
        if (container) {
          const work_dispatcher = new WorkDispatcher(
            /*labels of works*/ ["Work"],
            /*number of batches of work*/ [600],
            /*labels of devices*/ ["CUDA:0", "CUDA:1", "CPU"],
            /*device capacities*/ [150, 180, 50],
            /*device dispatch batch sizes*/ [50, 60, 25],
            /*device compute speed (batch/millisec)*/ [0.01, 0.012, 0.006],
            /*colors of devices*/ [
              "rgb(21,181,255)",
              "rgb(6, 149, 216)",
              "rgb(21,181,0)",
            ]
          );
          // Animate the work dispatching process
          const scheduler = new StaticHybridScheduler(work_dispatcher);
          const [canvas, timeline] = work_dispatcher.animate(scheduler);
          container.appendChild(canvas);
          document.querySelector(`.${controlsname} .play`).onclick =
            timeline.play;
          document.querySelector(`.${controlsname} .pause`).onclick =
            timeline.pause;
          document.querySelector(`.${controlsname} .restart`).onclick =
            timeline.restart;
        }
      });
    });
  }
}
