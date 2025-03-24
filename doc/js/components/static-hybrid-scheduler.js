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

class StaticHybridScheduler {
  constructor(workdispatcher) {
    this.work = workdispatcher.works[0];
    this.work_size = workdispatcher.work_sizes[0];
    this.devices = workdispatcher.devices;
    this.batch_sizes = workdispatcher.batch_sizes;
    this.capacities = workdispatcher.capacities;
    this.speeds = workdispatcher.speeds;
  }

  apply_defaults(options) {
    options.dispatch_speed ||= 500; // ms
    options.retrieve_speed ||= 500; // ms
  }

  // Find the next device with the least load
  next(loads) {
    const load_fractions = this.capacities.map((cap, i) =>
      Math.ceil((loads[i] / cap) * 100)
    );
    const min_load = Math.min(...load_fractions);
    return load_fractions.indexOf(min_load);
  }

  // Simulate the work dispatching process
  simulate(options = {}) {
    this.apply_defaults(options);

    let work_events = {};
    let queues = Array.from({ length: this.devices.length }, () => []);
    let time = 0;
    let loads = Array(this.devices.length).fill(0);
    let dev = this.next(loads);
    let queue = queues[dev];
    let next_work_id = 0;
    let next_batch_size = Math.min(this.batch_sizes[dev], this.work_size);
    while (this.work_size > 0) {
      // Dispatch if we can
      if (this.capacities[dev] - loads[dev] >= next_batch_size) {
        let queue = queues[dev];
        // Dispatch a batch of work to the device
        let work_event = {
          size: next_batch_size,
          dispatched_to: dev,
          dispatched_at: time,
          shifted_at: [],
          shifted_by: [],
          process_duration: Math.ceil(next_batch_size / this.speeds[dev]),
        };
        work_event.processed_at = time + options.dispatch_speed;
        work_event.retrieved_at =
          work_event.processed_at + work_event.process_duration;
        this.enqueue(queue, next_work_id, work_event, work_events, options);
        work_events[next_work_id] = work_event;
        // Finish dispatching
        time += options.dispatch_speed;
        this.work_size -= next_batch_size;
        loads[dev] += next_batch_size;
        dev = this.next(loads);
        queue = queues[dev];
        next_work_id++;
        next_batch_size = Math.min(this.batch_sizes[dev], this.work_size);
      } else {
        time = Number.MAX_SAFE_INTEGER;
        for (let next_dev = 0; next_dev < this.devices.length; next_dev++) {
          if (queues[next_dev].length > 0) {
            const first_in_queue = work_events[queues[next_dev][0]];
            const next_time =
              first_in_queue.retrieved_at + options.retrieve_speed;
            if (next_time < time) {
              time = next_time;
              dev = next_dev;
              queue = queues[dev];
            }
          }
        }
        const retrieved = this.dequeue(queue, work_events, options);
        loads[dev] -= retrieved.size;
        dev = this.next(loads);
        next_batch_size = Math.min(this.batch_sizes[dev], this.work_size);
      }
    }

    // Finish queues
    queues.forEach((queue) => {
      while (queue.length > 0) {
        this.dequeue(queue, work_events, options);
      }
    });

    this.consolidate_shifts(work_events);
    return [work_events];
  }

  dequeue(queue, work_events, options) {
    const first_in_queue = work_events[queue.shift()];
    queue.forEach((id) => {
      if (work_events[id].dispatched_at < first_in_queue.dispatched_at) return;
      const shifted_at = Math.max(
        first_in_queue.retrieved_at,
        work_events[id].dispatched_at + options.dispatch_speed
      );
      work_events[id].shifted_at.push(shifted_at);
      work_events[id].shifted_by.push(first_in_queue.size);
    });
    return first_in_queue;
  }

  enqueue(queue, id, next_work_event, work_events) {
    next_work_event.pos0 = 0;
    queue.forEach((i) => {
      next_work_event.pos0 += work_events[i].size;
    });
    if (queue.length === 0) {
      queue.push(id);
      return;
    }
    for (let i = 0; i < queue.length; i++) {
      const work_event = work_events[queue[i]];
      if (next_work_event.retrieved_at < work_event.retrieved_at) {
        queue.splice(i, 0, id);
        return;
      }
    }
    queue.push(id);
  }

  consolidate_shifts(work_events) {
    Object.values(work_events).forEach((work_event) => {
      let shift_by = {};
      work_event.shifted_at.forEach((shifted_at, j) => {
        if (shifted_at in shift_by) {
          shift_by[shifted_at] += work_event.shifted_by[j];
        } else shift_by[shifted_at] = work_event.shifted_by[j];
      });
      work_event.shifted_at = Object.keys(shift_by).map((k) => parseInt(k));
      work_event.shifted_by = Object.values(shift_by);
    });
  }
}
