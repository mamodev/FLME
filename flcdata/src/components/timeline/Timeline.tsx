import { debounce } from "@mui/material";
import { green } from "@mui/material/colors";
import React from "react";
import { draw, precompute } from "./draw";
import { ISimulation, ITimeline } from "./types";

export type TimelineProps = {
  sim: ISimulation;
  timeline: ITimeline;
}

export function Timeline(props: TimelineProps) {
  const ref = React.useRef<HTMLDivElement>(null);

  const { sim, timeline } = props;

  React.useEffect(() => { 
    if (ref.current) {
      const container = ref.current;

      const events = timeline.events;

      function getCanvas(): HTMLCanvasElement {
        let canvas = container.querySelector("canvas");
        if (!canvas) {
          canvas = document.createElement("canvas");
          container.appendChild(canvas);
        }

        return canvas!;
      }

      const canvas = getCanvas();
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      canvas.style.cursor = "move";
      canvas.style.backgroundColor = "white";
      canvas.style.position = "absolute";
      canvas.style.top = "0";
      canvas.style.left = "0";
      canvas.style.zIndex = "1";

      const __precomp = precompute(timeline, sim);

      let range = [0, Math.min(events.length - 1, 500)];

      function setRange(fn: (old: number[]) => number[]) {
        range = fn(range);
        draw(
          canvas,
          timeline,
          sim,
          __precomp,
          range[0],
          range[1]
        );
      }

      draw(
        canvas,
        timeline,
        sim,
        __precomp,
        range[0],
        range[1]
      );

      // add observer for resizing
      const resizeObserver = new ResizeObserver(() => {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        draw(
          canvas,
          timeline,
          sim,
          __precomp,
          range[0],
          range[1]
        );
      });

      // add event listeners for mouse events
      let mouseDown = false;

      let prevX = 0;
      let prevY = 0;

      let __mreminder: number | null = null;

      const handleMouseMove = (event: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // console.log(`Mouse moved to (${x}, ${y})`);
        if (mouseDown) {
          let fdx = x - prevX;
          if (__mreminder != null) {
            fdx += __mreminder;
            __mreminder = null;
          }

          const w_tick = canvas.width / (range[1] - range[0] + 1);

          let dx = Math.round(fdx / w_tick);
          if (dx == 0) {
            __mreminder = fdx;
          } else {
            setRange((range) => {
              if (dx < 0 && range[0] + dx < 0) {
                dx = 0;
              }

              if (dx > 0 && range[1] + dx > events.length - 1) {
                dx = events.length - 1 - range[1];
              }

              let r = [range[0] + dx, range[1] + dx];

              return r;
            });
          }
        }

        prevX = x;
        prevY = y;
      };

      let last_mouse_down = new Date();

      const handleMouseDown = (event: MouseEvent) => {
        console.log("Mouse down", event);

        const now = new Date();
        console.log(now.getTime() - last_mouse_down.getTime());
        if (now.getTime() - last_mouse_down.getTime() < 200) {
          if (range[0] !== 0) {
            setRange(() => [0, events.length - 1]);
          } else {
            const x = event.clientX;

            const rect = canvas.getBoundingClientRect();

            const t = (x * events.length) / rect.width;

            const zoom = events.length * 0.1;

            const skew = t / events.length;
            console.log("sle", skew);

            setRange(() => [
              Math.floor(t - zoom * skew),
              Math.floor(t + zoom * (1 - skew)),
            ]);
          }
        }

        last_mouse_down = now;
        mouseDown = true;
      };

      const handleMouseUp = (event: MouseEvent) => {
        console.log("Mouse up", event);
        mouseDown = false;
        __mreminder = null;
      };

      const handleMouseLeave = (event: MouseEvent) => {
        mouseDown = false;
        __mreminder = null;
      };

      let wheel_delta = 0;
      const zoom_factor = 10;

      const debouncedSetDelta = debounce(() => {
        const d = wheel_delta;
        wheel_delta = 0;

        setRange((range) => {
          const x = prevX;

          const rect = canvas.getBoundingClientRect();

          const time = range[1] - range[0];

          // x : rect.width = t : time
          let t = (x * time) / rect.width;

          const zoom = Math.round(time * (1 - 0.1 * d));
          console.log("zoom", zoom);

          const skew = t / time;

          const zoomRight = Math.floor(zoom * (1 - skew));
          const zoomLeft = Math.floor(zoom * skew);
          t = Math.floor(t) + range[0];
          console.log("time", time);
          console.log("t", t);
          console.log("sle", skew);
          console.log("zoomL", zoomLeft);
          console.log("zoomR", zoomRight);
          console.log("zoomSum", zoomLeft + zoomRight);
          console.log("prev", range);

          const newRange = [
            Math.max(
              Math.floor(t - zoomLeft),
              -Math.floor(events.length * 1.5)
            ),
            Math.min(
              Math.floor(t + zoomRight),
              Math.floor(events.length * 1.5)
            ),
          ];

          console.log("new", newRange);
          // return range
          return newRange;
        });
      }, 10);

      const handleMouseWheel = (event: WheelEvent) => {
        event.preventDefault();
        const delta = Math.sign(event.deltaY);
        wheel_delta += delta;
        debouncedSetDelta();
      };

      canvas.addEventListener("mousemove", handleMouseMove);
      canvas.addEventListener("mousedown", handleMouseDown);
      canvas.addEventListener("mouseup", handleMouseUp);
      canvas.addEventListener("wheel", handleMouseWheel);
      canvas.addEventListener("mouseleave", handleMouseLeave);
      resizeObserver.observe(container);

      return () => {
        canvas.removeEventListener("mousemove", handleMouseMove);
        canvas.removeEventListener("mousedown", handleMouseDown);
        canvas.removeEventListener("mouseup", handleMouseUp);
        canvas.removeEventListener("wheel", handleMouseWheel);
        canvas.removeEventListener("mouseleave", handleMouseLeave);
        resizeObserver.unobserve(container);
        resizeObserver.disconnect();
      };
    }
  }, [draw, sim, timeline]);

  return (
    <div
      ref={ref}
      style={{
        position: "relative",
        backgroundColor: green[50],
        flex: 1,
      }}
    >
      &nbsp;
   
    </div>
  );
}
