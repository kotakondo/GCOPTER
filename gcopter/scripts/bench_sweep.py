#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import re
import math
import time
import ast
import signal
from datetime import datetime

# ---------- small utils ----------
def frange(start, stop, step):
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        yield round(start + i * step, 6)

def safe_kill(p):
    if p is None:
        return
    try:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    except Exception:
        pass

_METRIC_LINE = re.compile(
    r'^\s*(solve \[ms\]|time \[s\]|path len \[m\]|jerk_cost|collisions)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)'
)

def parse_metrics(stdout_text):
    out = {
        "solve_ms_gc": None, "solve_ms_my": None,
        "time_s_gc": None,  "time_s_my": None,
        "path_len_gc": None, "path_len_my": None,
        "jerk_gc": None, "jerk_my": None,
        "collisions_gc": None, "collisions_my": None,
    }
    in_block = False
    for line in stdout_text.splitlines():
        if "==== Benchmark ====" in line:
            in_block = True
            continue
        if "===================" in line:
            in_block = False
            continue
        if not in_block:
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        metric, a, b, _delta = m.groups()
        a = float(a); b = float(b)
        if metric.startswith("solve [ms]"):
            out["solve_ms_gc"], out["solve_ms_my"] = a, b
        elif metric.startswith("time [s]"):
            out["time_s_gc"], out["time_s_my"] = a, b
        elif metric.startswith("path len [m]"):
            out["path_len_gc"], out["path_len_my"] = a, b
        elif metric.startswith("jerk_cost"):
            out["jerk_gc"], out["jerk_my"] = a, b
        elif metric.startswith("collisions"):
            out["collisions_gc"], out["collisions_my"] = int(a), int(b)
    have_any = any(v is not None for v in out.values())
    return out if have_any else None

def write_case_metrics_csv(case_csv_path, header, row):
    new_file = not os.path.exists(case_csv_path)
    with open(case_csv_path, "a") as f:
        if new_file:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")

# ---------- perimeter goals ----------
import math

def _linspace_inc(a, b, step):
    """Inclusive samples from a to b with |step| spacing, works both directions."""
    n = max(1, int(round(abs((b - a) / step))))
    s = step if b >= a else -abs(step)
    xs = [a + i * s for i in range(n)]
    if (s >= 0 and xs[-1] < b - 1e-12) or (s < 0 and xs[-1] > b + 1e-12):
        xs.append(b)
    else:
        xs[-1] = b  # snap
    return xs

def _same_xy(p, q, tol=1e-9):
    return abs(p[0]-q[0]) <= tol and abs(p[1]-q[1]) <= tol

def generate_perimeter_goals(x_min, x_max, y_min, y_max, z_goal, step,
                             start_corner="SW", clockwise=True, closed=False):
    """
    Return list of [x,y,z_goal] around rectangle perimeter with spacing 'step'.
    Corners are NOT duplicated across edges. If closed=True the start corner
    is appended at the end; otherwise it is not.
    """
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("x_min < x_max and y_min < y_max required")
    if step <= 0:
        raise ValueError("step must be > 0")

    def add_pts(xs, ys, fixed_axis, fixed_val, skip_first=False):
        pts = []
        if fixed_axis == 'y':
            for i, xv in enumerate(xs):
                if skip_first and i == 0:
                    continue
                pts.append([xv, fixed_val, z_goal])
        else:
            for i, yv in enumerate(ys):
                if skip_first and i == 0:
                    continue
                pts.append([fixed_val, yv, z_goal])
        return pts

    if clockwise:
        bottom_x = _linspace_inc(x_min, x_max, step)   # include both ends
        right_y  = _linspace_inc(y_min, y_max, step)
        top_x    = _linspace_inc(x_max, x_min, step)   # backwards
        left_y   = _linspace_inc(y_max, y_min, step)   # backwards
        perim = []
        perim += add_pts(bottom_x, None, 'y', y_min, skip_first=False)
        perim += add_pts(None, right_y, 'x', x_max,  skip_first=True)   # skip BR
        perim += add_pts(top_x,   None, 'y', y_max,  skip_first=True)   # skip TR
        perim += add_pts(None, left_y,  'x', x_min,  skip_first=True)   # skip TL
    else:
        left_y   = _linspace_inc(y_min, y_max, step)
        top_x    = _linspace_inc(x_min, x_max, step)
        right_y  = _linspace_inc(y_max, y_min, step)
        bottom_x = _linspace_inc(x_max, x_min, step)
        perim = []
        perim += add_pts(None, left_y,  'x', x_min,  skip_first=False)
        perim += add_pts(top_x,   None, 'y', y_max,  skip_first=True)
        perim += add_pts(None, right_y, 'x', x_max,  skip_first=True)
        perim += add_pts(bottom_x, None, 'y', y_min, skip_first=True)

    # rotate to requested start corner
    corners = {"SW": (x_min, y_min), "SE": (x_max, y_min),
               "NE": (x_max, y_max), "NW": (x_min, y_max)}
    if start_corner not in corners:
        raise ValueError("start_corner must be one of SW/SE/NE/NW")
    sx, sy = corners[start_corner]
    start_idx = next((i for i, (x, y, _) in enumerate(perim)
                      if abs(x - sx) < 1e-9 and abs(y - sy) < 1e-9), 0)
    perim = perim[start_idx:] + perim[:start_idx]

    # remove wrap-around duplicate unless a closed loop is requested
    if not closed and len(perim) > 1 and _same_xy(perim[0], perim[-1]):
        perim.pop()

    # tidy small FP noise
    for p in perim:
        p[0] = round(p[0], 3)
        p[1] = round(p[1], 3)
        p[2] = round(p[2], 3)
    return perim

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Sweep minco_bench_viz over rectangle-perimeter goals, while base.launch (RViz+map) stays alive."
    )
    # sweep controls
    # ap.add_argument("--out_root", default="/home/kkondo/data/gcopter_csv", help="Root directory for cases")
    ap.add_argument("--out_root", default="/media/kkondo/lucas_pro/mighty_gcopter_bench/sweep_bench", help="Root directory for cases")
    ap.add_argument("--v_min", type=float, default=1.0)
    ap.add_argument("--v_max", type=float, default=5.0)
    ap.add_argument("--v_step", type=float, default=1.0)
    ap.add_argument("--jerk_dec_min", type=int, default=-3)
    ap.add_argument("--jerk_dec_max", type=int, default=-2)

    # perimeter goals
    ap.add_argument("--x_min", type=float, default=-15.0)
    ap.add_argument("--x_max", type=float, default= 15.0)
    ap.add_argument("--y_min", type=float, default=-15.0)
    ap.add_argument("--y_max", type=float, default= 15.0)
    ap.add_argument("--z_goal", type=float, default=2.5)
    ap.add_argument("--perim_step", type=float, default=5.0)
    ap.add_argument("--start_corner", choices=["SW","SE","NE","NW"], default="SW")
    ap.add_argument("--clockwise", action="store_true", help="Traverse perimeter clockwise (default: CCW)")

    # ROS launch bits
    ap.add_argument("--pkg_base", default="gcopter")
    ap.add_argument("--base_launch", default="base.launch.py")   # keeps rviz + mockamap alive
    ap.add_argument("--pkg_node", default="gcopter")
    ap.add_argument("--node_launch", default="benchmark.launch.py")  # runs benchmark once
    ap.add_argument("--start", default="[0.0, 0.0, 0.5]")
    ap.add_argument("--sample_dt", default="0.01")
    ap.add_argument("--collision_dt", default="0.01")
    ap.add_argument("--sleep_after_base", type=float, default=3.0)
    ap.add_argument("--case_timeout", type=float, default=10.0, help="seconds to wait for each run before aborting")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # Build perimeter goal list
    goals = generate_perimeter_goals(
        args.x_min, args.x_max, args.y_min, args.y_max,
        args.z_goal, args.perim_step,
        start_corner=args.start_corner,
        clockwise=args.clockwise
    )
    if not goals:
        print("[bench] No goals generated from perimeter; check inputs.")
        sys.exit(1)

    print(f"[bench] Generated {len(goals)} perimeter goals:")
    for i, g in enumerate(goals):
        print(f"  {i+1:02d}: {g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}")


    # Prepare sweep grids
    v_vals  = list(frange(args.v_min, args.v_max, args.v_step))
    jw_vals = [f"1e{e}" for e in range(args.jerk_dec_min, args.jerk_dec_max)]
    print(f"[bench] V sweep: {v_vals}")
    print(f"[bench] Jerk weight sweep: {jw_vals}")

    # Start base.launch.py (RViz + mockamap) and keep it alive
    print("[bench] Starting base.launch.py…")
    base = subprocess.Popen([
        "ros2", "launch", args.pkg_base, args.base_launch,
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    bag_topics_to_record = ["/tf", "/tf_static", "/initialpose", "/visualizer/mesh", "/visualizer/edge", "/visualizer/route", "/visualizer/spheres", "/visualizer/waypoints", "/visualizer/trajectory", "visualizer/mighty/spheres", "visualizer/mighty/waypoints", "visualizer/mighty/trajectory"]

    try:
        time.sleep(args.sleep_after_base)

        trial = 0
        for v in v_vals:
            for jw in jw_vals:

                case_dir = os.path.join(
                    args.out_root,
                    f"case_{trial:05d}_v{v:.2f}_jw{jw}"
                )
                os.makedirs(case_dir, exist_ok=True)

                print(f"\n[bench] === case {trial} | v={v:.2f} jw={jw} ===")

                # Start rosbag for this case
                bag_out = os.path.join(case_dir, "bag")
                bag_cmd = ["ros2", "bag", "record", "-o", bag_out] + bag_topics_to_record
                bag = subprocess.Popen(bag_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                for gi, goal in enumerate(goals):

                    goal_str = f"[{goal[0]}, {goal[1]}, {goal[2]}]"
                    start_str = args.start

                    print(f"[bench] goal {gi+1}/{len(goals)}: {goal_str} | start: {start_str}")

                    case_goal_dir = os.path.join(case_dir, f"goal_{gi:03d}")

                    # Run the compute launch (minco_bench_viz once)
                    node_cmd = [
                        "ros2", "launch", args.pkg_node, args.node_launch,
                        f"start:={start_str}",
                        f"goal:={goal_str}",
                        f"max_vel:={v}",
                        f"mighty_jerk_weight:={jw}",
                        f"sample_dt:={args.sample_dt}",
                        f"collision_dt:={args.collision_dt}",
                        f"out_csv:={case_goal_dir}",
                        "do_benchmark:=true",  # ensure benchmarking mode
                    ]
                    t0 = time.time()
                    rc = subprocess.run(node_cmd, check=False).returncode   # <-- just wait
                    duration = time.time() - t0

                    print(f"[bench] done case {trial} | wall {duration:.2f}s | rc={rc}")
                    trial += 1

                 # stop bag for this case
                safe_kill(bag)

    except KeyboardInterrupt:
        print("\n[bench] Ctrl-C — shutting down.")
    finally:
        safe_kill(base)
        print("[bench] base.launch stopped.")

if __name__ == "__main__":
    main()
