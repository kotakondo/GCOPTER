#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import time
import signal

# ---------- small utils ----------
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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Run the SAME start/goal multiple times (e.g., 100) and save outputs just like bench_sweep.py."
    )
    # output / run controls
    ap.add_argument("--out_root", default="/media/kkondo/lucas_pro/mighty_gcopter_bench/simple_bench",
                    help="Root directory for cases (same default as bench_sweep.py)")
    ap.add_argument("--n_runs", type=int, default=2, help="How many repeated runs to execute")
    ap.add_argument("--sleep_after_base", type=float, default=3.0, help="Seconds to wait after starting base.launch")
    ap.add_argument("--case_timeout", type=float, default=10.0,
                    help="(Unused, kept for parity) Seconds to wait for each run before aborting")

    # fixed parameters (no sweeping)
    ap.add_argument("--max_vel", type=float, default=4.0, help="Fixed max velocity to pass to the benchmark")
    ap.add_argument("--mighty_jerk_weight", type=str, default="1e0",
                    help="Fixed jerk weight to pass to the benchmark (string like 1e-3)")

    # start / goal
    ap.add_argument("--start", default="[0.0, 1.0, 1.0]")
    ap.add_argument("--goal",  default="[-10.0, -10.0, 1.0]")

    # sampling/col-check settings
    ap.add_argument("--sample_dt", default="0.01")
    ap.add_argument("--collision_dt", default="0.01")

    # ROS launch bits
    ap.add_argument("--pkg_base", default="gcopter")
    ap.add_argument("--base_launch", default="base.launch.py")   # keeps rviz + mockamap alive
    ap.add_argument("--pkg_node", default="gcopter")
    ap.add_argument("--node_launch", default="benchmark.launch.py")  # runs benchmark once

    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # Single 'case' directory (keeps the same naming shape as bench_sweep.py)
    case_dir = os.path.join(
        args.out_root,
        f"case_{0:05d}_v{args.max_vel:.2f}_jw{args.mighty_jerk_weight}"
    )
    os.makedirs(case_dir, exist_ok=True)

    print(f"[bench] Repeated-run case dir: {case_dir}")
    print(f"[bench] start={args.start}  goal={args.goal}")
    print(f"[bench] max_vel={args.max_vel}  mighty_jerk_weight={args.mighty_jerk_weight}")
    print(f"[bench] n_runs={args.n_runs}")

    # Start base.launch.py (RViz + mockamap) and keep it alive
    print("[bench] Starting base.launch.py…")
    base = subprocess.Popen([
        "ros2", "launch", args.pkg_base, args.base_launch, "use_simple_case_benchmark:=true",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # topics matched to bench_sweep.py
    bag_topics_to_record = [
        "/tf", "/tf_static", "/initialpose",
        "/visualizer/mesh", "/visualizer/edge", "/visualizer/route",
        "/visualizer/spheres", "/visualizer/waypoints", "/visualizer/trajectory",
        "visualizer/mighty/spheres", "visualizer/mighty/waypoints", "visualizer/mighty/trajectory"
    ]

    try:
        time.sleep(args.sleep_after_base)

        # One rosbag per 'case', same as bench_sweep.py
        bag_out = os.path.join(case_dir, "bag")
        bag_cmd = ["ros2", "bag", "record", "-o", bag_out] + bag_topics_to_record
        bag = subprocess.Popen(bag_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for run_idx in range(100, 100 + args.n_runs):
            goal_str = args.goal
            start_str = args.start

            print(f"[bench] run {run_idx+1}/{args.n_runs}: goal: {goal_str} | start: {start_str}")

            # For parity with bench_sweep.py, each run writes into goal_XXX/
            case_goal_dir = os.path.join(case_dir, f"goal_{run_idx:03d}")

            # Run the compute launch (one benchmark per run)
            node_cmd = [
                "ros2", "launch", args.pkg_node, args.node_launch,
                f"start:={start_str}",
                f"goal:={goal_str}",
                f"max_vel:={args.max_vel}",
                f"mighty_jerk_weight:={args.mighty_jerk_weight}",
                f"sample_dt:={args.sample_dt}",
                f"collision_dt:={args.collision_dt}",
                f"out_csv:={case_goal_dir}",
                "do_benchmark:=true",  # ensure benchmarking mode
            ]
            t0 = time.time()
            rc = subprocess.run(node_cmd, check=False).returncode
            wall = time.time() - t0
            print(f"[bench] done run {run_idx:03d} | wall {wall:.2f}s | rc={rc}")

        # stop bag for this case
        safe_kill(bag)

    except KeyboardInterrupt:
        print("\n[bench] Ctrl-C — shutting down.")
    finally:
        safe_kill(base)
        print("[bench] base.launch stopped.")

if __name__ == "__main__":
    main()
