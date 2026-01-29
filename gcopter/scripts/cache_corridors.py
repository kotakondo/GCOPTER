#!/usr/bin/env python3
"""
Phase-1: compute global route + safe corridor once, save to disk, then exit.

This keeps base.launch.py (mockamap + rviz) alive, and repeatedly runs
benchmark.launch.py with corridorMode:=1 to export only.

Usage example:
  ./cache_corridors.py \
    --out_root /tmp/corridors \
    --n_runs 100 \
    --start "[0.0, 1.0, 1.0]" \
    --goal "[-7.0, -7.0, 1.0]" \
    --max_vel 1.0
"""
import argparse, os, time, signal, subprocess

def start_proc(cmd, name):
    print(f"[cache] starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)

def kill_proc_group(p, name, grace=5.0):
    if p is None:
        return
    try:
        pgid = os.getpgid(p.pid)
    except Exception:
        return
    try:
        print(f"[cache] stopping {name} (SIGINT pgid {pgid})")
        os.killpg(pgid, signal.SIGINT)
        p.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGTERM)
        p.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    finally:
        try:
            p.wait(timeout=2)
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/media/kkondo/T7/mighty_gcopter_bench/simple_bench/corridor_cache", help="Directory to store corridor files")
    ap.add_argument("--n_runs", type=int, default=100, help="Number of repeated exports (goal_XXX naming)")
    ap.add_argument("--sleep_after_base", type=float, default=1.0)
    ap.add_argument("--case_timeout", type=float, default=30.0)
    ap.add_argument("--start", default="[0.0, 1.0, 1.0]")
    ap.add_argument("--goal", default="[-7.0, -7.0, 1.0]")
    ap.add_argument("--max_vel", type=float, default=1.0)

    ap.add_argument("--pkg_base", default="gcopter")
    ap.add_argument("--base_launch", default="base.launch.py")
    ap.add_argument("--pkg_node", default="gcopter")
    ap.add_argument("--node_launch", default="benchmark.launch.py")

    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    base_cmd = ["ros2", "launch", args.pkg_base, args.base_launch, "use_simple_case_benchmark:=true"]
    base = start_proc(base_cmd, "base.launch")

    try:
        time.sleep(args.sleep_after_base)

        for run_idx in range(args.n_runs):
            out_dir = os.path.join(args.out_root, f"goal_{run_idx:03d}")
            os.makedirs(out_dir, exist_ok=True)
            corridor_file = os.path.join(out_dir, "corridor_cache.bin")

            node_cmd = [
                "ros2", "launch", args.pkg_node, args.node_launch,
                f"start:={args.start}",
                f"goal:={args.goal}",
                f"max_vel:={args.max_vel}",
                # new args you add to minco_bench_viz:
                "corridorMode:=1",  # 1=compute+save+exit
                f"corridorCacheFile:={corridor_file}",
                # don't export planner CSV in phase 1
                f"out_csv:={out_dir}",
                "do_benchmark:=true",
            ]
            node = start_proc(node_cmd, f"export[{run_idx:03d}]")
            t0 = time.time()
            try:
                rc = node.wait(timeout=args.case_timeout) if args.case_timeout > 0 else node.wait()
            except subprocess.TimeoutExpired:
                print(f"[cache] export[{run_idx:03d}] timed out; killing")
                kill_proc_group(node, f"export[{run_idx:03d}]")
                rc = -9
            print(f"[cache] done export[{run_idx:03d}] rc={rc} wall={time.time()-t0:.2f}s")

            if rc != 0:
                print("[cache] non-zero rc, aborting remaining exports.")
                break

    except KeyboardInterrupt:
        print("\n[cache] Ctrl-C — shutting down.")
    finally:
        kill_proc_group(base, "base.launch")

if __name__ == "__main__":
    main()