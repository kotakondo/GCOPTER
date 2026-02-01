#!/usr/bin/env python3
"""
Phase-2: run trajectory planners (GCOPTER + MIGHTY) using a previously saved corridor.

This script expects corridor_cache.bin files produced by cache_corridors.py.

Usage example:
  ./bench_from_corridors.py \
    --corridor_root /tmp/corridors \
    --n_runs 100 \
    --max_vel 1.0 \
    --mighty_jerk_weight 1e-3
"""
import argparse, os, glob, time, signal, subprocess

def start_proc(cmd, name):
    print(f"[bench2] starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=None, stderr=None, start_new_session=True)

def kill_proc_group(p, name, grace=5.0):
    if p is None:
        return
    try:
        pgid = os.getpgid(p.pid)
    except Exception:
        return
    try:
        print(f"[bench2] stopping {name} (SIGINT pgid {pgid})")
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
    ap.add_argument("--corridor_root", default="/media/kkondo/T7/mighty_gcopter_bench/simple_bench/corridor_cache", help="Root dir containing goal_XXX/corridor_cache.bin")
    ap.add_argument("--out_root", default="/media/kkondo/T7/mighty_gcopter_bench/simple_bench", help="Root dir for benchmark outputs")
    ap.add_argument("--n_runs", type=int, default=100, help="How many goal_XXX to run (starting from 0)")
    ap.add_argument("--sleep_after_base", type=float, default=1.0)
    ap.add_argument("--case_timeout", type=float, default=60.0)
    ap.add_argument("--use_cpp_batch", action="store_true",
                    help="Run all goals in one C++ loop (corridorMode=3) per weight")
    ap.add_argument("--goal_start", type=int, default=0, help="Start index for goal_XXX")

    ap.add_argument("--max_vel", type=float, default=1.0)
    ap.add_argument("--mighty_jerk_weight", type=float, default=1e-3)
    ap.add_argument("--sample_dt", default="0.01")
    ap.add_argument("--collision_dt", default="0.01")
    ap.add_argument("--vel_ref", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="Velocity reference vector (x y z)")
    ap.add_argument("--vel_ref_weight", type=float, default=0.0,
                    help="Soft vref weight used in soft/freeze modes")
    ap.add_argument("--vel_ref_knot", type=int, default=-1,
                    help="Knot index (<=0 => auto penultimate per case)")
    ap.add_argument("--vel_ref_log_every", type=int, default=0,
                    help="Per-iter logging period (0 disables)")
    ap.add_argument("--vel_ref_grad_check", action="store_true",
                    help="Enable one-shot FD check (prints once per case)")
    ap.add_argument("--pos_ref", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="Position reference vector (x y z)")
    ap.add_argument("--pos_ref_weight", type=float, default=0.0,
                    help="Soft pref weight used in soft/freeze modes")
    ap.add_argument("--pos_ref_knot", type=int, default=-1,
                    help="Knot index (<=0 => auto penultimate per case)")
    ap.add_argument("--pos_ref_log_every", type=int, default=0,
                    help="Per-iter logging period (0 disables)")
    ap.add_argument("--pos_ref_grad_check", action="store_true",
                    help="Enable one-shot FD check for position ref (prints once per case)")
    ap.add_argument("--pos_ref_enable", action="store_true",
                    help="Enable position reference term (only in soft/freeze modes)")
    ap.add_argument("--skip_freeze", action="store_true",
                    help="Skip the MIGHTY-freeze mode (run baseline + soft only)")
    ap.add_argument("--skip_baseline", action="store_true",
                    help="Skip the baseline mode (run soft only, or soft+freeze)")
    ap.add_argument("--no_sweep_mighty_jerk_weight", action="store_true", help="Disable sweep and use --mighty_jerk_weight")
    ap.add_argument("--sweep_min", type=float, default=1e-5, help="Sweep min value (log10 space)")
    ap.add_argument("--sweep_max", type=float, default=1e+3, help="Sweep max value (log10 space)")
    ap.add_argument("--sweep_steps", type=int, default=9, help="Number of sweep points (log-spaced)")

    ap.add_argument("--pkg_base", default="gcopter")
    ap.add_argument("--base_launch", default="base.launch.py")
    ap.add_argument("--pkg_node", default="gcopter")
    ap.add_argument("--node_launch", default="benchmark.launch.py")

    args = ap.parse_args()

    def fmt_weight(w):
        s = f"{w:.0e}"
        if "e" in s:
            base, exp = s.split("e")
            sign = exp[0]
            exp_val = exp[1:].lstrip("0") or "0"
            s = f"{base}e{sign}{exp_val}"
        return s

    if args.no_sweep_mighty_jerk_weight:
        weights = [args.mighty_jerk_weight]
    else:
        if args.sweep_steps <= 1:
            weights = [args.sweep_min]
        else:
            ratio = (args.sweep_max / args.sweep_min) ** (1.0 / (args.sweep_steps - 1))
            weights = [args.sweep_min * (ratio ** i) for i in range(args.sweep_steps)]

    base_cmd = ["ros2", "launch", args.pkg_base, args.base_launch, "use_simple_case_benchmark:=true"]
    base = start_proc(base_cmd, "base.launch")

    try:
        time.sleep(args.sleep_after_base)

        for w in weights:
            w_str = fmt_weight(w)
            print(f"[bench2] sweeping mighty_jerk_weight={w_str}")

            weight_root = os.path.join(args.out_root, f"weight_{w_str}")
            os.makedirs(weight_root, exist_ok=True)

            # Three modes per case
            modes = [
                ("baseline", False, False),
                ("soft", True, False),
                ("freeze", True, True),
            ]
            if args.skip_freeze:
                modes = [m for m in modes if m[0] != "freeze"]
            if args.skip_baseline:
                modes = [m for m in modes if m[0] != "baseline"]

            if args.use_cpp_batch:
                for mode_name, vref_enable, freeze_enable in modes:
                    pref_enable = args.pos_ref_enable and vref_enable
                    mode_root = os.path.join(weight_root, mode_name)
                    os.makedirs(mode_root, exist_ok=True)
                    node_cmd = [
                        "ros2", "launch", args.pkg_node, args.node_launch,
                        f"max_vel:={args.max_vel}",
                        f"mighty_jerk_weight:={w}",
                        f"sample_dt:={args.sample_dt}",
                        f"collision_dt:={args.collision_dt}",
                        "corridorMode:=3",  # 3=batch load+run
                        f"corridorBatchRoot:={args.corridor_root}",
                        f"corridorBatchOutRoot:={mode_root}",
                        f"corridorBatchStart:={args.goal_start}",
                        f"corridorBatchCount:={args.n_runs}",
                        "corridorBatchGoalWidth:=3",
                        "corridorNoMap:=true",
                        "do_benchmark:=true",
                        f"vel_ref_enable:={'true' if vref_enable else 'false'}",
                        f"vel_ref_knot:={args.vel_ref_knot}",
                        f"vel_ref:=[{args.vel_ref[0]}, {args.vel_ref[1]}, {args.vel_ref[2]}]",
                        f"vel_ref_weight:={args.vel_ref_weight}",
                        f"mighty_freeze_enable:={'true' if freeze_enable else 'false'}",
                        f"vel_ref_grad_check:={'true' if args.vel_ref_grad_check else 'false'}",
                        f"vel_ref_log_every:={args.vel_ref_log_every}",
                        f"pos_ref_enable:={'true' if pref_enable else 'false'}",
                        f"pos_ref_knot:={args.pos_ref_knot}",
                        f"pos_ref:=[{args.pos_ref[0]}, {args.pos_ref[1]}, {args.pos_ref[2]}]",
                        f"pos_ref_weight:={args.pos_ref_weight}",
                        f"pos_ref_grad_check:={'true' if args.pos_ref_grad_check else 'false'}",
                        f"pos_ref_log_every:={args.pos_ref_log_every}",
                    ]
                    node = start_proc(node_cmd, f"bench_batch[{w_str}][{mode_name}]")
                    t0 = time.time()
                    try:
                        if args.case_timeout > 0:
                            total_timeout = args.case_timeout * max(1, args.n_runs)
                            rc = node.wait(timeout=total_timeout)
                        else:
                            rc = node.wait()
                    except subprocess.TimeoutExpired:
                        print(f"[bench2] bench_batch[{w_str}][{mode_name}] timed out; killing")
                        kill_proc_group(node, f"bench_batch[{w_str}][{mode_name}]")
                        rc = -9
                    print(f"[bench2] done bench_batch[{w_str}][{mode_name}] rc={rc} wall={time.time()-t0:.2f}s")

                    if rc != 0:
                        print("[bench2] non-zero rc, aborting remaining runs.")
                        break
            else:
                for run_idx in range(args.goal_start, args.goal_start + args.n_runs):
                    corridor_goal_dir = os.path.join(args.corridor_root, f"goal_{run_idx:03d}")
                    corridor_file = os.path.join(corridor_goal_dir, "corridor_cache.bin")
                    if not os.path.exists(corridor_file):
                        print(f"[bench2] missing {corridor_file}, stopping.")
                        break

                    for mode_name, vref_enable, freeze_enable in modes:
                        pref_enable = args.pos_ref_enable and vref_enable
                        out_dir = os.path.join(weight_root, mode_name, f"goal_{run_idx:03d}")
                        os.makedirs(out_dir, exist_ok=True)

                        node_cmd = [
                            "ros2", "launch", args.pkg_node, args.node_launch,
                            # these args still matter for dynamics/weights, even though start/goal come from file
                            f"max_vel:={args.max_vel}",
                            f"mighty_jerk_weight:={w}",
                            f"sample_dt:={args.sample_dt}",
                            f"collision_dt:={args.collision_dt}",
                            # new args:
                            "corridorMode:=2",  # 2=load+run
                            f"corridorCacheFile:={corridor_file}",
                            # export into the weight-specific goal_XXX directory:
                            f"out_csv:={out_dir}",
                            "corridorNoMap:=true",
                            "do_benchmark:=true",
                            f"vel_ref_enable:={'true' if vref_enable else 'false'}",
                            f"vel_ref_knot:={args.vel_ref_knot}",
                            f"vel_ref:=[{args.vel_ref[0]}, {args.vel_ref[1]}, {args.vel_ref[2]}]",
                            f"vel_ref_weight:={args.vel_ref_weight}",
                            f"mighty_freeze_enable:={'true' if freeze_enable else 'false'}",
                            f"vel_ref_grad_check:={'true' if args.vel_ref_grad_check else 'false'}",
                            f"vel_ref_log_every:={args.vel_ref_log_every}",
                            f"pos_ref_enable:={'true' if pref_enable else 'false'}",
                            f"pos_ref_knot:={args.pos_ref_knot}",
                            f"pos_ref:=[{args.pos_ref[0]}, {args.pos_ref[1]}, {args.pos_ref[2]}]",
                            f"pos_ref_weight:={args.pos_ref_weight}",
                            f"pos_ref_grad_check:={'true' if args.pos_ref_grad_check else 'false'}",
                            f"pos_ref_log_every:={args.pos_ref_log_every}",
                        ]
                        node = start_proc(node_cmd, f"bench[{w_str}][{mode_name}][{run_idx:03d}]")
                        t0 = time.time()
                        try:
                            rc = node.wait(timeout=args.case_timeout) if args.case_timeout > 0 else node.wait()
                        except subprocess.TimeoutExpired:
                            print(f"[bench2] bench[{w_str}][{mode_name}][{run_idx:03d}] timed out; killing")
                            kill_proc_group(node, f"bench[{w_str}][{mode_name}][{run_idx:03d}]")
                            rc = -9
                        print(f"[bench2] done bench[{w_str}][{mode_name}][{run_idx:03d}] rc={rc} wall={time.time()-t0:.2f}s")

                        if rc != 0:
                            print("[bench2] non-zero rc, aborting remaining runs.")
                            break

    except KeyboardInterrupt:
        print("\n[bench2] Ctrl-C — shutting down.")
    finally:
        kill_proc_group(base, "base.launch")

if __name__ == "__main__":
    main()
