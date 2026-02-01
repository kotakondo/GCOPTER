#!/usr/bin/env python3
"""
Debug runner for velocity-reference experiments on a cached corridor.

Example:
  ./debug_ref_vel.py --goal_idx 3 --vel_ref_knot 4 --vel_ref_weight 1000 --vel_ref_grad_check
"""
import argparse
import os
import re
import signal
import subprocess
import time


def start_proc(cmd, name):
    print(f"[debug_ref_vel] starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)


def kill_proc_group(p, name, grace=5.0):
    if p is None:
        return
    try:
        pgid = os.getpgid(p.pid)
    except Exception:
        return
    try:
        print(f"[debug_ref_vel] stopping {name} (SIGINT pgid {pgid})")
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
    ap.add_argument("--corridor_root", default="/media/kkondo/T7/mighty_gcopter_bench/simple_bench/corridor_cache",
                    help="Root dir with goal_XXX/corridor_cache.bin")
    ap.add_argument("--goal_idx", type=int, default=0, help="Single-case index")
    ap.add_argument("--goal_start", type=int, default=0, help="Batch start index")
    ap.add_argument("--goal_count", type=int, default=1, help="Batch count (<=1 means single run)")
    ap.add_argument("--goal_width", type=int, default=3)
    ap.add_argument("--corridor_file", default="", help="Override full path to corridor_cache.bin")

    ap.add_argument("--vel_ref_enable", action="store_true")
    ap.add_argument("--vel_ref_knot", type=int, default=-1)
    ap.add_argument("--vel_ref", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--vel_ref_weight", type=float, default=0.0)
    ap.add_argument("--mighty_freeze_enable", action="store_true")
    ap.add_argument("--vel_ref_grad_check", action="store_true")
    ap.add_argument("--vel_ref_log_every", type=int, default=0)
    ap.add_argument("--pos_ref_enable", action="store_true")
    ap.add_argument("--pos_ref_knot", type=int, default=-1)
    ap.add_argument("--pos_ref_weight", type=float, default=0.0)
    ap.add_argument("--pos_ref_grad_check", action="store_true")
    ap.add_argument("--pos_ref_log_every", type=int, default=0)
    ap.add_argument("--rel_tol", type=float, default=1e-4, help="Relative error threshold for FD checks")
    ap.add_argument("--print_node_output", action="store_true",
                    help="Print full node output (debugging)")
    ap.add_argument("--case_timeout", type=float, default=120.0,
                    help="Per-case timeout in seconds (batch mode)")

    ap.add_argument("--max_vel", type=float, default=1.0)
    ap.add_argument("--mighty_jerk_weight", type=float, default=1e-3)
    ap.add_argument("--sample_dt", default="0.01")
    ap.add_argument("--collision_dt", default="0.01")
    ap.add_argument("--use_scaled_cost", default="true")
    ap.add_argument("--opt_timeout_ms", default="-1.0")

    ap.add_argument("--start_base", action="store_true",
                    help="Also start base.launch.py (mock map + rviz). Not required for corridor mode.")
    ap.add_argument("--base_launch", default="base.launch.py")
    ap.add_argument("--node_launch", default="debug_ref_vel.launch.py")

    args = ap.parse_args()

    def parse_grad_rels(output_text):
        rels = {
            "gcopter": {"vref": [], "pref": [], "full_dir": None, "full_coord": None},
            "mighty": {"vref": [], "pref": [], "full_dir": None, "full_coord": None},
        }
        current = None
        current_term = None
        for line in output_text.splitlines():
            if "[GCOPTER][vref-grad]" in line:
                current = "gcopter"
                current_term = "vref"
                continue
            if "[GCOPTER][pref-grad]" in line:
                current = "gcopter"
                current_term = "pref"
                continue
            if "[MIGHTY][vref-grad]" in line:
                current = "mighty"
                current_term = "vref"
                continue
            if "[MIGHTY][pref-grad]" in line:
                current = "mighty"
                current_term = "pref"
                continue
            if "[GCOPTER][full-grad]" in line:
                m1 = re.search(r"max_rel_dir=([0-9eE+\\.-]+)", line)
                m2 = re.search(r"max_rel_coord=([0-9eE+\\.-]+)", line)
                if m1:
                    rels["gcopter"]["full_dir"] = float(m1.group(1))
                if m2:
                    rels["gcopter"]["full_coord"] = float(m2.group(1))
                continue
            if "[MIGHTY][full-grad]" in line:
                m1 = re.search(r"max_rel_dir=([0-9eE+\\.-]+)", line)
                m2 = re.search(r"max_rel_coord=([0-9eE+\\.-]+)", line)
                if m1:
                    rels["mighty"]["full_dir"] = float(m1.group(1))
                if m2:
                    rels["mighty"]["full_coord"] = float(m2.group(1))
                continue
            if current is None or current_term is None:
                continue
            m = re.search(r"rel=([0-9eE+\\.-]+)", line)
            if m:
                try:
                    rels[current][current_term].append(float(m.group(1)))
                except ValueError:
                    pass
        return rels

    def run_one_case(corridor_file, goal_tag):
        vel_ref_str = f"[{args.vel_ref[0]}, {args.vel_ref[1]}, {args.vel_ref[2]}]"
        node_cmd = [
            "ros2", "launch", "gcopter", args.node_launch,
            f"corridor_cache:={corridor_file}",
            "corridor_no_map:=true",
            "do_benchmark:=true",
            f"max_vel:={args.max_vel}",
            f"mighty_jerk_weight:={args.mighty_jerk_weight}",
            f"sample_dt:={args.sample_dt}",
            f"collision_dt:={args.collision_dt}",
            f"use_scaled_cost:={args.use_scaled_cost}",
            f"opt_timeout_ms:={args.opt_timeout_ms}",
            f"vel_ref_enable:={'true' if args.vel_ref_enable else 'false'}",
            f"vel_ref_knot:={args.vel_ref_knot}",
            f"vel_ref:={vel_ref_str}",
            f"vel_ref_weight:={args.vel_ref_weight}",
            f"mighty_freeze_enable:={'true' if args.mighty_freeze_enable else 'false'}",
            f"vel_ref_grad_check:={'true' if args.vel_ref_grad_check else 'false'}",
            f"vel_ref_log_every:={args.vel_ref_log_every}",
            f"pos_ref_enable:={'true' if args.pos_ref_enable else 'false'}",
            f"pos_ref_knot:={args.pos_ref_knot}",
            f"pos_ref_weight:={args.pos_ref_weight}",
            f"pos_ref_grad_check:={'true' if args.pos_ref_grad_check else 'false'}",
            f"pos_ref_log_every:={args.pos_ref_log_every}",
        ]

        node = start_proc(node_cmd, f"debug_ref_vel[{goal_tag}]")
        try:
            out, _ = node.communicate(timeout=args.case_timeout if args.goal_count > 1 else None)
        except subprocess.TimeoutExpired:
            print(f"[debug_ref_vel] timeout for {goal_tag}; killing.")
            kill_proc_group(node, f"debug_ref_vel[{goal_tag}]")
            return 1, ""

        if args.print_node_output:
            print(out)

        return node.returncode or 0, out

    base = None
    if args.start_base:
        base_cmd = ["ros2", "launch", "gcopter", args.base_launch, "use_simple_case_benchmark:=true"]
        base = start_proc(base_cmd, "base.launch")
        time.sleep(1.0)

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    try:
        # Single case (default)
        if args.goal_count <= 1:
            if args.corridor_file:
                corridor_file = args.corridor_file
                goal_tag = os.path.basename(os.path.dirname(corridor_file)) or "corridor_file"
            else:
                goal_tag = f"goal_{args.goal_idx:0{args.goal_width}d}"
                corridor_file = os.path.join(args.corridor_root, goal_tag, "corridor_cache.bin")
            if not os.path.exists(corridor_file):
                raise SystemExit(f"Missing corridor file: {corridor_file}")

            rc, out = run_one_case(corridor_file, goal_tag)
            rels = parse_grad_rels(out)
            for planner in ("gcopter", "mighty"):
                full_dir = rels[planner]["full_dir"]
                full_coord = rels[planner]["full_coord"]
                if full_dir is not None or full_coord is not None:
                    max_rel = max([v for v in (full_dir, full_coord) if v is not None])
                    if max_rel > args.rel_tol:
                        print(f"{RED}{planner.upper()} {goal_tag}: full-grad max rel={max_rel:.3e}{RESET}")
                    else:
                        print(f"{GREEN}{planner.upper()} {goal_tag}: full-grad passed (max rel={max_rel:.3e}){RESET}")
                else:
                    rel_list = rels[planner]["vref"] + rels[planner]["pref"]
                    if rel_list:
                        max_rel = max(rel_list)
                        if max_rel > args.rel_tol:
                            print(f"{RED}{planner.upper()} {goal_tag}: ref-grad max rel={max_rel:.3e}{RESET}")
                        else:
                            print(f"{GREEN}{planner.upper()} {goal_tag}: ref-grad passed (max rel={max_rel:.3e}){RESET}")
                    else:
                        print(f"{YELLOW}{planner.upper()} {goal_tag}: no grad output found{RESET}")

            print(f"[debug_ref_vel] done rc={rc}")
            return

        # Batch mode
        fails = {"gcopter": [], "mighty": []}
        missing = {"gcopter": [], "mighty": []}

        for idx in range(args.goal_start, args.goal_start + args.goal_count):
            goal_tag = f"goal_{idx:0{args.goal_width}d}"
            corridor_file = os.path.join(args.corridor_root, goal_tag, "corridor_cache.bin")
            if not os.path.exists(corridor_file):
                print(f"{YELLOW}[debug_ref_vel] missing {goal_tag}; stopping batch.{RESET}")
                break

            rc, out = run_one_case(corridor_file, goal_tag)
            if rc != 0:
                print(f"{YELLOW}[debug_ref_vel] non-zero rc={rc} for {goal_tag}{RESET}")

            rels = parse_grad_rels(out)
            for planner in ("gcopter", "mighty"):
                full_dir = rels[planner]["full_dir"]
                full_coord = rels[planner]["full_coord"]
                if full_dir is not None or full_coord is not None:
                    max_rel = max([v for v in (full_dir, full_coord) if v is not None])
                    if max_rel > args.rel_tol:
                        fails[planner].append((goal_tag, max_rel))
                else:
                    rel_list = rels[planner]["vref"] + rels[planner]["pref"]
                    if rel_list:
                        max_rel = max(rel_list)
                        if max_rel > args.rel_tol:
                            fails[planner].append((goal_tag, max_rel))
                    else:
                        missing[planner].append(goal_tag)

        # Summary
        for planner in ("gcopter", "mighty"):
            if fails[planner]:
                for goal_tag, max_rel in fails[planner]:
                    print(f"{RED}{planner.upper()} {goal_tag}: max rel={max_rel:.3e}{RESET}")
            else:
                print(f"{GREEN}{planner.upper()}: gradient test passed (tol={args.rel_tol:.1e}){RESET}")
            if missing[planner]:
                print(f"{YELLOW}{planner.upper()}: missing grad output for {len(missing[planner])} case(s){RESET}")

    finally:
        kill_proc_group(base, "base.launch")


if __name__ == "__main__":
    main()
