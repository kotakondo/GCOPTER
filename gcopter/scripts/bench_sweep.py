#!/usr/bin/env python3
import argparse, subprocess, sys, os, math, random, ast

def frange(start, stop, step):
    x = start
    # Protect against floating rounding
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        yield round(start + i * step, 6)

def parse_bounds(s):
    """
    Parse bounds from a string.
    Accepts '[-25,25,-25,25,0,5]' or 'x_min x_max y_min y_max z_min z_max'.
    """
    if s.strip().startswith('['):
        vals = ast.literal_eval(s)
    else:
        vals = [float(x) for x in s.split()]
    if len(vals) != 6:
        raise ValueError("bounds must have 6 numbers: x_min x_max y_min y_max z_min z_max")
    x0,x1,y0,y1,z0,z1 = vals
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise ValueError("bounds must satisfy min<max for each axis")
    return vals

def sample_goal(bounds, start, min_dist, rng, max_tries=1000):
    """
    Uniformly sample a goal inside bounds with ||goal-start|| >= min_dist.
    If not found in max_tries, returns the farthest tried candidate.
    """
    x0,x1,y0,y1,z0,z1 = bounds
    best = None
    best_d = -1.0
    for _ in range(max_tries):
        gx = rng.uniform(x0, x1)
        gy = rng.uniform(y0, y1)
        gz = rng.uniform(z0, z1)
        dx = gx - start[0]
        dy = gy - start[1]
        dz = gz - start[2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d >= min_dist:
            return [round(gx,3), round(gy,3), round(gz,3)]
        if d > best_d:
            best_d = d
            best = (gx, gy, gz)
    # fallback: use farthest candidate
    gx,gy,gz = best
    return [round(gx,3), round(gy,3), round(gz,3)]

def main():
    ap = argparse.ArgumentParser(description="Sweep GCOPTER/MIGHTY over max velocity & MIGHTY jerk weight with randomized goals.")
    ap.add_argument("--out_csv", default="benchmark_results.csv")
    ap.add_argument("--N_sim", type=int, default=3, help="Trials per (v, jerk_w) cell")
    ap.add_argument("--v_min", type=float, default=1.0)
    ap.add_argument("--v_max", type=float, default=10.0)
    ap.add_argument("--v_step", type=float, default=1.0)
    ap.add_argument("--jerk_dec_min", type=int, default=-6)
    ap.add_argument("--jerk_dec_max", type=int, default=6)
    ap.add_argument("--bounds", default="[-25,25,-25,25,0,5]",
                    help="Map bounds: either JSON-like list or 6 floats 'x0 x1 y0 y1 z0 z1'")
    ap.add_argument("--goal_min_dist", type=float, default=15.0,
                    help="Minimum distance from start to goal (meters)")
    ap.add_argument("--sample_dt", default="0.02")
    ap.add_argument("--collision_dt", default="0.02")
    ap.add_argument("--pkg", default="gcopter", help="ROS2 package containing benchmark.launch.py")
    ap.add_argument("--launch", default="benchmark.launch.py")
    ap.add_argument("--rng_seed", type=int, default=12345, help="Seed for Python RNG that places goals")
    # start is fixed as requested
    args = ap.parse_args()

    bounds = parse_bounds(args.bounds)
    start = [0.0, 0.0, 0.5]

    # build sweep grids
    v_vals = list(frange(args.v_min, args.v_max, args.v_step))
    jw_vals = [f"1e{e}" for e in range(args.jerk_dec_min, args.jerk_dec_max + 1)]

    print(f"Sweeping |V| in {v_vals}")
    print(f"Sweeping jerk weight in {jw_vals}")
    print(f"Trials per cell: {args.N_sim}")
    print(f"Bounds: {bounds}, start: {start}, goal_min_dist: {args.goal_min_dist}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    trial = 0
    for v in v_vals:
        for jw in jw_vals:
            for rep in range(args.N_sim):
                # derive a reproducible RNG per trial
                seed = 1000 + 97 * trial + rep
                rng = random.Random(args.rng_seed * 65537 + seed)

                goal = sample_goal(bounds, start, args.goal_min_dist, rng)
                goal_str = f"[{goal[0]}, {goal[1]}, {goal[2]}]"
                start_str = f"[{start[0]}, {start[1]}, {start[2]}]"

                print(f"\n== trial={trial} seed={seed} V={v} jw={jw} goal={goal_str} ==")
                cmd = [
                    "ros2", "launch", args.pkg, args.launch,
                    f"seed:={seed}",
                    f"start:={start_str}",
                    f"goal:={goal_str}",
                    f"max_vel:={v}",
                    f"mighty_jerk_weight:={jw}",
                    f"sample_dt:={args.sample_dt}",
                    f"collision_dt:={args.collision_dt}",
                    f"out_csv:={os.path.abspath(args.out_csv)}",
                    f"trial_id:={trial}",
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Run failed: {e}", file=sys.stderr)
                trial += 1

    print("\nDone. Results ->", os.path.abspath(args.out_csv))

if __name__ == "__main__":
    main()
