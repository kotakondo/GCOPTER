# MIGHTY vs GCOPTER Benchmarking

This is a revised version for benchmarking based on Kumar Lab's repo (https://github.com/yuwei-wu/GCOPTER.git).
 
## Usage 

### Prerequisites

Before running the code, make sure you have ROS 2 (Humble) installed. You may also need to install some additional dependencies:

```bash
sudo apt update
sudo apt install ros-humble-pcl-conversions ros-humble-pcl-ros
sudo apt install libpcl-dev libompl-dev
```

### Build

```bash
mkdir -p gcopter_ws/src
cd gcopter_ws/src
git clone https://github.com/kotakondo/GCOPTER.git
cd ../
colcon build --cmake-args -DCMAKE_INSTALL_PREFIX="$HOME/.local" -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ ..
source install/setup.bash
```

### Advanced (FAST) Build Options

First export the number of parallel workers (optional, default: number of CPU cores):
```bash
export J=$(nproc)  # number of parallel workers
```

Then build with the following command:
```bash
colcon build --merge-install --symlink-install   --parallel-workers $J   --cmake-args     -DCMAKE_BUILD_TYPE=Release     -DBUILD_TESTING=OFF     -DCMAKE_UNITY_BUILD=ON     -DCMAKE_C_COMPILER_LAUNCHER=ccache     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache     -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -pipe -flto=thin -DNDEBUG"     -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -pipe -flto=thin -DEIGEN_NO_DEBUG -DNDEBUG"     -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -Wl,--threads=$J -Wl,--thinlto-jobs=$J"     -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld -Wl,--threads=$J -Wl,--thinlto-jobs=$J"
```

### Run Benchmark (simple case)

```bash
python3 src/GCOPTER/gcopter/scripts/bench_simple.py
```

### Velocity Reference Experiment (vref at a knot)

Set these ROS params (e.g., in `gcopter/config/global_planning.yaml` or via launch args):
- Baseline (no vref): `VelRefEnable: false`
- Soft vref only: `VelRefEnable: true`, `VelRefKnot: <interior index>`, `VelRef: [0,0,0]`, `VelRefWeight: <w>`
- MIGHTY-freeze (hard + soft): `VelRefEnable: true`, `MIGHTYFreezeEnable: true` (GCOPTER ignores the freeze flag)

### Run Sweep Benchmark (simple case)

First, cache the corridors:
```bash
python3 src/GCOPTER/gcopter/scripts/cache_corridors.py
```
Then run the sweep benchmark:
```bash
python3 src/GCOPTER/gcopter/scripts/bench_from_corridors.py --use_cpp_batch --n_runs 100 --goal_start 0
```

### Run Reference Position and Velocity Benchmark

This benchmark tests trajectory optimization with reference position and velocity constraints. The command below uses cached corridors and evaluates performance with soft constraints at all knots:

```bash
python3 src/GCOPTER/gcopter/scripts/bench_from_corridors.py \
  --corridor_root /media/kkondo/T7/mighty_gcopter_bench/simple_bench/corridor_cache \
  --out_root /media/kkondo/T7/mighty_gcopter_bench/simple_bench \
  --n_runs 100 --goal_start 0 \
  --max_vel 2.0 --mighty_jerk_weight 1e-5 \
  --vel_ref 0 0 0 \
  --vel_ref_weight 1000.0 --pos_ref_weight 1000.0 \
  --vel_ref_knot -1 --use_cpp_batch --skip_freeze --skip_baseline \
  --no_sweep_mighty_jerk_weight --pos_ref_enable
```

Key parameters:
- `--vel_ref 0 0 0`: Target reference velocity (hovering in this case)
- `--vel_ref_weight 1000.0`: Weight for velocity reference soft constraint
- `--pos_ref_weight 1000.0`: Weight for position reference soft constraint
- `--vel_ref_knot -1`: Apply velocity reference at all knots (-1 = all knots)
- `--pos_ref_enable`: Enable position reference constraints
- `--skip_freeze`: Skip MIGHTY-freeze variant (hard constraint)
- `--skip_baseline`: Skip baseline GCOPTER without references
- `--no_sweep_mighty_jerk_weight`: Don't sweep over jerk weight values

### Run Benchmark (complex case)

```bash
python3 src/GCOPTER/gcopter/scripts/bench_sweep.py
```

### Useful Scripts

- `analyze_data.ipynb`: Jupyter notebook for analyzing and visualizing the benchmark results.
- `bench_simple.py`: Script for running a simple benchmark case.
- `bench_sweep.py`: Script for running a sweep benchmark case.
- `bag_id_shift.py`: Script for shifting the ID's in ROS bag files so that we can visualize multiple runs in one Rviz without marker ID conflicts.

## About GCOPTER

__Author__: [Zhepei Wang](https://zhepeiwang.github.io) and [Fei Gao](https://scholar.google.com/citations?hl=en&user=4RObDv0AAAAJ) from [ZJU FAST Lab](http://zju-fast.com).

__Paper__: [Geometrically Constrained Trajectory Optimization for Multicopters](https://arxiv.org/abs/2103.00190), Zhepei Wang, Xin Zhou, Chao Xu, and Fei Gao, <em>[IEEE Transactions on Robotics](https://doi.org/10.1109/TRO.2022.3160022)</em> (__T-RO__), Regular Paper.
```
@article{WANG2022GCOPTER,
    title={Geometrically Constrained Trajectory Optimization for Multicopters}, 
    author={Wang, Zhepei and Zhou, Xin and Xu, Chao and Gao, Fei}, 
    journal={IEEE Transactions on Robotics}, 
    year={2022}, 
    volume={38}, 
    number={5}, 
    pages={3259-3278}, 
    doi={10.1109/TRO.2022.3160022}
}
```

## Powerful Submodules
- [SDLP: Seidel's Algorithm](https://github.com/ZJU-FAST-Lab/SDLP) on Linear-Complexity Linear Programming for Computational Geometry.
- [VertexEnumeration3D](https://github.com/ZJU-FAST-Lab/VertexEnumeration3D): Highly Efficient Vertex Enumeration for 3D Convex Polytopes (Outperforms [cddlib](https://github.com/cddlib/cddlib) in 3D).
- [LBFGS-Lite](https://github.com/ZJU-FAST-Lab/LBFGS-Lite): An Easy-to-Use Header-Only L-BFGS Solver.
