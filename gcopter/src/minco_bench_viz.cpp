// minco_bench_viz.cpp — GCOPTER + MIGHTY (LBFGS) side-by-side benchmark
// Drop-in replacement.
// - Plans with GCOPTER (as before)
// - Builds the same corridor for MIGHTY, maps weights/limits, runs LBFGS
// - Compares time / path length / jerk cost and prints a small table

#include "misc/visualizer.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"

// === MIGHTY / LBFGS solver ===
#include <dynus/lbfgs_solver.hpp> // declares SolverLBFGS, planner_params_t, LinearConstraint3D, etc.
// ^ this is the public header included by your C++ solver implementation. :contentReference[oaicite:1]{index=1}

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <decomp_ros_msgs/msg/polyhedron_array.hpp>
#include <decomp_rviz_plugins/data_ros_utils.hpp>

#include <fstream>
#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#endif

using lbfgs::planner_params_t;
using lbfgs::SolverLBFGS;
using lbfgs::Vec3;

struct Config
{
    // Map + inputs
    std::string mapTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound; // [xmin,xmax, ymin,ymax, zmin,zmax]
    std::vector<double> startXYZ; // [x,y,z]
    std::vector<double> goalXYZ;  // [x,y,z]
    double timeoutRRT;

    // GCOPTER limits
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;

    // Physical params
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;

    // Optimization weights
    double weightT;
    std::vector<double> chiVec; // {vel, bdr, tilt, minThr, maxThr}
    double smoothingEps;
    int integralIntervs;
    double relCostTol;

    // Export CSV
    std::string exportCSVDir;
    double sampleDt; // seconds

    // MIGHTY-specific
    double mightyJerkWeight; // Jerk weight for MIGHTY (LBFGS

    // Collision checking
    double collisionDt; // sampling step for collision checking

    // Safe corridor parameters
    double sfc_progress; // progress along the route for corridor generation
    double sfc_range;    // range around the route for corridor generation

    Config(rclcpp::Node &node)
    {
        // Declare parameters with default values
        node.declare_parameter("MapTopic", std::string("/voxel_map"));
        node.declare_parameter("DilateRadius", 0.5);
        node.declare_parameter("VoxelWidth", 0.25);
        node.declare_parameter("MapBound", std::vector<double>{-25.0, 25.0, -25.0, 25.0, 0.0, 5.0});
        node.declare_parameter("Start", std::vector<double>{0.0, 0.0, 0.5});
        node.declare_parameter("Goal", std::vector<double>{10.0, 3.0, 1.5});
        node.declare_parameter("TimeoutRRT", 0.1);
        node.declare_parameter("MaxVelMag", 4.0);
        node.declare_parameter("MaxBdrMag", 2.1);
        node.declare_parameter("MaxTiltAngle", 1.05);
        node.declare_parameter("MinThrust", 2.0);
        node.declare_parameter("MaxThrust", 12.0);
        node.declare_parameter("VehicleMass", 0.61);
        node.declare_parameter("GravAcc", 9.8);
        node.declare_parameter("HorizDrag", 0.70);
        node.declare_parameter("VertDrag", 0.80);
        node.declare_parameter("ParasDrag", 0.01);
        node.declare_parameter("SpeedEps", 1.0e-4);
        node.declare_parameter("WeightT", 20.0);
        node.declare_parameter("ChiVec", std::vector<double>{1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e5});
        node.declare_parameter("SmoothingEps", 1.0e-2);
        node.declare_parameter("IntegralIntervs", 16);
        node.declare_parameter("RelCostTol", 1.0e-5);
        node.declare_parameter("ExportCSVDir", std::string("")); // empty => no export
        node.declare_parameter("SampleDt", 0.02);                // seconds
        node.declare_parameter("MIGHTYJerkWeight", 0.1); // Jerk weight for MIGHTY (LBFGS)
        node.declare_parameter("CollisionDt", 0.01);
        node.declare_parameter("sfc_progress", 0.5); // progress along the route for corridor generation
        node.declare_parameter("sfc_range", 3.0);    // range around

        // Get parameters from the node
        node.get_parameter("MapTopic", mapTopic);
        node.get_parameter("DilateRadius", dilateRadius);
        node.get_parameter("VoxelWidth", voxelWidth);
        node.get_parameter("MapBound", mapBound);
        node.get_parameter("Start", startXYZ);
        node.get_parameter("Goal", goalXYZ);
        node.get_parameter("TimeoutRRT", timeoutRRT);
        node.get_parameter("MaxVelMag", maxVelMag);
        node.get_parameter("MaxBdrMag", maxBdrMag);
        node.get_parameter("MaxTiltAngle", maxTiltAngle);
        node.get_parameter("MinThrust", minThrust);
        node.get_parameter("MaxThrust", maxThrust);
        node.get_parameter("VehicleMass", vehicleMass);
        node.get_parameter("GravAcc", gravAcc);
        node.get_parameter("HorizDrag", horizDrag);
        node.get_parameter("VertDrag", vertDrag);
        node.get_parameter("ParasDrag", parasDrag);
        node.get_parameter("SpeedEps", speedEps);
        node.get_parameter("WeightT", weightT);
        node.get_parameter("ChiVec", chiVec);
        node.get_parameter("SmoothingEps", smoothingEps);
        node.get_parameter("IntegralIntervs", integralIntervs);
        node.get_parameter("RelCostTol", relCostTol);
        node.get_parameter("ExportCSVDir", exportCSVDir);
        node.get_parameter("SampleDt", sampleDt);
        node.get_parameter("MIGHTYJerkWeight", mightyJerkWeight);
        node.get_parameter("CollisionDt", collisionDt);
        node.get_parameter("sfc_progress", sfc_progress);
        node.get_parameter("sfc_range", sfc_range);
    }
};

// === Simple metrics shared by both planners
struct Metrics
{
    double time_s{0.0};    // total trajectory duration
    double path_len{0.0};  // ∫||v|| dt
    double jerk_cost{0.0}; // ∫||j|| dt  (same definition as your sampling metrics)
    double solve_ms{0.0};  // planner's wall time in milliseconds
    int n_collisions{0};   // # sampled positions that hit occupied voxels
};

// Degree-5 Bézier evaluator (pos,vel,acc,jer) at u∈[0,1] for one segment (same as before)
static inline void evalBezier5_PVAJ(const std::array<Eigen::Vector3d, 6> &CP,
                                    double T, double u,
                                    Eigen::Vector3d &p,
                                    Eigen::Vector3d &v,
                                    Eigen::Vector3d &a,
                                    Eigen::Vector3d &j)
{
    const double om = 1.0 - u;

    // position
    const double b0 = std::pow(om, 5);
    const double b1 = 5 * u * std::pow(om, 4);
    const double b2 = 10 * u * u * std::pow(om, 3);
    const double b3 = 10 * std::pow(u, 3) * om * om;
    const double b4 = 5 * std::pow(u, 4) * om;
    const double b5 = std::pow(u, 5);
    p = b0 * CP[0] + b1 * CP[1] + b2 * CP[2] + b3 * CP[3] + b4 * CP[4] + b5 * CP[5];

    // forward diffs
    std::array<Eigen::Vector3d, 5> D1;
    for (int i = 0; i < 5; ++i)
        D1[i] = CP[i + 1] - CP[i];
    std::array<Eigen::Vector3d, 4> D2;
    for (int i = 0; i < 4; ++i)
        D2[i] = D1[i + 1] - D1[i];
    std::array<Eigen::Vector3d, 3> D3;
    for (int i = 0; i < 3; ++i)
        D3[i] = D2[i + 1] - D2[i];

    const double invT = 1.0 / std::max(T, 1e-9);
    const double invT2 = invT * invT;
    const double invT3 = invT2 * invT;

    // velocity (deg-4)
    const double c0 = std::pow(om, 4);
    const double c1 = 4 * u * std::pow(om, 3);
    const double c2 = 6 * u * u * om * om;
    const double c3 = 4 * std::pow(u, 3) * om;
    const double c4 = std::pow(u, 4);
    v = 5.0 * invT * (c0 * D1[0] + c1 * D1[1] + c2 * D1[2] + c3 * D1[3] + c4 * D1[4]);

    // acceleration (deg-3)
    const double d0 = std::pow(om, 3);
    const double d1 = 3 * u * om * om;
    const double d2 = 3 * u * u * om;
    const double d3 = std::pow(u, 3);
    a = 20.0 * invT2 * (d0 * D2[0] + d1 * D2[1] + d2 * D2[2] + d3 * D2[3]);

    // jerk (deg-2)
    const double e0 = om * om;
    const double e1 = 2 * u * om;
    const double e2 = u * u;
    j = 60.0 * invT3 * (e0 * D3[0] + e1 * D3[1] + e2 * D3[2]);
}

// Sample the GCOPTER trajectory and count collisions against the voxel map.
// A sample is a "collision" if voxelMap.query(p) != 0
static int countCollisionsGCOPTER(const Trajectory<5> &traj,
                                  const voxel_map::VoxelMap &voxelMap,
                                  double dt)
{
    const double Ttot = traj.getTotalDuration();
    if (Ttot <= 0.0)
        return 0;

    const double h = std::max(1e-6, dt);
    const int N = std::max(1, (int)std::ceil(Ttot / h));
    int hits = 0;

    for (int i = 0; i <= N; ++i)
    {
        const double t = std::min(Ttot, i * h);
        const Eigen::Vector3d p = traj.getPos(t);
        if (voxelMap.query(p) != 0)
            ++hits;
    }
    return hits;
}

// Sample the MIGHTY (Bezier) trajectory and count collisions against the voxel map.
static int countCollisionsMIGHTY(const std::vector<std::array<Eigen::Vector3d, 6>> &CP,
                                 const std::vector<double> &T,
                                 const voxel_map::VoxelMap &voxelMap,
                                 double dt)
{
    const int Mseg = (int)CP.size();
    if (Mseg == 0)
        return 0;

    // cumulative time-edges
    std::vector<double> edges(Mseg + 1, 0.0);
    for (int s = 0; s < Mseg; ++s)
        edges[s + 1] = edges[s] + T[s];
    const double Ttot = edges.back();
    const double h = std::max(1e-6, dt);
    const int N = std::max(1, (int)std::ceil(Ttot / h));

    auto eval_pos = [&](double t) -> Eigen::Vector3d
    {
        t = std::clamp(t, 0.0, Ttot);
        int s = std::clamp((int)(std::upper_bound(edges.begin(), edges.end(), t) - edges.begin()) - 1, 0, Mseg - 1);
        const double Ts = std::max(1e-9, T[s]);
        const double u = std::clamp((t - edges[s]) / Ts, 0.0, 1.0);
        Eigen::Vector3d p, v, a, j;
        evalBezier5_PVAJ(CP[s], Ts, u, p, v, a, j);
        return p;
    };

    int hits = 0;
    for (int i = 0; i <= N; ++i)
    {
        const double t = std::min(Ttot, i * h);
        const Eigen::Vector3d p = eval_pos(t);
        if (voxelMap.query(p) != 0)
            ++hits;
    }
    return hits;
}

// Sample GCOPTER trajectory at a uniform dt and integrate speed and jerk-norm via trapezoid.
static Metrics computeMetricsGCOPTER_sampled_dt(const Trajectory<5> &traj, double dt)
{
    Metrics M;
    const double Ttot = traj.getTotalDuration();
    if (Ttot <= 0.0)
        return M;

    M.time_s = Ttot;

    const double h = std::max(1e-6, dt);
    const int N = std::max(1, (int)std::ceil(Ttot / h));

    double t_prev = 0.0;
    Eigen::Vector3d v_prev = traj.getVel(0.0);
    Eigen::Vector3d j_prev = traj.getJer(0.0);

    for (int i = 1; i <= N; ++i)
    {
        const double t = std::min(Ttot, i * h);
        const double hi = t - t_prev;

        const Eigen::Vector3d v = traj.getVel(t);
        const Eigen::Vector3d j = traj.getJer(t);

        // path length: ∫ ||v|| dt
        M.path_len += 0.5 * (v_prev.norm() + v.norm()) * hi;

        // jerk_cost: ∫ ||j|| dt   (change to j.squaredNorm() if you want ∫||j||^2)
        M.jerk_cost += 0.5 * (j_prev.norm() + j.norm()) * hi; // <-- change to .squaredNorm() for L2^2

        // advance
        t_prev = t;
        v_prev = v;
        j_prev = j;
    }
    return M;
}

// Sample MIGHTY trajectory at a uniform dt and integrate speed and jerk-norm via trapezoid.
static Metrics computeMetricsMIGHTY_sampled_dt(const std::vector<std::array<Eigen::Vector3d, 6>> &CP,
                                               const std::vector<double> &T,
                                               double dt)
{
    Metrics M;
    const int Mseg = (int)CP.size();
    if (Mseg == 0)
        return M;

    // cumulative edges
    std::vector<double> edges(Mseg + 1, 0.0);
    for (int s = 0; s < Mseg; ++s)
        edges[s + 1] = edges[s] + T[s];
    const double Ttot = edges.back();
    M.time_s = Ttot;

    const double h = std::max(1e-6, dt);
    const int N = std::max(1, (int)std::ceil(Ttot / h));

    // Helper to eval v/j at global time t
    auto eval_vj = [&](double t) -> std::pair<Eigen::Vector3d, Eigen::Vector3d>
    {
        t = std::clamp(t, 0.0, Ttot);
        // find segment
        int s = std::clamp(static_cast<int>(std::upper_bound(edges.begin(), edges.end(), t) - edges.begin()) - 1, 0, Mseg - 1);
        const double Ts = std::max(1e-9, T[s]);
        const double u = (t - edges[s]) / Ts;
        Eigen::Vector3d p, v, a, j;
        evalBezier5_PVAJ(CP[s], Ts, std::clamp(u, 0.0, 1.0), p, v, a, j);
        return {v, j};
    };

    double t_prev = 0.0;
    auto [v_prev, j_prev] = eval_vj(0.0);

    for (int i = 1; i <= N; ++i)
    {
        const double t = std::min(Ttot, i * h);
        const double hi = t - t_prev;

        auto [v, j] = eval_vj(t);

        // path length: ∫ ||v|| dt
        M.path_len += 0.5 * (v_prev.norm() + v.norm()) * hi;

        // jerk_cost: ∫ ||j|| dt   (change to j.squaredNorm() if you want ∫||j||^2)
        M.jerk_cost += 0.5 * (j_prev.norm() + j.norm()) * hi; // <-- change to .squaredNorm() for L2^2

        t_prev = t;
        v_prev = v;
        j_prev = j;
    }
    return M;
}

static std::vector<LinearConstraint3D>
toLinearConstraints(const std::vector<Eigen::MatrixX4d> &hPolys)
{
    std::vector<LinearConstraint3D> out;
    out.reserve(hPolys.size());

    for (const auto &H : hPolys)
    {
        if (H.rows() == 0 || H.cols() != 4)
        {
            out.emplace_back();
            continue;
        }

        const int m = static_cast<int>(H.rows());
        Eigen::MatrixXd A(m, 3);
        Eigen::VectorXd b(m);

        A = H.leftCols<3>();
        b = -H.col(3); // just convention difference
        out.emplace_back(A, b);
    }
    return out;
}

// Pretty print
static void printCompare(const std::string &tagA, const Metrics &A,
                         const std::string &tagB, const Metrics &B)
{
    const int wName = 16;
    const int wNum = 12;

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(2);

    auto pr = [&](const std::string &nm, double a, double b)
    {
        std::cout << std::left << std::setw(wName) << nm
                  << std::right << std::setw(wNum) << a
                  << std::right << std::setw(wNum) << b
                  << std::right << std::setw(wNum) << (b - a)
                  << "\n";
    };
    auto pri = [&](const std::string &nm, int a, int b)
    {
        std::cout << std::left << std::setw(wName) << nm
                  << std::right << std::setw(wNum) << a
                  << std::right << std::setw(wNum) << b
                  << std::right << std::setw(wNum) << (b - a)
                  << "\n";
    };

    std::cout << "\n==== Benchmark ====\n";
    std::cout << std::left << std::setw(wName) << "metric"
              << std::right << std::setw(wNum) << tagA
              << std::right << std::setw(wNum) << tagB
              << std::right << std::setw(wNum) << "Δ(B-A)\n";

    pr("solve [ms]", A.solve_ms, B.solve_ms);
    pr("time [s]", A.time_s, B.time_s);
    pr("path len [m]", A.path_len, B.path_len);
    pr("jerk_cost", A.jerk_cost, B.jerk_cost);
    pri("collisions", A.n_collisions, B.n_collisions);

    std::cout << "===================\n";
    std::cout.unsetf(std::ios::floatfield);
}

// MIGHTY wrapper (reconstruct CP/T from zopt locally; no hidden getters needed)
struct MightyOut
{
    std::vector<std::array<Eigen::Vector3d, 6>> CP;
    std::vector<double> T;
    double obj{0.0};
    double wall_ms{0.0};
};

static MightyOut runMighty(
    const vec_Vecf<3> &global_wps,
    const std::vector<LinearConstraint3D> &safe_corridor,
    const state &initial_state,
    const state &final_state,
    const planner_params_t &params)
{
    MightyOut out;
    using Clock = std::chrono::high_resolution_clock;

    // 2) Build solver; prepare problem
    auto solver = std::make_shared<SolverLBFGS>();
    solver->initializeSolver(params); // immutable settings (weights, limits, etc) :contentReference[oaicite:2]{index=2}

    std::vector<std::shared_ptr<dynTraj>> obstacles; // none for now

    double t0 = 0.0, ig_ms = 0.0;
    solver->prepareSolverForReplan(
        t0, global_wps, safe_corridor, obstacles,
        initial_state, final_state, ig_ms, false); // :contentReference[oaicite:3]{index=3}

    // Get the single initial guess we just built
    const std::vector<Eigen::VectorXd> &z0_list = solver->getInitialGuesses();
    if (z0_list.empty())
    {
        std::cerr << "[MIGHTY] No initial guess available.\n";
        return out;
    }
    Eigen::VectorXd z0 = z0_list.front();

    // 3) Optimize
    lbfgs::lbfgs_parameter_t lb;
    lb.mem_size = (int)z0.size();
    lb.past = 20;
    lb.max_linesearch = 64;
    lb.max_iterations = 300;
    lb.delta = 1e-6;
    lb.g_epsilon = 1e-6;

    auto t_start = Clock::now();
    Eigen::VectorXd zopt;
    double fopt = 0.0;
    solver->optimize(z0, zopt, fopt, lb); // LBFGS entrypoint (objective+analytic grad) :contentReference[oaicite:4]{index=4}
    auto t_end = Clock::now();

    out.wall_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-3;
    out.obj = fopt;

    // 4) Recover CP/T from zopt (no need to touch solver internals)
    {
        std::vector<Vec3> P, V, A;
        std::vector<std::array<Vec3, 6>> CPv; // CP[s][0..5]
        std::vector<double> Tv;
        solver->reconstruct(zopt, P, V, A, CPv, Tv); // public; fills all fields :contentReference[oaicite:5]{index=5}

        // Copy to Eigen::Vector3d form the rest of this file uses
        out.CP.resize(CPv.size());
        for (size_t s = 0; s < CPv.size(); ++s)
        {
            for (int j = 0; j < 6; ++j)
                out.CP[s][j] = Eigen::Vector3d(CPv[s][j].x(), CPv[s][j].y(), CPv[s][j].z());
        }
        out.T = Tv;
    }

    return out;
}

// === ROS2 Planner node (mostly your original, with a "MIGHTY" section appended)
class GlobalPlanner : public rclcpp::Node
{
private:
    Config config;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mapSub;
    Visualizer visualizer;
    voxel_map::VoxelMap voxelMap;

    std::vector<Eigen::Vector3d> startGoal;
    Trajectory<5> traj;
    double trajStamp = 0.0;
    bool mapInitialized = false;
    bool plannedOnce = false;

    // Cache for periodic re-publish
    std::vector<Eigen::Vector3d> routeCache_;
    std::vector<Eigen::MatrixX4d> hPolysCache_;
    rclcpp::TimerBase::SharedPtr repubTimer_;
    MightyOut M_mighty_;

    vec_E<Polyhedron<3>> poly_whole_;
    rclcpp::Publisher<decomp_ros_msgs::msg::PolyhedronArray>::SharedPtr pub_poly_whole_;

    double mightyStamp_{0.0};
    std::vector<double> mightyEdges_;          // cumulative segment times
    std::vector<Eigen::Vector3d> mightyKnots_; // P0..PM (segment endpoints)

    static inline double clamp(double x, double lo, double hi)
    {
        return std::max(lo, std::min(x, hi));
    }

public:
    GlobalPlanner() : Node("minco_bench_viz"), config(*this), visualizer(*this)
    {
        Eigen::Vector3i xyz(
            static_cast<int>((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth),
            static_cast<int>((config.mapBound[3] - config.mapBound[2]) / config.voxelWidth),
            static_cast<int>((config.mapBound[5] - config.mapBound[4]) / config.voxelWidth));
        Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            config.mapTopic, rclcpp::SensorDataQoS(),
            std::bind(&GlobalPlanner::mapCallBack, this, std::placeholders::_1));

        // Keep RViz markers alive
        repubTimer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&GlobalPlanner::republishMarkers_, this));

        pub_poly_whole_ = this->create_publisher<decomp_ros_msgs::msg::PolyhedronArray>("poly_whole", 10);

        RCLCPP_INFO(this->get_logger(), "minco_bench_viz ready. Waiting for map on %s",
                    config.mapTopic.c_str());
    }

    void mapCallBack(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Ingest map once
        if (!mapInitialized)
        {
            const size_t total = msg->data.size() / msg->point_step;
            const float *fdata = reinterpret_cast<const float *>(&msg->data[0]);
            const size_t stride = msg->point_step / sizeof(float);
            for (size_t i = 0; i < total; ++i)
            {
                const size_t cur = stride * i;
                const float x = fdata[cur + 0];
                const float y = fdata[cur + 1];
                const float z = fdata[cur + 2];
                if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
                    voxelMap.setOccupied(Eigen::Vector3d(x, y, z));
            }
            voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));
            mapInitialized = true;
            RCLCPP_INFO(this->get_logger(), "Map ingested (%zu pts) & dilated.", total);
        }

        // Plan once after map ready using Start/Goal params
        if (mapInitialized && !plannedOnce)
        {
            plannedOnce = true;

            auto S = config.startXYZ, G = config.goalXYZ;
            if (S.size() != 3 || G.size() != 3)
            {
                RCLCPP_ERROR(this->get_logger(), "Start/Goal must be [x,y,z]");
                return;
            }
            // Clamp to bounds
            S[0] = clamp(S[0], config.mapBound[0], config.mapBound[1]);
            S[1] = clamp(S[1], config.mapBound[2], config.mapBound[3]);
            S[2] = clamp(S[2], config.mapBound[4], config.mapBound[5]);
            G[0] = clamp(G[0], config.mapBound[0], config.mapBound[1]);
            G[1] = clamp(G[1], config.mapBound[2], config.mapBound[3]);
            G[2] = clamp(G[2], config.mapBound[4], config.mapBound[5]);

            Eigen::Vector3d start(S[0], S[1], S[2]);
            Eigen::Vector3d goal(G[0], G[1], G[2]);

            if (voxelMap.query(start) != 0)
            {
                RCLCPP_WARN(this->get_logger(), "Start occupied; nudging up by DilateRadius.");
                start.z() = clamp(start.z() + config.dilateRadius, config.mapBound[4], config.mapBound[5]);
            }
            if (voxelMap.query(goal) != 0)
            {
                RCLCPP_WARN(this->get_logger(), "Goal occupied; nudging up by DilateRadius.");
                goal.z() = clamp(goal.z() + config.dilateRadius, config.mapBound[4], config.mapBound[5]);
            }

            startGoal.clear();
            startGoal.emplace_back(start);
            startGoal.emplace_back(goal);

            // Show start/goal
            visualizer.visualizeStartGoal(start, 0.5, 0);
            visualizer.visualizeStartGoal(goal, 0.5, 1);

            plan(); // run both planners
        }
    }

    void plan()
    {
        if (startGoal.size() != 2)
            return;

        // === 1) Global route (RRT), corridor from voxel surface
        std::vector<Eigen::Vector3d> route;
        sfc_gen::planPath<voxel_map::VoxelMap>(
            startGoal[0], startGoal[1],
            voxelMap.getOrigin(), voxelMap.getCorner(),
            &voxelMap, config.timeoutRRT, route);

        if (route.size() <= 1)
        {
            RCLCPP_WARN(this->get_logger(), "Global route failed.");
            return;
        }

        std::vector<Eigen::MatrixX4d> hPolys;
        std::vector<Eigen::Vector3d> pc;
        voxelMap.getSurf(pc);
        sfc_gen::convexCover(route, pc, voxelMap.getOrigin(), voxelMap.getCorner(), config.sfc_progress, config.sfc_range, hPolys);

        sfc_gen::shortCut(hPolys);
        printf("Route size: %zu\n and hPolys size: %zu\n", route.size(), hPolys.size());

        // Visualize SFC once (timer keeps them alive)
        visualizer.visualizePolytope(hPolys);

        // Boundary states (pos, vel=0, acc=0)
        Eigen::Matrix3d iniState, finState;
        iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

        // === 2) GCOPTER
        gcopter::GCOPTER_PolytopeSFC gcopter;
        Eigen::VectorXd magnitudeBounds(5), penaltyWeights(5), physicalParams(6);
        magnitudeBounds << config.maxVelMag, config.maxBdrMag, config.maxTiltAngle,
            config.minThrust, config.maxThrust;
        penaltyWeights << config.chiVec[0], config.chiVec[1], config.chiVec[2],
            config.chiVec[3], config.chiVec[4];
        physicalParams << config.vehicleMass, config.gravAcc, config.horizDrag,
            config.vertDrag, config.parasDrag, config.speedEps;

        traj.clear();
        if (!gcopter.setup(config.weightT, iniState, finState, hPolys,
                           INFINITY, config.smoothingEps, config.integralIntervs,
                           magnitudeBounds, penaltyWeights, physicalParams))
        {
            RCLCPP_ERROR(this->get_logger(), "GCOPTER setup failed.");
            return;
        }

        using Clock = std::chrono::high_resolution_clock;
        auto gc_start = Clock::now();
        if (std::isinf(gcopter.optimize(traj, config.relCostTol)))
        {
            RCLCPP_ERROR(this->get_logger(), "GCOPTER optimize failed.");
            return;
        }
        auto gc_end = Clock::now();
        double gc_ms = std::chrono::duration_cast<std::chrono::microseconds>(gc_end - gc_start).count() * 1e-3;

        if (traj.getPieceNum() > 0)
        {
            trajStamp = this->now().seconds();
            visualizer.visualize(traj, route);
            visualizer.visualizePolytope(hPolys);
            routeCache_ = route;
            hPolysCache_ = hPolys;
        }

        // === 3) Metrics for GCOPTER
        Metrics M_gc = computeMetricsGCOPTER_sampled_dt(traj, config.sampleDt);
        M_gc.solve_ms = gc_ms;
        M_gc.n_collisions = countCollisionsGCOPTER(traj, voxelMap, config.collisionDt);

        // Get GCOPTER's initial guess for MIGHTY
        Eigen::Matrix3Xd init_points;
        Eigen::VectorXd init_times;
        gcopter.getInitialGuess(init_points, init_times);

        int init_size = init_points.cols();
        printf("init_points size: %d, hPolys size: %zu\n", init_size, hPolys.size());

        // === 4) Build MIGHTY params mapped from config
        planner_params_t mighty_cfg{};
        mighty_cfg.verbose = false;
        mighty_cfg.V_nom = magnitudeBounds[0];
        mighty_cfg.V_max = magnitudeBounds[0];
        mighty_cfg.A_max = 0.0;                           // not used here
        mighty_cfg.J_max = 0.0;                           // not used here
        mighty_cfg.time_weight = config.weightT;          // you can tune this
        mighty_cfg.dyn_weight = 0.0;                      // no moving obstacles in this bench
        mighty_cfg.stat_weight = config.chiVec[0];        // keep corridor penalties active
        mighty_cfg.jerk_weight = config.mightyJerkWeight; // you can tune this
        mighty_cfg.dyn_constr_vel_weight = config.chiVec[1];
        mighty_cfg.dyn_constr_acc_weight = 0.0; // not used here
        mighty_cfg.omega_weight = config.chiVec[2];
        mighty_cfg.theta_weight = config.chiVec[3];
        mighty_cfg.thrust_min_weight = config.chiVec[4];
        mighty_cfg.thrust_max_weight = config.chiVec[4];
        mighty_cfg.num_dyn_obst_samples = 64;
        mighty_cfg.init_turn_bf = 40.0; // degrees
        mighty_cfg.Co = 0.05;           // corridor “soft margin”
        mighty_cfg.Cw = 0.40;           // dyn obstacle radius (unused here)
        mighty_cfg.BIG = 1e9;
        mighty_cfg.dc = 0.01; // sampling step for setpoints
        mighty_cfg.second_to_last_vel_scale = 1.0;

        // Prepare initial/goal states for MIGHTY (pos/vel/acc)
        state init_state, goal_state;
        init_state.pos = Vec3(route.front().x(), route.front().y(), route.front().z());
        init_state.vel = Vec3::Zero();
        init_state.accel = Vec3::Zero();
        goal_state.pos = Vec3(route.back().x(), route.back().y(), route.back().z());
        goal_state.vel = Vec3::Zero();
        goal_state.accel = Vec3::Zero();

        // Convert route (Eigen) to MIGHTY vector type
        vec_Vecf<3> route_m;
        route_m.reserve(init_points.size() + 2); // +2 for start/goal
        route_m.emplace_back(init_state.pos.x(), init_state.pos.y(), init_state.pos.z());
        for (const auto &p : init_points.colwise())
            route_m.emplace_back(p.x(), p.y(), p.z());
        route_m.emplace_back(goal_state.pos.x(), goal_state.pos.y(), goal_state.pos.z());

        // === 5) Run MIGHTY
        // Build exactly-one-poly-per-segment corridor for MIGHTY

        // Ensure interior points lie inside both adjacent polytopes (nudged inward by 2 cm)
        bool ok_fix = enforceRouteInsideCorridor(route_m, hPolys, /*inward_margin=*/0.02, /*max_passes=*/20);
        if (!ok_fix)
        {
            RCLCPP_WARN(this->get_logger(), "Some waypoints still near/outside corridor after projection.");
        }

        // print out path_v for debugging
        std::cout << "Path for MIGHTY:\n";
        for (const auto &p : route_m)
        {
            std::cout << "  " << p.transpose() << "\n";
        }

        // Get obstacle points from the voxel map for the ROS2 decomp
        // vec_Vecf<3> vec_o;
        // voxelMap.getSurfaceObsForDecomp(vec_o);         // recommended
        // or: voxelMap.getObsForDecomp(vec_o, /*surface_only=*/false, /*stride=*/2);

        // Run your decomposer
        // std::vector<LinearConstraint3D> l_constraints;
        // bool ok = cvxEllipsoidDecomp(route_m, vec_o, l_constraints, poly_whole_);
        // if (!ok) {
        //     RCLCPP_WARN(get_logger(), "Ellipsoid decomposition failed");
        // }

        // Convert GCOPTER’s hPolys to MIGHTY constraints
        std::vector<LinearConstraint3D> l_constraints = toLinearConstraints(hPolys);

        printf("route_m size: %zu, l_constraints size: %zu\n",
               route_m.size(), l_constraints.size());

        MightyOut M_mighty = runMighty(route_m, l_constraints, init_state, goal_state, mighty_cfg);

        // Draw MIGHTY trajectory in green, with its own namespace
        visualizer.visualizeBezier(M_mighty.CP, M_mighty.T, /*ns=*/"mighty", /*width=*/0.06,
                                   /*samples=*/120, /*r=*/0.0f, /*g=*/1.0f, /*b=*/0.0f, /*a=*/1.0f,
                                   /*frame_id=*/"world");

        // Cache cumulative edges for fast lookup
        mightyEdges_.assign(M_mighty.T.size() + 1, 0.0);
        for (size_t s = 0; s < M_mighty.T.size(); ++s)
            mightyEdges_[s + 1] = mightyEdges_[s] + M_mighty.T[s];
        mightyStamp_ = this->now().seconds();

        // Build MIGHTY knot positions from Bezier CPs: P0 = CP[0][0], P_{s+1} = CP[s][5]
        mightyKnots_.clear();
        if (!M_mighty.CP.empty())
        {
            mightyKnots_.push_back(M_mighty.CP.front()[0]);
            for (size_t s = 0; s < M_mighty.CP.size(); ++s)
                mightyKnots_.push_back(M_mighty.CP[s][5]);
        }

        // Show knots as green spheres (persistent)
        visualizer.visualizePoints(mightyKnots_,
                                   /*radius=*/0.07f,
                                   /*r=*/0.0f, /*g=*/1.0f, /*b=*/0.0f, /*a=*/1.0f,
                                   /*frame=*/"odom",
                                   /*ns=*/"mighty_knots",
                                   /*ttl=*/0.0);

        // === 6) Metrics for MIGHTY
        Metrics M_m = computeMetricsMIGHTY_sampled_dt(M_mighty.CP, M_mighty.T, /*dt=*/config.sampleDt);
        M_m.solve_ms = M_mighty.wall_ms;
        M_m.n_collisions = countCollisionsMIGHTY(M_mighty.CP, M_mighty.T, voxelMap, config.collisionDt);

        // === 7) Print comparison & wall times
        printCompare("GCOPTER", M_gc, "MIGHTY", M_m);
        std::cout << "GCOPTER solve time [ms]: (printed by library or external timer)\n";
        std::cout << "MIGHTY  solve time [ms]: " << M_mighty.wall_ms << "\n";
        std::cout << "MIGHTY  final objective : " << M_mighty.obj << "\n";

        M_mighty_ = M_mighty;

        // === 8) Export VAJ histories to CSV (for Python plotting)
        // Export GCOPTER and MIGHTY VAJ CSVs if requested
        if (!config.exportCSVDir.empty())
        {
            const std::string dir = config.exportCSVDir;
            const std::string f_gc = (dir.back() == '/' ? dir : dir + "/") + "gcopter_vaj.csv";
            const std::string f_my = (dir.back() == '/' ? dir : dir + "/") + "mighty_vaj.csv";
            writeCSV_GCOPTER(traj, f_gc, config.sampleDt);
            writeCSV_MIGHTY(M_mighty.CP, M_mighty.T, f_my, config.sampleDt);
            RCLCPP_INFO(this->get_logger(), "Exported VAJ CSVs:\n  %s\n  %s", f_gc.c_str(), f_my.c_str());
        }
    }

    bool cvxEllipsoidDecomp(const vec_Vecf<3> &path,
                            const vec_Vecf<3> &vec_o,
                            std::vector<LinearConstraint3D> &l_constraints,
                            vec_E<Polyhedron<3>> &poly_out)
    {

        // Initialize result.
        bool result = true;

        // For decomposition
        EllipsoidDecomp3D ellip_decomp_util;

        // Get occupied cells.
        ellip_decomp_util.set_obs(vec_o);

        // Set the local bounding box and z constraints.
        ellip_decomp_util.set_local_bbox(Vec3f(4.0, 4.0, 4.0));
        ellip_decomp_util.set_z_min_and_max(-1.0, 5.0); // buffer for the drone size

        // Find convex polyhedra.
        ellip_decomp_util.dilate(path, result);

        if (!result)
            return false;

        // Get the polyhedra.
        auto polys = ellip_decomp_util.get_polyhedrons();

        // Preallocate the constraints vector.
        size_t numConstraints = (path.size() > 0) ? (path.size() - 1) : 0;
        l_constraints.clear();
        l_constraints.resize(numConstraints);

        // Flag to record if any thread finds an error.
        bool errorFound = false;

        // Parallelize the constraint computation loop.
        for (int i = 0; i < static_cast<int>(numConstraints); i++)
        {

            // Compute the midpoint between consecutive path points.
            auto pt_inside = (path[i] + path[i + 1]) / 2.0;
            LinearConstraint3D cs(pt_inside, polys[i].hyperplanes(), polys[i]);

            // If either matrix A_ or vector b_ contains NaN, mark an error.
            if (cs.A_.hasNaN() || cs.b_.hasNaN())
            {
                errorFound = true;
            }
            else
            {
                l_constraints[i] = cs;
            }
        }

        // If an error was detected, report and exit.
        if (errorFound)
        {
            std::cout << "A_ or b_ has NaN" << std::endl;
            return false;
        }

        // Return the computed polyhedra.
        poly_out = std::move(polys);

        return true;
    }

    void republishMarkers_()
    {
        if (traj.getPieceNum() <= 0)
            return;

        if (!hPolysCache_.empty())
            visualizer.visualizePolytope(hPolysCache_);
        if (!routeCache_.empty())
            visualizer.visualize(traj, routeCache_);
        if (startGoal.size() == 2)
        {
            visualizer.visualizeStartGoal(startGoal[0], 0.5, 0);
            visualizer.visualizeStartGoal(startGoal[1], 0.5, 1);
        }

        if (!mightyKnots_.empty())
            visualizer.visualizePoints(mightyKnots_, 0.35f, 0.f, 1.f, 0.f, 1.f, "odom", "mighty_knots", 0.0);
        if (!M_mighty_.CP.empty())
            visualizer.visualizeBezier(M_mighty_.CP, M_mighty_.T, "mighty", 0.06, 120, 0.f, 1.f, 0.f, 1.f, "odom");
    }

    void process()
    {
        if (traj.getPieceNum() <= 0)
            return;

        double delta = this->now().seconds() - trajStamp;
        if (!(delta < 0.0 || delta > traj.getTotalDuration()))
        {
            double thr;
            Eigen::Vector4d quat;
            Eigen::Vector3d omg;

            Eigen::VectorXd physicalParams(6);
            physicalParams << config.vehicleMass, config.gravAcc, config.horizDrag,
                config.vertDrag, config.parasDrag, config.speedEps;

            flatness::FlatnessMap flatmap;
            flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                          physicalParams(3), physicalParams(4), physicalParams(5));

            flatmap.forward(traj.getVel(delta), traj.getAcc(delta), traj.getJer(delta),
                            0.0, 0.0, thr, quat, omg);

            // Current pose & velocity
            Eigen::Vector3d pos = traj.getPos(delta);
            Eigen::Vector3d vel = traj.getVel(delta);
            double speed = vel.norm();
            double bodyratemag = omg.norm();
            double tiltangle = std::acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));

            std_msgs::msg::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
            speedMsg.data = speed;
            thrMsg.data = thr;
            tiltMsg.data = tiltangle;
            bdrMsg.data = bodyratemag;

            visualizer.speedPub->publish(speedMsg);
            visualizer.thrPub->publish(thrMsg);
            visualizer.tiltPub->publish(tiltMsg);
            visualizer.bdrPub->publish(bdrMsg);

            visualizer.visualizeSphere(pos, config.dilateRadius);

            // Print speed next to it (two decimals)
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << speed << " m/s";
                visualizer.visualizeText(pos, ss.str(),
                                         /*scale_z=*/2.0,
                                         /*r=*/0.f, /*g=*/0.f, /*b=*/0.f, /*a=*/1.f,
                                         /*frame_id=*/"odom",
                                         /*ns=*/"speed_text",
                                         /*id=*/0,
                                         /*ttl_sec=*/0.15,
                                         /*planner=*/"gcopter");
            }
        }

        if (!M_mighty_.CP.empty() && !M_mighty_.T.empty())
        {
            const double now_s = this->now().seconds();
            const double Ttot_m = mightyEdges_.empty() ? std::accumulate(M_mighty_.T.begin(), M_mighty_.T.end(), 0.0)
                                                       : mightyEdges_.back();
            const double delta_m = now_s - mightyStamp_;
            if (delta_m >= 0.0 && delta_m <= Ttot_m)
            {
                // locate segment
                auto it = std::upper_bound(mightyEdges_.begin(), mightyEdges_.end(), delta_m);
                int idx = static_cast<int>(std::distance(mightyEdges_.begin(), it)) - 1;
                int s = std::clamp(idx, 0, (int)M_mighty_.CP.size() - 1);

                const double Ts = std::max(1e-9, M_mighty_.T[s]);
                const double u = std::clamp((delta_m - mightyEdges_[s]) / Ts, 0.0, 1.0);

                Eigen::Vector3d p, v, a, j;
                evalBezier5_PVAJ(M_mighty_.CP[s], Ts, u, p, v, a, j);

                // Current position sphere (green)
                visualizer.visualizeSphereColor(p, /*radius=*/config.dilateRadius,
                                                /*r=*/0.0f, /*g=*/1.0f, /*b=*/0.0f, /*a=*/1.0f,
                                                /*frame=*/"odom",
                                                /*ns=*/"mighty_curr",
                                                /*id=*/0,
                                                /*ttl=*/0.15);

                // Velocity text just above the sphere
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << v.norm() << " m/s";
                visualizer.visualizeText(p + Eigen::Vector3d(0, 0, 0.15), ss.str(),
                                         /*scale_z=*/2.0,
                                         /*r=*/0.f, /*g=*/0.6f, /*b=*/0.f, /*a=*/1.f,
                                         /*frame_id=*/"odom",
                                         /*ns=*/"mighty_speed",
                                         /*id=*/0,
                                         /*ttl_sec=*/0.15,
                                         /*planner=*/"mighty");
            }
        }

        if (M_mighty_.CP.empty() || M_mighty_.T.empty())
        {
            return;
        }

        visualizer.visualizeBezier(M_mighty_.CP, M_mighty_.T, /*ns=*/"mighty", /*width=*/0.06,
                                   /*samples=*/120, /*r=*/0.0f, /*g=*/1.0f, /*b=*/0.0f, /*a=*/1.0f,
                                   /*frame_id=*/"odom");

        if (poly_whole_.empty())
        {
            return;
        }

        // For whole trajectory
        if (!poly_whole_.empty())
        {
            decomp_ros_msgs::msg::PolyhedronArray poly_whole_msg = DecompROS::polyhedron_array_to_ros(poly_whole_);
            poly_whole_msg.header.stamp = this->now();
            poly_whole_msg.header.frame_id = "odom";
            poly_whole_msg.lifetime = rclcpp::Duration::from_seconds(1.0);
            pub_poly_whole_->publish(poly_whole_msg);
        }
    }

    // Check & fix interior waypoints against segment corridors (H rows: [nx ny nz d], inside if n·x + d <= 0).
    // route_m.size() == hPolys.size() + 1 expected (one poly per segment).
    static bool enforceRouteInsideCorridor(vec_Vecf<3> &route_m,
                                           const std::vector<Eigen::MatrixX4d> &hPolys,
                                           double inward_margin = 0.02, // how far to push inside after projection [m]
                                           int max_passes = 5,          // POCS passes per waypoint
                                           double tol = 1e-9)           // numeric tolerance
    {
        const int M = static_cast<int>(route_m.size()) - 1; // number of segments
        if (M <= 0 || (int)hPolys.size() < M)
        {
            // Nothing to do or corridor size mismatch (be permissive).
            return true;
        }

        auto inside_with_margin = [&](const Eigen::Vector3d &p,
                                      const Eigen::MatrixX4d &H,
                                      double margin) -> bool
        {
            // inside if max_k (n_k · p + d_k + margin) <= 0
            const Eigen::VectorXd vals = H.leftCols<3>() * p + H.rightCols<1>();
            return (vals.array() + margin).maxCoeff() <= 0.0 + tol;
        };

        bool all_ok = true;

        // Interior points only: idx = 1..M-1
        for (int idx = 1; idx < M; ++idx)
        {
            // Clamp corridor indices defensively
            const int sL = std::max(0, std::min(idx - 1, (int)hPolys.size() - 1));
            const int sR = std::max(0, std::min(idx, (int)hPolys.size() - 1));

            // Work copy (Eigen) then write back to route_m
            Eigen::Vector3d p = route_m[idx].template cast<double>();

            // Early exit if already inside both with margin
            bool okL = inside_with_margin(p, hPolys[sL], inward_margin);
            bool okR = inside_with_margin(p, hPolys[sR], inward_margin);
            if (okL && okR)
            {
                // write back (no-op usually, keeps numeric type consistent)
                route_m[idx] = p;
                continue;
            }

            // Cyclic projections onto violating planes of both polytopes
            for (int pass = 0; pass < max_passes; ++pass)
            {
                // Left segment poly
                {
                    const Eigen::MatrixX4d &H = hPolys[sL];
                    const int K = (int)H.rows();
                    for (int k = 0; k < K; ++k)
                    {
                        const Eigen::Vector3d n = H.block<1, 3>(k, 0).transpose();
                        const double d = H(k, 3);
                        // we want n·p + d <= -inward_margin
                        const double val = n.dot(p) + d + inward_margin;
                        if (val > 0.0)
                        {
                            const double nn = n.squaredNorm() + 1e-12;
                            // project onto plane n·x + d = -inward_margin (so we land *inside* by margin)
                            p -= (val / nn) * n;
                        }
                    }
                }
                // Right segment poly
                {
                    const Eigen::MatrixX4d &H = hPolys[sR];
                    const int K = (int)H.rows();
                    for (int k = 0; k < K; ++k)
                    {
                        const Eigen::Vector3d n = H.block<1, 3>(k, 0).transpose();
                        const double d = H(k, 3);
                        const double val = n.dot(p) + d + inward_margin;
                        if (val > 0.0)
                        {
                            const double nn = n.squaredNorm() + 1e-12;
                            p -= (val / nn) * n;
                        }
                    }
                }

                // Check stop condition
                bool nowL = inside_with_margin(p, hPolys[sL], inward_margin);
                bool nowR = inside_with_margin(p, hPolys[sR], inward_margin);
                if (nowL && nowR)
                    break;
            }

            // Write back the (possibly adjusted) point
            route_m[idx] = p;

            // Final check
            bool finalL = inside_with_margin(p, hPolys[sL], inward_margin);
            bool finalR = inside_with_margin(p, hPolys[sR], inward_margin);
            if (!(finalL && finalR))
            {
                all_ok = false; // still outside after POCS; keep going to fix others
            }
        }

        return all_ok;
    }

    // Degree-5 Bezier evaluator (pos,vel,acc,jer) at u∈[0,1] for one segment
    static inline void evalBezier5_PVAJ(const std::array<Eigen::Vector3d, 6> &CP,
                                        double T, double u,
                                        Eigen::Vector3d &p,
                                        Eigen::Vector3d &v,
                                        Eigen::Vector3d &a,
                                        Eigen::Vector3d &j)
    {
        const double om = 1.0 - u;

        // position
        const double b0 = std::pow(om, 5);
        const double b1 = 5 * u * std::pow(om, 4);
        const double b2 = 10 * u * u * std::pow(om, 3);
        const double b3 = 10 * std::pow(u, 3) * om * om;
        const double b4 = 5 * std::pow(u, 4) * om;
        const double b5 = std::pow(u, 5);
        p = b0 * CP[0] + b1 * CP[1] + b2 * CP[2] + b3 * CP[3] + b4 * CP[4] + b5 * CP[5];

        // forward differences
        std::array<Eigen::Vector3d, 5> D1;
        for (int i = 0; i < 5; ++i)
            D1[i] = CP[i + 1] - CP[i];
        std::array<Eigen::Vector3d, 4> D2;
        for (int i = 0; i < 4; ++i)
            D2[i] = D1[i + 1] - D1[i];
        std::array<Eigen::Vector3d, 3> D3;
        for (int i = 0; i < 3; ++i)
            D3[i] = D2[i + 1] - D2[i];

        const double invT = 1.0 / std::max(T, 1e-9);
        const double invT2 = invT * invT;
        const double invT3 = invT2 * invT;

        // velocity (deg-4)
        const double c0 = std::pow(om, 4);
        const double c1 = 4 * u * std::pow(om, 3);
        const double c2 = 6 * u * u * om * om;
        const double c3 = 4 * std::pow(u, 3) * om;
        const double c4 = std::pow(u, 4);
        v = 5.0 * invT * (c0 * D1[0] + c1 * D1[1] + c2 * D1[2] + c3 * D1[3] + c4 * D1[4]);

        // acceleration (deg-3)
        const double d0 = std::pow(om, 3);
        const double d1 = 3 * u * om * om;
        const double d2 = 3 * u * u * om;
        const double d3 = std::pow(u, 3);
        a = 20.0 * invT2 * (d0 * D2[0] + d1 * D2[1] + d2 * D2[2] + d3 * D2[3]);

        // jerk (deg-2)
        const double e0 = om * om;
        const double e1 = 2 * u * om;
        const double e2 = u * u;
        j = 60.0 * invT3 * (e0 * D3[0] + e1 * D3[1] + e2 * D3[2]);
    }

    // GCOPTER: sample [0, T_tot] every dt and write CSV
    static void writeCSV_GCOPTER(const Trajectory<5> &traj,
                                 const std::string &filepath,
                                 double dt)
    {
#if __cplusplus >= 201703L
        if (!filepath.empty())
        {
            fs::create_directories(fs::path(filepath).parent_path());
        }
#endif
        std::ofstream f(filepath);
        if (!f.is_open())
        {
            std::cerr << "Failed to open " << filepath << "\n";
            return;
        }
        f << "t,px,py,pz,vx,vy,vz,v_norm,ax,ay,az,a_norm,jx,jy,jz,j_norm\n";

        const double Ttot = traj.getTotalDuration();
        if (Ttot <= 0.0)
        {
            f.close();
            return;
        }

        const int N = std::max(1, (int)std::ceil(Ttot / dt));
        for (int i = 0; i <= N; ++i)
        {
            double t = std::min(Ttot, i * dt);
            Eigen::Vector3d p = traj.getPos(t);
            Eigen::Vector3d v = traj.getVel(t);
            Eigen::Vector3d a = traj.getAcc(t);
            Eigen::Vector3d j = traj.getJer(t);
            f << std::fixed << std::setprecision(6)
              << t << ","
              << p.x() << "," << p.y() << "," << p.z() << ","
              << v.x() << "," << v.y() << "," << v.z() << "," << v.norm() << ","
              << a.x() << "," << a.y() << "," << a.z() << "," << a.norm() << ","
              << j.x() << "," << j.y() << "," << j.z() << "," << j.norm() << "\n";
        }
        f.close();
    }

    // MIGHTY (Bezier CP/T): sample [0, sum(T)] every dt and write CSV
    static void writeCSV_MIGHTY(const std::vector<std::array<Eigen::Vector3d, 6>> &CP,
                                const std::vector<double> &T,
                                const std::string &filepath,
                                double dt)
    {
#if __cplusplus >= 201703L
        if (!filepath.empty())
        {
            fs::create_directories(fs::path(filepath).parent_path());
        }
#endif
        std::ofstream f(filepath);
        if (!f.is_open())
        {
            std::cerr << "Failed to open " << filepath << "\n";
            return;
        }
        f << "t,px,py,pz,vx,vy,vz,v_norm,ax,ay,az,a_norm,jx,jy,jz,j_norm\n";

        const int M = (int)CP.size();
        if (M == 0)
        {
            f.close();
            return;
        }

        // cumulative durations
        std::vector<double> edges(M + 1, 0.0);
        for (int s = 0; s < M; ++s)
            edges[s + 1] = edges[s] + T[s];
        const double Ttot = edges.back();
        const int N = std::max(1, (int)std::ceil(Ttot / dt));

        int seg = 0;
        for (int i = 0; i <= N; ++i)
        {
            double t = std::min(Ttot, i * dt);

            // advance segment
            while (seg < M - 1 && t > edges[seg + 1])
                ++seg;

            const double Ts = T[seg];
            const double t0 = edges[seg];
            const double u = (Ts <= 0.0) ? 0.0 : (t - t0) / Ts;

            Eigen::Vector3d p, v, a, j;
            evalBezier5_PVAJ(CP[seg], Ts, std::clamp(u, 0.0, 1.0), p, v, a, j);

            f << std::fixed << std::setprecision(6)
              << t << ","
              << p.x() << "," << p.y() << "," << p.z() << ","
              << v.x() << "," << v.y() << "," << v.z() << "," << v.norm() << ","
              << a.x() << "," << a.y() << "," << a.z() << "," << a.norm() << ","
              << j.x() << "," << j.y() << "," << j.z() << "," << j.norm() << "\n";
        }
        f.close();
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GlobalPlanner>();

    rclcpp::Rate rate(1000);
    while (rclcpp::ok())
    {
        node->process();
        rclcpp::spin_some(node);
        rate.sleep();
    }
    rclcpp::shutdown();
    return 0;
}
