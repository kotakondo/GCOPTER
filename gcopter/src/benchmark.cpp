// benchmark.cpp — Headless GCOPTER vs MIGHTY benchmark runner
// Builds corridor from voxel map, plans both, samples metrics, checks collision,
// writes two CSV rows, then exits.
//
// Depends on the same libs as your minco_bench_viz node (GCOPTER, MIGHTY LBFGS).
//
// Build: add_executable(benchmark src/benchmark.cpp) and install in CMakeLists.txt.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <Eigen/Eigen>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#endif

// GCOPTER bits
#include "gcopter/trajectory.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/voxel_map.hpp"

// MIGHTY / LBFGS
#include <dynus/lbfgs_solver.hpp> // SolverLBFGS, planner_params_t, LinearConstraint3D

using lbfgs::planner_params_t;
using lbfgs::SolverLBFGS;
using lbfgs::Vec3;
template <int Dim>

// ---------- config read ----------
struct Config
{
  // Map
  std::string MapTopic{"/voxel_map"};
  double DilateRadius{0.5};
  double VoxelWidth{0.25};
  std::vector<double> MapBound{-25, 25, -25, 25, 0, 5};

  // Start/Goal
  std::vector<double> Start{0.0, 0.0, 0.5};
  std::vector<double> Goal{10.0, 3.0, 1.5};

  // Corridor gen
  double SFC_Progress{7.0};
  double SFC_Range{3.0};
  double TimeoutRRT{0.1};

  // Limits, weights, physicals (GCOPTER)
  double MaxVelMag{4.0};
  double MaxBdrMag{2.1};
  double MaxTiltAngle{1.05};
  double MinThrust{2.0};
  double MaxThrust{12.0};

  double VehicleMass{0.61};
  double GravAcc{9.8};
  double HorizDrag{0.70};
  double VertDrag{0.80};
  double ParasDrag{0.01};
  double SpeedEps{1.0e-4};

  double WeightT{20.0};
  std::vector<double> ChiVec{1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e5};
  double SmoothingEps{1.0e-2};
  int IntegralIntervs{16};
  double RelCostTol{1.0e-5};

  // Sampling & IO
  double SampleDt{0.02};
  double CollisionDt{0.02};
  std::string OutCSV{""};
  bool QuitOnFinish{true};

  // MIGHTY
  double MIGHTYJerkWeight{0.1};

  // Book-keeping
  int TrialID{0};
  int MapSeed{0};

  Config(rclcpp::Node &node)
  {
    // declare/get
    node.declare_parameter("MapTopic", MapTopic);
    node.declare_parameter("DilateRadius", DilateRadius);
    node.declare_parameter("VoxelWidth", VoxelWidth);
    node.declare_parameter("MapBound", MapBound);

    node.declare_parameter("Start", Start);
    node.declare_parameter("Goal", Goal);

    node.declare_parameter("SFC_Progress", SFC_Progress);
    node.declare_parameter("SFC_Range", SFC_Range);
    node.declare_parameter("TimeoutRRT", TimeoutRRT);

    node.declare_parameter("MaxVelMag", MaxVelMag);
    node.declare_parameter("MaxBdrMag", MaxBdrMag);
    node.declare_parameter("MaxTiltAngle", MaxTiltAngle);
    node.declare_parameter("MinThrust", MinThrust);
    node.declare_parameter("MaxThrust", MaxThrust);

    node.declare_parameter("VehicleMass", VehicleMass);
    node.declare_parameter("GravAcc", GravAcc);
    node.declare_parameter("HorizDrag", HorizDrag);
    node.declare_parameter("VertDrag", VertDrag);
    node.declare_parameter("ParasDrag", ParasDrag);
    node.declare_parameter("SpeedEps", SpeedEps);

    node.declare_parameter("WeightT", WeightT);
    node.declare_parameter("ChiVec", ChiVec);
    node.declare_parameter("SmoothingEps", SmoothingEps);
    node.declare_parameter("IntegralIntervs", IntegralIntervs);
    node.declare_parameter("RelCostTol", RelCostTol);

    node.declare_parameter("SampleDt", SampleDt);
    node.declare_parameter("CollisionDt", CollisionDt);
    node.declare_parameter("OutCSV", OutCSV);
    node.declare_parameter("QuitOnFinish", QuitOnFinish);

    node.declare_parameter("MIGHTYJerkWeight", MIGHTYJerkWeight);

    node.declare_parameter("TrialID", TrialID);
    node.declare_parameter("MapSeed", MapSeed);

    // get
    node.get_parameter("MapTopic", MapTopic);
    node.get_parameter("DilateRadius", DilateRadius);
    node.get_parameter("VoxelWidth", VoxelWidth);
    node.get_parameter("MapBound", MapBound);

    node.get_parameter("Start", Start);
    node.get_parameter("Goal", Goal);

    node.get_parameter("SFC_Progress", SFC_Progress);
    node.get_parameter("SFC_Range", SFC_Range);
    node.get_parameter("TimeoutRRT", TimeoutRRT);

    node.get_parameter("MaxVelMag", MaxVelMag);
    node.get_parameter("MaxBdrMag", MaxBdrMag);
    node.get_parameter("MaxTiltAngle", MaxTiltAngle);
    node.get_parameter("MinThrust", MinThrust);
    node.get_parameter("MaxThrust", MaxThrust);

    node.get_parameter("VehicleMass", VehicleMass);
    node.get_parameter("GravAcc", GravAcc);
    node.get_parameter("HorizDrag", HorizDrag);
    node.get_parameter("VertDrag", VertDrag);
    node.get_parameter("ParasDrag", ParasDrag);
    node.get_parameter("SpeedEps", SpeedEps);

    node.get_parameter("WeightT", WeightT);
    node.get_parameter("ChiVec", ChiVec);
    node.get_parameter("SmoothingEps", SmoothingEps);
    node.get_parameter("IntegralIntervs", IntegralIntervs);
    node.get_parameter("RelCostTol", RelCostTol);

    node.get_parameter("SampleDt", SampleDt);
    node.get_parameter("CollisionDt", CollisionDt);
    node.get_parameter("OutCSV", OutCSV);
    node.get_parameter("QuitOnFinish", QuitOnFinish);

    node.get_parameter("MIGHTYJerkWeight", MIGHTYJerkWeight);

    node.get_parameter("TrialID", TrialID);
    node.get_parameter("MapSeed", MapSeed);
  }
};

// ---------- shared metrics ----------
struct Metrics
{
  double time_s{0.0};
  double path_len{0.0};
  double jerk_cost{0.0}; // ∫||j|| dt
};

// Bezier degree-5 evaluator (pos, vel, acc, jer)
static inline void evalBezier5_PVAJ(const std::array<Eigen::Vector3d, 6> &CP,
                                    double T, double u,
                                    Eigen::Vector3d &p,
                                    Eigen::Vector3d &v,
                                    Eigen::Vector3d &a,
                                    Eigen::Vector3d &j)
{
  const double om = 1.0 - u;

  const double b0 = std::pow(om, 5);
  const double b1 = 5 * u * std::pow(om, 4);
  const double b2 = 10 * u * u * std::pow(om, 3);
  const double b3 = 10 * std::pow(u, 3) * om * om;
  const double b4 = 5 * std::pow(u, 4) * om;
  const double b5 = std::pow(u, 5);
  p = b0 * CP[0] + b1 * CP[1] + b2 * CP[2] + b3 * CP[3] + b4 * CP[4] + b5 * CP[5];

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

  const double c0 = std::pow(om, 4);
  const double c1 = 4 * u * std::pow(om, 3);
  const double c2 = 6 * u * u * om * om;
  const double c3 = 4 * std::pow(u, 3) * om;
  const double c4 = std::pow(u, 4);
  v = 5.0 * invT * (c0 * D1[0] + c1 * D1[1] + c2 * D1[2] + c3 * D1[3] + c4 * D1[4]);

  const double d0 = std::pow(om, 3);
  const double d1 = 3 * u * om * om;
  const double d2 = 3 * u * u * om;
  const double d3 = std::pow(u, 3);
  a = 20.0 * invT2 * (d0 * D2[0] + d1 * D2[1] + d2 * D2[2] + d3 * D2[3]);

  const double e0 = om * om;
  const double e1 = 2 * u * om;
  const double e2 = u * u;
  j = 60.0 * invT3 * (e0 * D3[0] + e1 * D3[1] + e2 * D3[2]);
}

// ∫||v|| dt and ∫||j|| dt for GCOPTER by uniform sampling
static Metrics computeMetricsGCOPTER(const Trajectory<5> &traj, double dt)
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

    M.path_len += 0.5 * (v_prev.norm() + v.norm()) * hi;
    M.jerk_cost += 0.5 * (j_prev.norm() + j.norm()) * hi;

    t_prev = t;
    v_prev = v;
    j_prev = j;
  }
  return M;
}

// ∫||v|| dt and ∫||j|| dt for MIGHTY by uniform sampling
static Metrics computeMetricsMIGHTY(const std::vector<std::array<Eigen::Vector3d, 6>> &CP,
                                    const std::vector<double> &T,
                                    double dt)
{
  Metrics M;
  const int Mseg = (int)CP.size();
  if (Mseg == 0)
    return M;

  std::vector<double> edges(Mseg + 1, 0.0);
  for (int s = 0; s < Mseg; ++s)
    edges[s + 1] = edges[s] + T[s];
  const double Ttot = edges.back();
  M.time_s = Ttot;

  const double h = std::max(1e-6, dt);
  const int N = std::max(1, (int)std::ceil(Ttot / h));

  auto eval_vj = [&](double t) -> std::pair<Eigen::Vector3d, Eigen::Vector3d>
  {
    t = std::clamp(t, 0.0, Ttot);
    int s = std::clamp((int)(std::upper_bound(edges.begin(), edges.end(), t) - edges.begin()) - 1, 0, Mseg - 1);
    const double Ts = std::max(1e-9, T[s]);
    const double u = std::clamp((t - edges[s]) / Ts, 0.0, 1.0);
    Eigen::Vector3d p, v, a, j;
    evalBezier5_PVAJ(CP[s], Ts, u, p, v, a, j);
    return {v, j};
  };

  double t_prev = 0.0;
  auto [v_prev, j_prev] = eval_vj(0.0);

  for (int i = 1; i <= N; ++i)
  {
    const double t = std::min(Ttot, i * h);
    const double hi = t - t_prev;

    auto [v, j] = eval_vj(t);
    M.path_len += 0.5 * (v_prev.norm() + v.norm()) * hi;
    M.jerk_cost += 0.5 * (j_prev.norm() + j.norm()) * hi;

    t_prev = t;
    v_prev = v;
    j_prev = j;
  }
  return M;
}

// Collision check by sampling positions
static bool collidesGCOPTER(const voxel_map::VoxelMap &vm, const Trajectory<5> &traj, double dt)
{
  const double Ttot = traj.getTotalDuration();
  if (Ttot <= 0.0)
    return false;
  const int N = std::max(1, (int)std::ceil(Ttot / std::max(dt, 1e-6)));
  for (int i = 0; i <= N; ++i)
  {
    double t = std::min(Ttot, i * dt);
    Eigen::Vector3d p = traj.getPos(t);
    if (vm.query(p) != 0)
      return true;
  }
  return false;
}

static bool collidesMIGHTY(const voxel_map::VoxelMap &vm,
                           const std::vector<std::array<Eigen::Vector3d, 6>> &CP,
                           const std::vector<double> &T,
                           double dt)
{
  const int Mseg = (int)CP.size();
  if (Mseg == 0)
    return false;

  std::vector<double> edges(Mseg + 1, 0.0);
  for (int s = 0; s < Mseg; ++s)
    edges[s + 1] = edges[s] + T[s];
  const double Ttot = edges.back();

  const int N = std::max(1, (int)std::ceil(Ttot / std::max(dt, 1e-6)));

  int s = 0;
  for (int i = 0; i <= N; ++i)
  {
    double t = std::min(Ttot, i * dt);
    while (s < Mseg - 1 && t > edges[s + 1])
      ++s;
    const double Ts = std::max(1e-9, T[s]);
    const double u = std::clamp((t - edges[s]) / Ts, 0.0, 1.0);
    Eigen::Vector3d p, v, a, j;
    evalBezier5_PVAJ(CP[s], Ts, u, p, v, a, j);
    if (vm.query(p) != 0)
      return true;
  }
  return false;
}

// GCOPTER hPolys (rows [nx ny nz d] with n·x + d <= 0) to MIGHTY Ax >= b (A x >= b)
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
    Eigen::MatrixXd A = H.leftCols<3>();
    Eigen::VectorXd b = -H.col(3); // n·x + d <= 0  =>  n·x >= -d  =>  A x >= b
    out.emplace_back(A, b);
  }
  return out;
}

struct MightyOut
{
  std::vector<std::array<Eigen::Vector3d, 6>> CP;
  std::vector<double> T;
  double obj{0.0};
  double wall_ms{0.0};
};

static MightyOut runMighty(
    const vec_Vecf<3> &route_m,
    const std::vector<LinearConstraint3D> &safe_corridor,
    const state &initial_state,
    const state &final_state,
    const planner_params_t &params)
{
  MightyOut out;

  auto solver = std::make_shared<SolverLBFGS>();
  solver->initializeSolver(params);

  std::vector<std::shared_ptr<dynTraj>> obstacles; // none
  double t0 = 0.0, ig_ms = 0.0;

  solver->prepareSolverForReplan(
      t0, route_m, safe_corridor, obstacles,
      initial_state, final_state, ig_ms, false);

  const auto &z0s = solver->getInitialGuesses();
  if (z0s.empty())
    return out;

  lbfgs::lbfgs_parameter_t lb{};
  lb.mem_size = (int)z0s.front().size();
  lb.past = 20;
  lb.max_linesearch = 64;
  lb.max_iterations = 300;
  lb.delta = 1e-6;
  lb.g_epsilon = 1e-6;

  auto t_start = std::chrono::steady_clock::now();
  Eigen::VectorXd zopt;
  double fopt = 0.0;
  solver->optimize(z0s.front(), zopt, fopt, lb);
  auto t_end = std::chrono::steady_clock::now();

  out.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  out.obj = fopt;

  // Recover CP/T
  {
    std::vector<Vec3> P, V, A;
    std::vector<std::array<Vec3, 6>> CPv;
    std::vector<double> Tv;
    solver->reconstruct(zopt, P, V, A, CPv, Tv);

    out.CP.resize(CPv.size());
    for (size_t s = 0; s < CPv.size(); ++s)
      for (int j = 0; j < 6; ++j)
        out.CP[s][j] = Eigen::Vector3d(CPv[s][j].x(), CPv[s][j].y(), CPv[s][j].z());
    out.T = std::move(Tv);
  }
  return out;
}

// Append one row to CSV (create & header if needed)
static void appendCSVRow(const std::string &path,
                         const std::vector<std::string> &fields)
{
#if __cplusplus >= 201703L
  if (!path.empty())
    fs::create_directories(fs::path(path).parent_path());
#endif
  const bool need_header = !std::ifstream(path).good() || (std::ifstream(path).peek(), std::ifstream(path).rdbuf()->in_avail() == 0);
  std::ofstream f(path, std::ios::app);
  if (!f.is_open())
  {
    std::cerr << "Cannot open CSV: " << path << "\n";
    return;
  }

  if (need_header)
  {
    f << "trial,seed,max_vel,mighty_jerk_weight,planner,solve_ms,travel_time_s,path_len_m,jerk_int,collision\n";
  }
  for (size_t i = 0; i < fields.size(); ++i)
  {
    if (i)
      f << ",";
    f << fields[i];
  }
  f << "\n";
  f.close();
}

// --------- node ----------
class BenchmarkNode : public rclcpp::Node
{
public:
  BenchmarkNode() : Node("benchmark"), cfg_(*this)
  {
    // Build voxel map from bounds & voxel size
    Eigen::Vector3i xyz(
        (int)((cfg_.MapBound[1] - cfg_.MapBound[0]) / cfg_.VoxelWidth),
        (int)((cfg_.MapBound[3] - cfg_.MapBound[2]) / cfg_.VoxelWidth),
        (int)((cfg_.MapBound[5] - cfg_.MapBound[4]) / cfg_.VoxelWidth));
    Eigen::Vector3d offset(cfg_.MapBound[0], cfg_.MapBound[2], cfg_.MapBound[4]);
    vmap_ = voxel_map::VoxelMap(xyz, offset, cfg_.VoxelWidth);

    mapSub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cfg_.MapTopic, rclcpp::SensorDataQoS(),
        std::bind(&BenchmarkNode::mapCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "benchmark waiting for map on %s", cfg_.MapTopic.c_str());
  }

private:
  void mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (mapReady_)
      return;

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
        vmap_.setOccupied(Eigen::Vector3d(x, y, z));
    }
    vmap_.dilate(std::ceil(cfg_.DilateRadius / vmap_.getScale()));
    mapReady_ = true;

    RCLCPP_INFO(get_logger(), "Map ingested: %zu pts (dilated).", total);

    // trigger once
    planAndRecord_();
    if (cfg_.QuitOnFinish)
    {
      rclcpp::shutdown();
    }
  }

  void planAndRecord_()
  {
    // clamp start/goal to bounds
    auto S = cfg_.Start, G = cfg_.Goal;
    auto clamp = [&](double v, double lo, double hi)
    { return std::max(lo, std::min(v, hi)); };
    for (int i = 0; i < 3; ++i)
    {
      S[i] = clamp(S[i], cfg_.MapBound[2 * i + 0], cfg_.MapBound[2 * i + 1]);
      G[i] = clamp(G[i], cfg_.MapBound[2 * i + 0], cfg_.MapBound[2 * i + 1]);
    }
    Eigen::Vector3d start(S[0], S[1], S[2]), goal(G[0], G[1], G[2]);
    if (vmap_.query(start) != 0)
      start.z() = clamp(start.z() + cfg_.DilateRadius, cfg_.MapBound[4], cfg_.MapBound[5]);
    if (vmap_.query(goal) != 0)
      goal.z() = clamp(goal.z() + cfg_.DilateRadius, cfg_.MapBound[4], cfg_.MapBound[5]);

    // RRT route + corridor
    std::vector<Eigen::Vector3d> route;
    sfc_gen::planPath<voxel_map::VoxelMap>(
        start, goal, vmap_.getOrigin(), vmap_.getCorner(),
        &vmap_, cfg_.TimeoutRRT, route);

    if (route.size() < 2)
    {
      RCLCPP_ERROR(get_logger(), "Global route failed — not writing CSV.");
      return;
    }

    std::vector<Eigen::MatrixX4d> hPolys;
    std::vector<Eigen::Vector3d> pc;
    vmap_.getSurf(pc);
    sfc_gen::convexCover(route, pc, vmap_.getOrigin(), vmap_.getCorner(),
                         cfg_.SFC_Progress, cfg_.SFC_Range, hPolys);
    sfc_gen::shortCut(hPolys);

    // boundary states
    Eigen::Matrix3d ini, fin;
    ini << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
    fin << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

    // === GCOPTER ===
    gcopter::GCOPTER_PolytopeSFC solver;
    Eigen::VectorXd magnitudeBounds(5), penaltyWeights(5), physicalParams(6);
    magnitudeBounds << cfg_.MaxVelMag, cfg_.MaxBdrMag, cfg_.MaxTiltAngle, cfg_.MinThrust, cfg_.MaxThrust;
    penaltyWeights << cfg_.ChiVec[0], cfg_.ChiVec[1], cfg_.ChiVec[2], cfg_.ChiVec[3], cfg_.ChiVec[4];
    physicalParams << cfg_.VehicleMass, cfg_.GravAcc, cfg_.HorizDrag, cfg_.VertDrag, cfg_.ParasDrag, cfg_.SpeedEps;

    Trajectory<5> traj;
    if (!solver.setup(cfg_.WeightT, ini, fin, hPolys,
                      INFINITY, cfg_.SmoothingEps, cfg_.IntegralIntervs,
                      magnitudeBounds, penaltyWeights, physicalParams))
    {
      RCLCPP_ERROR(get_logger(), "GCOPTER setup failed.");
      return;
    }

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    double optCost = solver.optimize(traj, cfg_.RelCostTol);
    auto t1 = Clock::now();
    (void)optCost;

    const double gc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    Metrics m_gc = computeMetricsGCOPTER(traj, cfg_.SampleDt);
    const bool coll_gc = collidesGCOPTER(vmap_, traj, cfg_.CollisionDt);

    // initial guess that GCOPTER used (optional)
    Eigen::Matrix3Xd init_pts;
    Eigen::VectorXd init_times;
    solver.getInitialGuess(init_pts, init_times);

    // === MIGHTY ===
    planner_params_t mighty{};
    mighty.verbose = false;
    mighty.V_nom = cfg_.MaxVelMag;
    mighty.V_max = cfg_.MaxVelMag;
    mighty.A_max = 0.0;
    mighty.J_max = 0.0;
    mighty.time_weight = cfg_.WeightT;
    mighty.dyn_weight = 0.0;
    mighty.stat_weight = cfg_.ChiVec[0];
    mighty.jerk_weight = cfg_.MIGHTYJerkWeight;
    mighty.dyn_constr_vel_weight = cfg_.ChiVec[1];
    mighty.dyn_constr_acc_weight = 0.0;
    mighty.omega_weight = cfg_.ChiVec[2];
    mighty.theta_weight = cfg_.ChiVec[3];
    mighty.thrust_min_weight = cfg_.ChiVec[4];
    mighty.thrust_max_weight = cfg_.ChiVec[4];
    mighty.num_dyn_obst_samples = 64;
    mighty.init_turn_bf = 40.0;
    mighty.Co = 0.05;
    mighty.Cw = 0.40;
    mighty.BIG = 1e9;
    mighty.dc = 0.01;
    mighty.second_to_last_vel_scale = 1.0;

    state s0, s1;
    s0.pos = Vec3(route.front().x(), route.front().y(), route.front().z());
    s0.vel = Vec3::Zero();
    s0.accel = Vec3::Zero();
    s1.pos = Vec3(route.back().x(), route.back().y(), route.back().z());
    s1.vel = Vec3::Zero();
    s1.accel = Vec3::Zero();

    vec_Vecf<3> route_m;
    route_m.reserve(route.size());
    for (const auto &p : route)
      route_m.emplace_back(p.x(), p.y(), p.z());

    std::vector<LinearConstraint3D> lcorr = toLinearConstraints(hPolys);

    MightyOut mout = runMighty(route_m, lcorr, s0, s1, mighty);
    Metrics m_my = computeMetricsMIGHTY(mout.CP, mout.T, cfg_.SampleDt);
    const bool coll_my = collidesMIGHTY(vmap_, mout.CP, mout.T, cfg_.CollisionDt);

    // write CSV (two rows)
    auto rowGC = std::vector<std::string>{
        std::to_string(cfg_.TrialID),
        std::to_string(cfg_.MapSeed),
        std::to_string(cfg_.MaxVelMag),
        std::to_string(cfg_.MIGHTYJerkWeight),
        "GCOPTER",
        std::to_string((long long)gc_ms),
        fmt6(m_gc.time_s), fmt6(m_gc.path_len), fmt6(m_gc.jerk_cost),
        coll_gc ? "1" : "0"};
    appendCSVRow(cfg_.OutCSV, rowGC);

    auto rowMY = std::vector<std::string>{
        std::to_string(cfg_.TrialID),
        std::to_string(cfg_.MapSeed),
        std::to_string(cfg_.MaxVelMag),
        std::to_string(cfg_.MIGHTYJerkWeight),
        "MIGHTY",
        std::to_string((long long)std::llround(mout.wall_ms)),
        fmt6(m_my.time_s), fmt6(m_my.path_len), fmt6(m_my.jerk_cost),
        coll_my ? "1" : "0"};
    appendCSVRow(cfg_.OutCSV, rowMY);

    RCLCPP_INFO(get_logger(), "Wrote rows to %s", cfg_.OutCSV.c_str());
  }

  // helper to format double with 6 decimals
  static std::string fmt6(double x)
  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << x;
    return ss.str();
  }

  // members
  Config cfg_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mapSub_;
  voxel_map::VoxelMap vmap_;
  bool mapReady_{false};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BenchmarkNode>());
  rclcpp::shutdown();
  return 0;
}
