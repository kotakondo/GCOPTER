/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <dynus/lbfgs_solver.hpp>
#include <chrono>

using namespace lbfgs;

// -----------------------------------------------------------------------------

SolverLBFGS::SolverLBFGS()
{
    // Constructor implementation
}

// -----------------------------------------------------------------------------

SolverLBFGS::~SolverLBFGS()
{
    // Destructor implementation
    // Clean up resources if necessary
}

// -----------------------------------------------------------------------------

void SolverLBFGS::initializeSolver(const planner_params_t &params)
{
    // Initialize the solver with parameters that won't change throughout the mission
    verbose_ = params.verbose;                                   // Verbosity level
    V_nom_ = params.V_nom;                                       // Nominal velocity
    V_max_ = params.V_max;                                       // Max velocity
    A_max_ = params.A_max;                                       // Max acceleration
    J_max_ = params.J_max;                                       // Max jerk
    num_perturbation_ = params.num_perturbation;                 // Number of perturbations for initial guesses
    r_max_ = params.r_max;                                       // Perturbation radius for initial guesses
    time_weight_ = params.time_weight;                           // Weight for time in the objective
    dyn_weight_ = params.dyn_weight;                             // Weight for dynamic avoidance in the objective
    stat_weight_ = params.stat_weight;                           // Weight for static avoidance in the objective
    jerk_weight_ = params.jerk_weight;                           // Weight for jerk in the objective
    dyn_constr_vel_weight_ = params.dyn_constr_vel_weight;       // Weight for dynamic velocity constraints
    dyn_constr_acc_weight_ = params.dyn_constr_acc_weight;       // Weight for dynamic acceleration constraints
    theta_weight_ = params.theta_weight;                         // Weight for tilt angle constraints
    thrust_min_weight_ = params.thrust_min_weight;               // Weight for minimum thrust constraints
    thrust_max_weight_ = params.thrust_max_weight;               // Weight for maximum thrust constraints
    omega_weight_ = params.omega_weight;                         // Weight for angular velocity constraints
    omega_max_ = params.omega_max;                               // Maximum angular velocity
    theta_max_ = params.theta_max;                               // Maximum tilt angle
    thrust_min_ = params.thrust_min;                             // Minimum thrust
    thrust_max_ = params.thrust_max;                             // Maximum thrust
    num_dyn_obst_samples_ = params.num_dyn_obst_samples;         // Number of dynamic obstacle samples
    Co_ = params.Co;                                             // Clearance distance for static obstacle avoidance
    Cw_ = params.Cw;                                             // Clearance distance for dynamic obstacle avoidance
    BIG_ = params.BIG;                                           // A large constant for static constraints
    dc_ = params.dc;                                             // Discretization constant
    second_to_last_vel_scale_ = params.second_to_last_vel_scale; // Scale for the second to last velocity vector
    V_min_ = 0.0;                                                // Minimum speed
    turn_buf_ = params.init_turn_bf * M_PI / 180;                // 15° in radians
    turn_span_ = M_PI - turn_buf_;                               // over which we ramp down
    cos_thresh_ = std::cos(turn_buf_ * M_PI / 180.0);
    // Precompute constants
    Cw2_ = Cw_ * Cw_;

    // --- 2) parameters ---
}

// -----------------------------------------------------------------------------

void SolverLBFGS::pushWaypointsByStaticCorridor(std::vector<Vec3> &wps)
{

    int N = int(wps.size());
    // push every point except the very first
    for (int idx = 1; idx < N; ++idx)
    {
        // skip nearly‐straight interior knots
        if (idx < N - 1)
        {
            Vec3 v_prev = wps[idx] - wps[idx - 1];
            Vec3 v_next = wps[idx + 1] - wps[idx];
            double n1 = v_prev.norm(), n2 = v_next.norm();
            if (n1 > 1e-6 && n2 > 1e-6)
            {
                double cosang = v_prev.dot(v_next) / (n1 * n2);
                if (cosang >= cos_thresh_)
                    continue;
            }
        }

        Vec3 p0 = wps[idx];
        Vec3 sum_move = Vec3::Zero();
        int cnt = 0;

        // gather all individual pushes
        for (int seg : {idx - 1, idx})
        {
            if (seg < 0 || seg >= int(A_stat_.size()))
                continue;

            auto const &A = A_stat_[seg]; // (m × 3)
            auto const &b = b_stat_[seg]; // (m)

            for (int k = 0; k < A.rows(); ++k)
            {
                Eigen::RowVector3d arow = A.row(k);
                double bi = b[k];
                double na = arow.norm();
                if (na < 1e-12)
                    continue;

                Vec3 n = arow.transpose() / na;             // unit normal
                double d_signed = (arow.dot(p0) - bi) / na; // signed distance

                if (d_signed < Co_)
                {
                    sum_move += (Co_ - d_signed) * n;
                    ++cnt;
                }
            }
        }

        // if there were any pushes, average them and test feasibility
        if (cnt > 0)
        {
            Vec3 p_new = p0 + (sum_move / double(cnt));

            // verify p_new satisfies A·p ≥ b (no violation)
            bool feasible = true;
            for (int seg : {idx - 1, idx})
            {
                if (seg < 0 || seg >= int(A_stat_.size()))
                    continue;
                auto const &A = A_stat_[seg];
                auto const &b = b_stat_[seg];
                for (int k = 0; k < A.rows(); ++k)
                {
                    Eigen::RowVector3d arow = A.row(k);
                    double bi = b[k];
                    double d_new = arow.dot(p_new) - bi;
                    if (d_new < 0.0)
                    {
                        feasible = false;
                        break;
                    }
                }
                if (!feasible)
                    break;
            }

            // commit the averaged push only if we stay inside
            if (feasible)
                wps[idx] = p_new;
        }
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::prepareSolverForReplan(double t0,
                                         const vec_Vec3f &global_wps,
                                         const std::vector<LinearConstraint3D> &safe_corridor,
                                         const std::vector<std::shared_ptr<dynTraj>> &obstacles,
                                         const state &initial_state,
                                         const state &goal_state,
                                         double &initial_guess_computation_time,
                                         bool use_multiple_initial_guesses)
{
    // Initialize the solver with parameters that changes for replanning
    t0_ = t0; // Current time in the trajectory

    // Global waypoints
    global_wps_.clear();
    for (const auto &wp : global_wps)
    {
        global_wps_.emplace_back(wp.x(), wp.y(), wp.z());
    }

    // Number of segments
    M_ = global_wps_.size() - 1; // Number of segments is one less than the number of waypoints

    // joint_indices_
    for (int i = 1; i < M_; ++i)
    {
        joint_indices_.push_back(i);
    }

    // free control points indices
    K_cp_ = 9 * (M_ + 1); // Number of decision variables (CPs + slack times)
    K_sig_ = M_;          // Number of slack times (K_sig_ = M_)
    K_ = K_cp_ + K_sig_;  // Total number of decision variables

    // Set initial and goal states
    x0_ = initial_state.pos;
    v0_ = initial_state.vel;
    a0_ = initial_state.accel;

    // Copy the dynamic obstacles
    obstacles_.clear();
    obstacles_ = obstacles;

    // Create A_stat_ and b_stat_ from safe corridor
    setStaticConstraints(safe_corridor);

    // Push the waypoints by static corridor
    // pushWaypointsByStaticCorridor(global_wps_);

    xf_ = global_wps_.back();
    vf_ = goal_state.vel;
    af_ = goal_state.accel;

    if (use_multiple_initial_guesses)
    {
        // Create constraint sets (will be used in buildInitialGuesses) from A_stat_ and b_stat_
        ConstraintBlocks constraint_sets;
        for (int idx : joint_indices_)
        {
            std::vector<PlaneBlock> blocks;
            if (A_stat_[idx - 1].rows() > 0)
                blocks.emplace_back(A_stat_[idx - 1], b_stat_[idx - 1]);
            if (A_stat_[idx].rows() > 0)
                blocks.emplace_back(A_stat_[idx], b_stat_[idx]);
            constraint_sets.push_back(std::move(blocks));
        }

        // Generate initial guesses
        auto start_time = std::chrono::high_resolution_clock::now();
        buildInitialGuesses(constraint_sets);
        auto end_time = std::chrono::high_resolution_clock::now();
        // compute initial guess time in ms
        initial_guess_computation_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    else
    {
        // Single initial guess: use min-jerk helper
        initial_guess_wps_.clear();
        initial_guess_wps_.push_back(global_wps_);

        list_z0_.clear();

        // Allocate containers
        Eigen::VectorXd z0;
        std::vector<Vec3> P, V, A;
        std::vector<double> T;

        // Use the min-jerk initializer
        auto start_time = std::chrono::high_resolution_clock::now();
        findInitialGuess(T, V, A);
        auto end_time = std::chrono::high_resolution_clock::now();
        // compute initial guess time in ms
        initial_guess_computation_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        for (int i = 0; i < M_ + 1; ++i)
        {
            P.push_back(global_wps_[i]);
        }

        // Generate the initial guess in z form
        packDecisionVariables(P, V, A, T, z0);

        // Store the results
        list_z0_.push_back(z0);
    }

    // Sanity check
    sanityCheck();
}

// -----------------------------------------------------------------------------

void SolverLBFGS::getGlobalPath(vec_Vecf<3> &global_path)
{
    global_path.clear();
    global_path.reserve(global_wps_.size());
    for (auto const &wp : global_wps_)
    {
        // wp is Eigen::Vector3d; Vec3f is Eigen::Matrix<double,3,1>
        global_path.emplace_back(wp.x(), wp.y(), wp.z());
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::packDecisionVariables(
    const std::vector<Vec3> &P,
    const std::vector<Vec3> &V,
    const std::vector<Vec3> &A,
    const std::vector<double> &T,
    VecXd &z) const
{

    z.resize(K_);

    // 1) pack all (p,v,a) for each knot i=0..M_
    for (int i = 0; i < M_ + 1; ++i)
    {
        int base = 9 * i;
        const Vec3 &pi = P[i];
        const Vec3 &vi = V[i];
        const Vec3 &ai = A[i];

        z(base + 0) = pi.x();
        z(base + 1) = pi.y();
        z(base + 2) = pi.z();

        z(base + 3) = vi.x();
        z(base + 4) = vi.y();
        z(base + 5) = vi.z();

        z(base + 6) = ai.x();
        z(base + 7) = ai.y();
        z(base + 8) = ai.z();
    }

    // 2) pack log‐times σ[s] = log(T[s]) for s=0..M_-1
    for (int s = 0; s < M_; ++s)
    {
        z[K_cp_ + s] = std::log(T[s]);
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::reconstruct(
    const VecXd &z,
    std::vector<Vec3> &P,
    std::vector<Vec3> &V,
    std::vector<Vec3> &A,
    std::vector<std::array<Vec3, 6>> &CP,
    std::vector<double> &T) const
{
    // Number of knots = M_+1, segments = M_
    int knotCount = M_ + 1;

    // Clear and resize outputs
    P.resize(knotCount);
    V.resize(knotCount);
    A.resize(knotCount);
    T.resize(M_);
    CP.resize(M_);

    // 1) Unpack P, V, A from z[0 .. 9*(M_+1)-1]
    for (int i = 0; i < knotCount; ++i)
    {
        int base = 9 * i;
        P[i].x() = z[base + 0];
        P[i].y() = z[base + 1];
        P[i].z() = z[base + 2];

        V[i].x() = z[base + 3];
        V[i].y() = z[base + 4];
        V[i].z() = z[base + 5];

        A[i].x() = z[base + 6];
        A[i].y() = z[base + 7];
        A[i].z() = z[base + 8];
    }

    // 2) Unpack segment durations σ → T
    for (int s = 0; s < M_; ++s)
    {
        T[s] = std::exp(z[K_cp_ + s]);
    }

    // 3) Build Hermite‐quintic control points in CP
    //    Each segment s has 6 control points in R³
    for (int s = 0; s < M_; ++s)
    {
        const Vec3 &p0 = P[s];
        const Vec3 &v0s = V[s];
        const Vec3 &a0s = A[s];

        const Vec3 &p1 = P[s + 1];
        const Vec3 &v1s = V[s + 1];
        const Vec3 &a1s = A[s + 1];

        double Ts = T[s];
        double T2 = Ts * Ts;

        // Fill CP[s] = array<Vec3,6>
        auto &c = CP[s];

        c[0] = p0;
        c[1] = p0 + v0s * (Ts / 5.0);
        c[2] = p0 + v0s * (2.0 * Ts / 5.0) + a0s * (T2 / 20.0);
        c[3] = p1 - v1s * (2.0 * Ts / 5.0) + a1s * (T2 / 20.0);
        c[4] = p1 - v1s * (Ts / 5.0);
        c[5] = p1;
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::findInitialGuess(
    std::vector<double> &T,
    std::vector<Vec3> &V,
    std::vector<Vec3> &A)
{
    // --- 1) clear outputs and reserve ---
    T.clear();
    T.reserve(M_);
    V.clear();
    V.reserve(M_ + 1);
    A.clear();
    A.reserve(M_ + 1);

    // --- 3) precompute segment directions ---
    std::vector<Vec3> dirs(M_);
    for (int i = 0; i < M_; ++i)
    {
        Vec3 Δ = global_wps_[i + 1] - global_wps_[i];
        double d = Δ.norm();
        dirs[i] = (d > 0 ? (Δ / d).eval() : Vec3::UnitX());
    }

    // --- 4) per‐waypoint speeds and velocity vectors ---
    std::vector<double> s(M_ + 1);
    std::vector<Vec3> v_vec(M_ + 1);

    s[0] = v0_.norm();
    s[M_] = vf_.norm();

    for (int i = 1; i < M_; ++i)
    {
        double ca = std::clamp(dirs[i - 1].dot(dirs[i]), -1.0, 1.0);
        double angle = std::acos(ca); // 0..π

        double factor;
        if (angle <= turn_buf_)
        {
            factor = 1.0; // within buffer, no slow-down
        }
        else
        {
            // linearly ramp from 1 at turn_buf_ → 0 at π
            factor = 1.0 - (angle - turn_buf_) / turn_span_;
            factor = std::clamp(factor, 0.0, 1.0);
        }

        s[i] = std::max(V_min_, factor * V_nom_);
    }

    // build velocity vectors at each waypoint
    v_vec[0] = dirs[0] * s[0];
    for (int i = 1; i < M_; ++i)
    {
        Vec3 bis = dirs[i - 1] + dirs[i];
        double n = bis.norm();
        if (n > 0)
            bis /= n;
        else
            bis = dirs[i];
        v_vec[i] = bis * s[i];
    }
    v_vec[M_] = dirs[M_ - 1] * s[M_];
    v_vec[M_ - 1] *= second_to_last_vel_scale_;

    // --- 5) allocate time per segment using average vector velocity ---
    for (int i = 0; i < M_; ++i)
    {
        double dist = (global_wps_[i + 1] - global_wps_[i]).norm();
        Vec3 v_avg = 0.5 * (v_vec[i] + v_vec[i + 1]);
        double speed = std::max(V_min_, v_avg.norm());
        T.push_back(dist / speed);
    }

    // --- 6) generate initial V, A via your min-jerk helper ---
    solveMinJerkVelAcc(global_wps_, T, v0_, a0_, vf_, af_, V, A);
}

// -----------------------------------------------------------------------------

//  Closed-form quintic control‐points for one segment
std::array<Vec3, 6> SolverLBFGS::computeQuinticCP(
    const Vec3 &P0, const Vec3 &V0, const Vec3 &A0,
    const Vec3 &P1, const Vec3 &V1, const Vec3 &A1,
    double T)
{
    double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
    Vec3 c0 = P0;
    Vec3 c1 = V0;
    Vec3 c2 = 0.5 * A0;
    Vec3 c3 = (-20.0 * P0 + 20.0 * P1 - (8.0 * V1 + 12.0 * V0) * T - (3.0 * A0 - A1) * T2) / (2.0 * T3);
    Vec3 c4 = (30.0 * P0 - 30.0 * P1 + (14.0 * V1 + 16.0 * V0) * T + (3.0 * A0 - 2.0 * A1) * T2) / (2.0 * T4);
    Vec3 c5 = (-12.0 * P0 + 12.0 * P1 - (6.0 * V1 + 6.0 * V0) * T - (A0 - A1) * T2) / (2.0 * T5);

    std::array<Vec3, 6> CP;
    for (int i = 0; i < 6; ++i)
    {
        double t = T * (double(i) / 5.0);
        double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
        CP[i] = c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t4 + c5 * t5;
    }
    return CP;
}

// -----------------------------------------------------------------------------

inline Eigen::Matrix4d SolverLBFGS::K_r(double T)
{
    double t2 = T * T, t3 = t2 * T;
    double i2 = 1.0 / t2, i3 = 1.0 / t3;
    Eigen::Matrix4d K;
    K << 192 * i3, 36 * i2, 168 * i3, -24 * i2,
        36 * i2, 9.0 / T, 24 * i2, -3.0 / T,
        168 * i3, 24 * i2, 192 * i3, -36 * i2,
        -24 * i2, -3.0 / T, -36 * i2, 9.0 / T;
    return K;
}

// -----------------------------------------------------------------------------

inline std::array<double, 4> SolverLBFGS::k_r(double T, double P0, double P1)
{
    // Example placeholder — you must substitute your real formulas:
    double k0 = (-66960 * P0 * T + 66960 * P1 * T) / (2 * T * T * T * T * T) + (-24480 * P0 * T + 24480 * P1 * T) / (2 * T * T * T * T * T) + (4320 * P0 * T - 4320 * P1 * T) / (2 * T * T * T * T * T) + (25920 * P0 * T - 25920 * P1 * T) / (2 * T * T * T * T * T) + (61920 * P0 * T - 61920 * P1 * T) / (2 * T * T * T * T * T);

    double k1 = (-11880 * P0 * T * T + 11880 * P1 * T * T) / (2 * T * T * T * T * T) + (-5400 * P0 * T * T + 5400 * P1 * T * T) / (2 * T * T * T * T * T) + (1080 * P0 * T * T - 1080 * P1 * T * T) / (2 * T * T * T * T * T) + (4320 * P0 * T * T - 4320 * P1 * T * T) / (2 * T * T * T * T * T) + (12000 * P0 * T * T - 12000 * P1 * T * T) / (2 * T * T * T * T * T);

    double k2 = (-62640 * P0 * T + 62640 * P1 * T) / (2 * T * T * T * T * T) + (-18720 * P0 * T + 18720 * P1 * T) / (2 * T * T * T * T * T) + (2880 * P0 * T - 2880 * P1 * T) / (2 * T * T * T * T * T) + (25920 * P0 * T - 25920 * P1 * T) / (2 * T * T * T * T * T) + (53280 * P0 * T - 53280 * P1 * T) / (2 * T * T * T * T * T);

    double k3 = (-7680 * P0 * T * T + 7680 * P1 * T * T) / (2 * T * T * T * T * T) + (-4320 * P0 * T * T + 4320 * P1 * T * T) / (2 * T * T * T * T * T) + (-360 * P0 * T * T + 360 * P1 * T * T) / (2 * T * T * T * T * T) + (2520 * P0 * T * T - 2520 * P1 * T * T) / (2 * T * T * T * T * T) + (9720 * P0 * T * T - 9720 * P1 * T * T) / (2 * T * T * T * T * T);
    return {k0, k1, k2, k3};
}

// -----------------------------------------------------------------------------

void SolverLBFGS::assemble_H_b(
    const std::vector<Vec3> &wps,
    const std::vector<double> &T,
    const Vec3 &v0, const Vec3 &a0,
    const Vec3 &vf, const Vec3 &af,
    Eigen::MatrixXd &H,
    Eigen::MatrixXd &b)
{
    int M = (int)T.size();
    int N = M - 1; // # interior junctions
    int D = 3;     // 3d

    H = Eigen::MatrixXd::Zero(2 * N, 2 * N);
    b = Eigen::MatrixXd::Zero(2 * N, D);

    // for each segment r = 0..M-1
    for (int r = 0; r < M; ++r)
    {
        // 1) Hessian block
        Eigen::Matrix4d Kr = K_r(T[r]);

        // 2) constant-term block (4×3)
        Eigen::Matrix<double, 4, 3> kr;
        for (int dim = 0; dim < 3; ++dim)
        {
            auto c = k_r(T[r],
                         wps[r](dim),
                         wps[r + 1](dim));
            for (int i = 0; i < 4; ++i)
                kr(i, dim) = c[i];
        }

        // 3) left half → junction r-1
        if (r >= 1)
        {
            int i = r - 1;
            H.block<2, 2>(2 * i, 2 * i) += Kr.block<2, 2>(0, 0);
            b.block<2, 3>(2 * i, 0) += kr.block<2, 3>(0, 0);
        }

        // 4) right half → junction r
        if (r <= M - 2)
        {
            int j = r;
            H.block<2, 2>(2 * j, 2 * j) += Kr.block<2, 2>(2, 2);
            b.block<2, 3>(2 * j, 0) += kr.block<2, 3>(2, 0);
        }

        // 5) coupling off-diagonals (H only)
        if (r >= 1 && r <= M - 2)
        {
            int i = r - 1, j = r;
            H.block<2, 2>(2 * i, 2 * j) += Kr.block<2, 2>(0, 2);
            H.block<2, 2>(2 * j, 2 * i) += Kr.block<2, 2>(2, 0);
        }
    }

    // 6) endpoint cross-terms into b
    {
        // segment 0: x0 (known) → x1 interior
        Eigen::Matrix4d K0 = K_r(T[0]);
        Eigen::Matrix<double, 2, 3> v0a0;
        v0a0.row(0) = v0.transpose();
        v0a0.row(1) = a0.transpose();
        // b[0:2] += K0[2:4,0:2]*[v0;a0]
        b.block<2, 3>(0, 0) += K0.block<2, 2>(2, 0) * v0a0;
    }
    {
        // segment M-1: x_{M} known → x_{M-1} interior
        Eigen::Matrix4d KM = K_r(T.back());
        Eigen::Matrix<double, 2, 3> vfaf;
        vfaf.row(0) = vf.transpose();
        vfaf.row(1) = af.transpose();
        int i = 2 * (M - 2);
        // b[i:i+2] += KM[0:2,2:4]*[vf;af]
        b.block<2, 3>(i, 0) += KM.block<2, 2>(0, 2) * vfaf;
    }
}

// -----------------------------------------------------------------------------

//  Build and solve the banded min-jerk system:
void SolverLBFGS::solveMinJerkVelAcc(
    const std::vector<Vec3> &wps, // size M+1
    const std::vector<double> &T, // size M
    const Vec3 &v0, const Vec3 &a0,
    const Vec3 &vf, const Vec3 &af,
    std::vector<Vec3> &V, // out size M+1
    std::vector<Vec3> &A  // out size M+1
)
{

    int M = (int)T.size();

    Eigen::MatrixXd H, b;
    assemble_H_b(wps, T, v0, a0, vf, af, H, b);

    // Solve H x = -b   →   x is (2N×3)
    Eigen::MatrixXd X = H.ldlt().solve(-b);

    // Unpack into V,A
    V.clear();
    A.clear();
    V.resize(M + 1);
    A.resize(M + 1);
    V[0] = v0;
    A[0] = a0;
    V[M] = vf;
    A[M] = af;
    for (int i = 1; i < M; ++i)
    {
        // row 2*(i-1) of X is the velocity at junction i
        V[i] = X.row(2 * (i - 1)).transpose(); // Vec3 = (3×1)
        // row 2*(i-1)+1 of X is the acceleration at junction i
        A[i] = X.row(2 * (i - 1) + 1).transpose();
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::getGoalSetpoints(std::vector<state> &goal_setpoints)
{
    // 1) total time & sample count
    double total_time = std::accumulate(T_opt_.begin(), T_opt_.end(), 0.0);
    // BUG #1: truncating here can undershoot the end of the trajectory.
    // It’s usually better to do:
    //   int N = static_cast<int>(std::ceil(total_time / dc_));
    int N = static_cast<int>(total_time / dc_);
    N = std::max(N, 2); // at least two points

    // 2) resize output
    goal_setpoints.resize(N);

    // 3) precompute timestamps
    std::vector<double> timestamps(N);
    for (int i = 0; i < N; ++i)
    {
        timestamps[i] = (i + 1) * dc_;
    }

    // 4) cumulative segment‑end times
    int M = static_cast<int>(T_opt_.size());
    std::vector<double> ends(M);
    if (M > 0)
    {
        ends[0] = T_opt_[0];
        for (int s = 1; s < M; ++s)
        {
            ends[s] = ends[s - 1] + T_opt_[s];
        }
    }

// 5) parallel fill of goal_setpoints
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        double ti = timestamps[i];

        // find which segment we're in
        auto it = std::upper_bound(ends.begin(), ends.end(), ti);
        int seg = static_cast<int>(it - ends.begin());
        seg = std::min(seg, std::max(0, M - 1));

        double t0 = (seg > 0 ? ends[seg - 1] : 0.0);
        double tau = (T_opt_[seg] > 0.0)
                         ? (ti - t0) / T_opt_[seg]
                         : 0.0;

        StateDeriv D = evalStateDeriv(seg, tau);

        state st;
        st.setTimeStamp(t0_ + ti);
        st.setPos(D.pos.x(), D.pos.y(), D.pos.z());
        st.setVel(D.vel.x(), D.vel.y(), D.vel.z());
        st.setAccel(D.accel.x(), D.accel.y(), D.accel.z());
        st.setJerk(D.jerk.x(), D.jerk.y(), D.jerk.z());
        goal_setpoints[i] = st;
    }

    // 6) zero out final derivatives explicitly
    {
        auto &last = goal_setpoints.back();
        // BUG #2: .transpose() is unnecessary if vel/accel are Vector3d
        last.vel = Eigen::Vector3d::Zero();
        last.accel = Eigen::Vector3d::Zero();
        last.jerk = Eigen::Vector3d::Zero();
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::getPieceWisePol(PieceWiseQuinticPol &pwp)
{
    // 1) reset
    pwp.clear();

    // 2) build the time‐breaks
    pwp.times.push_back(t0_);
    double t_acc = 0.0;
    for (int i = 0; i < M_; ++i)
    {
        t_acc += T_opt_[i];
        pwp.times.push_back(t0_ + t_acc);
    }

    // 3) for each segment, convert Hermite → power basis:
    for (int s = 0; s < M_; ++s)
    {
        double T = T_opt_[s];
        double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;

        // Hermite endpoints
        const auto &P0 = P_opt_[s], &P1 = P_opt_[s + 1];
        const auto &V0 = V_opt_[s], &V1 = V_opt_[s + 1];
        const auto &A0 = A_opt_[s], &A1 = A_opt_[s + 1];

        // helper deltas
        Eigen::Vector3d C0 = P1 - (P0 + V0 * T + 0.5 * A0 * T2);
        Eigen::Vector3d C1 = V1 - (V0 + A0 * T);
        Eigen::Vector3d C2 = A1 - A0;

        // build coeff vector [a0, a1, a2, a3, a4, a5]^T per axis
        Eigen::Matrix<double, 6, 1> cx, cy, cz;

        // a0, a1, a2
        cx(0) = P0.x();
        cx(1) = V0.x();
        cx(2) = 0.5 * A0.x();
        // a3, a4, a5
        cx(3) = (10 * C0.x() - 4 * C1.x() + 0.5 * C2.x()) / T3;
        cx(4) = (-15 * C0.x() + 7 * C1.x() - C2.x()) / T4;
        cx(5) = (6 * C0.x() - 3 * C1.x() + 0.5 * C2.x()) / T5;

        // same for y
        cy(0) = P0.y();
        cy(1) = V0.y();
        cy(2) = 0.5 * A0.y();
        cy(3) = (10 * C0.y() - 4 * C1.y() + 0.5 * C2.y()) / T3;
        cy(4) = (-15 * C0.y() + 7 * C1.y() - C2.y()) / T4;
        cy(5) = (6 * C0.y() - 3 * C1.y() + 0.5 * C2.y()) / T5;

        // …and z
        cz(0) = P0.z();
        cz(1) = V0.z();
        cz(2) = 0.5 * A0.z();
        cz(3) = (10 * C0.z() - 4 * C1.z() + 0.5 * C2.z()) / T3;
        cz(4) = (-15 * C0.z() + 7 * C1.z() - C2.z()) / T4;
        cz(5) = (6 * C0.z() - 3 * C1.z() + 0.5 * C2.z()) / T5;

        // store it
        pwp.coeff_x.push_back(cx);
        pwp.coeff_y.push_back(cy);
        pwp.coeff_z.push_back(cz);
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::getControlPoints(std::vector<Eigen::Matrix<double, 3, 6>> &cps)
{
    // Resize the control points vector
    cps.resize(M_);

    // Fill the control points
    for (int i = 0; i < M_; i++)
    {
        Eigen::Matrix<double, 3, 6> cp_i;
        for (int j = 0; j < 6; j++)
        {
            cp_i(0, j) = CP_opt_[i][j][0];
            cp_i(1, j) = CP_opt_[i][j][1];
            cp_i(2, j) = CP_opt_[i][j][2];
        }
        cps[i] = cp_i;
    }
}

// -----------------------------------------------------------------------------

inline StateDeriv SolverLBFGS::evalStateDeriv(int s, double tau) const
{
    // endpoints and duration
    const auto &P0 = P_opt_[s];
    const auto &V0 = V_opt_[s];
    const auto &A0 = A_opt_[s];
    const auto &P1 = P_opt_[s + 1];
    const auto &V1 = V_opt_[s + 1];
    const auto &A1 = A_opt_[s + 1];
    double T = T_opt_[s];
    double t2 = tau * tau, t3 = t2 * tau, t4 = t3 * tau, t5 = t4 * tau;

    // Quintic Hermite basis H_i(τ)
    double h0 = 1 - 10 * t3 + 15 * t4 - 6 * t5;
    double h1 = tau - 6 * t3 + 8 * t4 - 3 * t5;
    double h2 = 0.5 * (t2 - 3 * t3 + 3 * t4 - t5);
    double h3 = 10 * t3 - 15 * t4 + 6 * t5;
    double h4 = -4 * t3 + 7 * t4 - 3 * t5;
    double h5 = 0.5 * (t3 - 2 * t4 + t5);

    // position
    Eigen::Vector3d p =
        P0 * h0 + V0 * (h1 * T) + A0 * (h2 * T * T) + P1 * h3 + V1 * (h4 * T) + A1 * (h5 * T * T);

    // first derivatives of H_i(τ)
    double dh0 = -30 * t2 + 60 * t3 - 30 * t4;
    double dh1 = 1 - 18 * t2 + 32 * t3 - 15 * t4;
    double dh2 = 0.5 * (2 * tau - 9 * t2 + 12 * t3 - 5 * t4);
    double dh3 = 30 * t2 - 60 * t3 + 30 * t4;
    double dh4 = -12 * t2 + 28 * t3 - 15 * t4;
    double dh5 = 0.5 * (3 * t2 - 8 * t3 + 5 * t4);

    // velocity = (1/T) * d/dτ
    Eigen::Vector3d v = (P0 * dh0 + V0 * (dh1 * T) + A0 * (dh2 * T * T) + P1 * dh3 + V1 * (dh4 * T) + A1 * (dh5 * T * T)) / T;

    // second derivatives of H_i(τ)
    double d2h0 = -60 * tau + 180 * t2 - 120 * t3;
    double d2h1 = -36 * tau + 96 * t2 - 60 * t3;
    double d2h2 = 0.5 * (2 - 18 * tau + 36 * t2 - 20 * t3);
    double d2h3 = 60 * tau - 180 * t2 + 120 * t3;
    double d2h4 = -24 * tau + 84 * t2 - 60 * t3;
    double d2h5 = 0.5 * (6 * tau - 24 * t2 + 20 * t3);

    // acceleration = (1/T^2) * d²/dτ²
    Eigen::Vector3d a = (P0 * d2h0 + V0 * (d2h1 * T) + A0 * (d2h2 * T * T) + P1 * d2h3 + V1 * (d2h4 * T) + A1 * (d2h5 * T * T)) / (T * T);

    // third derivatives of H_i(τ)
    double d3h0 = -60 + 360 * tau - 360 * t2;
    double d3h1 = -36 + 192 * tau - 180 * t2;
    double d3h2 = 0.5 * (-18 + 72 * tau - 60 * t2);
    double d3h3 = 60 - 360 * tau + 360 * t2;
    double d3h4 = -24 + 168 * tau - 180 * t2;
    double d3h5 = 0.5 * (6 - 48 * tau + 60 * t2);

    // jerk = (1/T^3) * d³/dτ³
    Eigen::Vector3d j = (P0 * d3h0 + V0 * (d3h1 * T) + A0 * (d3h2 * T * T) + P1 * d3h3 + V1 * (d3h4 * T) + A1 * (d3h5 * T * T)) / (T * T * T);

    return {p, v, a, j};
}

// -----------------------------------------------------------------------------

void SolverLBFGS::buildInitialGuesses(const ConstraintBlocks &constraint_sets)
{
    auto sampled_wps = sample_systematic_perturbed_waypoints(constraint_sets);

    initial_guess_wps_.clear();
    initial_guess_wps_.reserve(sampled_wps.size() + 1);
    initial_guess_wps_.push_back(global_wps_);
    for (auto &s : sampled_wps)
        initial_guess_wps_.push_back(std::move(s));

    list_z0_.clear();
    list_z0_.reserve(initial_guess_wps_.size());

    for (auto const &wps : initial_guess_wps_)
    {
        Eigen::VectorXd z0;
        std::vector<Vec3> P, V, A;
        std::vector<double> T;

        // Use the min-jerk initializer
        findInitialGuess(T, V, A);
        for (int i = 0; i < M_ + 1; ++i)
        {
            P.push_back(global_wps_[i]);
        }

        // Generate the initial guess in z form
        packDecisionVariables(P, V, A, T, z0);

        list_z0_.push_back(z0);
    }
}

// -----------------------------------------------------------------------------

std::vector<std::vector<Vec3>>
SolverLBFGS::sample_systematic_perturbed_waypoints(
    const ConstraintBlocks &constraint_sets) const
{
    int K = (int)global_wps_.size();
    int J = (int)joint_indices_.size();

    // 1) build bisector‐plane bases for each joint
    std::vector<std::pair<Vec3, Vec3>> bases;
    bases.reserve(J);
    for (int j = 0; j < J; ++j)
    {
        int idx = joint_indices_[j];
        Vec3 Pprev = global_wps_[idx - 1];
        Vec3 P = global_wps_[idx];
        Vec3 Pnext = global_wps_[idx + 1];

        Vec3 d1 = (P - Pprev).normalized();
        Vec3 d2 = (Pnext - P).normalized();
        Vec3 pn = d1.cross(d2);
        if (pn.norm() < 1e-8)
            pn = d1.cross(Vec3(1, 0, 0));
        pn.normalize();

        Vec3 bis_n = (d1 + d2);
        if (bis_n.norm() < 1e-8)
            bis_n = d1;
        bis_n.normalize();

        Vec3 v_bis = bis_n.cross(pn);
        v_bis.normalize();

        bases.emplace_back(pn, v_bis);
    }

    // 2) num_perturbation_ equally‐spaced angles
    std::vector<double> angles(num_perturbation_);
    for (int k = 0; k < num_perturbation_; ++k)
        angles[k] = 2.0 * M_PI * double(k) / double(num_perturbation_);

    std::vector<std::vector<Vec3>> samples;
    samples.reserve(num_perturbation_);

    for (double phi : angles)
    {
        auto cand = global_wps_;
        bool feasible = true;

        for (int j = 0; j < J; ++j)
        {
            int idx = joint_indices_[j];
            auto [pn, v_bis] = bases[j];
            Vec3 dir = std::cos(phi) * pn + std::sin(phi) * v_bis;
            Vec3 pt = global_wps_[idx] + r_max_ * dir;

            // ---- HERE: loop over the _two_ blocks, each block is an (n×3),(n) pair ----
            for (auto const &block : constraint_sets[j])
            {

                // If block is empty, skip it (sometimes segment doesn't have any constraints)
                if (block.first.rows() == 0 || block.second.size() == 0)
                {
                    std::cout << "Skipping empty block for joint index " << idx << "\n";
                    continue;
                }

                auto const &Ablock = block.first;  // (n_planes x 3)
                auto const &bblock = block.second; // (n_planes)
                // compute per‐plane residuals
                Eigen::VectorXd h = Ablock * pt - bblock;
                // if *any* plane in this block is violated, reject
                for (int r = 0; r < h.size(); ++r)
                {
                    if (h(r) <= 0.0)
                    {
                        feasible = false;
                        break;
                    }
                }
                if (!feasible)
                    break;
            }
            if (!feasible)
                break;
            cand[idx] = pt;
        }

        if (feasible)
            samples.push_back(std::move(cand));
    }

    return samples;
}

// -----------------------------------------------------------------------------

// degree-5 Bézier: p(u) = Σ_{j=0..5} B5_j(u) * CP[j]
inline Eigen::Vector3d evalBezierQuintic(const std::array<Eigen::Vector3d, 6> &cp, double u)
{
    u = std::min(std::max(u, 0.0), 1.0);
    const double omu = 1.0 - u;
    const double omu2 = omu * omu;
    const double omu3 = omu2 * omu;
    const double omu4 = omu3 * omu;
    const double omu5 = omu4 * omu;
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double u4 = u3 * u;
    const double u5 = u4 * u;

    const double b0 = omu5;
    const double b1 = 5.0 * u * omu4;
    const double b2 = 10.0 * u2 * omu3;
    const double b3 = 10.0 * u3 * omu2;
    const double b4 = 5.0 * u4 * omu;
    const double b5 = u5;

    return b0 * cp[0] + b1 * cp[1] + b2 * cp[2] + b3 * cp[3] + b4 * cp[4] + b5 * cp[5];
}

// Sample robot positions at absolute times t_i = t0 + i*dt, i=1..N
// Uses a single pass over segments (monotone times) for speed.
inline void sampleRobotPositionsUniform(
    const std::vector<std::array<Eigen::Vector3d, 6>> &CP,
    const std::vector<double> &T,
    double t0, int N,
    std::vector<double> &t_samples,
    std::vector<Eigen::Vector3d> &p_samples)
{
    const int M = static_cast<int>(T.size());
    t_samples.resize(N);
    p_samples.resize(N);

    // absolute edges: [t0, t0+T0, t0+T0+T1, ...]
    std::vector<double> edges(M + 1, t0);
    for (int i = 1; i <= M; ++i)
        edges[i] = edges[i - 1] + T[i - 1];

    const double total_T = edges.back() - t0;
    const double dt = (N > 0) ? (total_T / N) : 0.0;

    int s = 0; // current segment
    for (int i = 0; i < N; ++i)
    {
        const double ti = t0 + (i + 1) * dt; // [t0+dt, ..., t0+N*dt]
        t_samples[i] = ti;

        while (s + 1 < (int)edges.size() && ti > edges[s + 1])
            ++s;
        const double Ts = T[std::min(s, M - 1)];
        double u = 0.0;
        if (Ts > 0.0)
        {
            u = (ti - edges[s]) / Ts;
            if (u < 0.0)
                u = 0.0;
            if (u > 1.0)
                u = 1.0;
        }
        p_samples[i] = evalBezierQuintic(CP[std::min(s, M - 1)], u);
    }
}

// -----------------------------------------------------------------------------

double SolverLBFGS::evaluateObjective(const VecXd &z) const
{
    // 1) reconstruct P, V, A, CP, and T
    std::vector<Vec3> P, V, A;
    std::vector<std::array<Vec3, 6>> CP; // CP[s][0..5]
    std::vector<double> T;               // T[s]
    reconstruct(z, P, V, A, CP, T);

    // 2) time cost
    double J_time = std::accumulate(T.begin(), T.end(), 0.0);

    // 3) absolute knot times
    t_abs_.clear();
    t_abs_.resize(M_ + 1);
    t_abs_[0] = t0_;
    for (int i = 1; i <= M_; ++i)
        t_abs_[i] = t_abs_[i - 1] + T[i - 1];

    // 4) dynamic‐obstacle (unchanged; right Riemann)
    int N = (num_dyn_obst_samples_ > 0 ? num_dyn_obst_samples_ : 10);
    N = std::max(N, 1);
    std::vector<double> t_samples;
    std::vector<Vec3> p_samples;
    sampleRobotPositionsUniform(CP, T, t0_, N, t_samples, p_samples);
    const double total_T = t_abs_.back() - t0_;
    const double dt = total_T / N;

    double J_dyn = 0.0;
    for (const auto &obs : obstacles_)
    {
        for (int i = 0; i < N; ++i)
        {
            const double t = t_samples[i];
            const Vec3 &pi = p_samples[i];
            const Vec3 ki = obs->eval(t);
            const Vec3 d = pi - ki;
            const double d2 = d.squaredNorm();
            const double v = Cw2_ - d2;
            if (v > 0.0)
                J_dyn += v * v * v * dt;
        }
    }

    // 5) static corridor via CPs (unchanged)
    double J_stat = 0.0;
    for (int i = 1; i < M_; ++i)
    {
        Vec3 pi = P[i];
        int segs[2] = {i - 1, i};
        int nseg = (i < M_) ? 2 : 1;
        for (int si = 0; si < nseg; ++si)
        {
            int p = segs[si];
            auto const &Aseg = A_stat_[p];
            auto const &bseg = b_stat_[p];
            int nplanes = Aseg.rows();
            for (int k = 0; k < nplanes; ++k)
            {
                double h = Aseg.row(k).dot(pi) - bseg[k];
                double viol = Co_ - h;
                if (viol > 0.0)
                    J_stat += viol * viol * viol;

                // if actually h < 0.0, then the point is inside the obstacle -> absolutely infeasible
                if (h < 0.0)
                {
                    // std::cout << "Infeasible static corridor constraint at segment " << p
                    //           << ", point " << pi.transpose() << ", Aseg.row(k) = "
                    //           << Aseg.row(k).transpose() << ", bseg[k] = " << bseg[k] << "\n";
                    return BIG_;
                }
            }
        }
    }

    // 6) jerk L2 (unchanged)
    double J_jerk = 0.0;
    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        const double Cs = 3600.0 / std::pow(Ts, 5);
        const Vec3 d30 = CP[s][3] - 3.0 * CP[s][2] + 3.0 * CP[s][1] - CP[s][0];
        const Vec3 d31 = CP[s][4] - 3.0 * CP[s][3] + 3.0 * CP[s][2] - CP[s][1];
        const Vec3 d32 = CP[s][5] - 3.0 * CP[s][4] + 3.0 * CP[s][3] - CP[s][2];
        J_jerk += Cs * (d30.squaredNorm() + d31.squaredNorm() + d32.squaredNorm());
    }

    // 7) vel/acc constraints via derivative CPs (unchanged)
    double J_vel_constr = 0.0, J_acc_constr = 0.0;
    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        const double invT = 1.0 / (Ts + 1e-16);
        const double alpha = 5.0 * invT;
        const double beta = 20.0 * invT * invT;

        for (int j = 0; j <= 4; ++j)
        {
            const Vec3 U = alpha * (CP[s][j + 1] - CP[s][j]);
            const double dv = std::max(U.norm() - V_max_, 0.0);
            if (dv > 0.0)
                J_vel_constr += dv * dv * dv;
        }
        for (int j = 0; j <= 3; ++j)
        {
            const Vec3 W = beta * (CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j]);
            const double da = std::max(W.norm() - A_max_, 0.0);
            if (da > 0.0)
                J_acc_constr += da * da * da;
        }
    }

    // 8) NEW: sampling-free thrust/tilt/body-rate penalties
    const Vec3 e3(0.0, 0.0, 1.0);
    const double g = 9.81; // gravity magnitude (e.g., 9.81)
    double J_thr_max = 0.0, J_thr_min = 0.0, J_theta = 0.0, J_omega = 0.0;

    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        if (Ts <= 0.0)
            continue;

        const double invT = 1.0 / (Ts + 1e-16);
        const double beta = 20.0 * invT * invT;         // accel CP scale
        const double gamma = 60.0 * invT * invT * invT; // jerk  CP scale

        // thrust/tilt @ accel CPs j=0..3
        for (int j = 0; j <= 3; ++j)
        {
            const Vec3 d2 = CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j];
            const Vec3 W = beta * d2;
            const Vec3 Wt = W + g * e3;

            // thrust max
            if (thrust_max_weight_ > 0.0)
            {
                const double v = Wt.norm() - thrust_max_;
                if (v > 0.0)
                    J_thr_max += v * v * v;
            }

            // thrust min (conservative: on vertical component)
            if (thrust_min_weight_ > 0.0)
            {
                const double v = thrust_min_ - Wt.z();
                if (v > 0.0)
                    J_thr_min += v * v * v;
            }

            // tilt: ||Wt_xy|| <= tan(theta_max) * Wt_z
            if (theta_weight_ > 0.0)
            {
                const double tan_theta_max_ = std::tan(theta_max_);
                const double nxy = std::sqrt(Wt.x() * Wt.x() + Wt.y() * Wt.y());
                const double v = nxy - tan_theta_max_ * Wt.z();
                if (v > 0.0)
                    J_theta += v * v * v;
            }
        }

        // body-rate @ jerk CPs j=0..2, coupled with Wt_z at same j
        if (omega_weight_ > 0.0)
        {
            for (int j = 0; j <= 2; ++j)
            {
                const Vec3 d3 = CP[s][j + 3] - 3.0 * CP[s][j + 2] + 3.0 * CP[s][j + 1] - CP[s][j];
                const Vec3 Jv = gamma * d3;

                // vertical thrust at same j from accel CPs
                const Vec3 d2 = CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j];
                const Vec3 W = beta * d2;
                const double Wt_z = W.z() + g;

                const double v = Jv.norm() - omega_max_ * Wt_z;
                if (v > 0.0)
                    J_omega += v * v * v;
            }
        }
    }

    // std::cout << "J_time: " << J_time
    //           << ", J_dyn: " << J_dyn
    //           << ", J_stat: " << J_stat
    //           << ", J_jerk: " << J_jerk
    //           << ", J_vel_constr: " << J_vel_constr
    //           << ", J_acc_constr: " << J_acc_constr
    //           << ", J_thr_max: " << J_thr_max
    //           << ", J_thr_min: " << J_thr_min
    //           << ", J_theta: " << J_theta
    //           << ", J_omega: " << J_omega
    //           << std::endl;

    // 9) weighted sum
    return time_weight_ * J_time + dyn_weight_ * J_dyn + stat_weight_ * J_stat + jerk_weight_ * J_jerk + dyn_constr_vel_weight_ * J_vel_constr + dyn_constr_acc_weight_ * J_acc_constr + thrust_max_weight_ * J_thr_max + thrust_min_weight_ * J_thr_min + theta_weight_ * J_theta + omega_weight_ * J_omega;
}

// -----------------------------------------------------------------------------

void SolverLBFGS::computeAnalyticalGrad(
    const Eigen::VectorXd &z,
    Eigen::VectorXd &grad) const
{
    // 1) Reconstruct P, V, A, CP, T
    std::vector<Vec3> P, V, A;
    std::vector<std::array<Vec3, 6>> CP;
    std::vector<double> T;
    reconstruct(z, P, V, A, CP, T);

    // 2) Zero output
    grad = Eigen::VectorXd::Zero(K_);

    // 3) Time cost: d/dσ_r ( time_weight * sum_s T_s ) = time_weight * T_r
    for (int r = 0; r < M_; ++r)
        grad[K_cp_ + r] += time_weight_ * T[r];

    // 4) Dynamic-obstacle cost (uniform samples, already implemented)
    if (dyn_weight_ != 0.0 && !obstacles_.empty())
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_dyn_dz(z, P, V, A, CP, T, g); // unweighted
        grad += dyn_weight_ * g;
    }

    // 5) Static corridor penalty
    if (stat_weight_ != 0.0)
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_stat_dz(z, P, T, g); // unweighted
        grad += stat_weight_ * g;
    }

    // 6) Jerk L2 penalty
    if (jerk_weight_ != 0.0)
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_jerk_dz(z, P, V, A, CP, T, g); // unweighted
        grad += jerk_weight_ * g;
    }

    // 7) Velocity-constraint penalty
    if (dyn_constr_vel_weight_ != 0.0)
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_vel_constr_dz(z, P, V, A, CP, T, g); // unweighted
        grad += dyn_constr_vel_weight_ * g;
    }

    // 8) Acceleration-constraint penalty
    if (dyn_constr_acc_weight_ != 0.0)
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_acc_constr_dz(z, P, V, A, CP, T, g); // unweighted
        grad += dyn_constr_acc_weight_ * g;
    }

    // 9) NEW: Thrust/tilt/body-rate penalties (sampling-free, Bézier-CP based)
    if (thrust_max_weight_ != 0.0)
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_thrust_max_dz(z, P, V, A, CP, T, g); // unweighted
        grad += thrust_max_weight_ * g;
    }
    if (thrust_min_weight_ != 0.0)
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_thrust_min_dz(z, P, V, A, CP, T, g); // unweighted
        grad += thrust_min_weight_ * g;
    }
    if (theta_weight_ != 0.0)
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_theta_dz(z, P, V, A, CP, T, g); // unweighted
        grad += theta_weight_ * g;
    }
    if (omega_weight_ != 0.0)
    {
        Eigen::VectorXd g(K_);
        g.setZero();
        dJ_omega_dz(z, P, V, A, CP, T, g); // unweighted
        grad += omega_weight_ * g;
    }

    // 10) Zero endpoints: (p0,v0,a0) and (pM,vM,aM)
    for (int i = 0; i < 9; ++i)
    {
        grad[i] = 0.0;             // P0, V0, A0
        grad[K_cp_ - 9 + i] = 0.0; // Pf, Vf, Af
    }
}

// -----------------------------------------------------------------------------

double SolverLBFGS::evaluateObjectiveAndGradient(
    const Eigen::VectorXd &z,
    Eigen::VectorXd &g) const
{

    // auto t_objective_start = std::chrono::high_resolution_clock::now();

    // 1) compute the objective alone
    double f = evaluateObjective(z);

    // auto t_objective_end = std::chrono::high_resolution_clock::now();
    // auto duration_objective = std::chrono::duration_cast<std::chrono::microseconds>(t_objective_end - t_objective_start);
    // std::cout << "Objective evaluation took: "
    //           << duration_objective.count() / 1000.0 << " ms." << std::endl;

    // auto t_gradient_start = std::chrono::high_resolution_clock::now();

    // 2) compute analytic gradient into g
    g.resize(z.size());
    computeAnalyticalGrad(z, g);

    // auto t_gradient_end = std::chrono::high_resolution_clock::now();
    // auto duration_gradient = std::chrono::duration_cast<std::chrono::microseconds>(t_gradient_end - t_gradient_start);
    // std::cout << "Gradient evaluation took: "
    //           << duration_gradient.count() / 1000.0 << " ms." << std::endl;

    return f;
}

// -----------------------------------------------------------------------------

int SolverLBFGS::progressCallback(
    void *instance,
    const Eigen::VectorXd &z,
    const Eigen::VectorXd &g,
    const double f,
    const double step,
    const int k,
    const int ls)
{
    // std::cout << "iter=" << k << " f=" << f << " z=" << z.transpose() << " |g|=" << g.norm() << std::endl;
    // std::cout << "iter=" << k << " f=" << f << " |g|=" << g.norm() << std::endl;
    printf("iter=%d f=%.6f |g|=%.6f\n", k, f, g.norm());
    return 0; // return non‐zero to abort optimization early
}

// -----------------------------------------------------------------------------

double SolverLBFGS::evalObjGradCallback(
    void *instance,
    const Eigen::VectorXd &x,
    Eigen::VectorXd &g)
{
    // dispatch into the instance
    return static_cast<SolverLBFGS *>(instance)
        ->evaluateObjectiveAndGradient(x, g);
}

// -----------------------------------------------------------------------------

int SolverLBFGS::optimize(
    const Eigen::VectorXd &z0,
    Eigen::VectorXd &z_opt,
    double &f_opt,
    const lbfgs::lbfgs_parameter_t &param) const
{
    // copy initial guess
    Eigen::VectorXd z = z0;

    int status = lbfgs::lbfgs_optimize(
        z,                                 // in/out decision vector
        f_opt,                             // out final cost
        &SolverLBFGS::evalObjGradCallback, // objective+gradient callback
        /*stepbound*/ nullptr,
        /*progress*/ &SolverLBFGS::progressCallback,
        // /*progress*/ nullptr,
        const_cast<SolverLBFGS *>(this),
        param);

    // copy result
    z_opt = z;
    return status;
}

// -----------------------------------------------------------------------------

void SolverLBFGS::setStaticConstraints(
    const std::vector<LinearConstraint3D> &cons)
{
    A_stat_.clear();
    b_stat_.clear();
    A_stat_.reserve(cons.size());
    b_stat_.reserve(cons.size());

    for (auto const &lc : cons)
    {
        // lc.A() is Eigen::Matrix<decimal_t,Dynamic,3>
        // lc.b() is Eigen::Matrix<decimal_t,Dynamic,1>
        // cast to double if decimal_t isn’t already double:
        Eigen::Matrix<double, Eigen::Dynamic, 3> A =
            lc.A().template cast<double>();
        Eigen::VectorXd b =
            lc.b().template cast<double>();

        A = -A;
        b = -b;

        A_stat_.push_back(std::move(A));
        b_stat_.push_back(std::move(b));
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::reconstructPVATCPopt(const Eigen::VectorXd &z)
{

    reconstruct(z, P_opt_, V_opt_, A_opt_, CP_opt_, T_opt_);

    // std::cout << "Reconstructed P_opt size: " << P_opt_.size() << std::endl;
    // for (const auto &p : P_opt_)
    // {
    //     std::cout << p.transpose() << std::endl;
    // }
    // std::cout << "Reconstructed V_opt size: " << V_opt_.size() << std::endl;
    // for (const auto &v : V_opt_)
    // {
    //     std::cout << v.transpose() << std::endl;
    // }
    // std::cout << "Reconstructed A_opt size: " << A_opt_.size() << std::endl;
    // for (const auto &a : A_opt_)
    // {
    //     std::cout << a.transpose() << std::endl;
    // }
    // std::cout << "Reconstructed T_opt size: " << T_opt_.size() << std::endl;
    // for (const auto &t : T_opt_)
    // {
    //     std::cout << t << " ";
    // }
}

//------------------------------------------------------------------------------

inline void bernstein5(double u, double B[6])
{
    const double um = 1.0 - u;
    const double u2 = u * u, u3 = u2 * u, u4 = u3 * u, u5 = u4 * u;
    const double m2 = um * um, m3 = m2 * um, m4 = m3 * um, m5 = m4 * um;
    B[0] = m5;
    B[1] = 5.0 * u * m4;
    B[2] = 10.0 * u2 * m3;
    B[3] = 10.0 * u3 * m2;
    B[4] = 5.0 * u4 * um;
    B[5] = u5;
}

inline void bernstein4(double u, double b[5])
{
    const double um = 1.0 - u;
    const double u2 = u * u, u3 = u2 * u, u4 = u3 * u;
    const double m2 = um * um, m3 = m2 * um, m4 = m3 * um;
    b[0] = m4;
    b[1] = 4.0 * u * m3;
    b[2] = 6.0 * u2 * m2;
    b[3] = 4.0 * u3 * um;
    b[4] = u4;
}

inline void accumulate_cp_to_pvaT(
    int s,
    const std::array<Vec3, 6> &gCPs,
    const std::vector<Vec3> &V,
    const std::vector<Vec3> &A,
    const std::vector<double> &T,
    std::vector<Vec3> &gP,
    std::vector<Vec3> &gV,
    std::vector<Vec3> &gA,
    std::vector<double> &gT_cp)
{
    const double Ts = T[s];
    const double Ts2 = Ts * Ts;

    // left endpoint (s)
    gP[s] += gCPs[0] + gCPs[1] + gCPs[2];
    gV[s] += (Ts / 5.0) * gCPs[1] + (2.0 * Ts / 5.0) * gCPs[2];
    gA[s] += (Ts2 / 20.0) * gCPs[2];

    // right endpoint (s+1)
    gP[s + 1] += gCPs[3] + gCPs[4] + gCPs[5];
    gV[s + 1] += (-2.0 * Ts / 5.0) * gCPs[3] + (-Ts / 5.0) * gCPs[4];
    gA[s + 1] += (Ts2 / 20.0) * gCPs[3];

    // ∂/∂T via CP(T) dependence
    gT_cp[s] += gCPs[1].dot(V[s] / 5.0);
    gT_cp[s] += gCPs[2].dot((2.0 / 5.0) * V[s] + (Ts / 10.0) * A[s]);
    gT_cp[s] += gCPs[3].dot((-2.0 / 5.0) * V[s + 1] + (Ts / 10.0) * A[s + 1]);
    gT_cp[s] += gCPs[4].dot((-1.0 / 5.0) * V[s + 1]);
}

//------------------------------------------------------------------------------

void SolverLBFGS::dJ_dyn_dz(const VecXd &z,
                            const std::vector<Vec3> &P,
                            const std::vector<Vec3> &V,
                            const std::vector<Vec3> &A,
                            const std::vector<std::array<Vec3, 6>> &CP,
                            const std::vector<double> &T,
                            VecXd &grad) const
{
    const int M = M_;
    const int N = std::max(1, num_dyn_obst_samples_);
    const double Cw2 = Cw2_;

    // absolute knot times
    std::vector<double> edges(M + 1);
    edges[0] = t0_;
    for (int i = 1; i <= M; ++i)
        edges[i] = edges[i - 1] + T[i - 1];

    const double total_T = edges[M] - edges[0];
    const double invN = 1.0 / static_cast<double>(N);
    const double dt = total_T * invN;

    std::vector<Vec3> gP(M + 1, Vec3::Zero());
    std::vector<Vec3> gV(M + 1, Vec3::Zero());
    std::vector<Vec3> gA(M + 1, Vec3::Zero());
    std::vector<double> gT_cp(M, 0.0);

    std::vector<std::array<Vec3, 6>> gCP(M);
    for (int s = 0; s < M; ++s)
        for (int j = 0; j < 6; ++j)
            gCP[s][j].setZero();

    for (const auto &obs : obstacles_)
    {
        for (int i = 1; i <= N; ++i)
        {
            const double i_over_N = static_cast<double>(i) * invN;
            const double t_abs = edges[0] + i_over_N * total_T;

            int s = int(std::upper_bound(edges.begin(), edges.end(), t_abs) - edges.begin()) - 1;
            if (s < 0)
                s = 0;
            if (s >= M)
                s = M - 1;

            const double Ts = T[s];
            if (Ts <= 0.0)
                continue;

            const double t_rel = t_abs - edges[s];
            double u = t_rel / Ts;
            if (u < 0.0)
                u = 0.0;
            else if (u > 1.0)
                u = 1.0;

            double B[6], b4[5];
            bernstein5(u, B);
            bernstein4(u, b4);

            Vec3 p = Vec3::Zero();
            for (int j = 0; j < 6; ++j)
                p += B[j] * CP[s][j];

            const Vec3 k = obs->eval(t_abs);
            const Vec3 v_obs = obs->velocity(t_abs);

            const Vec3 diff = p - k;
            const double d2 = diff.squaredNorm();

            if (d2 < Cw2)
            {
                const double h = (Cw2 - d2);
                const double factor = -6.0 * h * h;

                for (int j = 0; j < 6; ++j)
                    gCP[s][j] += (factor * dt * B[j]) * diff;

                Vec3 dp_du = Vec3::Zero();
                for (int j = 0; j < 5; ++j)
                    dp_du += b4[j] * (CP[s][j + 1] - CP[s][j]);
                dp_du *= 5.0;

                const double term_u = diff.dot(dp_du);

                for (int r = 0; r < M; ++r)
                {
                    const double is_lt = (r < s) ? 1.0 : 0.0;
                    const double is_eq = (r == s) ? 1.0 : 0.0;
                    const double du_dTr = (i_over_N - is_lt - u * is_eq) / Ts;
                    grad[K_cp_ + r] += (factor * dt) * term_u * du_dTr * T[r];
                }

                const double inner_obs = -diff.dot(v_obs);
                for (int r = 0; r < M; ++r)
                    grad[K_cp_ + r] += (factor * dt) * inner_obs * i_over_N * T[r];

                const double hinge3 = h * h * h;
                for (int r = 0; r < M; ++r)
                    grad[K_cp_ + r] += hinge3 * invN * T[r];
            }
            else
            {
                // outside hinge: only dt scaling term contributes 0 anyway (hinge^3 == 0)
                // so nothing to do.
            }
        }
    }

    // push gCP -> (p,v,a,T)
    for (int s = 0; s < M; ++s)
        accumulate_cp_to_pvaT(s, gCP[s], V, A, T, gP, gV, gA, gT_cp);

    // scatter into z
    for (int i = 0; i <= M; ++i)
    {
        const int base = 9 * i;
        grad.segment<3>(base + 0) += gP[i];
        grad.segment<3>(base + 3) += gV[i];
        grad.segment<3>(base + 6) += gA[i];
    }
    for (int s = 0; s < M; ++s)
        grad[K_cp_ + s] += gT_cp[s] * T[s];
}

//------------------------------------------------------------------------------
int SolverLBFGS::findSegment(double ti,
                             const std::vector<double> &T) const
{
    int M = int(T.size());
    static std::vector<double> ends;
    ends.resize(M);
    ends[0] = T[0];
    for (int i = 1; i < M; ++i)
        ends[i] = ends[i - 1] + T[i];
    auto it = std::upper_bound(ends.begin(), ends.end(), ti);
    int s = int(it - ends.begin());
    return (s < M ? s : M - 1);
}

//------------------------------------------------------------------------------

// Fast and stable version
void SolverLBFGS::evalBernstein5(double tau, double B[6], double dB[6]) const
{
    // clamp into [0,1] to avoid tiny overshoots
    if (tau <= 0.0)
        tau = 0.0;
    else if (tau >= 1.0)
        tau = 1.0;

    double u = tau;
    double v = 1.0 - tau;

    // compute all powers with just multiplies
    double u2 = u * u;
    double u3 = u2 * u;
    double u4 = u3 * u;
    double u5 = u4 * u;

    double v2 = v * v;
    double v3 = v2 * v;
    double v4 = v3 * v;
    double v5 = v4 * v;

    // degree‐5 Bernstein basis
    B[0] = v5;
    B[1] = 5.0 * u * v4;
    B[2] = 10.0 * u2 * v3;
    B[3] = 10.0 * u3 * v2;
    B[4] = 5.0 * u4 * v;
    B[5] = u5;

    // degree‐4 Bernstein used for the derivative formula
    //    B4[k] = C(4,k) u^k v^(4−k)
    double B4_0 = v4;
    double B4_1 = 4 * u * v3;
    double B4_2 = 6 * u2 * v2;
    double B4_3 = 4 * u3 * v;
    double B4_4 = u4;

    // dB[j] = 5 * ( B4[j−1] − B4[j] ), with B4[−1]=B4[5]=0
    dB[0] = -5.0 * B4_0;
    dB[1] = 5.0 * (B4_0 - B4_1);
    dB[2] = 5.0 * (B4_1 - B4_2);
    dB[3] = 5.0 * (B4_2 - B4_3);
    dB[4] = 5.0 * (B4_3 - B4_4);
    dB[5] = 5.0 * B4_4;
}

//------------------------------------------------------------------------------

Eigen::Vector3d SolverLBFGS::evalSample(
    const std::vector<std::array<Eigen::Vector3d, 6>> &CP,
    const std::vector<double> &T,
    int sample_i,
    int num_samples,
    int &seg,
    double &tau,
    double &t0) const
{
    // 1) absolute time of this sample
    double total_time = std::accumulate(T.begin(), T.end(), 0.0);
    double alpha = double(sample_i) / double(num_samples - 1);
    double ti = total_time * alpha;

    // 2) find segment index
    seg = findSegment(ti, T);

    // 3) compute t0 and τ
    t0 = (seg > 0)
             ? std::accumulate(T.begin(), T.begin() + seg, 0.0)
             : 0.0;
    double Ti = T[seg];
    tau = (ti - t0) / Ti;

    // 4) Bernstein basis at τ
    double B[6], dB[6];
    evalBernstein5(tau, B, dB);

    // 5) form position
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    for (int j = 0; j < 6; ++j)
    {
        p += CP[seg][j] * B[j];
    }
    return p;
}

//------------------------------------------------------------------------------

Eigen::Vector3d SolverLBFGS::evalObs(
    const Eigen::Matrix<double, 6, 1> &cx,
    const Eigen::Matrix<double, 6, 1> &cy,
    const Eigen::Matrix<double, 6, 1> &cz,
    double t) const
{
    double tp = 1.0;
    double x = 0, y = 0, zv = 0;
    for (int j = 0; j < 6; ++j)
    {
        x += cx[j] * tp;
        y += cy[j] * tp;
        zv += cz[j] * tp;
        tp *= t;
    }
    return {x, y, zv};
}

// -----------------------------------------------------------------------------

void SolverLBFGS::dJ_stat_dz(const VecXd &z,
                             const std::vector<Vec3> &P,
                             const std::vector<double> &T,
                             VecXd &grad) const
{
    // 2) Loop over interior knots i=1..M_-1
    for (int i = 1; i < M_; ++i)
    {
        const Vec3 &Pi = P[i];
        // Determine which segments’ planes apply
        int segs[2] = {i - 1, i};
        int nseg = (i < M_) ? 2 : 1;

        for (int si = 0; si < nseg; ++si)
        {
            int segIdx = segs[si];
            // A_stat_[segIdx]: (nplanes × 3), b_stat_[segIdx]: length nplanes
            const auto &Aseg = A_stat_[segIdx];
            const auto &bseg = b_stat_[segIdx];
            int nplanes = Aseg.rows();

            for (int p = 0; p < nplanes; ++p)
            {
                // signed distance
                double h = Aseg.row(p).dot(Pi) - bseg[p];
                double viol = Co_ - h;
                if (viol <= 0.0)
                    continue;

                // gradient w.r.t. P_i
                double coeff = -3.0 * viol * viol;
                Vec3 dJ_dPi = coeff * Aseg.row(p).transpose();

                // accumulate into grad
                grad.segment<3>(9 * i) += dJ_dPi;
            }
        }
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::dJ_jerk_dz(const VecXd &z,
                             const std::vector<Vec3> &P,
                             const std::vector<Vec3> &V,
                             const std::vector<Vec3> &A,
                             const std::vector<std::array<Vec3, 6>> &CP,
                             const std::vector<double> &T,
                             VecXd &grad) const
{

    // 2) loop over segments
    for (int s = 0; s < M_; ++s)
    {
        double Ts = T[s];
        double T2 = Ts * Ts;
        double invT5 = 1.0 / std::pow(Ts, 5);
        double invT6 = invT5 / Ts;
        double C = 3600.0 * invT5;
        double dC_dT = -5.0 * 3600.0 * invT6;

        // compute the three third‐differences
        Vec3 d30 = CP[s][3] - 3.0 * CP[s][2] + 3.0 * CP[s][1] - CP[s][0];
        Vec3 d31 = CP[s][4] - 3.0 * CP[s][3] + 3.0 * CP[s][2] - CP[s][1];
        Vec3 d32 = CP[s][5] - 3.0 * CP[s][4] + 3.0 * CP[s][3] - CP[s][2];

        // sum of squared norms
        double sum_norm2 = d30.squaredNorm() + d31.squaredNorm() + d32.squaredNorm();

        // 3) build dJ/dCP array
        std::array<Vec3, 6> dJ_dCP;
        for (auto &v : dJ_dCP)
            v.setZero();

        // m=0: coeffs [-1,3,-3,1] at j=0,1,2,3
        dJ_dCP[0] += 2.0 * C * (-1.0) * d30;
        dJ_dCP[1] += 2.0 * C * (3.0) * d30;
        dJ_dCP[2] += 2.0 * C * (-3.0) * d30;
        dJ_dCP[3] += 2.0 * C * (1.0) * d30;
        // m=1: [-1,3,-3,1] at j=1,2,3,4
        dJ_dCP[1] += 2.0 * C * (-1.0) * d31;
        dJ_dCP[2] += 2.0 * C * (3.0) * d31;
        dJ_dCP[3] += 2.0 * C * (-3.0) * d31;
        dJ_dCP[4] += 2.0 * C * (1.0) * d31;
        // m=2: [-1,3,-3,1] at j=2,3,4,5
        dJ_dCP[2] += 2.0 * C * (-1.0) * d32;
        dJ_dCP[3] += 2.0 * C * (3.0) * d32;
        dJ_dCP[4] += 2.0 * C * (-3.0) * d32;
        dJ_dCP[5] += 2.0 * C * (1.0) * d32;

        // 4) propagate into z‐entries:
        int base_s = 9 * s;
        int base_sp1 = 9 * (s + 1);

        // P[s]
        grad.segment<3>(base_s) += dJ_dCP[0] + dJ_dCP[1] + dJ_dCP[2];
        // V[s]
        grad.segment<3>(base_s + 3) += dJ_dCP[1] * (Ts / 5.0) + dJ_dCP[2] * (2.0 * Ts / 5.0);
        // A[s]
        grad.segment<3>(base_s + 6) += dJ_dCP[2] * (T2 / 20.0);

        // P[s+1]
        grad.segment<3>(base_sp1) += dJ_dCP[3] + dJ_dCP[4] + dJ_dCP[5];
        // V[s+1]
        grad.segment<3>(base_sp1 + 3) += dJ_dCP[3] * (-2.0 * Ts / 5.0) + dJ_dCP[4] * (-Ts / 5.0);
        // A[s+1]
        grad.segment<3>(base_sp1 + 6) += dJ_dCP[3] * (T2 / 20.0);

        // 5) propagate into σ_s
        //    a) through the CP‐dependence (each term picks up a *Ts)
        Vec3 Vs = V[s], As = A[s];
        Vec3 Vsp = V[s + 1], Asp = A[s + 1];
        double term =
            dJ_dCP[1].dot(Vs / 5.0) +
            dJ_dCP[2].dot(2.0 * Vs / 5.0 + Ts * As / 10.0) +
            dJ_dCP[3].dot(-2.0 * Vsp / 5.0 + Ts * Asp / 10.0) +
            dJ_dCP[4].dot(-Vsp / 5.0);
        grad[K_cp_ + s] += Ts * term;

        //    b) through the C_s factor itself
        grad[K_cp_ + s] += dC_dT * sum_norm2 * Ts;
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::dJ_vel_constr_dz(const VecXd &z,
                                   const std::vector<Vec3> &P,
                                   const std::vector<Vec3> &V,
                                   const std::vector<Vec3> &A,
                                   const std::vector<std::array<Vec3, 6>> &CP,
                                   const std::vector<double> &T,
                                   VecXd &grad) const
{
    const double eps = 1e-12;

    // accumulators
    std::vector<Vec3> gP(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gV(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gA(M_ + 1, Vec3::Zero());
    std::vector<double> gT(M_, 0.0);

    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        const double invT = 1.0 / (Ts + 1e-16);
        const double alpha = 5.0 * invT; // 5/T
        const double dalpha_dT = -5.0 / (Ts * Ts + 1e-16);

        // gradient wrt the 6 CPs of segment s
        std::array<Vec3, 6> gCP_s;
        for (auto &g : gCP_s)
            g.setZero();

        for (int j = 0; j <= 4; ++j)
        {
            const Vec3 d1 = CP[s][j + 1] - CP[s][j];
            const Vec3 U = alpha * d1;
            const double nU = U.norm();

            if (nU <= V_max_ + eps)
                continue;

            // dJ/dU = 3*(||U|| - Vmax)^2 * U/||U||
            const double coeff = 3.0 * std::pow(nU - V_max_, 2);
            const Vec3 gU = (coeff / std::max(nU, eps)) * U; // (3,)

            // chain to CP: U = alpha * (CP_{j+1} - CP_j)
            gCP_s[j] += -alpha * gU;
            gCP_s[j + 1] += alpha * gU;

            // direct dJ/dT via alpha(T)
            const Vec3 dU_dT = dalpha_dT * d1;
            gT[s] += gU.dot(dU_dT);
        }

        // push gCP_s to (p,v,a,T) through Hermite->Bézier map
        const double Ts2 = Ts * Ts;

        // p_s, v_s, a_s
        gP[s] += gCP_s[0] + gCP_s[1] + gCP_s[2];
        gV[s] += (Ts / 5.0) * gCP_s[1] + (2.0 * Ts / 5.0) * gCP_s[2];
        gA[s] += (Ts2 / 20.0) * gCP_s[2];

        // p_{s+1}, v_{s+1}, a_{s+1}
        gP[s + 1] += gCP_s[3] + gCP_s[4] + gCP_s[5];
        gV[s + 1] += (-2.0 * Ts / 5.0) * gCP_s[3] + (-Ts / 5.0) * gCP_s[4];
        gA[s + 1] += (Ts2 / 20.0) * gCP_s[3];

        // dJ/dT via CP(T) dependence
        gT[s] += gCP_s[1].dot(V[s] / 5.0);
        gT[s] += gCP_s[2].dot((2.0 / 5.0) * V[s] + (Ts / 10.0) * A[s]);
        gT[s] += gCP_s[3].dot((-2.0 / 5.0) * V[s + 1] + (Ts / 10.0) * A[s + 1]);
        gT[s] += gCP_s[4].dot((-1.0 / 5.0) * V[s + 1]);
    }

    // scatter to z (T = exp σ ⇒ d/dσ = T * d/dT)
    for (int i = 0; i <= M_; ++i)
    {
        const int base = 9 * i;
        grad.segment<3>(base + 0) += gP[i];
        grad.segment<3>(base + 3) += gV[i];
        grad.segment<3>(base + 6) += gA[i];
    }
    for (int s = 0; s < M_; ++s)
    {
        grad[K_cp_ + s] += gT[s] * T[s];
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::dJ_acc_constr_dz(const VecXd &z,
                                   const std::vector<Vec3> &P,
                                   const std::vector<Vec3> &V,
                                   const std::vector<Vec3> &A,
                                   const std::vector<std::array<Vec3, 6>> &CP,
                                   const std::vector<double> &T,
                                   VecXd &grad) const
{
    const double eps = 1e-12;

    // accumulators
    std::vector<Vec3> gP(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gV(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gA(M_ + 1, Vec3::Zero());
    std::vector<double> gT(M_, 0.0);

    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        const double invT = 1.0 / (Ts + 1e-16);
        const double beta = 20.0 * invT * invT; // 20/T^2
        const double dbeta_dT = -40.0 / (std::pow(Ts, 3) + 1e-16);

        // gradient wrt the 6 CPs of segment s
        std::array<Vec3, 6> gCP_s;
        for (auto &g : gCP_s)
            g.setZero();

        for (int j = 0; j <= 3; ++j)
        {
            const Vec3 d2 = CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j];
            const Vec3 W = beta * d2;
            const double nW = W.norm();

            if (nW <= A_max_ + eps)
                continue;

            // dJ/dW = 3*(||W|| - Amax)^2 * W/||W||
            const double coeff = 3.0 * std::pow(nW - A_max_, 2);
            const Vec3 gW = (coeff / std::max(nW, eps)) * W;

            // chain to CP: W = beta * (CP_{j+2} - 2 CP_{j+1} + CP_j)
            gCP_s[j] += beta * gW;
            gCP_s[j + 1] += -2.0 * beta * gW;
            gCP_s[j + 2] += beta * gW;

            // direct dJ/dT via beta(T)
            const Vec3 dW_dT = dbeta_dT * d2;
            gT[s] += gW.dot(dW_dT);
        }

        // push gCP_s to (p,v,a,T) through Hermite->Bézier map
        const double Ts2 = Ts * Ts;

        gP[s] += gCP_s[0] + gCP_s[1] + gCP_s[2];
        gV[s] += (Ts / 5.0) * gCP_s[1] + (2.0 * Ts / 5.0) * gCP_s[2];
        gA[s] += (Ts2 / 20.0) * gCP_s[2];

        gP[s + 1] += gCP_s[3] + gCP_s[4] + gCP_s[5];
        gV[s + 1] += (-2.0 * Ts / 5.0) * gCP_s[3] + (-Ts / 5.0) * gCP_s[4];
        gA[s + 1] += (Ts2 / 20.0) * gCP_s[3];

        // dJ/dT via CP(T) dependence
        gT[s] += gCP_s[1].dot(V[s] / 5.0);
        gT[s] += gCP_s[2].dot((2.0 / 5.0) * V[s] + (Ts / 10.0) * A[s]);
        gT[s] += gCP_s[3].dot((-2.0 / 5.0) * V[s + 1] + (Ts / 10.0) * A[s + 1]);
        gT[s] += gCP_s[4].dot((-1.0 / 5.0) * V[s + 1]);
    }

    // scatter to z
    for (int i = 0; i <= M_; ++i)
    {
        const int base = 9 * i;
        grad.segment<3>(base + 0) += gP[i];
        grad.segment<3>(base + 3) += gV[i];
        grad.segment<3>(base + 6) += gA[i];
    }
    for (int s = 0; s < M_; ++s)
    {
        grad[K_cp_ + s] += gT[s] * T[s];
    }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::dJ_thrust_max_dz(const VecXd &z,
                                   const std::vector<Vec3> &P,
                                   const std::vector<Vec3> &V,
                                   const std::vector<Vec3> &A,
                                   const std::vector<std::array<Vec3, 6>> &CP,
                                   const std::vector<double> &T,
                                   VecXd &grad) const
{
    const double eps = 1e-12;
    const Vec3 e3(0, 0, 1);
    const double g = g_;

    // accumulators
    std::vector<Vec3> gP(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gV(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gA(M_ + 1, Vec3::Zero());
    std::vector<double> gT(M_, 0.0);

    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        if (Ts <= 0.0)
            continue;

        const double invT = 1.0 / (Ts + 1e-16);
        const double beta = 20.0 * invT * invT; // 20/T^2
        const double dbeta_dT = -40.0 / (std::pow(Ts, 3) + 1e-16);

        std::array<Vec3, 6> gCP_s;
        for (auto &gcp : gCP_s)
            gcp.setZero();

        for (int j = 0; j <= 3; ++j)
        {
            const Vec3 d2 = CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j];
            const Vec3 W = beta * d2;
            const Vec3 Wt = W + g * e3;

            const double n = std::sqrt(Wt.squaredNorm() + eps * eps);
            const double viol = n - thrust_max_;
            if (viol <= 0.0)
                continue;

            // dJ/dWt = 3*viol^2 * Wt / n
            const Vec3 gWt = (3.0 * viol * viol / n) * Wt;

            // chain: W = beta * d2
            gCP_s[j] += beta * gWt;
            gCP_s[j + 1] += -2.0 * beta * gWt;
            gCP_s[j + 2] += beta * gWt;

            // direct dJ/dT through beta(T)
            const Vec3 dW_dT = dbeta_dT * d2;
            gT[s] += gWt.dot(dW_dT);
        }

        // push to (P,V,A,T) through Hermite→Bézier map
        accumulate_cp_to_pvaT(s, gCP_s, V, A, T, gP, gV, gA, gT);
    }

    // scatter to z
    for (int i = 0; i <= M_; ++i)
    {
        const int base = 9 * i;
        grad.segment<3>(base + 0) += gP[i];
        grad.segment<3>(base + 3) += gV[i];
        grad.segment<3>(base + 6) += gA[i];
    }
    for (int s = 0; s < M_; ++s)
        grad[K_cp_ + s] += gT[s] * T[s];
}

// -----------------------------------------------------------------------------

void SolverLBFGS::dJ_thrust_min_dz(const VecXd &z,
                                   const std::vector<Vec3> &P,
                                   const std::vector<Vec3> &V,
                                   const std::vector<Vec3> &A,
                                   const std::vector<std::array<Vec3, 6>> &CP,
                                   const std::vector<double> &T,
                                   VecXd &grad) const
{
    const Vec3 e3(0, 0, 1);
    const double g = g_;

    std::vector<Vec3> gP(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gV(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gA(M_ + 1, Vec3::Zero());
    std::vector<double> gT(M_, 0.0);

    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        if (Ts <= 0.0)
            continue;

        const double invT = 1.0 / (Ts + 1e-16);
        const double beta = 20.0 * invT * invT;
        const double dbeta_dT = -40.0 / (std::pow(Ts, 3) + 1e-16);

        std::array<Vec3, 6> gCP_s;
        for (auto &gcp : gCP_s)
            gcp.setZero();

        for (int j = 0; j <= 3; ++j)
        {
            const Vec3 d2 = CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j];
            const Vec3 W = beta * d2;
            const double Wt_z = W.z() + g;

            const double viol = thrust_min_ - Wt_z;
            if (viol <= 0.0)
                continue;

            // dJ/dWt_z = -3 * viol^2
            const Vec3 gWt(0.0, 0.0, -3.0 * viol * viol);

            // chain via W = beta * d2
            gCP_s[j] += beta * gWt;
            gCP_s[j + 1] += -2.0 * beta * gWt;
            gCP_s[j + 2] += beta * gWt;

            const Vec3 dW_dT = dbeta_dT * d2;
            gT[s] += gWt.dot(dW_dT);
        }

        accumulate_cp_to_pvaT(s, gCP_s, V, A, T, gP, gV, gA, gT);
    }

    for (int i = 0; i <= M_; ++i)
    {
        const int base = 9 * i;
        grad.segment<3>(base + 0) += gP[i];
        grad.segment<3>(base + 3) += gV[i];
        grad.segment<3>(base + 6) += gA[i];
    }
    for (int s = 0; s < M_; ++s)
        grad[K_cp_ + s] += gT[s] * T[s];
}

// -----------------------------------------------------------------------------

void SolverLBFGS::dJ_theta_dz(const VecXd &z,
                              const std::vector<Vec3> &P,
                              const std::vector<Vec3> &V,
                              const std::vector<Vec3> &A,
                              const std::vector<std::array<Vec3, 6>> &CP,
                              const std::vector<double> &T,
                              VecXd &grad) const
{
    const double eps = 1e-12;
    const Vec3 e3(0, 0, 1);
    const double g = g_;

    std::vector<Vec3> gP(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gV(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gA(M_ + 1, Vec3::Zero());
    std::vector<double> gT(M_, 0.0);

    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        if (Ts <= 0.0)
            continue;

        const double invT = 1.0 / (Ts + 1e-16);
        const double beta = 20.0 * invT * invT;
        const double dbeta_dT = -40.0 / (std::pow(Ts, 3) + 1e-16);

        std::array<Vec3, 6> gCP_s;
        for (auto &gcp : gCP_s)
            gcp.setZero();

        for (int j = 0; j <= 3; ++j)
        {
            const Vec3 d2 = CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j];
            const Vec3 W = beta * d2;
            const Vec3 Wt = W + g * e3;
            const double tan_theta_max_ = std::tan(theta_max_);
            const double nxy = std::sqrt(Wt.x() * Wt.x() + Wt.y() * Wt.y() + eps * eps);
            const double viol = nxy - tan_theta_max_ * Wt.z();
            if (viol <= 0.0)
                continue;

            // dJ/dWt = 3*viol^2 * [ Wx/nxy, Wy/nxy, -tan(theta_max) ]
            Vec3 gWt(Wt.x() / nxy, Wt.y() / nxy, -tan_theta_max_);
            gWt *= (3.0 * viol * viol);

            // chain via W = beta * d2
            gCP_s[j] += beta * gWt;
            gCP_s[j + 1] += -2.0 * beta * gWt;
            gCP_s[j + 2] += beta * gWt;

            const Vec3 dW_dT = dbeta_dT * d2;
            gT[s] += gWt.dot(dW_dT);
        }

        accumulate_cp_to_pvaT(s, gCP_s, V, A, T, gP, gV, gA, gT);
    }

    for (int i = 0; i <= M_; ++i)
    {
        const int base = 9 * i;
        grad.segment<3>(base + 0) += gP[i];
        grad.segment<3>(base + 3) += gV[i];
        grad.segment<3>(base + 6) += gA[i];
    }
    for (int s = 0; s < M_; ++s)
        grad[K_cp_ + s] += gT[s] * T[s];
}

// -----------------------------------------------------------------------------

void SolverLBFGS::dJ_omega_dz(const VecXd &z,
                              const std::vector<Vec3> &P,
                              const std::vector<Vec3> &V,
                              const std::vector<Vec3> &A,
                              const std::vector<std::array<Vec3, 6>> &CP,
                              const std::vector<double> &T,
                              VecXd &grad) const
{
    const double eps = 1e-12;
    const Vec3 e3(0, 0, 1);
    const double g = g_;

    std::vector<Vec3> gP(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gV(M_ + 1, Vec3::Zero());
    std::vector<Vec3> gA(M_ + 1, Vec3::Zero());
    std::vector<double> gT(M_, 0.0);

    for (int s = 0; s < M_; ++s)
    {
        const double Ts = T[s];
        if (Ts <= 0.0)
            continue;

        const double invT = 1.0 / (Ts + 1e-16);
        const double beta = 20.0 * invT * invT; // accel scale
        const double dbeta_dT = -40.0 / (std::pow(Ts, 3) + 1e-16);
        const double gamma = 60.0 * invT * invT * invT; // jerk  scale
        const double dgamma_dT = -180.0 / (std::pow(Ts, 4) + 1e-16);

        std::array<Vec3, 6> gCP_s;
        for (auto &gcp : gCP_s)
            gcp.setZero();

        for (int j = 0; j <= 2; ++j)
        {
            // Jerk CP (quad) at j
            const Vec3 d3 = CP[s][j + 3] - 3.0 * CP[s][j + 2] + 3.0 * CP[s][j + 1] - CP[s][j];
            const Vec3 Jv = gamma * d3;
            const double nJ = std::sqrt(Jv.squaredNorm() + eps * eps);

            // vertical thrust at same j from accel CPs
            const Vec3 d2 = CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j];
            const Vec3 W = beta * d2;
            const double Wt_z = W.z() + g;

            const double viol = nJ - omega_max_ * Wt_z;
            if (viol <= 0.0)
                continue;

            // part 1: through J
            const Vec3 gJ = (3.0 * viol * viol / nJ) * Jv; // dJ/dJv

            // chain to CP via J = gamma * d3  (coeffs [-1,3,-3,1])
            gCP_s[j] += -gamma * gJ;
            gCP_s[j + 1] += 3.0 * gamma * gJ;
            gCP_s[j + 2] += -3.0 * gamma * gJ;
            gCP_s[j + 3] += gamma * gJ;

            // direct d/dT via gamma(T)
            gT[s] += gJ.dot(dgamma_dT * d3);

            // part 2: through Wt_z term (−ω_max * Wt_z)
            const Vec3 gWt(0.0, 0.0, -omega_max_ * 3.0 * viol * viol);

            // chain via W = beta * d2  (uses j..j+2)
            gCP_s[j] += beta * gWt;
            gCP_s[j + 1] += -2.0 * beta * gWt;
            gCP_s[j + 2] += beta * gWt;

            gT[s] += gWt.dot(dbeta_dT * d2); // direct T via beta(T)
        }

        accumulate_cp_to_pvaT(s, gCP_s, V, A, T, gP, gV, gA, gT);
    }

    for (int i = 0; i <= M_; ++i)
    {
        const int base = 9 * i;
        grad.segment<3>(base + 0) += gP[i];
        grad.segment<3>(base + 3) += gV[i];
        grad.segment<3>(base + 6) += gA[i];
    }
    for (int s = 0; s < M_; ++s)
        grad[K_cp_ + s] += gT[s] * T[s];
}

// -----------------------------------------------------------------------------

void SolverLBFGS::sanityCheck() const
{

    // If fewer than 2 segments, that's not valid
    if (M_ < 2)
    {
        std::cout << "\033[31mError: M_ must be at least 2.\033[0m" << std::endl;
    }

    if (K_ <= 0)
    {
        std::cout << "\033[31mError: K_ must be positive.\033[0m" << std::endl;
    }

    // Check if the number of segments matches (the number of global waypoints - 1)
    if (M_ != global_wps_.size() - 1)
    {
        // red color
        std::cout << "\033[31mError: Number of segments does not match the number of global waypoints - 1.\033[0m" << std::endl;
        std::cout << "M_ = " << M_ << ", global_wps_.size() = " << global_wps_.size() << std::endl;
    }
}