/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef GCOPTER_HPP
#define GCOPTER_HPP

#include "gcopter/minco.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>

namespace gcopter
{

    class GCOPTER_PolytopeSFC
    {
    public:
        typedef Eigen::Matrix3Xd PolyhedronV;
        typedef Eigen::MatrixX4d PolyhedronH;
        typedef std::vector<PolyhedronV> PolyhedraV;
        typedef std::vector<PolyhedronH> PolyhedraH;
        Eigen::Matrix3Xd initial_guess_points_;
        Eigen::VectorXd initial_guess_times_;
        Eigen::VectorXd initial_xi_;
        double computation_time_ms_;

    private:
        minco::MINCO_S3NU minco;
        flatness::FlatnessMap flatmap;

        double rho;
        Eigen::Matrix3d headPVA;
        Eigen::Matrix3d tailPVA;

        PolyhedraV vPolytopes;
        PolyhedraH hPolytopes;
        Eigen::Matrix3Xd shortPath;

        Eigen::VectorXi pieceIdx;
        Eigen::VectorXi vPolyIdx;
        Eigen::VectorXi hPolyIdx;

        int polyN;
        int pieceN;

        int spatialDim;
        int temporalDim;

        double smoothEps;
        int integralRes;
        Eigen::VectorXd magnitudeBd;
        Eigen::VectorXd penaltyWt;
        Eigen::VectorXd physicalPm;
        double allocSpeed;

        lbfgs::lbfgs_parameter_t lbfgs_params;

        Eigen::Matrix3Xd points;
        Eigen::VectorXd times;
        Eigen::Matrix3Xd gradByPoints;
        Eigen::VectorXd gradByTimes;
        Eigen::MatrixX3d partialGradByCoeffs;
        Eigen::VectorXd partialGradByTimes;

        // Velocity reference settings (knot velocity soft cost)
        bool vel_ref_enable_{false};
        int vel_ref_knot_{-1};
        Eigen::Vector3d vel_ref_{Eigen::Vector3d::Zero()};
        double vel_ref_weight_{0.0};
        bool vel_ref_grad_check_{false};
        mutable bool vel_ref_grad_check_done_{false};
        int vel_ref_log_every_{0};
        // Position reference settings (knot position soft cost)
        bool pos_ref_enable_{false};
        int pos_ref_knot_{-1};
        Eigen::Vector3d pos_ref_{Eigen::Vector3d::Zero()};
        double pos_ref_weight_{0.0};
        bool pos_ref_grad_check_{false};
        mutable bool pos_ref_grad_check_done_{false};
        int pos_ref_log_every_{0};
        // Full objective gradient check (GCOPTER)
        bool full_grad_check_enable_{false};
        int full_grad_check_dirs_{8};
        int full_grad_check_max_coords_{256};
        double full_grad_check_eps_{1e-5};
        mutable bool full_grad_check_done_{false};
        mutable bool full_grad_check_running_{false};
        mutable Eigen::Vector3d last_vref_v_{Eigen::Vector3d::Zero()};
        mutable double last_vref_err_{0.0};
        mutable double last_vref_cost_{0.0};
        mutable bool last_vref_valid_{false};
        mutable Eigen::Vector3d last_pref_p_{Eigen::Vector3d::Zero()};
        mutable double last_pref_err_{0.0};
        mutable double last_pref_cost_{0.0};
        mutable bool last_pref_valid_{false};

        std::vector<Eigen::MatrixX4d> piece_corridor_;

        // --- Wall-clock time budget control ---
        mutable bool time_budget_enabled_ = false;
        mutable double time_budget_ms_ = 0.0;
        mutable std::chrono::steady_clock::time_point deadline_;

        // Best-so-far snapshot (so we can return the best iterate on early exit)
        mutable Eigen::VectorXd best_x_;
        mutable double best_f_ = std::numeric_limits<double>::infinity();

    private:
        inline void setTimeBudgetMs(double ms)
        {
            time_budget_ms_ = ms;
            deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds((int)ms);
            time_budget_enabled_ = (ms > 0.0);
        }

        // Called once per *iteration* (not per function eval)
        static int progressTimeGuard(
            void *instance,
            const Eigen::VectorXd &x,
            const Eigen::VectorXd &g,
            const double fx,
            const double /*step*/,
            const int k,
            const int /*ls*/)
        {
            auto *self = static_cast<GCOPTER_PolytopeSFC *>(instance);

            // Track best-so-far
            if (fx < self->best_f_)
            {
                self->best_f_ = fx;
                self->best_x_ = x; // copy
            }

            const bool log_vref = self->vel_ref_enable_ && self->vel_ref_log_every_ > 0 &&
                                  (k % self->vel_ref_log_every_ == 0);
            const bool log_pref = self->pos_ref_enable_ && self->pos_ref_log_every_ > 0 &&
                                  (k % self->pos_ref_log_every_ == 0);
            if (log_vref || log_pref)
            {
                std::cout << "[GCOPTER][iter " << k << "] ";
                if (log_vref)
                {
                    std::cout << "vref_knot=" << self->vel_ref_knot_
                              << " v_i=" << self->last_vref_v_.transpose()
                              << " |e|=" << self->last_vref_err_
                              << " J_vref=" << self->last_vref_cost_;
                }
                if (log_pref)
                {
                    if (log_vref)
                        std::cout << " ";
                    std::cout << "pref_knot=" << self->pos_ref_knot_
                              << " p_i=" << self->last_pref_p_.transpose()
                              << " |e_p|=" << self->last_pref_err_
                              << " J_pref=" << self->last_pref_cost_;
                }
                std::cout << " f=" << fx << "\n";
            }

            if (!self->time_budget_enabled_)
                return 0;

            // Hard stop on wall-clock
            if (std::chrono::steady_clock::now() >= self->deadline_)
            {
                return 1; // non-zero => cancel optimization
            }
            return 0;
        }

        // Called right before each line search to cap the step size
        static double stepBoundTimeGuard(
            void *instance,
            const Eigen::VectorXd &xp,
            const Eigen::VectorXd &d)
        {
            auto *self = static_cast<GCOPTER_PolytopeSFC *>(instance);
            (void)xp;
            (void)d;

            if (!self->time_budget_enabled_)
            {
                // No extra cap; let the library clamp to param.max_step.
                return self->lbfgs_params.max_step;
            }

            // If past the deadline, force a tiny step so the algorithm returns to the
            // progress callback quickly. This keeps the overrun to ~one iteration.
            if (std::chrono::steady_clock::now() >= self->deadline_)
            {
                return self->lbfgs_params.min_step; // as small as the library allows
            }

            return self->lbfgs_params.max_step;
        }

        static inline void forwardT(const Eigen::VectorXd &tau,
                                    Eigen::VectorXd &T)
        {
            const int sizeTau = tau.size();
            T.resize(sizeTau);
            for (int i = 0; i < sizeTau; i++)
            {
                T(i) = tau(i) > 0.0
                           ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                           : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
            }
            return;
        }

        template <typename EIGENVEC>
        static inline void backwardT(const Eigen::VectorXd &T,
                                     EIGENVEC &tau)
        {
            const int sizeT = T.size();
            tau.resize(sizeT);
            for (int i = 0; i < sizeT; i++)
            {
                tau(i) = T(i) > 1.0
                             ? (sqrt(2.0 * T(i) - 1.0) - 1.0)
                             : (1.0 - sqrt(2.0 / T(i) - 1.0));
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradT(const Eigen::VectorXd &tau,
                                         const Eigen::VectorXd &gradT,
                                         EIGENVEC &gradTau)
        {
            const int sizeTau = tau.size();
            gradTau.resize(sizeTau);
            double denSqrt;
            for (int i = 0; i < sizeTau; i++)
            {
                if (tau(i) > 0)
                {
                    gradTau(i) = gradT(i) * (tau(i) + 1.0);
                }
                else
                {
                    denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
                    gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
                }
            }

            return;
        }

        static inline void forwardP(const Eigen::VectorXd &xi,
                                    const Eigen::VectorXi &vIdx,
                                    const PolyhedraV &vPolys,
                                    Eigen::Matrix3Xd &P)
        {
            const int sizeP = vIdx.size();
            P.resize(3, sizeP);
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k).normalized().head(k - 1);
                P.col(i) = vPolys[l].rightCols(k - 1) * q.cwiseProduct(q) +
                           vPolys[l].col(0);
            }
            return;
        }

        static inline double costTinyNLS(void *ptr,
                                         const Eigen::VectorXd &xi,
                                         Eigen::VectorXd &gradXi)
        {
            const int n = xi.size();
            const Eigen::Matrix3Xd &ovPoly = *(Eigen::Matrix3Xd *)ptr;

            const double sqrNormXi = xi.squaredNorm();
            const double invNormXi = 1.0 / sqrt(sqrNormXi);
            const Eigen::VectorXd unitXi = xi * invNormXi;
            const Eigen::VectorXd r = unitXi.head(n - 1);
            const Eigen::Vector3d delta = ovPoly.rightCols(n - 1) * r.cwiseProduct(r) +
                                          ovPoly.col(1) - ovPoly.col(0);

            double cost = delta.squaredNorm();
            gradXi.head(n - 1) = (ovPoly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                                 r.array() * 2.0;
            gradXi(n - 1) = 0.0;
            gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

            const double sqrNormViolation = sqrNormXi - 1.0;
            if (sqrNormViolation > 0.0)
            {
                double c = sqrNormViolation * sqrNormViolation;
                const double dc = 3.0 * c;
                c *= sqrNormViolation;
                cost += c;
                gradXi += dc * 2.0 * xi;
            }

            return cost;
        }

        template <typename EIGENVEC>
        static inline void backwardP(const Eigen::Matrix3Xd &P,
                                     const Eigen::VectorXi &vIdx,
                                     const PolyhedraV &vPolys,
                                     EIGENVEC &xi)
        {
            const int sizeP = P.cols();

            double minSqrD;
            lbfgs::lbfgs_parameter_t tiny_nls_params;
            tiny_nls_params.past = 0;
            tiny_nls_params.delta = 1.0e-5;
            tiny_nls_params.g_epsilon = FLT_EPSILON;
            tiny_nls_params.max_iterations = 128;

            Eigen::Matrix3Xd ovPoly;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();

                ovPoly.resize(3, k + 1);
                ovPoly.col(0) = P.col(i);
                ovPoly.rightCols(k) = vPolys[l];
                Eigen::VectorXd x(k);
                x.setConstant(sqrt(1.0 / k));
                lbfgs::lbfgs_optimize(x,
                                      minSqrD,
                                      &GCOPTER_PolytopeSFC::costTinyNLS,
                                      nullptr,
                                      nullptr,
                                      &ovPoly,
                                      tiny_nls_params);

                xi.segment(j, k) = x;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradP(const Eigen::VectorXd &xi,
                                         const Eigen::VectorXi &vIdx,
                                         const PolyhedraV &vPolys,
                                         const Eigen::Matrix3Xd &gradP,
                                         EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double normInv;
            Eigen::VectorXd q, gradQ, unitQ;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k);
                normInv = 1.0 / q.norm();
                unitQ = q * normInv;
                gradQ.resize(k);
                gradQ.head(k - 1) = (vPolys[l].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradXi.segment(j, k) = (gradQ - unitQ * unitQ.dot(gradQ)) * normInv;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void normRetrictionLayer(const Eigen::VectorXd &xi,
                                               const Eigen::VectorXi &vIdx,
                                               const PolyhedraV &vPolys,
                                               double &cost,
                                               EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double sqrNormQ, sqrNormViolation, c, dc;
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k; i < sizeP; i++, j += k)
            {
                k = vPolys[vIdx(i)].cols();

                q = xi.segment(j, k);
                sqrNormQ = q.squaredNorm();
                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradXi.segment(j, k) += dc * 2.0 * q;
                }
            }

            return;
        }

        static inline bool smoothedL1(const double &x,
                                      const double &mu,
                                      double &f,
                                      double &df)
        {
            if (x < 0.0)
            {
                return false;
            }
            else if (x > mu)
            {
                f = x - 0.5 * mu;
                df = 1.0;
                return true;
            }
            else
            {
                const double xdmu = x / mu;
                const double sqrxdmu = xdmu * xdmu;
                const double mumxd2 = mu - 0.5 * x;
                f = mumxd2 * sqrxdmu * xdmu;
                df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
                return true;
            }
        }

        // Soft velocity reference at a knot (uses left segment endpoint).
        // Adds cost and partial gradients in coefficient/time space.
        static inline bool attachVelRefFunctional(const Eigen::VectorXd &T,
                                                  const Eigen::MatrixX3d &coeffs,
                                                  const int knot_i,
                                                  const Eigen::Vector3d &v_ref,
                                                  const double weight,
                                                  double &cost,
                                                  Eigen::VectorXd &gradT,
                                                  Eigen::MatrixX3d &gradC,
                                                  Eigen::Vector3d *v_out = nullptr,
                                                  Eigen::Vector3d *a_out = nullptr)
        {
            const int pieceNum = T.size();
            if (weight <= 0.0 || knot_i <= 0 || knot_i >= pieceNum)
                return false;

            const int s = knot_i - 1;
            const double t = T(s);
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;

            Eigen::Matrix<double, 6, 1> beta1, beta2;
            beta1 << 0.0, 1.0, 2.0 * t, 3.0 * t2, 4.0 * t3, 5.0 * t4;
            beta2 << 0.0, 0.0, 2.0, 6.0 * t, 12.0 * t2, 20.0 * t3;

            const Eigen::Matrix<double, 6, 3> c = coeffs.block<6, 3>(s * 6, 0);
            const Eigen::Vector3d v = c.transpose() * beta1;
            const Eigen::Vector3d a = c.transpose() * beta2;
            const Eigen::Vector3d e = v - v_ref;
            const Eigen::Vector3d gv = weight * e;

            cost += 0.5 * weight * e.squaredNorm();
            gradC.block<6, 3>(s * 6, 0) += beta1 * gv.transpose();
            gradT(s) += gv.dot(a);

            if (v_out)
                *v_out = v;
            if (a_out)
                *a_out = a;

            return true;
        }

        // Soft position reference at a knot (uses left segment endpoint).
        // Adds cost and partial gradients in coefficient/time space.
        static inline bool attachPosRefFunctional(const Eigen::VectorXd &T,
                                                  const Eigen::MatrixX3d &coeffs,
                                                  const int knot_i,
                                                  const Eigen::Vector3d &p_ref,
                                                  const double weight,
                                                  double &cost,
                                                  Eigen::VectorXd &gradT,
                                                  Eigen::MatrixX3d &gradC,
                                                  Eigen::Vector3d *p_out = nullptr,
                                                  Eigen::Vector3d *v_out = nullptr)
        {
            const int pieceNum = T.size();
            if (weight <= 0.0 || knot_i <= 0 || knot_i >= pieceNum)
                return false;

            const int s = knot_i - 1;
            const double t = T(s);
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            const double t5 = t4 * t;

            Eigen::Matrix<double, 6, 1> beta0, beta1;
            beta0 << 1.0, t, t2, t3, t4, t5;
            beta1 << 0.0, 1.0, 2.0 * t, 3.0 * t2, 4.0 * t3, 5.0 * t4;

            const Eigen::Matrix<double, 6, 3> c = coeffs.block<6, 3>(s * 6, 0);
            const Eigen::Vector3d p = c.transpose() * beta0;
            const Eigen::Vector3d v = c.transpose() * beta1;
            const Eigen::Vector3d e = p - p_ref;
            const Eigen::Vector3d gp = weight * e;

            cost += 0.5 * weight * e.squaredNorm();
            gradC.block<6, 3>(s * 6, 0) += beta0 * gp.transpose();
            gradT(s) += gp.dot(v);

            if (p_out)
                *p_out = p;
            if (v_out)
                *v_out = v;

            return true;
        }

        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        static inline void attachPenaltyFunctional(const Eigen::VectorXd &T,
                                                   const Eigen::MatrixX3d &coeffs,
                                                   const Eigen::VectorXi &hIdx,
                                                   const PolyhedraH &hPolys,
                                                   const double &smoothFactor,
                                                   const int &integralResolution,
                                                   const Eigen::VectorXd &magnitudeBounds,
                                                   const Eigen::VectorXd &penaltyWeights,
                                                   flatness::FlatnessMap &flatMap,
                                                   double &cost,
                                                   Eigen::VectorXd &gradT,
                                                   Eigen::MatrixX3d &gradC)
        {
            const double velSqrMax = magnitudeBounds(0) * magnitudeBounds(0);
            const double omgSqrMax = magnitudeBounds(1) * magnitudeBounds(1);
            const double thetaMax = magnitudeBounds(2);
            const double thrustMean = 0.5 * (magnitudeBounds(3) + magnitudeBounds(4));
            const double thrustRadi = 0.5 * fabs(magnitudeBounds(4) - magnitudeBounds(3));
            const double thrustSqrRadi = thrustRadi * thrustRadi;

            const double weightPos = penaltyWeights(0);
            const double weightVel = penaltyWeights(1);
            const double weightOmg = penaltyWeights(2);
            const double weightTheta = penaltyWeights(3);
            const double weightThrust = penaltyWeights(4);

            Eigen::Vector3d pos, vel, acc, jer, sna;
            Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
            double totalGradPsi, totalGradPsiD;
            double thr, cos_theta;
            Eigen::Vector4d quat;
            Eigen::Vector3d omg;
            double gradThr;
            Eigen::Vector4d gradQuat;
            Eigen::Vector3d gradPos, gradVel, gradOmg;

            double step, alpha;
            double s1, s2, s3, s4, s5;
            Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;
            Eigen::Vector3d outerNormal;
            int K, L;
            double violaPos, violaVel, violaOmg, violaTheta, violaThrust;
            double violaPosPenaD, violaVelPenaD, violaOmgPenaD, violaThetaPenaD, violaThrustPenaD;
            double violaPosPena, violaVelPena, violaOmgPena, violaThetaPena, violaThrustPena;
            double node, pena;

            const int pieceNum = T.size();
            const double integralFrac = 1.0 / integralResolution;
            for (int i = 0; i < pieceNum; i++)
            {
                const Eigen::Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
                step = T(i) * integralFrac;
                for (int j = 0; j <= integralResolution; j++)
                {
                    s1 = j * step;
                    s2 = s1 * s1;
                    s3 = s2 * s1;
                    s4 = s2 * s2;
                    s5 = s4 * s1;
                    beta0(0) = 1.0, beta0(1) = s1, beta0(2) = s2, beta0(3) = s3, beta0(4) = s4, beta0(5) = s5;
                    beta1(0) = 0.0, beta1(1) = 1.0, beta1(2) = 2.0 * s1, beta1(3) = 3.0 * s2, beta1(4) = 4.0 * s3, beta1(5) = 5.0 * s4;
                    beta2(0) = 0.0, beta2(1) = 0.0, beta2(2) = 2.0, beta2(3) = 6.0 * s1, beta2(4) = 12.0 * s2, beta2(5) = 20.0 * s3;
                    beta3(0) = 0.0, beta3(1) = 0.0, beta3(2) = 0.0, beta3(3) = 6.0, beta3(4) = 24.0 * s1, beta3(5) = 60.0 * s2;
                    beta4(0) = 0.0, beta4(1) = 0.0, beta4(2) = 0.0, beta4(3) = 0.0, beta4(4) = 24.0, beta4(5) = 120.0 * s1;
                    pos = c.transpose() * beta0;
                    vel = c.transpose() * beta1;
                    acc = c.transpose() * beta2;
                    jer = c.transpose() * beta3;
                    sna = c.transpose() * beta4;

                    flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

                    violaVel = vel.squaredNorm() - velSqrMax;
                    violaOmg = omg.squaredNorm() - omgSqrMax;
                    cos_theta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
                    violaTheta = acos(cos_theta) - thetaMax;
                    violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;

                    gradThr = 0.0;
                    gradQuat.setZero();
                    gradPos.setZero(), gradVel.setZero(), gradOmg.setZero();
                    pena = 0.0;

                    L = hIdx(i);
                    K = hPolys[L].rows();
                    for (int k = 0; k < K; k++)
                    {
                        outerNormal = hPolys[L].block<1, 3>(k, 0);
                        violaPos = outerNormal.dot(pos) + hPolys[L](k, 3);
                        if (smoothedL1(violaPos, smoothFactor, violaPosPena, violaPosPenaD))
                        {
                            gradPos += weightPos * violaPosPenaD * outerNormal;
                            pena += weightPos * violaPosPena;
                        }
                    }

                    if (smoothedL1(violaVel, smoothFactor, violaVelPena, violaVelPenaD))
                    {
                        gradVel += weightVel * violaVelPenaD * 2.0 * vel;
                        pena += weightVel * violaVelPena;
                    }

                    if (smoothedL1(violaOmg, smoothFactor, violaOmgPena, violaOmgPenaD))
                    {
                        gradOmg += weightOmg * violaOmgPenaD * 2.0 * omg;
                        pena += weightOmg * violaOmgPena;
                    }

                    if (smoothedL1(violaTheta, smoothFactor, violaThetaPena, violaThetaPenaD))
                    {
                        gradQuat += weightTheta * violaThetaPenaD /
                                    sqrt(1.0 - cos_theta * cos_theta) * 4.0 *
                                    Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
                        pena += weightTheta * violaThetaPena;
                    }

                    if (smoothedL1(violaThrust, smoothFactor, violaThrustPena, violaThrustPenaD))
                    {
                        gradThr += weightThrust * violaThrustPenaD * 2.0 * (thr - thrustMean);
                        pena += weightThrust * violaThrustPena;
                    }

                    flatMap.backward(gradPos, gradVel, gradThr, gradQuat, gradOmg,
                                     totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                                     totalGradPsi, totalGradPsiD);

                    node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
                    alpha = j * integralFrac;
                    gradC.block<6, 3>(i * 6, 0) += (beta0 * totalGradPos.transpose() +
                                                    beta1 * totalGradVel.transpose() +
                                                    beta2 * totalGradAcc.transpose() +
                                                    beta3 * totalGradJer.transpose()) *
                                                   node * step;
                    gradT(i) += (totalGradPos.dot(vel) +
                                 totalGradVel.dot(acc) +
                                 totalGradAcc.dot(jer) +
                                 totalGradJer.dot(sna)) *
                                    alpha * node * step +
                                node * integralFrac * pena;
                    cost += node * step * pena;
                }
            }

            return;
        }

        static inline double costFunctional(void *ptr,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &g)
        {
            GCOPTER_PolytopeSFC &obj = *(GCOPTER_PolytopeSFC *)ptr;
            const int dimTau = obj.temporalDim;
            const int dimXi = obj.spatialDim;
            const double weightT = obj.rho;
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimXi);
            Eigen::Map<Eigen::VectorXd> gradTau(g.data(), dimTau);
            Eigen::Map<Eigen::VectorXd> gradXi(g.data() + dimTau, dimXi);

            forwardT(tau, obj.times);
            forwardP(xi, obj.vPolyIdx, obj.vPolytopes, obj.points);

            double cost;
            obj.minco.setParameters(obj.points, obj.times);
            obj.minco.getEnergy(cost);
            obj.minco.getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs);
            obj.minco.getEnergyPartialGradByTimes(obj.partialGradByTimes);

            attachPenaltyFunctional(obj.times, obj.minco.getCoeffs(),
                                    obj.hPolyIdx, obj.hPolytopes,
                                    obj.smoothEps, obj.integralRes,
                                    obj.magnitudeBd, obj.penaltyWt, obj.flatmap,
                                    cost, obj.partialGradByTimes, obj.partialGradByCoeffs);

            // Velocity reference soft cost (knot velocity)
            if (obj.vel_ref_enable_)
            {
                Eigen::Vector3d v_i;
                obj.last_vref_valid_ = attachVelRefFunctional(obj.times, obj.minco.getCoeffs(),
                                                             obj.vel_ref_knot_, obj.vel_ref_, obj.vel_ref_weight_,
                                                             cost, obj.partialGradByTimes, obj.partialGradByCoeffs,
                                                             &v_i, nullptr);
                if (obj.last_vref_valid_)
                {
                    const Eigen::Vector3d e = v_i - obj.vel_ref_;
                    obj.last_vref_v_ = v_i;
                    obj.last_vref_err_ = e.norm();
                    obj.last_vref_cost_ = 0.5 * obj.vel_ref_weight_ * e.squaredNorm();
                }
                else
                {
                    obj.last_vref_v_.setZero();
                    obj.last_vref_err_ = 0.0;
                    obj.last_vref_cost_ = 0.0;
                }
            }
            else
            {
                obj.last_vref_valid_ = false;
                obj.last_vref_v_.setZero();
                obj.last_vref_err_ = 0.0;
                obj.last_vref_cost_ = 0.0;
            }

            // Position reference soft cost (knot position)
            if (obj.pos_ref_enable_)
            {
                Eigen::Vector3d p_i;
                obj.last_pref_valid_ = attachPosRefFunctional(obj.times, obj.minco.getCoeffs(),
                                                             obj.pos_ref_knot_, obj.pos_ref_, obj.pos_ref_weight_,
                                                             cost, obj.partialGradByTimes, obj.partialGradByCoeffs,
                                                             &p_i, nullptr);
                if (obj.last_pref_valid_)
                {
                    const Eigen::Vector3d e = p_i - obj.pos_ref_;
                    obj.last_pref_p_ = p_i;
                    obj.last_pref_err_ = e.norm();
                    obj.last_pref_cost_ = 0.5 * obj.pos_ref_weight_ * e.squaredNorm();
                }
                else
                {
                    obj.last_pref_p_.setZero();
                    obj.last_pref_err_ = 0.0;
                    obj.last_pref_cost_ = 0.0;
                }
            }
            else
            {
                obj.last_pref_valid_ = false;
                obj.last_pref_p_.setZero();
                obj.last_pref_err_ = 0.0;
                obj.last_pref_cost_ = 0.0;
            }

            obj.minco.propogateGrad(obj.partialGradByCoeffs, obj.partialGradByTimes,
                                    obj.gradByPoints, obj.gradByTimes);

            cost += weightT * obj.times.sum();
            obj.gradByTimes.array() += weightT;

            backwardGradT(tau, obj.gradByTimes, gradTau);
            backwardGradP(xi, obj.vPolyIdx, obj.vPolytopes, obj.gradByPoints, gradXi);
            normRetrictionLayer(xi, obj.vPolyIdx, obj.vPolytopes, cost, gradXi);

            // Optional one-shot gradient check for vref term only
            if (obj.vel_ref_grad_check_)
            {
                obj.checkVelRefGradOnce(x);
            }

            // Optional one-shot gradient check for pref term only
            if (obj.pos_ref_grad_check_)
            {
                obj.checkPosRefGradOnce(x);
            }

            // Optional one-shot gradient check for full objective
            if (obj.full_grad_check_enable_ && !obj.full_grad_check_running_)
            {
                obj.checkFullGradOnce(x, gradTau, gradXi);
            }

            return cost;
        }

        inline void checkVelRefGradOnce(const Eigen::VectorXd &x)
        {
            if (!vel_ref_grad_check_ || vel_ref_grad_check_done_)
                return;
            if (!vel_ref_enable_ || vel_ref_weight_ <= 0.0)
                return;
            if (vel_ref_knot_ <= 0 || vel_ref_knot_ >= temporalDim)
                return;

            vel_ref_grad_check_done_ = true;

            const int dimTau = temporalDim;
            const int dimXi = spatialDim;
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimXi);

            // Helper: vref-only cost
            auto costOnly = [&](const Eigen::VectorXd &x_in) -> double
            {
                Eigen::Map<const Eigen::VectorXd> tau_in(x_in.data(), dimTau);
                Eigen::Map<const Eigen::VectorXd> xi_in(x_in.data() + dimTau, dimXi);
                forwardT(tau_in, times);
                forwardP(xi_in, vPolyIdx, vPolytopes, points);
                minco.setParameters(points, times);
                double cost_local = 0.0;
                partialGradByCoeffs.setZero();
                partialGradByTimes.setZero();
                attachVelRefFunctional(times, minco.getCoeffs(), vel_ref_knot_, vel_ref_, vel_ref_weight_,
                                       cost_local, partialGradByTimes, partialGradByCoeffs,
                                       nullptr, nullptr);
                return cost_local;
            };

            // Analytic gradient for vref-only term
            forwardT(tau, times);
            forwardP(xi, vPolyIdx, vPolytopes, points);
            minco.setParameters(points, times);
            partialGradByCoeffs.setZero();
            partialGradByTimes.setZero();
            double cost_vref = 0.0;
            attachVelRefFunctional(times, minco.getCoeffs(), vel_ref_knot_, vel_ref_, vel_ref_weight_,
                                   cost_vref, partialGradByTimes, partialGradByCoeffs,
                                   nullptr, nullptr);
            minco.propogateGrad(partialGradByCoeffs, partialGradByTimes,
                                gradByPoints, gradByTimes);

            Eigen::VectorXd gradTau(dimTau);
            Eigen::VectorXd gradXi(dimXi);
            backwardGradT(tau, gradByTimes, gradTau);
            backwardGradP(xi, vPolyIdx, vPolytopes, gradByPoints, gradXi);

            Eigen::VectorXd gradAnalytic(x.size());
            gradAnalytic.setZero();
            gradAnalytic.head(dimTau) = gradTau;
            gradAnalytic.tail(dimXi) = gradXi;

            const double eps = 1e-6;
            std::cout << "[GCOPTER][vref-grad] knot=" << vel_ref_knot_ << "\n";

            // Check one tau dimension (segment tied to knot)
            const int s_check = std::clamp(vel_ref_knot_ - 1, 0, dimTau - 1);
            {
                Eigen::VectorXd xp = x, xm = x;
                xp[s_check] += eps;
                xm[s_check] -= eps;
                const double fd = (costOnly(xp) - costOnly(xm)) / (2.0 * eps);
                const double ad = gradAnalytic[s_check];
                const double denom = std::max(1e-12, std::abs(fd) + std::abs(ad));
                const double rel = std::abs(fd - ad) / denom;
                std::cout << "  tau[" << s_check << "] fd=" << fd << " ad=" << ad << " rel=" << rel << "\n";
            }

            // Check one xi dimension with largest magnitude (if any)
            if (dimXi > 0)
            {
                int idx_xi = 0;
                double max_abs = 0.0;
                for (int i = 0; i < dimXi; ++i)
                {
                    const double a = std::abs(gradXi[i]);
                    if (a > max_abs)
                    {
                        max_abs = a;
                        idx_xi = i;
                    }
                }
                const int idx = dimTau + idx_xi;
                Eigen::VectorXd xp = x, xm = x;
                xp[idx] += eps;
                xm[idx] -= eps;
                const double fd = (costOnly(xp) - costOnly(xm)) / (2.0 * eps);
                const double ad = gradAnalytic[idx];
                const double denom = std::max(1e-12, std::abs(fd) + std::abs(ad));
                const double rel = std::abs(fd - ad) / denom;
                std::cout << "  xi[" << idx_xi << "] fd=" << fd << " ad=" << ad << " rel=" << rel << "\n";
            }
        }

        inline void checkPosRefGradOnce(const Eigen::VectorXd &x)
        {
            if (!pos_ref_grad_check_ || pos_ref_grad_check_done_)
                return;
            if (!pos_ref_enable_ || pos_ref_weight_ <= 0.0)
                return;
            if (pos_ref_knot_ <= 0 || pos_ref_knot_ >= temporalDim)
                return;

            pos_ref_grad_check_done_ = true;

            const int dimTau = temporalDim;
            const int dimXi = spatialDim;
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimXi);

            // Helper: pref-only cost
            auto costOnly = [&](const Eigen::VectorXd &x_in) -> double
            {
                Eigen::Map<const Eigen::VectorXd> tau_in(x_in.data(), dimTau);
                Eigen::Map<const Eigen::VectorXd> xi_in(x_in.data() + dimTau, dimXi);
                forwardT(tau_in, times);
                forwardP(xi_in, vPolyIdx, vPolytopes, points);
                minco.setParameters(points, times);
                double cost_local = 0.0;
                partialGradByCoeffs.setZero();
                partialGradByTimes.setZero();
                attachPosRefFunctional(times, minco.getCoeffs(), pos_ref_knot_, pos_ref_, pos_ref_weight_,
                                       cost_local, partialGradByTimes, partialGradByCoeffs,
                                       nullptr, nullptr);
                return cost_local;
            };

            // Analytic gradient for pref-only term
            forwardT(tau, times);
            forwardP(xi, vPolyIdx, vPolytopes, points);
            minco.setParameters(points, times);
            partialGradByCoeffs.setZero();
            partialGradByTimes.setZero();
            double cost_pref = 0.0;
            attachPosRefFunctional(times, minco.getCoeffs(), pos_ref_knot_, pos_ref_, pos_ref_weight_,
                                   cost_pref, partialGradByTimes, partialGradByCoeffs,
                                   nullptr, nullptr);
            minco.propogateGrad(partialGradByCoeffs, partialGradByTimes,
                                gradByPoints, gradByTimes);

            Eigen::VectorXd gradTau(dimTau);
            Eigen::VectorXd gradXi(dimXi);
            backwardGradT(tau, gradByTimes, gradTau);
            backwardGradP(xi, vPolyIdx, vPolytopes, gradByPoints, gradXi);

            Eigen::VectorXd gradAnalytic(x.size());
            gradAnalytic.setZero();
            gradAnalytic.head(dimTau) = gradTau;
            gradAnalytic.tail(dimXi) = gradXi;

            const double eps = 1e-6;
            std::cout << "[GCOPTER][pref-grad] knot=" << pos_ref_knot_ << "\n";

            // Check one tau dimension (segment tied to knot)
            const int s_check = std::clamp(pos_ref_knot_ - 1, 0, dimTau - 1);
            {
                Eigen::VectorXd xp = x, xm = x;
                xp[s_check] += eps;
                xm[s_check] -= eps;
                const double fd = (costOnly(xp) - costOnly(xm)) / (2.0 * eps);
                const double ad = gradAnalytic[s_check];
                const double denom = std::max(1e-12, std::abs(fd) + std::abs(ad));
                const double rel = std::abs(fd - ad) / denom;
                std::cout << "  tau[" << s_check << "] fd=" << fd << " ad=" << ad << " rel=" << rel << "\n";
            }

            // Check one xi dimension with largest magnitude (if any)
            if (dimXi > 0)
            {
                int idx_xi = 0;
                double max_abs = 0.0;
                for (int i = 0; i < dimXi; ++i)
                {
                    const double a = std::abs(gradXi[i]);
                    if (a > max_abs)
                    {
                        max_abs = a;
                        idx_xi = i;
                    }
                }
                const int idx = dimTau + idx_xi;
                Eigen::VectorXd xp = x, xm = x;
                xp[idx] += eps;
                xm[idx] -= eps;
                const double fd = (costOnly(xp) - costOnly(xm)) / (2.0 * eps);
                const double ad = gradAnalytic[idx];
                const double denom = std::max(1e-12, std::abs(fd) + std::abs(ad));
                const double rel = std::abs(fd - ad) / denom;
                std::cout << "  xi[" << idx_xi << "] fd=" << fd << " ad=" << ad << " rel=" << rel << "\n";
            }
        }

        inline void checkFullGradOnce(const Eigen::VectorXd &x,
                                      const Eigen::VectorXd &gradTau,
                                      const Eigen::VectorXd &gradXi)
        {
            if (!full_grad_check_enable_ || full_grad_check_done_ || full_grad_check_running_)
                return;
            full_grad_check_done_ = true;

            const int dimTau = temporalDim;
            const int dimXi = spatialDim;
            Eigen::VectorXd gradAnalytic(x.size());
            gradAnalytic.setZero();
            gradAnalytic.head(dimTau) = gradTau;
            gradAnalytic.tail(dimXi) = gradXi;

            auto costOnly = [&](const Eigen::VectorXd &xin) -> double
            {
                full_grad_check_running_ = true;
                Eigen::VectorXd gtmp = Eigen::VectorXd::Zero(xin.size());
                double c = GCOPTER_PolytopeSFC::costFunctional(this, xin, gtmp);
                full_grad_check_running_ = false;
                return c;
            };

            std::mt19937 rng(42);
            std::normal_distribution<double> N(0.0, 1.0);

            double max_rel_dir = 0.0;
            for (int k = 0; k < full_grad_check_dirs_; ++k)
            {
                Eigen::VectorXd d(x.size());
                for (int i = 0; i < d.size(); ++i)
                    d[i] = N(rng);
                const double n = d.norm();
                if (n < 1e-12)
                {
                    --k;
                    continue;
                }
                d /= n;
                const double eps = full_grad_check_eps_;
                const double fd = (costOnly(x + eps * d) - costOnly(x - eps * d)) / (2.0 * eps);
                const double ad = gradAnalytic.dot(d);
                const double denom = std::max(1e-12, std::abs(fd) + std::abs(ad));
                const double rel = std::abs(fd - ad) / denom;
                max_rel_dir = std::max(max_rel_dir, rel);
                if (rel > 1e-3)
                    std::cout << "[GCOPTER][full-grad][dir " << k << "] fd=" << fd << " ad=" << ad
                              << " rel_err=" << rel << "\n";
            }

            std::vector<int> idx(x.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            if ((int)idx.size() > full_grad_check_max_coords_)
                idx.resize(full_grad_check_max_coords_);

            double max_rel_coord = 0.0;
            for (int i : idx)
            {
                Eigen::VectorXd ei = Eigen::VectorXd::Zero(x.size());
                ei[i] = 1.0;
                const double eps = full_grad_check_eps_;
                const double fd = (costOnly(x + eps * ei) - costOnly(x - eps * ei)) / (2.0 * eps);
                const double ad = gradAnalytic[i];
                const double denom = std::max(1e-12, std::abs(fd) + std::abs(ad));
                const double rel = std::abs(fd - ad) / denom;
                max_rel_coord = std::max(max_rel_coord, rel);
                if (rel > 1e-3)
                    std::cout << "[GCOPTER][full-grad][idx " << i << "] g_fd=" << fd << " g_ad=" << ad
                              << " rel_err=" << rel << "\n";
            }

            std::cout << "[GCOPTER][full-grad] max_rel_dir=" << max_rel_dir
                      << " max_rel_coord=" << max_rel_coord << "\n";
        }

        static inline double costDistance(void *ptr,
                                          const Eigen::VectorXd &xi,
                                          Eigen::VectorXd &gradXi)
        {
            void **dataPtrs = (void **)ptr;
            const double &dEps = *((const double *)(dataPtrs[0]));
            const Eigen::Vector3d &ini = *((const Eigen::Vector3d *)(dataPtrs[1]));
            const Eigen::Vector3d &fin = *((const Eigen::Vector3d *)(dataPtrs[2]));
            const PolyhedraV &vPolys = *((PolyhedraV *)(dataPtrs[3]));

            double cost = 0.0;
            const int overlaps = vPolys.size() / 2;

            Eigen::Matrix3Xd gradP = Eigen::Matrix3Xd::Zero(3, overlaps);
            Eigen::Vector3d a, b, d;
            Eigen::VectorXd r;
            double smoothedDistance;
            for (int i = 0, j = 0, k = 0; i <= overlaps; i++, j += k)
            {
                a = i == 0 ? ini : b;
                if (i < overlaps)
                {
                    k = vPolys[2 * i + 1].cols();
                    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                    r = q.normalized().head(k - 1);
                    b = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                        vPolys[2 * i + 1].col(0);
                }
                else
                {
                    b = fin;
                }

                d = b - a;
                smoothedDistance = sqrt(d.squaredNorm() + dEps);
                cost += smoothedDistance;

                if (i < overlaps)
                {
                    gradP.col(i) += d / smoothedDistance;
                }
                if (i > 0)
                {
                    gradP.col(i - 1) -= d / smoothedDistance;
                }
            }

            Eigen::VectorXd unitQ;
            double sqrNormQ, invNormQ, sqrNormViolation, c, dc;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                Eigen::Map<Eigen::VectorXd> gradQ(gradXi.data() + j, k);
                sqrNormQ = q.squaredNorm();
                invNormQ = 1.0 / sqrt(sqrNormQ);
                unitQ = q * invNormQ;
                gradQ.head(k - 1) = (vPolys[2 * i + 1].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradQ = (gradQ - unitQ * unitQ.dot(gradQ)).eval() * invNormQ;

                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradQ += dc * 2.0 * q;
                }
            }

            return cost;
        }

        inline void getShortestPath(const Eigen::Vector3d &ini,
                                    const Eigen::Vector3d &fin,
                                    const PolyhedraV &vPolys,
                                    const double &smoothD,
                                    Eigen::Matrix3Xd &path)
        {
            const int overlaps = vPolys.size() / 2;
            Eigen::VectorXi vSizes(overlaps);
            for (int i = 0; i < overlaps; i++)
            {
                vSizes(i) = vPolys[2 * i + 1].cols();
            }
            Eigen::VectorXd xi(vSizes.sum());
            for (int i = 0, j = 0; i < overlaps; i++)
            {
                xi.segment(j, vSizes(i)).setConstant(sqrt(1.0 / vSizes(i)));
                j += vSizes(i);
            }

            double minDistance;
            void *dataPtrs[4];
            dataPtrs[0] = (void *)(&smoothD);
            dataPtrs[1] = (void *)(&ini);
            dataPtrs[2] = (void *)(&fin);
            dataPtrs[3] = (void *)(&vPolys);
            lbfgs::lbfgs_parameter_t shortest_path_params;
            shortest_path_params.past = 3;
            shortest_path_params.delta = 1.0e-3;
            shortest_path_params.g_epsilon = 1.0e-5;

            lbfgs::lbfgs_optimize(xi,
                                  minDistance,
                                  &GCOPTER_PolytopeSFC::costDistance,
                                  nullptr,
                                  nullptr,
                                  dataPtrs,
                                  shortest_path_params);

            // initialize initial_xi_
            initial_xi_ = xi;

            path.resize(3, overlaps + 2);
            path.leftCols<1>() = ini;
            path.rightCols<1>() = fin;
            Eigen::VectorXd r;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                r = q.normalized().head(k - 1);
                path.col(i + 1) = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                                  vPolys[2 * i + 1].col(0);
            }

            return;
        }

        static inline bool processCorridor(const PolyhedraH &hPs,
                                           PolyhedraV &vPs)
        {
            const int sizeCorridor = hPs.size() - 1;

            vPs.clear();
            vPs.reserve(2 * sizeCorridor + 1);

            int nv;
            PolyhedronH curIH;
            PolyhedronV curIV, curIOB;
            for (int i = 0; i < sizeCorridor; i++)
            {
                if (!geo_utils::enumerateVs(hPs[i], curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);

                curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);
                curIH.topRows(hPs[i].rows()) = hPs[i];
                curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];
                if (!geo_utils::enumerateVs(curIH, curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);
            }

            if (!geo_utils::enumerateVs(hPs.back(), curIV))
            {
                return false;
            }
            nv = curIV.cols();
            curIOB.resize(3, nv);
            curIOB.col(0) = curIV.col(0);
            curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
            vPs.push_back(curIOB);

            return true;
        }

        static inline void setInitial(const Eigen::Matrix3Xd &path,
                                      const double &speed,
                                      const Eigen::VectorXi &intervalNs,
                                      Eigen::Matrix3Xd &innerPoints,
                                      Eigen::VectorXd &timeAlloc)
        {
            const int sizeM = intervalNs.size();
            const int sizeN = intervalNs.sum();
            innerPoints.resize(3, sizeN - 1);
            timeAlloc.resize(sizeN);

            Eigen::Vector3d a, b, c;
            for (int i = 0, j = 0, k = 0, l; i < sizeM; i++)
            {
                l = intervalNs(i);
                a = path.col(i);
                b = path.col(i + 1);
                c = (b - a) / l;
                timeAlloc.segment(j, l).setConstant(c.norm() / speed);
                j += l;
                for (int m = 0; m < l; m++)
                {
                    if (i > 0 || m > 0)
                    {
                        innerPoints.col(k++) = a + c * m;
                    }
                }
            }
        }

    public:
        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        inline bool setup(const double &timeWeight,
                          const Eigen::Matrix3d &initialPVA,
                          const Eigen::Matrix3d &terminalPVA,
                          const PolyhedraH &safeCorridor,
                          const double &lengthPerPiece,
                          const double &smoothingFactor,
                          const int &integralResolution,
                          const Eigen::VectorXd &magnitudeBounds,
                          const Eigen::VectorXd &penaltyWeights,
                          const Eigen::VectorXd &physicalParams)
        {
            rho = timeWeight;
            headPVA = initialPVA;
            tailPVA = terminalPVA;

            hPolytopes = safeCorridor;
            for (size_t i = 0; i < hPolytopes.size(); i++)
            {
                const Eigen::ArrayXd norms =
                    hPolytopes[i].leftCols<3>().rowwise().norm();
                hPolytopes[i].array().colwise() /= norms;
            }
            if (!processCorridor(hPolytopes, vPolytopes))
            {
                return false;
            }

            polyN = hPolytopes.size();
            smoothEps = smoothingFactor;
            integralRes = integralResolution;
            magnitudeBd = magnitudeBounds;
            penaltyWt = penaltyWeights;
            physicalPm = physicalParams;
            allocSpeed = magnitudeBd(0) * 3.0;

            getShortestPath(headPVA.col(0), tailPVA.col(0),
                            vPolytopes, smoothEps, shortPath);
            const Eigen::Matrix3Xd deltas = shortPath.rightCols(polyN) - shortPath.leftCols(polyN);
            pieceIdx = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            pieceIdx.array() += 1;
            pieceN = pieceIdx.sum();

            temporalDim = pieceN;
            spatialDim = 0;
            vPolyIdx.resize(pieceN - 1);
            hPolyIdx.resize(pieceN);
            for (int i = 0, j = 0, k; i < polyN; i++)
            {
                k = pieceIdx(i);
                for (int l = 0; l < k; l++, j++)
                {
                    if (l < k - 1)
                    {
                        vPolyIdx(j) = 2 * i;
                        spatialDim += vPolytopes[2 * i].cols();
                    }
                    else if (i < polyN - 1)
                    {
                        vPolyIdx(j) = 2 * i + 1;
                        spatialDim += vPolytopes[2 * i + 1].cols();
                    }
                    hPolyIdx(j) = i;
                }
            }

            // Setup for MINCO_S3NU, FlatnessMap, and L-BFGS solver
            minco.setConditions(headPVA, tailPVA, pieceN);
            flatmap.reset(physicalPm(0), physicalPm(1), physicalPm(2),
                          physicalPm(3), physicalPm(4), physicalPm(5));

            // Allocate temp variables
            points.resize(3, pieceN - 1);
            times.resize(pieceN);
            gradByPoints.resize(3, pieceN - 1);
            gradByTimes.resize(pieceN);
            partialGradByCoeffs.resize(6 * pieceN, 3);
            partialGradByTimes.resize(pieceN);

            return true;
        }

        inline void setVelRef(const bool enable,
                              const int knot,
                              const Eigen::Vector3d &vref,
                              const double weight,
                              const bool grad_check = false,
                              const int log_every = 0)
        {
            vel_ref_enable_ = enable;
            vel_ref_knot_ = knot;
            vel_ref_ = vref;
            vel_ref_weight_ = weight;
            vel_ref_grad_check_ = grad_check;
            vel_ref_grad_check_done_ = false;
            vel_ref_log_every_ = log_every;
            if (vel_ref_enable_)
            {
                if (vel_ref_knot_ <= 0 || vel_ref_knot_ >= pieceN)
                {
                    std::cout << "[GCOPTER] warning: VelRefKnot=" << vel_ref_knot_
                              << " out of range (valid: 1.." << (pieceN - 1) << ")\n";
                }
                std::cout << "[GCOPTER] vref enabled: gradients injected in coeff/time space, "
                          << "then propagated by MINCO.\n";
            }
        }

        inline void setPosRef(const bool enable,
                              const int knot,
                              const Eigen::Vector3d &pref,
                              const double weight,
                              const bool grad_check = false,
                              const int log_every = 0)
        {
            pos_ref_enable_ = enable;
            pos_ref_knot_ = knot;
            pos_ref_ = pref;
            pos_ref_weight_ = weight;
            pos_ref_grad_check_ = grad_check;
            pos_ref_grad_check_done_ = false;
            pos_ref_log_every_ = log_every;
            if (pos_ref_enable_)
            {
                if (pos_ref_knot_ <= 0 || pos_ref_knot_ >= pieceN)
                {
                    std::cout << "[GCOPTER] warning: PosRefKnot=" << pos_ref_knot_
                              << " out of range (valid: 1.." << (pieceN - 1) << ")\n";
                }
                std::cout << "[GCOPTER] pref enabled: gradients injected in coeff/time space, "
                          << "then propagated by MINCO.\n";
            }
        }

        inline void setFullGradCheck(const bool enable,
                                     const int dirs,
                                     const int max_coords,
                                     const double eps)
        {
            full_grad_check_enable_ = enable;
            full_grad_check_dirs_ = std::max(1, dirs);
            full_grad_check_max_coords_ = std::max(1, max_coords);
            full_grad_check_eps_ = (eps > 0.0 ? eps : 1e-5);
            full_grad_check_done_ = false;
        }

        inline bool getVelRefInfo(Eigen::Vector3d &v_out,
                                  double &err_out,
                                  double &J_out) const
        {
            if (!vel_ref_enable_ || vel_ref_weight_ <= 0.0)
                return false;
            if (vel_ref_knot_ <= 0 || vel_ref_knot_ >= times.size())
                return false;
            const int s = vel_ref_knot_ - 1;
            const double t = times(s);
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            Eigen::Matrix<double, 6, 1> beta1;
            beta1 << 0.0, 1.0, 2.0 * t, 3.0 * t2, 4.0 * t3, 5.0 * t4;
            const Eigen::Matrix<double, 6, 3> c = minco.getCoeffs().block<6, 3>(s * 6, 0);
            v_out = c.transpose() * beta1;
            const Eigen::Vector3d e = v_out - vel_ref_;
            err_out = e.norm();
            J_out = 0.5 * vel_ref_weight_ * e.squaredNorm();
            return true;
        }

        inline bool getPosRefInfo(Eigen::Vector3d &p_out,
                                  double &err_out,
                                  double &J_out) const
        {
            if (!pos_ref_enable_ || pos_ref_weight_ <= 0.0)
                return false;
            if (pos_ref_knot_ <= 0 || pos_ref_knot_ >= times.size())
                return false;
            const int s = pos_ref_knot_ - 1;
            const double t = times(s);
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            const double t5 = t4 * t;
            Eigen::Matrix<double, 6, 1> beta0;
            beta0 << 1.0, t, t2, t3, t4, t5;
            const Eigen::Matrix<double, 6, 3> c = minco.getCoeffs().block<6, 3>(s * 6, 0);
            p_out = c.transpose() * beta0;
            const Eigen::Vector3d e = p_out - pos_ref_;
            err_out = e.norm();
            J_out = 0.5 * pos_ref_weight_ * e.squaredNorm();
            return true;
        }

        inline void getVPolytopes(PolyhedraV &vPs) const
        {
            vPs = vPolytopes;
        }

        // Get shortest path computed during setup()
        // shortPath includes start and goal: [start, inner1, inner2, ..., goal]
        inline void getShortPath(Eigen::Matrix3Xd &path) const
        {
            path = shortPath;
        }

        inline double optimize_with_timeout(Trajectory<5> &traj,
                               const double relCostTol,
                               const double time_budget_ms,
                               const int mem_size = 256,
                               const int max_linesearch = 20,
                               const int past = 3,
                               const double min_step = 1e-32,
                               const int max_iterations = 0,
                               const double g_epsilon = 1e-5)
        {
            // Tell the solver whether we have a budget
            setTimeBudgetMs(time_budget_ms);

            Eigen::VectorXd x(temporalDim + spatialDim);
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

            setInitial(shortPath, allocSpeed, pieceIdx, points, times);
            backwardT(times, tau);
            backwardP(points, vPolyIdx, vPolytopes, xi);

            // Save initial guess for later inspection (you already do this)
            initial_guess_points_ = points;
            initial_guess_times_ = times;
            initial_xi_.resize(xi.size());
            initial_xi_ = xi;

            // Initialize best-so-far
            best_x_ = x;
            best_f_ = std::numeric_limits<double>::infinity();

            double minCostFunctional;
            lbfgs_params.mem_size = mem_size;
            lbfgs_params.max_linesearch = max_linesearch;
            lbfgs_params.past = past;
            lbfgs_params.min_step = min_step;
            lbfgs_params.max_iterations = max_iterations;
            lbfgs_params.g_epsilon = g_epsilon;
            lbfgs_params.delta = relCostTol;

            // save initial guess
            using Clock = std::chrono::steady_clock;
            auto start = Clock::now();

            int ret = lbfgs::lbfgs_optimize(
                x,
                minCostFunctional,
                &GCOPTER_PolytopeSFC::costFunctional,     // fused f+g (already in GCOPTER)
                nullptr, // new (can be nullptr if you prefer)
                &GCOPTER_PolytopeSFC::progressTimeGuard,  // new
                this,
                lbfgs_params);

            auto end = Clock::now();
            computation_time_ms_ =
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3;

            // If we canceled due to time, restore best-so-far iterate.
            if (ret == lbfgs::LBFGS_CANCELED && time_budget_enabled_ && best_x_.size() == x.size())
            {
                x = best_x_;
                minCostFunctional = best_f_;
            }

            // Build trajectory from the (possibly best-so-far) solution
            forwardT(tau, times);
            forwardP(xi, vPolyIdx, vPolytopes, points);
            minco.setParameters(points, times);
            minco.getTrajectory(traj);

            return minCostFunctional;
        }

        // Keep old API as a wrapper with “no time limit”:
        inline double optimize(Trajectory<5> &traj, const double &relCostTol)
        {
            return optimize_with_timeout(traj, relCostTol, /*time_budget_ms=*/-1.0);
        }

        double getComputationTime() const
        {
            return computation_time_ms_;
        }

        void getInitialGuess(Eigen::Matrix3Xd &points,
                             Eigen::VectorXd &times,
                             Eigen::VectorXd &xi) const
        {
            points = initial_guess_points_;
            times = initial_guess_times_;
            xi = initial_xi_;
        }
    };

}

#endif
