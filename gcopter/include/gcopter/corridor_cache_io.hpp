#pragma once
/*
 * Corridor cache IO utilities.
 *
 * Binary format is intentionally identical to the one used in DYNUS's
 * corridor_generator_node.cpp (magic "MYSCO2\0\0", version 1) so you can
 * reuse tools across planners.
 *
 * Constraints are stored as A x <= b for each segment (polytope).
 * In GCOPTER/minco_bench_viz, corridor polytopes are represented as H with rows
 * [n_x n_y n_z d] meaning n^T x + d <= 0. We convert via:
 *   A = n
 *   b = -d
 */

#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

namespace corridor_cache_io
{
    namespace fs = std::filesystem;

    // Save route + corridor H-polys to DYNUS-compatible binary file.
    static inline void saveCorridorBinary(
        const fs::path &filepath,
        const Eigen::Vector3d &start,
        const Eigen::Vector3d &goal,
        const std::vector<Eigen::Vector3d> &route,
        const std::vector<Eigen::MatrixX4d> &hPolys)
    {
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs)
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());

        const char magic[8] = {'M', 'Y', 'S', 'C', 'O', '2', '\0', '\0'};
        const uint32_t version = 1;

        auto writeU32 = [&](uint32_t v)
        { ofs.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
        auto writeD = [&](double v)
        { ofs.write(reinterpret_cast<const char *>(&v), sizeof(v)); };

        ofs.write(magic, 8);
        writeU32(version);

        // start / goal
        for (int k = 0; k < 3; ++k)
            writeD(start[k]);
        for (int k = 0; k < 3; ++k)
            writeD(goal[k]);

        // route points
        writeU32(static_cast<uint32_t>(route.size()));
        for (const auto &p : route)
        {
            writeD(p.x());
            writeD(p.y());
            writeD(p.z());
        }

        const uint32_t n_seg = static_cast<uint32_t>(hPolys.size());
        writeU32(n_seg);

        for (size_t i = 0; i < hPolys.size(); ++i)
        {
            const auto &H = hPolys[i];
            if (H.cols() != 4)
                throw std::runtime_error("H-poly must have 4 columns [n_x n_y n_z d].");

            const uint32_t num_planes = static_cast<uint32_t>(H.rows());
            writeU32(num_planes);

            // A: plane normals
            for (uint32_t r = 0; r < num_planes; ++r)
            {
                writeD(H(r, 0));
                writeD(H(r, 1));
                writeD(H(r, 2));
            }
            // b: convert d -> b = -d
            for (uint32_t r = 0; r < num_planes; ++r)
                writeD(-H(r, 3));
        }

        if (!ofs.good())
            throw std::runtime_error("Failed while writing corridor file: " + filepath.string());
    }

    struct CorridorCache
    {
        Eigen::Vector3d start{0, 0, 0};
        Eigen::Vector3d goal{0, 0, 0};
        std::vector<Eigen::Vector3d> route;
        std::vector<Eigen::MatrixX4d> hPolys; // one per segment
    };

    // Load DYNUS-compatible corridor binary file into route + H-polys.
    static inline CorridorCache loadCorridorBinarySeeded(const fs::path &filepath,
                                                         double poly_seed_eps = 1e-6)
    {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs)
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());

        auto readU32 = [&]() -> uint32_t
        {
            uint32_t v{};
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (!ifs)
                throw std::runtime_error("EOF while reading u32");
            return v;
        };
        auto readD = [&]() -> double
        {
            double v{};
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (!ifs)
                throw std::runtime_error("EOF while reading double");
            return v;
        };

        char magic[8];
        ifs.read(magic, 8);
        if (!ifs)
            throw std::runtime_error("EOF while reading magic");
        const std::string magic_s(magic, magic + 8);
        if (magic_s.rfind("MYSCO2", 0) != 0)
            throw std::runtime_error("Bad magic; expected MYSCO2, got: " + magic_s);

        const uint32_t version = readU32();
        if (version != 1)
            throw std::runtime_error("Unsupported corridor cache version: " + std::to_string(version));

        CorridorCache out;
        auto x = readD(); // unused
        auto y = readD();
        auto z = readD();
        out.start = Eigen::Vector3d(x, y, z);
        x = readD(); // unused
        y = readD();
        z = readD();
        out.goal = Eigen::Vector3d(x, y, z);

        const uint32_t n_pts = readU32();
        out.route.resize(n_pts);
        for (uint32_t i = 0; i < n_pts; ++i)
        {
            x = readD();
            y = readD();
            z = readD();
            out.route[i] = Eigen::Vector3d(x, y, z);
        }

        const uint32_t n_seg = readU32();

        if (out.route.size() < 2)
            throw std::runtime_error("route too short");

        // Authoritative endpoints (prevents bogus start/goal fields)
        out.start = out.route.front();
        out.goal = out.route.back();

        out.hPolys.resize(n_seg);

        for (uint32_t si = 0; si < n_seg; ++si)
        {
            const uint32_t m = readU32(); // planes

            Eigen::MatrixXd A(m, 3);
            Eigen::VectorXd b(m);

            for (uint32_t r = 0; r < m; ++r)
            {
                A(r, 0) = readD();
                A(r, 1) = readD();
                A(r, 2) = readD();
            }
            for (uint32_t r = 0; r < m; ++r)
                b(r) = readD();

            // Convert A x <= b -> GCOPTER H: n·x + d <= 0 with d = -b
            Eigen::MatrixX4d H(m, 4);
            H.leftCols<3>() = A;
            H.col(3) = -b;
            out.hPolys[si] = H;
        }

        return out;
    }

} // namespace corridor_cache_io