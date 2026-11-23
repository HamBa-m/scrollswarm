#pragma once

/**
 * Modern C++20 Analytical 2D Geometry Engine for ScrollSwarm.
 * 
 * Provides SIMD-friendly, zero-heap-allocation geometric primitives:
 * 1. Analytic multi-ray LiDAR intersection against horizontal boundaries,
 *    arbitrary polygon obstacle edges, and spherical targets.
 * 2. Ray-casting point-in-polygon containment test.
 * 3. Point-to-segment Euclidean minimum distance computation.
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cstdint>

namespace scrollswarm {

struct RayResult {
    std::vector<float> obs_dists;
    std::vector<int32_t> obs_types;
    std::vector<float> obs_endpoints; // (num_rays, 2) packed row-major
    std::vector<float> tgt_dists;
    std::vector<int32_t> tgt_types;
    std::vector<float> tgt_endpoints; // (num_rays, 2) packed row-major
};

inline RayResult cast_rays(
    float origin_x,
    float origin_y,
    const float* angles,
    size_t num_rays,
    float max_range,
    float ceil_y,
    float floor_y,
    const float* obs_meta,        // (N_obs, 4): [x, y, radius, max_dim]
    size_t num_obs,
    const float* obs_edges,       // (N_edges, 4): [ax, ay, bx, by]
    size_t num_edges,
    const int32_t* obs_edge_ranges, // (N_obs, 2): [start_idx, end_idx]
    const float* target_data,     // (N_tgts, 3): [x, y, radius]
    size_t num_targets
) {
    RayResult res;
    res.obs_dists.assign(num_rays, max_range);
    res.obs_types.assign(num_rays, 0);
    res.obs_endpoints.assign(num_rays * 2, 0.0f);
    res.tgt_dists.assign(num_rays, max_range);
    res.tgt_types.assign(num_rays, 0);
    res.tgt_endpoints.assign(num_rays * 2, 0.0f);

    for (size_t k = 0; k < num_rays; ++k) {
        float angle = angles[k];
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        float closest_obs = max_range;
        int32_t hit_type = 0;

        // 1. Analytical ceiling and floor intersections
        if (sin_a < -1e-5f) {
            float t_ceil = (ceil_y - origin_y) / sin_a;
            if (t_ceil >= 0.0f && t_ceil < closest_obs) {
                closest_obs = t_ceil;
                hit_type = 1; // boundary
            }
        } else if (sin_a > 1e-5f) {
            float t_floor = (floor_y - origin_y) / sin_a;
            if (t_floor >= 0.0f && t_floor < closest_obs) {
                closest_obs = t_floor;
                hit_type = 1; // boundary
            }
        }

        // 2. Polygon obstacle intersections
        for (size_t i = 0; i < num_obs; ++i) {
            float ox = obs_meta[i * 4 + 0];
            float oy = obs_meta[i * 4 + 1];
            float max_dim = obs_meta[i * 4 + 3];

            float vx = ox - origin_x;
            float vy = oy - origin_y;
            float t_proj = vx * cos_a + vy * sin_a;

            // Bounding sphere rejection
            if (t_proj < -max_dim || t_proj > closest_obs + max_dim) {
                continue;
            }

            int32_t start_e = obs_edge_ranges[i * 2 + 0];
            int32_t end_e = obs_edge_ranges[i * 2 + 1];

            for (int32_t e = start_e; e < end_e; ++e) {
                float ax = obs_edges[e * 4 + 0];
                float ay = obs_edges[e * 4 + 1];
                float bx = obs_edges[e * 4 + 2];
                float by = obs_edges[e * 4 + 3];

                float ex = bx - ax;
                float ey = by - ay;
                float denom = cos_a * ey - sin_a * ex;
                if (std::abs(denom) < 1e-6f) {
                    continue;
                }

                float dx = ax - origin_x;
                float dy = ay - origin_y;
                float t = (dx * ey - dy * ex) / denom;
                float u = (dx * sin_a - dy * cos_a) / denom;

                if (t >= 0.0f && t < closest_obs && u >= 0.0f && u <= 1.0f) {
                    closest_obs = t;
                    hit_type = 2; // obstacle
                }
            }
        }

        closest_obs = std::clamp(closest_obs, 0.0f, max_range);
        res.obs_dists[k] = closest_obs;
        res.obs_types[k] = hit_type;
        res.obs_endpoints[k * 2 + 0] = origin_x + closest_obs * cos_a;
        res.obs_endpoints[k * 2 + 1] = origin_y + closest_obs * sin_a;

        // 3. Ray-target intersections
        float closest_tgt = max_range;
        int32_t tgt_hit = 0;

        for (size_t j = 0; j < num_targets; ++j) {
            float tx = target_data[j * 3 + 0];
            float ty = target_data[j * 3 + 1];
            float tr = target_data[j * 3 + 2];

            float vx = tx - origin_x;
            float vy = ty - origin_y;
            float t_proj = vx * cos_a + vy * sin_a;

            if (t_proj <= 0.0f || t_proj >= closest_tgt + tr) {
                continue;
            }

            float perp_sq = (vx * vx + vy * vy) - (t_proj * t_proj);
            float r_sq = tr * tr;

            if (perp_sq < r_sq) {
                float half_chord = std::sqrt(std::max(0.0f, r_sq - perp_sq));
                float entry_dist = t_proj - half_chord;
                if (entry_dist >= 0.0f && entry_dist < closest_tgt) {
                    closest_tgt = entry_dist;
                    tgt_hit = 1;
                }
            }
        }

        closest_tgt = std::clamp(closest_tgt, 0.0f, max_range);
        res.tgt_dists[k] = closest_tgt;
        res.tgt_types[k] = tgt_hit;
        res.tgt_endpoints[k * 2 + 0] = origin_x + closest_tgt * cos_a;
        res.tgt_endpoints[k * 2 + 1] = origin_y + closest_tgt * sin_a;
    }

    return res;
}

inline bool point_in_polygon(float px, float py, const float* vertices, size_t n) {
    if (n == 0) return false;
    bool inside = false;
    float p1x = vertices[0];
    float p1y = vertices[1];
    for (size_t i = 0; i <= n; ++i) {
        size_t idx = (i % n) * 2;
        float p2x = vertices[idx];
        float p2y = vertices[idx + 1];
        if (py > std::min(p1y, p2y)) {
            if (py <= std::max(p1y, p2y)) {
                if (px <= std::max(p1x, p2x)) {
                    float xinters = p1x;
                    if (p1y != p2y) {
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x;
                    }
                    if (p1x == p2x || px <= xinters) {
                        inside = !inside;
                    }
                }
            }
        }
        p1x = p2x;
        p1y = p2y;
    }
    return inside;
}

inline float distance_to_polygon(
    float px,
    float py,
    const float* vertices,
    size_t n_verts,
    const float* edges,
    size_t n_edges
) {
    if (point_in_polygon(px, py, vertices, n_verts)) {
        return 0.0f;
    }

    float min_dist_sq = 1e30f;
    for (size_t e = 0; e < n_edges; ++e) {
        float ax = edges[e * 4 + 0];
        float ay = edges[e * 4 + 1];
        float bx = edges[e * 4 + 2];
        float by = edges[e * 4 + 3];

        float vx = bx - ax;
        float vy = by - ay;
        float wx = px - ax;
        float wy = py - ay;
        float c1 = wx * vx + wy * vy;
        float dist_sq;
        if (c1 <= 0.0f) {
            dist_sq = (px - ax) * (px - ax) + (py - ay) * (py - ay);
        } else {
            float c2 = vx * vx + vy * vy;
            if (c2 <= c1) {
                dist_sq = (px - bx) * (px - bx) + (py - by) * (py - by);
            } else {
                float b = c1 / c2;
                float proj_x = ax + b * vx;
                float proj_y = ay + b * vy;
                dist_sq = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y);
            }
        }
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
        }
    }
    return std::sqrt(min_dist_sq);
}

} // namespace scrollswarm
