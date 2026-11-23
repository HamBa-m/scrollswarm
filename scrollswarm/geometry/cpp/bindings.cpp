#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "geometry.hpp"

namespace py = pybind11;

py::tuple cast_rays_py(
    float origin_x,
    float origin_y,
    py::array_t<float, py::array::c_style | py::array::forcecast> angles,
    float max_range,
    float ceil_y,
    float floor_y,
    py::array_t<float, py::array::c_style | py::array::forcecast> obs_meta,
    py::array_t<float, py::array::c_style | py::array::forcecast> obs_edges,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> obs_edge_ranges,
    py::array_t<float, py::array::c_style | py::array::forcecast> target_data
) {
    size_t num_rays = angles.shape(0);
    size_t num_obs = obs_meta.shape(0);
    size_t num_edges = obs_edges.shape(0);
    size_t num_targets = target_data.shape(0);

    auto result = scrollswarm::cast_rays(
        origin_x, origin_y,
        angles.data(), num_rays,
        max_range, ceil_y, floor_y,
        obs_meta.data(), num_obs,
        obs_edges.data(), num_edges,
        obs_edge_ranges.data(),
        target_data.data(), num_targets
    );

    // Return as NumPy arrays
    py::array_t<float> py_obs_dists(num_rays, result.obs_dists.data());
    py::array_t<int32_t> py_obs_types(num_rays, result.obs_types.data());
    
    std::vector<size_t> shape2d = {num_rays, 2};
    std::vector<size_t> strides2d = {2 * sizeof(float), sizeof(float)};
    py::array_t<float> py_obs_ends(shape2d, strides2d, result.obs_endpoints.data());

    py::array_t<float> py_tgt_dists(num_rays, result.tgt_dists.data());
    py::array_t<int32_t> py_tgt_types(num_rays, result.tgt_types.data());
    py::array_t<float> py_tgt_ends(shape2d, strides2d, result.tgt_endpoints.data());

    return py::make_tuple(py_obs_dists, py_obs_types, py_obs_ends, py_tgt_dists, py_tgt_types, py_tgt_ends);
}

bool point_in_polygon_py(
    float px,
    float py,
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices
) {
    size_t n = vertices.shape(0);
    return scrollswarm::point_in_polygon(px, py, vertices.data(), n);
}

float distance_to_polygon_py(
    float px,
    float py,
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<float, py::array::c_style | py::array::forcecast> edges
) {
    size_t n_verts = vertices.shape(0);
    size_t n_edges = edges.shape(0);
    return scrollswarm::distance_to_polygon(px, py, vertices.data(), n_verts, edges.data(), n_edges);
}

PYBIND11_MODULE(_geometry_cpp, m) {
    m.doc() = "High-performance C++20 geometry acceleration kernel for ScrollSwarm";
    m.def("cast_rays_cpp", &cast_rays_py, "Analytic raycasting against boundaries, polygons, and targets");
    m.def("point_in_polygon_cpp", &point_in_polygon_py, "Ray-casting point in polygon containment test");
    m.def("distance_to_polygon_cpp", &distance_to_polygon_py, "Euclidean point to polygon distance");
}
