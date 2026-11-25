"""
Automated Build Script for ScrollSwarm C++ Geometry Kernel.

Compiles `scrollswarm/geometry/cpp/bindings.cpp` into a high-performance
native extension (`_geometry_cpp.pyd` / `_geometry_cpp.so`).

Supports:
- Windows (GCC MinGW-w64 with static runtime linkage or MSVC)
- Linux / macOS (GCC or Clang)
"""

import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11


def build_extension():
    project_root = Path(__file__).resolve().parent.parent
    cpp_dir = project_root / "scrollswarm" / "geometry" / "cpp"
    out_dir = project_root / "scrollswarm" / "geometry"

    source_file = cpp_dir / "bindings.cpp"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if os.name == "nt" else ".so"
    )
    output_file = out_dir / f"_geometry_cpp{ext_suffix}"

    print(f"Building ScrollSwarm C++ Geometry Kernel...")
    print(f"Source: {source_file}")
    print(f"Target: {output_file}")

    py_include = sysconfig.get_path("include")
    pybind_include = pybind11.get_include()

    includes = [
        f"-I{cpp_dir}",
        f"-I{py_include}",
        f"-I{pybind_include}",
    ]

    is_windows = platform.system() == "Windows"

    cmd = [
        "g++",
        "-O3",
        "-shared",
        "-std=c++20",
        *includes,
        str(source_file),
        "-o",
        str(output_file),
    ]

    if is_windows:
        cmd.extend(
            [
                "-static-libgcc",
                "-static-libstdc++",
                "-static",
            ]
        )
        lib_dir = os.path.join(sysconfig.get_config_var("installed_base"), "libs")
        py_ver_nodot = f"{sys.version_info.major}{sys.version_info.minor}"
        cmd.extend(
            [
                f"-L{lib_dir}",
                f"-lpython{py_ver_nodot}",
            ]
        )
    else:
        cmd.append("-fPIC")

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(
            f"Compilation Failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    print("Compilation successful!")

    # Verify import
    if is_windows and hasattr(os, "add_dll_directory"):
        base_prefix = getattr(sys, "base_prefix", sys.prefix)
        if os.path.exists(base_prefix):
            os.add_dll_directory(base_prefix)

    sys.path.insert(0, str(out_dir))
    sys.path.insert(0, str(project_root))

    try:
        from scrollswarm.geometry import _geometry_cpp

        print(
            f"Verification passed: successfully imported {getattr(_geometry_cpp, '__file__', '_geometry_cpp')}"
        )
    except Exception as e:
        print(f"Verification warning: direct import check failed: {e}")

    return output_file


if __name__ == "__main__":
    build_extension()
