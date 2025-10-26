from setuptools import setup, Extension
import sys, os, platform
import pybind11
from pathlib import Path

# -----------------------------
# Platform & feature switches
# -----------------------------
is_darwin = sys.platform == "darwin"
arch = platform.machine().lower()

use_openmp = os.environ.get("PCC_USE_OPENMP", "0") == "1"  # requires llvm-openmp in conda
use_ofast  = os.environ.get("USE_OFAST",  "1") == "1"      # keep your original toggle (not used below)
use_lto    = os.environ.get("USE_LTO",    "1") == "1"
use_pgo_gen= os.environ.get("PGO_GEN",    "0") == "1"
pgo_use    = os.environ.get("PGO_USE",    "")
cpu_flag   = os.environ.get("PCC_CPU", "apple-m2")

# Detect conda prefix for headers/libs/rpath
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
conda_inc = Path(CONDA_PREFIX, "include")
conda_lib = Path(CONDA_PREFIX, "lib")
libcxx_inc = Path(conda_inc, "c++", "v1")   # conda-forge libcxx headers

# -----------------------------
# Baseline compile/link flags
# -----------------------------
extra_compile_args = []
extra_link_args = []
define_macros = [
    ("NDEBUG","1"),
    ("PYBIND11_DETAILED_ERROR_MESSAGES","0"),
    # On macOS, disable availability markup to avoid mismatches when using newer libc++ headers.
    ("_LIBCPP_DISABLE_AVAILABILITY","1"),
]

if sys.platform == "win32":
    raise SystemExit("This setup is macOS/clang tuned (use conda-forge clang/llvm-openmp).")

# Common C++ flags (clang on macOS via conda-forge)
extra_compile_args += ["-std=c++20", "-fvisibility=hidden"]
# Numerics: keep your choices
extra_compile_args += ["-O3", "-ffast-math", "-ffp-contract=fast"]

# Use -stdlib=libc++ both at compile and link time to bind against libc++ from conda-forge
extra_compile_args += ["-stdlib=libc++"]
extra_link_args    += ["-stdlib=libc++"]

# CPU tuning: Apple Silicon gets -mcpu, otherwise fallback to -march=native
if is_darwin and arch in ("arm64", "aarch64"):
    extra_compile_args += [f"-mcpu={cpu_flag}"]
else:
    extra_compile_args += ["-march=native"]

# ThinLTO (works with Apple ld64; fine in conda clang too)
if use_lto:
    extra_compile_args += ["-flto=thin"]
    extra_link_args    += ["-flto=thin"]

# Dead-strip unused sections on macOS
if is_darwin:
    extra_link_args += ["-Wl,-dead_strip"]

# OpenMP via llvm-openmp (conda-forge):
#   -fopenmp at compile & link, and ensure rpath points to $CONDA_PREFIX/lib
if use_openmp:
    define_macros.append(("USE_OPENMP","1"))
    extra_compile_args += ["-fopenmp"]
    extra_link_args    += ["-fopenmp"]

# PGO (profile-generate / profile-use)
if use_pgo_gen:
    extra_compile_args += ["-fprofile-instr-generate", "-fcoverage-mapping"]
    extra_link_args    += ["-fprofile-instr-generate"]
if pgo_use:
    extra_compile_args += [f"-fprofile-instr-use={pgo_use}"]
    extra_link_args    += [f"-fprofile-instr-use={pgo_use}"]

# -----------------------------
# Include/Library search paths
# -----------------------------
include_dirs = [
    pybind11.get_include(),              # pybind11 headers (system)
    pybind11.get_include(user=True),     # pybind11 headers (user site)
]
library_dirs = []
rpaths = []

# If running inside a conda env, prefer its headers/libs so the .so binds to conda libc++
if CONDA_PREFIX:
    # libc++ headers and general includes from conda toolchain
    include_dirs.append(str(conda_inc))
    if libcxx_inc.exists():
        include_dirs.append(str(libcxx_inc))

    # Libraries from conda and corresponding runtime search path
    library_dirs.append(str(conda_lib))
    rpaths.append(str(conda_lib))  # ensure dlopen finds libomp/libc++.1 at runtime

# Translate rpaths into linker flags
for rp in rpaths:
    extra_link_args += [f"-Wl,-rpath,{rp}"]

# macOS minimum deployment target (optional but helps keep ABI stable)
# If you want to pin it, export MACOSX_DEPLOYMENT_TARGET in env; otherwise we don't force one here.
mdt = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
if mdt:
    extra_compile_args += [f"-mmacosx-version-min={mdt}"]
    extra_link_args    += [f"-mmacosx-version-min={mdt}"]

# -----------------------------
# Extension definition
# -----------------------------
ext_modules = [
    Extension(
        "pccik_native._core",
        sources=["cpp/pccik.cpp"],  # adjust if your source path differs
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        # Do NOT list libraries=["c++"] here; -stdlib=libc++ already wires it
    )
]

# -----------------------------
# Setup entry
# -----------------------------
setup(
    name="pccik_native",
    version="0.0.0",
    ext_modules=ext_modules,
    zip_safe=False,
)
