from setuptools import setup, Extension
import sys, os, platform
import pybind11
from pathlib import Path

# -----------------------------
# Platform & feature switches
# -----------------------------
is_darwin = sys.platform == "darwin"
arch = platform.machine().lower()

use_openmp = os.environ.get("PCC_USE_OPENMP", "0") == "1"  # requires libomp
use_ofast = os.environ.get("USE_OFAST", "1") == "1"  # -Ofast optimizations
use_lto = os.environ.get("USE_LTO", "1") == "1"
use_pgo_gen = os.environ.get("PGO_GEN", "0") == "1"
pgo_use = os.environ.get("PGO_USE", "")  # path to .profdata
pgo_cov = os.environ.get("PGO_COV", "0") == "1"  # coverage mapping (optional)
cpu_flag = os.environ.get("PCC_CPU", "apple-m2")
pcc_real = os.environ.get("PCC_REAL", "").strip()  # e.g. "float" or "double"

# Detect conda prefix for headers/libs/rpath
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
conda_inc = Path(CONDA_PREFIX, "include")
conda_lib = Path(CONDA_PREFIX, "lib")
libcxx_inc = Path(conda_inc, "c++", "v1")  # conda-forge libcxx headers

# Homebrew LLVM lib path (for libomp runtime if not using conda)
brew_llvm_lib = Path("/opt/homebrew/opt/llvm/lib")

# -----------------------------
# Baseline compile/link flags
# -----------------------------
extra_compile_args = []
extra_link_args = []
define_macros = [
    ("NDEBUG", "1"),
    ("PYBIND11_DETAILED_ERROR_MESSAGES", "0"),
    ("_LIBCPP_DISABLE_AVAILABILITY", "1"),
]

if sys.platform == "win32":
    raise SystemExit(
        "This setup is macOS/clang tuned (use conda-forge clang/llvm-openmp)."
    )

extra_compile_args += [
    "-std=c++20",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
]

if use_ofast and not is_darwin:
    extra_compile_args += ["-Ofast"]
else:
    extra_compile_args += ["-O3", "-ffast-math"]

extra_compile_args += [
    "-ffp-contract=fast",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-fstrict-aliasing",
    "-ffunction-sections",
    "-fdata-sections",
]

extra_compile_args += ["-stdlib=libc++"]
extra_link_args += ["-stdlib=libc++"]

if is_darwin and arch in ("arm64", "aarch64"):
    extra_compile_args += [f"-mcpu={cpu_flag}"]
else:
    extra_compile_args += ["-march=native"]

if use_lto:
    extra_compile_args += ["-flto=thin"]
    extra_link_args += ["-flto=thin"]
    cache_dir = str(Path(".lto-cache").absolute())
    extra_link_args += [f"-Wl,-cache_path_lto,{cache_dir}"]

if is_darwin:
    extra_link_args += ["-Wl,-dead_strip", "-Wl,-dead_strip_dylibs"]

if use_openmp:
    define_macros.append(("USE_OPENMP", "1"))
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp", "-lomp"]

# -----------------------------
# PGO
# -----------------------------
if use_pgo_gen:
    extra_compile_args += ["-fprofile-instr-generate"]
    extra_link_args += ["-fprofile-instr-generate"]

if pgo_use:
    extra_compile_args += [f"-fprofile-instr-use={pgo_use}"]
    extra_link_args += [f"-fprofile-instr-use={pgo_use}"]

# -----------------------------
# Include/Library search paths
# -----------------------------
include_dirs = [
    pybind11.get_include(),
    pybind11.get_include(user=True),
]
library_dirs = []
rpaths = set()

if CONDA_PREFIX:
    include_dirs.append(str(conda_inc))
    if libcxx_inc.exists():
        include_dirs.append(str(libcxx_inc))
    library_dirs.append(str(conda_lib))
    rpaths.add(str(conda_lib))

brew_omp_lib = Path("/opt/homebrew/opt/libomp/lib")
if use_openmp and brew_omp_lib.exists():
    rpaths.add(str(brew_omp_lib))

rpaths.add("@loader_path")

for rp in sorted(rpaths):
    extra_link_args += [f"-Wl,-rpath,{rp}"]


# -----------------------------
# Extension definition
# -----------------------------
ext_modules = [
    Extension(
        "pccik_native._core",
        sources=["cpp/pccik.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
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
