from setuptools import setup, Extension
import sys, os, platform
import pybind11

is_darwin = sys.platform == "darwin"
arch = platform.machine().lower()
use_openmp = os.environ.get("PCC_USE_OPENMP", "0") == "1"
use_ofast  = os.environ.get("USE_OFAST", "1") == "1"       
use_lto    = os.environ.get("USE_LTO", "1") == "1"         
use_pgo_gen= os.environ.get("PGO_GEN", "0") == "1"        
pgo_use    = os.environ.get("PGO_USE", "")                 
cpu_flag   = os.environ.get("PCC_CPU", "apple-m2")         

extra_compile_args = []
extra_link_args = []
define_macros = [("NDEBUG","1"),("PYBIND11_DETAILED_ERROR_MESSAGES","0")]

if sys.platform == "win32":
    raise SystemExit("This setup is macOS/clang tuned.")
else:
    extra_compile_args += ["-std=c++20", "-fvisibility=hidden"]
    extra_compile_args += ["-O3", "-ffast-math", "-ffp-contract=fast"]

if is_darwin and arch in ("arm64","aarch64"):
    extra_compile_args += [f"-mcpu={cpu_flag}"]
else:
    extra_compile_args += ["-march=native"]

if use_lto:
    extra_compile_args += ["-flto=thin"]
    extra_link_args    += ["-flto=thin"]
if is_darwin:
    extra_link_args += ["-Wl,-dead_strip"]

if use_openmp:
    define_macros.append(("USE_OPENMP","1"))
    if is_darwin:
        extra_compile_args += ["-Xpreprocessor","-fopenmp"]
        extra_link_args    += ["-lomp"]


if use_pgo_gen:
    extra_compile_args += ["-fprofile-instr-generate","-fcoverage-mapping"]
    extra_link_args    += ["-fprofile-instr-generate"]
if pgo_use:
    extra_compile_args += [f"-fprofile-instr-use={pgo_use}"]
    extra_link_args    += [f"-fprofile-instr-use={pgo_use}"]

ext_modules = [
    Extension(
        "pccik_native._core",
        sources=["cpp/pccik.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )
]

setup(
    name="pccik_native",
    version="0.0.0",
    ext_modules=ext_modules,
)
