try:
    from ._core import SegmentBounds, SolverConst, solve
except Exception as e:
    raise RuntimeError(
        "Failed to import native extension pccik_native._core. "
        "Try `python setup.py build_ext --inplace` first."
    ) from e

__all__ = ["SegmentBounds", "SolverConst", "solve"]
