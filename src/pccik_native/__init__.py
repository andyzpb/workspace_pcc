try:
    from ._core import SegmentBounds, SolverConst, phi_scan, evaluate_once, brent_refine_phi, lm_polish
except Exception as e:
    raise RuntimeError(
        "Failed to import native extension pccik_native._core. "
        "Try `python setup.py build_ext --inplace` first."
    ) from e

__all__ = ["SegmentBounds", "SolverConst", "phi_scan"]
