from dataclasses import dataclass
from typing import Literal, Optional, Union
from typing import List, Dict, Any
import numpy as np

PhiCoupling = Optional[Union[str, float]]


@dataclass
class TranslationSpec:
    d_min: float
    d_max: float
    samples: int = 1
    d_step: Optional[float] = None


@dataclass
class SegmentSpec:
    name: str
    L_min: float
    L_max: float
    samples_length: int = 1
    L_step: Optional[float] = None

    passive_L_min: float = 0.0
    passive_L_max: float = 0.0
    samples_passive_length: int = 1
    passive_L_step: Optional[float] = None
    active_L_max: float = 0.006
    theta_max: float = 0.0
    theta_sign: Optional[bool] = None
    samples_theta: int = 1

    theta_min: Optional[float] = None
    theta_step: Optional[float] = None

    samples_phi: int = 1
    phi_coupling: PhiCoupling = None
    phi_min: Optional[float] = None
    phi_max: Optional[float] = None
    phi_step: Optional[float] = None

    is_inner: bool = False
    bevel_angle_deg: float = 45.0
    roll_offset_deg: float = 0.0


@dataclass
class TouchPointSpec:
    coordinates: tuple[float, float, float]
    normal: tuple[float, float, float]

    def __str__(self) -> str:
        x, y, z = self.coordinates
        nx, ny, nz = self.normal
        return (
            f"Touch point at ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"with normal ({nx:.3f}, {ny:.3f}, {nz:.3f})."
        )


@dataclass
class IKOptions:
    pos_tol: float = 1e-3
    ang_tol_deg: float = 5.0
    topk: int = 1
    require_frontside: bool = True
    angle_target_deg: float = 45.0

    front_safe_cos: float = 0.0
    use_bevel_alignment: bool = False
    bevel_tol_deg: float = 2.0
    enforce_axis_band: bool = True
    nms_enable: bool = True
    nms_theta_deg: float = 2.0
    nms_th_phi_deg: float = 5.0
    nms_translation: float = 1e-3

    enforce_coplanar_plane: bool = True
    enforce_bevel_coplanar_plane: bool = True
    plane_mode: Literal["snap", "filter"] = "snap"
    phi_coplanar_tol_deg: float = 2.0
    active_first: bool = True
    active_first_tol: float = 2e-5


@dataclass
class SegmentSolution:
    name: str
    theta: float
    phi: float
    L_total: float
    L_passive: float
    L_active: float


@dataclass
class IKSolution:
    reachable: bool
    pos_err: float
    ang_err_deg: float
    translation: float
    segments: List[SegmentSolution]
    end_T: np.ndarray
    end_p: np.ndarray
    meta: Dict[str, Any]
