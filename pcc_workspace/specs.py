from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class SegmentSpec:
    name: str
    L_min: float
    L_max: float
    theta_max: float
    theta_sign: bool = True 
    phi_coupling: Optional[Union[str, float]] = None
    samples_theta: int = 60
    samples_phi: int = 60
    samples_length: int = 1


@dataclass
class TranslationSpec:
    d_min: float = 0.0
    d_max: float = 0.0
    samples: int = 1
