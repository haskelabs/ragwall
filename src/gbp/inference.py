"""
GBP inference-time adjustment utilities (minimal subset).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SeverityConfig:
    fn: str = "sigmoid"
    tau: float = 5.0
    center: float = 0.50
    alpha_mult: float = 1.0


@dataclass
class ConditionalPolicy:
    enabled: bool = False
    low_conf: float = 0.60
    high_conf: float = 0.85
    low_scale: float = 0.90
    high_scale: float = 0.30
    default_scale: float = 0.60
    elderly_agebin: str = '>65'
    elderly_conf_thresh: float = 0.70
    elderly_scale: float = 1.00
    female_conf_thresh: float = 0.70
    female_scale: float = 1.00
    race_minority_conf_thresh: float = 0.70
    race_minority_scale: float = 1.00
    scale_cap: float = 1.50
