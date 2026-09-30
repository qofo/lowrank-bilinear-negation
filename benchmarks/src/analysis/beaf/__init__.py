"""
BEAF (Benchmark Evaluation & Analysis Framework) Subpackage.
"""

from analysis.beaf.vision_mechanisms import (
    extract_vision_features_unified,
    compute_vision_pipeline_breakdown,
    compute_vision_svd_sweep,
    compute_vision_linear_probe,
    compute_vision_direction_preservation,
)
from analysis.beaf.beaf_loader import (
    load_beaf_csv,
    load_and_verify_counterfactual_pairs,
)
from analysis.beaf.probe_factory import (
    SUPPORTED_PROBES,
    LowRankBilinearPyTorch,
    FullBilinearPyTorch,
    MLPVisionPyTorch,
    ElementWiseNonLinearPyTorch,
    PyTorchProbeEstimator,
    create_probe_classifier,
    get_param_candidates,
    format_params,
)

__all__ = [
    # Vision Mechanisms
    "extract_vision_features_unified",
    "compute_vision_pipeline_breakdown",
    "compute_vision_svd_sweep",
    "compute_vision_linear_probe",
    "compute_vision_direction_preservation",
    # Data Loading
    "load_beaf_csv",
    "load_and_verify_counterfactual_pairs",
    # Probe Factory
    "SUPPORTED_PROBES",
    "LowRankBilinearPyTorch",
    "FullBilinearPyTorch",
    "MLPVisionPyTorch",
    "ElementWiseNonLinearPyTorch",
    "PyTorchProbeEstimator",
    "create_probe_classifier",
    "get_param_candidates",
    "format_params",
]
