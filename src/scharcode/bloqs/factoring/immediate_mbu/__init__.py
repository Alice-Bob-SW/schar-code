"""First version of the oracle (immediate MBU): each Lookup is uncomputed right after use."""

from scharcode.bloqs.factoring.immediate_mbu.accumulate_residue import AccumulateResidue
from scharcode.bloqs.factoring.immediate_mbu.approx_mod_exp import ApproxModExp
from scharcode.bloqs.factoring.immediate_mbu.compute_exponent import ComputeExponent
from scharcode.bloqs.factoring.immediate_mbu.compute_residue import (
    ComputeResidue,
    ComputeResidueAdjoint,
)
from scharcode.bloqs.factoring.immediate_mbu.underflow_phasefix import UnderflowPhasefix

__all__ = [
    "AccumulateResidue",
    "ApproxModExp",
    "ComputeExponent",
    "ComputeResidue",
    "ComputeResidueAdjoint",
    "UnderflowPhasefix",
]
