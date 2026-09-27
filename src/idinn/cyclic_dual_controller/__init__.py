"""
idinn.cyclic_dual_controller -- Controllers for periodic (cyclic) dual-sourcing inventory systems.
"""

from .cyclic_dual_neural import CyclicDualNeuralController
from .dp_bound import DynamicProgrammingController
from .dp_bound_parallel import DynamicProgrammingController as DynamicProgrammingParallelController

__all__ = [
    "CyclicDualNeuralController",
    "DynamicProgrammingController",
    "DynamicProgrammingParallelController",
]
