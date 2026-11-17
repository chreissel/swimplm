"""swimplm: NPLM anomaly tests in learned embeddings with SparKer, Falkon and SWIM."""

from .methods import Falkon, SparKer, Swim
from .nplm import NPLMResult, p_value_empirical, z_asymptotic, z_empirical
from .toys import run_toys

__all__ = [
    "Falkon",
    "NPLMResult",
    "SparKer",
    "Swim",
    "p_value_empirical",
    "run_toys",
    "z_asymptotic",
    "z_empirical",
]
