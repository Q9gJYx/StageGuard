"""StageGuard: backbone-agnostic physiological constraints for neural sleep staging."""

__version__ = "1.0.0"

from .config import ModalityConfig
from .decoder import SemiMarkovDecoder
from .losses import SoftTransitionPenalty, stageguard_loss
from .wrapper import StageGuardWrapper
