"""Stage orchestration and terminal display for training runs."""

from ._display import Display
from ._pipeline import STAGES, Pipeline

__all__ = ["Display", "Pipeline", "STAGES"]
