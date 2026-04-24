"""rlm — Reinforcement Learning with Language Models.

A library for training and evaluating language models using reinforcement
learning techniques, including RLHF, PPO, and reward modeling.

Personal fork: using this to experiment with custom reward shaping
for code generation tasks.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rlm")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__author__ = "alexzhang13"
__license__ = "MIT"

from rlm.trainer import RLMTrainer
from rlm.reward import RewardModel
from rlm.config import RLMConfig

__all__ = [
    "__version__",
    "RLMTrainer",
    "RewardModel",
    "RLMConfig",
]
