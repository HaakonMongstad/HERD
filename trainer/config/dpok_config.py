import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

from trl.trainer import DDPOConfig


@dataclass
class DPOKConfig(DDPOConfig):
    """
    Configuration class for DPOKTrainer.
    """

    # common parameters
    sample_batch_size: int = 6
    """Batch size (per GPU!) to use for sampling."""
    sample_num_batches_per_epoch: int = 2
    """Number of batches to sample per epoch."""
    train_batch_size: int = 3
    """Batch size (per GPU!) to use for training."""

    # specific parameters
    hindsight_batch_size: int = 1
    """ Number of additional goals to sample from the replay buffer for each transition. """
