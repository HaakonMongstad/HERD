import os
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import whoami
from trl.models import DDPOStableDiffusionPipeline
from trl.trainer import BaseTrainer, DDPOConfig, DDPOTrainer
from trl.trainer.utils import PerPromptStatTracker

logger = get_logger(__name__)


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- her
- diffusers
- reinforcement-learning
- text-to-image
- stable-diffusion
---

# {model_name}

This is a diffusion model that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for image generation conditioned with text.

"""


class HERTrainer(DDPOTrainer):

    _tag_names = ["trl", "her"]

    def __init__(
        self,
        config: DDPOConfig,
        reward_function: Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DDPOStableDiffusionPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        super().__init__(
            config,
            reward_function,
            prompt_function,
            sd_pipeline,
            image_samples_hook=image_samples_hook,
        )

        # Add trajectory memory
        self.memory = []

    def step(self, epoch: int, global_step: int):

        # Get samples
        original_trajectories, prompt_image_data = self._generate_samples(
            iterations=self.config.sample_num_batches_per_epoch,
            batch_size=self.config.sample_batch_size,
        )

        # Store original trajectories
        self.memory.extend(original_trajectories)
