import importlib
import os
from dataclasses import dataclass, field

import ImageReward
import numpy as np
import torch
import torch.nn as nn
from transformers import HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

from trainer.ddpg_trainer import DDPGTrainer
from trainer.dpok_trainer import DPOKTrainer
from trainer.config.herd_config import HERDConfig
from trainer.herd import HERDTrainer

from reward_models import ImageRewardModel
from utils import image_outputs_logger, ScriptArguments, prompt_fn


if __name__ == "__main__":

    with torch.cuda.device(0):
        # Set config type
        CONFIG_TYPE = HERDConfig

        # Parse arguments
        parser = HfArgumentParser((ScriptArguments, CONFIG_TYPE))

        # Create config file
        args, trainer_config = parser.parse_args_into_dataclasses()
        trainer_config.project_kwargs = {
            "logging_dir": "./logs",
            "automatic_checkpoint_naming": True,
            "total_limit": 5,
            "project_dir": "./save",
        }

        pipeline = DefaultDDPOStableDiffusionPipeline(
            args.pretrained_model,
            pretrained_model_revision=args.pretrained_revision,
            use_lora=args.use_lora,
        )
        trainer_config.log_with = "wandb"

        trainer = HERDTrainer(
            trainer_config,
            # aesthetic_scorer(
            #     args.hf_hub_aesthetic_model_id, args.hf_hub_aesthetic_model_filename
            # ),
            ImageRewardModel("ImageReward-v1.0"),
            prompt_fn,
            pipeline,
            image_samples_hook=image_outputs_logger,
        )

        trainer.train()

        trainer.push_to_hub(args.hf_hub_model_id)
