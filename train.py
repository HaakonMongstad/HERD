import argparse

from transformers import HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

from reward_models import ImageRewardModel, aesthetic_scorer
from trainer.config.dpok_config import DPOKConfig
from trainer.config.herd_config import HERDConfig
from trainer.ddpg_trainer import DDPGTrainer
from trainer.dpok_trainer import DPOKTrainer
from trainer.herd import HERDTrainer
from utils import PromptFn, ScriptArguments, image_outputs_logger


def main(cli_args):
    # Reward and Trainer classes
    trainer_classes = {
        "herd": HERDTrainer,
        "ddpg": DDPGTrainer,
        "dpok": DPOKTrainer,
        "ddpo": DDPOTrainer,
    }

    # Set config type
    if cli_args.algorithm == "herd":
        CONFIG_TYPE = HERDConfig
    elif cli_args.algorithm == "dpok":
        CONFIG_TYPE = DPOKConfig
    elif cli_args.algorithm == "ddpo" or cli_args.algorithm == "ddpg":
        CONFIG_TYPE = DDPOConfig
    else:
        raise ValueError("Invalid algorithm")

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
    trainer_config.log_with = cli_args.log_with

    # Create prompt function
    prompt_fn = PromptFn(cli_args.prompt)

    # Create trainer
    trainer_class = trainer_classes[cli_args.algorithm]
    trainer = trainer_class(
        trainer_config,
        (
            aesthetic_scorer(
                args.hf_hub_aesthetic_model_id, args.hf_hub_aesthetic_model_filename
            )
            if cli_args.reward_model == "aesthetic"
            else ImageRewardModel("ImageReward-v1.0")
        ),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()
    trainer.push_to_hub(args.hf_hub_model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument(
        "--reward_model",
        type=str,
        default="aesthetic",
        help="Select reward model (e.g. aesthetic,)",
        choices=["aesthetic", "imagereward"],
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="herd",
        help="Select algorithm (e.g. herd)",
        choices=["herd", "ddpg", "dpok", "ddpo"],
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default="wandb",
        help="Select logging platform (e.g. wandb)",
        choices=["wandb", "tensorboard"],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A black cat and golden retriever dog. A hot ocean side beach. Dramatic atmosphere, centered, rule of thirds, professional photo.",
        help="Prompt for to generate images",
    )
    args = parser.parse_args()

    main(args)
