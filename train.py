from transformers import HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

from reward_models import ImageRewardModel, aesthetic_scorer
from trainer.config.dpok_config import DPOKConfig
from trainer.config.herd_config import HERDConfig
from trainer.ddpg_trainer import DDPGTrainer
from trainer.dpok_trainer import DPOKTrainer
from trainer.herd import HERDTrainer
from utils import PromptFn, ScriptArguments, image_outputs_logger


def get_reward_model():
    reward_model = input("Select reward model (e.g. aesthetic): ")
    if reward_model not in ["aesthetic", "imagereward"]:
        print("Invalid reward model. Please choose from 'aesthetic' or 'imagereward'.")
        return get_reward_model()
    return reward_model


def get_algorithm():
    algorithm = input("Select algorithm (e.g. herd): ")
    if algorithm not in ["herd", "ddpg", "dpok", "ddpo"]:
        print(
            "Invalid algorithm. Please choose from 'herd', 'ddpg', 'dpok', or 'ddpo'."
        )
        return get_algorithm()
    return algorithm


def get_logging_platform():
    log_with = input("Select logging platform (e.g. wandb): ")
    if log_with not in ["wandb", "tensorboard"]:
        print("Invalid logging platform. Please choose from 'wandb' or 'tensorboard'.")
        return get_logging_platform()
    return log_with


def get_prompt():
    prompt_input = input(
        "Enter prompt(s) for generating images (separated by comma if multiple): "
    )
    prompts = [prompt.strip() for prompt in prompt_input.split(",")]
    return prompts


def main():
    reward_model = get_reward_model()
    algorithm = get_algorithm()
    log_with = get_logging_platform()
    prompt = get_prompt()

    # Reward and Trainer classes
    trainer_classes = {
        "herd": HERDTrainer,
        "ddpg": DDPGTrainer,
        "dpok": DPOKTrainer,
        "ddpo": DDPOTrainer,
    }

    # Set config type
    if algorithm == "herd":
        CONFIG_TYPE = HERDConfig
    elif algorithm == "dpok":
        CONFIG_TYPE = DPOKConfig
    elif algorithm == "ddpo" or algorithm == "ddpg":
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
    trainer_config.log_with = log_with

    # Create prompt function
    prompt_fn = PromptFn(prompt)

    # Create trainer
    trainer_class = trainer_classes[algorithm]
    trainer = trainer_class(
        trainer_config,
        (
            aesthetic_scorer(
                args.hf_hub_aesthetic_model_id, args.hf_hub_aesthetic_model_filename
            )
            if reward_model == "aesthetic"
            else ImageRewardModel("ImageReward-v1.0")
        ),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()
    trainer.push_to_hub(args.hf_hub_model_id)


if __name__ == "__main__":
    main()
