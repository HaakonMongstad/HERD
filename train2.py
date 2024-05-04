import importlib
import os
from dataclasses import dataclass, field

import ImageReward
import numpy as np
import torch
import torch.nn as nn
import torchvision
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from trl.import_utils import is_npu_available, is_xpu_available

from trainer.config.herd_config import HERConfig
from trainer.herd import HERTrainer


class ImageRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = ImageReward.load(model_path)
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def forward(self, images, prompts, metadata):
        scores = []
        for i in range(len(images)):
            score = torch.tensor(
                self.model.score(
                    prompt=prompts[i],
                    image=torchvision.transforms.ToPILImage()(images[i]),
                )
            )
            scores.append(score)

        return torch.tensor(scores).to(self.device), {}


@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "the pretrained model to use"},
    )
    pretrained_revision: str = field(
        default="main", metadata={"help": "the pretrained model revision to use"}
    )
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion",
        metadata={"help": "HuggingFace repo to save model weights to"},
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={
            "help": "HuggingFace model filename for aesthetic scorer model weights"
        },
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def aesthetic_scorer(hub_model_id, model_filename):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=torch.float32,
    )
    if is_npu_available():
        scorer = scorer.npu()
    elif is_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


animals = [
    # "cat",
    # "dog",
    # "horse",
    # "monkey",
    # "rabbit",
    # "zebra",
    # "spider",
    # "bird",
    # "sheep",
    # "deer",
    # "cow",
    # "goat",
    # "lion",
    # "frog",
    # "chicken",
    # "duck",
    # "goose",
    # "bee",
    # "pig",
    # "turkey",
    # "fly",
    # "llama",
    # "camel",
    # "bat",
    # "gorilla",
    # "hedgehog",
    # "kangaroo",
    # "A bear washing dishes.",
    # "A cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal.",
    # "Batman, cute modern disney style, Pixar 3d portrait, ultra detailed, gorgeous, 3d zbrush, trending on dribbble, 8k render",
    # "muscular black male covered in chocolate with bunny ears. portrait",
    # "Scenic view of Yosemite National Park waterfall during sunset in the winter time",
    # "A cat under the snow with blue eyes, covered by snow. Cinematic style, medium shot. Professional photo, animal.",
    # "2 cats in a basket, one is looking at the camera. 1 dog barking in the background.",
    "A black cat and golden retriever dog. A hot ocean side beach. Dramatic atmosphere, centered, rule of thirds, professional photo.",
]


def prompt_fn():
    return np.random.choice(animals), {}


def image_outputs_logger(image_data, global_step, accelerate_logger):
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    with torch.cuda.device(1):
        # Set config type
        CONFIG_TYPE = HERConfig

        # Parse arguments
        parser = HfArgumentParser((ScriptArguments, CONFIG_TYPE))

        # Create config file
        args, ddpo_config = parser.parse_args_into_dataclasses()
        ddpo_config.project_kwargs = {
            "logging_dir": "./logs2",
            "automatic_checkpoint_naming": True,
            "total_limit": 5,
            "project_dir": "./save2",
        }

        pipeline = DefaultDDPOStableDiffusionPipeline(
            args.pretrained_model,
            pretrained_model_revision=args.pretrained_revision,
            use_lora=args.use_lora,
        )
        ddpo_config.log_with = "wandb"
        ddpo_config.sample_batch_size = 6
        ddpo_config.train_batch_size = 3
        ddpo_config.sample_num_batches_per_epoch = 2
        ddpo_config.num_epochs = 100
        # ddpo_config.hindsight_batch_size = 1

        trainer = HERTrainer(
            ddpo_config,
            aesthetic_scorer(
                args.hf_hub_aesthetic_model_id, args.hf_hub_aesthetic_model_filename
            ),
            # ImageRewardModel("ImageReward-v1.0"),
            prompt_fn,
            pipeline,
            image_samples_hook=image_outputs_logger,
        )

        trainer.train()

        trainer.push_to_hub(args.hf_hub_model_id)
