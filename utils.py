import numpy as np
from dataclasses import dataclass, field


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
