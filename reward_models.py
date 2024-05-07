from dataclasses import dataclass, field
import ImageReward
import numpy as np
import torch
import torch.nn as nn
import torchvision
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
import os
from trl.import_utils import is_npu_available, is_xpu_available


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


class ImageRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = ImageReward.load(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
