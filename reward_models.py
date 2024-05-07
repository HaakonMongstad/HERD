import importlib
import os
from dataclasses import dataclass, field

import ImageReward
import numpy as np
import torch
import torch.nn as nn
import torchvision

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