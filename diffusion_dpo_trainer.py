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
- diffusion_dpo
- diffusers
- reinforcement-learning
- text-to-image
- stable-diffusion
---

# {model_name}

This is a diffusion model that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for image generation conditioned with text.

"""


class DiffusionDPOTrainer(DDPOTrainer):

    _tag_names = ["trl", "diffusion_dpo"]

    def __init__(
        self,
        config: DDPOConfig,
        reward_function: Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DDPOStableDiffusionPipeline,
        ref_pipeline: DDPOStableDiffusionPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        super().__init__(
            config,
            reward_function,
            prompt_function,
            sd_pipeline,
            image_samples_hook=image_samples_hook,
        )

        self.ref_pipeline = ref_pipeline
        self.ref_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        self.ref_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.ref_pipeline.text_encoder.to(
            self.accelerator.device, dtype=inference_dtype
        )
        self.ref_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

        # Freeze vae, text_encoder, and reference unet
        self.ref_pipeline.vae.requires_grad_(False)
        self.ref_pipeline.text_encoder.requires_grad_(False)
        self.ref_pipeline.unet.requires_grad_(False)

    def calculate_loss(
        self, latents, timesteps, next_latents, log_probs, advantages, embeds
    ):
        target = torch.randn_like(latents)

        with self.autocast():
            if self.config.train_cfg:
                model_pred = self.sd_pipeline.unet(
                    torch.cat([latents] * 2),
                    torch.cat([timesteps] * 2),
                    embeds,
                ).sample
                # use for kl divergence and clip
                noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            else:
                model_pred = self.sd_pipeline.unet(
                    latents,
                    timesteps,
                    embeds,
                ).sample
                noise_pred = model_pred

            # model_losses = (model_pred - target).pow(2).mean(dim=[1, 2, 3])
            model_losses_w, model_losses_l = model_pred.chunk(2)

            model_diff = model_losses_w - model_losses_l

            with torch.no_grad():
                if self.config.train_cfg:
                    ref_pred = self.ref_pipeline.unet(
                        torch.cat([latents] * 2),
                        torch.cat([timesteps] * 2),
                        embeds,
                    ).sample
                else:
                    ref_pred = self.ref_pipeline.unet(
                        latents,
                        timesteps,
                        embeds,
                    ).sample

                # ref_losses = (ref_pred - target).pow(2).mean(dim=[1, 2, 3])
                ref_losses_w, ref_losses_l = ref_pred.chunk(2)

                ref_diff = ref_losses_w - ref_losses_l

            # Maybe take this from configs later
            beta_dpo = 5000
            scale_term = -0.5 * beta_dpo

            inside_term = scale_term * (model_diff - ref_diff)
            # implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
            loss = -1 * F.logsigmoid(inside_term).mean()

            scheduler_step_output = self.sd_pipeline.scheduler_step(
                noise_pred,
                timesteps,
                latents,
                eta=self.config.sample_eta,
                prev_sample=next_latents,
            )

            log_prob = scheduler_step_output.log_probs

            advantages = torch.clamp(
                advantages,
                -self.config.train_adv_clip_max,
                self.config.train_adv_clip_max,
            )

            ratio = torch.exp(log_prob - log_probs)

            approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)

            clipfrac = torch.mean(
                (torch.abs(ratio - 1.0) > self.config.train_clip_range).float()
            )

            return loss, approx_kl, clipfrac
