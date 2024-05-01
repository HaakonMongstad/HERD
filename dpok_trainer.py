import copy
import os
import random
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
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


class ConditionalLinear(nn.Module):
    """Conditional linear."""

    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()
        torch.nn.init.xavier_normal_(self.lin.weight)

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


class ValueModel(nn.Module):
    def __init__(self, num_steps, img_shape):
        super(ValueModel, self).__init__()
        self.lin1 = ConditionalLinear(int(np.prod(img_shape)) + 768, 256, num_steps)
        self.lin2 = ConditionalLinear(256, 256, num_steps)
        self.lin3 = ConditionalLinear(256, 256, num_steps)
        self.lin4 = nn.Linear(256, 1)
        torch.nn.init.xavier_normal_(self.lin4.weight)

    def forward(self, img, txt_emb, t):
        # going to change this later for accounting for different prompts within one batch but for now
        if len(txt_emb.shape) == 3:
            txt_emb = txt_emb[0, :]

        txt_emb = txt_emb[0, :]
        # x = img.view(img.shape[0], -1)

        # x = x.reshape(1, -1)
        x = img.flatten()
        txt_emb = txt_emb.unsqueeze(0)
        x = x.unsqueeze(0)
        x = torch.cat([x, txt_emb], dim=1)
        # x = torch.cat([x, txt_emb], dim=1)
        x = F.relu(self.lin1(x, t))
        x = F.relu(self.lin2(x, t))
        x = F.relu(self.lin3(x, t))
        return self.lin4(x)


class DPOKTrainer(DDPOTrainer):

    _tag_names = ["trl", "dpok"]

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

        self.value_function = ValueModel(50, (4, 64, 64))
        self.value_optimizer = torch.optim.AdamW(
            self.value_function.parameters(), lr=1e-4
        )
        self.value_function, self.value_optimizer = self.accelerator.prepare(
            self.value_function, self.value_optimizer
        )

        # prob get these from config
        self.v_steps = 5
        self.v_batch_size = 1

    # dont use save because of issue with ddpo not expecting accelerator to have 2 models
    def _save_model_hook(self, models, weights, output_dir):
        pass

    def _load_model_hook(self, models, input_dir):
        pass

    def compute_rewards(self, prompt_image_pairs, is_async=False):
        if not is_async:
            rewards = []
            for images, prompts, prompt_metadata in prompt_image_pairs:
                reward, reward_metadata = self.reward_fn(
                    images, prompts, prompt_metadata
                )
                rewards.append(
                    (
                        torch.as_tensor(reward, device=self.accelerator.device),
                        reward_metadata,
                    )
                )
        else:
            rewards = self.executor.map(
                lambda x: self.reward_fn(*x), prompt_image_pairs
            )
            rewards = [
                (
                    torch.as_tensor(reward.result(), device=self.accelerator.device),
                    reward_metadata.result(),
                )
                for reward, reward_metadata in rewards
            ]

        return zip(*rewards)

    def train_value_function(self, samples, batch_reward, train_index, time_index):
        # get random indices from batch samples

        batch_state = samples["latents"][train_index][time_index]
        batch_timestep = samples["timesteps"][train_index][time_index]
        batch_prompt_embeds = samples["prompt_embeds"][train_index]

        pred_value = self.value_function(
            batch_state.cuda().detach(),
            batch_prompt_embeds.cuda().detach(),
            torch.tensor(time_index).cuda().detach(),
            # batch_timestep.cuda().detach(),
        )

        batch_reward = torch.tensor(batch_reward)

        value_loss = F.mse_loss(
            pred_value.float(),
            batch_reward.cuda().detach(),
        )

        self.accelerator.backward(value_loss / self.v_steps)

        del batch_state, batch_timestep, batch_reward, batch_prompt_embeds, pred_value

        return value_loss.item() / self.v_steps

    def step(self, epoch: int, global_step: int):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step, and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.

        """
        samples, prompt_image_data = self._generate_samples(
            iterations=self.config.sample_num_batches_per_epoch,
            batch_size=self.config.sample_batch_size,
        )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        rewards, rewards_metadata = self.compute_rewards(
            prompt_image_data, is_async=self.config.async_reward_computation
        )

        for i, image_data in enumerate(prompt_image_data):
            image_data.extend([rewards[i], rewards_metadata[i]])

        if self.image_samples_callback is not None:
            self.image_samples_callback(
                prompt_image_data, global_step, self.accelerator.trackers[0]
            )

        rewards = torch.cat(rewards)
        rewards = self.accelerator.gather(rewards).cpu().numpy()

        self.accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        if self.config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = self.sd_pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages;  keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        for inner_epoch in range(self.config.train_num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            # still trying to understand the code below
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=self.accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )

            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=self.accelerator.device)[
                        :, None
                    ],
                    perms,
                ]

            original_keys = samples.keys()
            original_values = samples.values()
            # rebatch them as user defined train_batch_size is different from sample_batch_size
            reshaped_values = [
                v.reshape(-1, self.config.train_batch_size, *v.shape[1:])
                for v in original_values
            ]

            # Transpose the list of original values
            transposed_values = zip(*reshaped_values)
            # Create new dictionaries for each row of transposed values
            samples_batched = [
                dict(zip(original_keys, row_values)) for row_values in transposed_values
            ]

            for batch_num in range(len(samples_batched)):
                sample = samples_batched[batch_num]
                reward = rewards[batch_num]
                for train_batch_num in range(self.config.train_batch_size):

                    # value function training steps
                    val_loss = 0
                    self.value_optimizer.zero_grad()
                    v_time_indices = np.random.choice(
                        sample["latents"][train_batch_num].shape[0],
                        self.v_steps,
                        replace=False,
                    )
                    for v_step in range(self.v_steps):
                        if v_step < self.v_steps - 1:
                            with self.accelerator.no_sync(self.value_function):
                                val_loss += self.train_value_function(
                                    sample,
                                    reward,
                                    train_batch_num,
                                    v_time_indices[v_step],
                                )
                        else:
                            val_loss += self.train_value_function(
                                sample, reward, train_batch_num, v_time_indices[v_step]
                            )
                    self.value_optimizer.step()
                    self.value_optimizer.zero_grad()

                    self.accelerator.log(
                        {
                            "value_loss": val_loss,
                        }
                    )
                    torch.cuda.empty_cache()
                    # --------------------------------------------

                    self.sd_pipeline.unet.train()
                    self.optimizer.zero_grad()
                    info = defaultdict(list)

                    for j in range(self.num_train_timesteps):
                        if j < self.num_train_timesteps - 1:
                            with self.accelerator.no_sync(self.sd_pipeline.unet):
                                self.train_policy_function(sample, reward, j, info)
                        else:
                            self.train_policy_function(sample, reward, j, info)

                        # if self.accelerator.sync_gradients:
                    # log training-related stuff
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = self.accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    # self.accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)

            # ensure optimization step at the end of the inner epoch
            if not self.accelerator.sync_gradients:
                raise ValueError(
                    "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                )

        if (
            epoch != 0
            and epoch % self.config.save_freq == 0
            and self.accelerator.is_main_process
        ):
            self.accelerator.save_state()

        return global_step

    def _generate_samples(self, iterations, batch_size):
        """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (List[Dict[str, torch.Tensor]]), prompt_image_pairs (List[List[Any]])
        """
        samples = []
        prompt_image_pairs = []
        self.sd_pipeline.unet.eval()

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        for _ in range(iterations):
            prompts, prompt_metadata = zip(
                *[self.prompt_fn() for _ in range(batch_size)]
            )

            prompt_ids = self.sd_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            with self.autocast():
                sd_output = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

                images = sd_output.images
                latents = sd_output.latents
                log_probs = sd_output.log_probs

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(
                batch_size, 1
            )  # (batch_size, num_steps)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                }
            )
            prompt_image_pairs.append([images, prompts, prompt_metadata])

        return samples, prompt_image_pairs

    def train_policy_function(self, sample, reward, time_step, info):
        """
        Train the policy function

        Args:
            sample (Dict[str, torch.Tensor]): The sample to train on
            info (Dict[str, List]): The dictionary to store the training information

        Side Effects:
            - The policy function is trained
            - The training information is stored in the info dictionary
        """

        if self.config.train_cfg:
            # concat negative prompts to sample prompts to avoid two forward passes
            embeds = torch.cat(
                [sample["negative_prompt_embeds"], sample["prompt_embeds"]]
            )
        else:
            embeds = sample["prompt_embeds"]
        with self.autocast():
            model_pred = self.sd_pipeline.unet(
                torch.cat([sample["latents"][:, time_step]] * 2),
                torch.cat([sample["timesteps"][:, time_step]] * 2),
                embeds,
            ).sample

            # maybe do this... not quite sure what for
            model_pred_uncond, model_pred_text = model_pred.chunk(2)
            model_pred = model_pred_uncond + self.config.sample_guidance_scale * (
                model_pred_text - model_pred_uncond
            )

            scheduler_step_output = self.sd_pipeline.scheduler_step(
                model_pred,
                sample["timesteps"][:, time_step],
                sample["latents"][:, time_step],
                eta=self.config.sample_eta,
                prev_sample=sample["next_latents"][:, time_step],
            )

            log_prob = scheduler_step_output.log_probs

        with torch.no_grad():
            advantages = reward - self.value_function(
                sample["latents"][:, time_step],
                embeds,
                torch.tensor(time_step).cuda().detach(),
            ).reshape([self.config.train_batch_size, 1])

        ratio = torch.exp(log_prob - sample["log_probs"][:, time_step])
        ratio_clip = 1e-4  # get this froms self.config.train_clip_range later
        ratio = torch.clamp(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
        reward_weight = 100  # will pass this in too
        aprrox_kl = 0.5 * torch.mean(
            (log_prob - sample["log_probs"][:, time_step]) ** 2
        )

        loss = (
            -reward_weight
            * advantages.detach().float()
            * ratio.float().reshape([self.config.train_batch_size, 1])
        ).mean()
        loss = loss / self.num_train_timesteps

        info["aprrox_kl"].append(aprrox_kl)
        info["loss"].append(loss)

        self.accelerator.backward(loss)

    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            inner_epoch (int): The current inner epoch
            epoch (int): The current epoch
            global_step (int): The current global step
            batched_samples (List[Dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        info = defaultdict(list)
        for _i, sample in enumerate(batched_samples):
            self.optimizer.zero_grad()
            for j in range(self.num_train_timesteps):
                if j < self.num_train_timesteps - 1:
                    with self.accelerator.no_sync(self.sd_pipeline.unet):
                        self.train_policy_function(sample, info)
                else:
                    self.train_policy_function(sample, info)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    # log training-related stuff
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = self.accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    self.accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)
        return global_step

    def train(self, epochs: Optional[int] = None):
        """
        Train the model for a given number of epochs
        """
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        for epoch in range(self.first_epoch, epochs):
            global_step = self.step(epoch, global_step)
