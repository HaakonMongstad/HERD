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
        self.memory = {}

    def step(self, epoch: int, global_step: int):

        # Get samples
        samples, prompt_image_data = self._generate_samples(
            iterations=self.config.sample_num_batches_per_epoch,
            batch_size=self.config.sample_batch_size,
        )

        # Save the original prompt image data
        original_prompt_image_data = copy.deepcopy(prompt_image_data)

        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        rewards, rewards_metadata = self.compute_rewards(
            prompt_image_data, is_async=self.config.async_reward_computation
        )

        # Add rewards to prompt_image_data
        for i, image_data in enumerate(prompt_image_data):
            image_data.extend([rewards[i], rewards_metadata[i]])

        if self.image_samples_callback is not None:
            self.image_samples_callback(
                prompt_image_data, global_step, self.accelerator.trackers[0]
            )

        # Concatenate rewards
        rewards = torch.cat(rewards)
        rewards = self.accelerator.gather(rewards).cpu().numpy()

        # Log stats
        self.accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        # Use memory when not in the first step
        if global_step == 0:
            if self.config.per_prompt_stat_tracking:
                # gather the prompts across processes
                prompt_ids = (
                    self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                )
                prompts = self.sd_pipeline.tokenizer.batch_decode(
                    prompt_ids, skip_special_tokens=True
                )
                advantages = self.stat_tracker.update(prompts, rewards)
            else:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            # Factor in memory (per_prompt_stat_tracking probably won't work)
            all_rewards = np.concatenate((rewards, self.memory["rewards"]))
            if self.config.per_prompt_stat_tracking:
                # gather the prompts across processes
                prompt_ids = (
                    self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                )
                prompts = self.sd_pipeline.tokenizer.batch_decode(
                    prompt_ids, skip_special_tokens=True
                )
                advantages = self.stat_tracker.update(prompts, all_rewards)
            else:
                advantages = (all_rewards - all_rewards.mean()) / (
                    all_rewards.std() + 1e-8
                )

            memory_advantages = advantages[len(rewards) :]
            advantages = advantages[: len(rewards)]

            # ungather advantages
            for i in range(len(memory_advantages) // self.config.train_batch_size):
                self.memory["trajectories"][i]["advantages"] = (
                    torch.as_tensor(
                        [
                            memory_advantages[
                                i
                                * self.config.train_batch_size : (i + 1)
                                * self.config.train_batch_size
                            ]
                        ]
                    )
                    .reshape(self.accelerator.num_processes, -1)[
                        self.accelerator.process_index
                    ]
                    .to(self.accelerator.device)
                )

            del self.memory["rewards"]

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

            # Add hindsight trajectories to samples batch
            samples_batched.extend(self.memory["trajectories"]) if self.memory else None

            # Train the model
            self.sd_pipeline.unet.train()
            global_step = self._train_batched_samples(
                inner_epoch, epoch, global_step, samples_batched
            )

            # Add sampled trajectories to the memory
            hindsight_trajectories, hindsight_rewards = (
                self._sample_hindsight_trajectories(
                    samples_batched[: total_batch_size // self.config.train_batch_size],
                    original_prompt_image_data,
                    perm,
                )
            )

            # Log stats
            self.accelerator.log(
                {
                    "hindsight_reward_mean": hindsight_rewards.mean(),
                    "hindsight_reward_std": hindsight_rewards.std(),
                },
                step=global_step,
            )

            # Clear the memory (might want to delete this if you want to keep the memory for all future steps)
            self.memory.clear()

            # Save the trajectories to memory
            self.memory["trajectories"] = hindsight_trajectories
            self.memory["rewards"] = hindsight_rewards

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

    def _generate_new_prompt_data(self, prompt_image_data, index):
        self.sd_pipeline.unet.eval()

        # Shuffle the prompt
        row = (index // self.config.sample_batch_size).cpu().numpy()
        col = (index % self.config.sample_batch_size).cpu().numpy()
        shuffled_prompt = self._change_prompt(prompt_image_data[row][1][col])

        # Save the prompt
        prompt_image_pair = [
            prompt_image_data[row][0][col].unsqueeze(0),
            tuple([shuffled_prompt]),
            prompt_image_data[row][2][col],
        ]

        # Tokenize the prompt
        prompt_id = self.sd_pipeline.tokenizer(
            tuple([shuffled_prompt]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.sd_pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.accelerator.device)

        # Create prompt embeddings
        prompt_embeds = self.sd_pipeline.text_encoder(prompt_id)[0]
        return prompt_embeds, prompt_image_pair

    def _compute_new_reward(self, prompt_image_pair):
        # Compute rewards
        reward, _ = self.reward_fn(
            prompt_image_pair[0], prompt_image_pair[1], prompt_image_pair[2]
        )

        return torch.as_tensor(reward, device=self.accelerator.device)

    def _unpack_prev_trajectory(self, trajectories):
        reshaped_trajectories = [
            {} for _ in range(len(trajectories) * self.config.train_batch_size)
        ]

        for i, trajectory_bunch in enumerate(trajectories):
            for key, tensor in trajectory_bunch.items():
                for j in range(self.config.train_batch_size):
                    reshaped_trajectories[i * self.config.train_batch_size + j][key] = (
                        tensor[j].unsqueeze(0)
                    )

        return reshaped_trajectories

    def _repack_new_trajectory(self, trajectories):
        reshaped_trajectories = []

        for i, trajectory in enumerate(trajectories[:: self.config.train_batch_size]):
            trajectory_bunch = {}
            for key, _ in trajectory.items():
                trajectory_bunch[key] = torch.cat(
                    [
                        trajectories[i * self.config.train_batch_size + j][key]
                        for j in range(self.config.train_batch_size)
                    ]
                )

            reshaped_trajectories.append(trajectory_bunch)

        return reshaped_trajectories

    def _sample_hindsight_trajectories(
        self, prev_trajectories, original_prompt_image_data, perm
    ):
        hindsight_trajectories = []
        hindsight_rewards = []

        prev_trajectories = self._unpack_prev_trajectory(prev_trajectories)

        for _ in range(len(prev_trajectories)):
            # Sample a trajectory from the memory
            index, trajectory = random.choice(list(enumerate(prev_trajectories)))

            # Randomly shuffle the prompt
            prompt_embed, prompt_image_pair = self._generate_new_prompt_data(
                original_prompt_image_data, perm[index]
            )

            # Create new reward for the trajectory
            reward = self._compute_new_reward(prompt_image_pair)

            # Concatenate rewards
            reward = self.accelerator.gather(reward).cpu().numpy()
            hindsight_rewards.extend(reward)

            # Add the new trajectory to memory
            hindsight_trajectories.append(
                {
                    "prompt_embeds": prompt_embed,
                    "timesteps": trajectory["timesteps"],
                    "latents": trajectory["latents"],
                    "next_latents": trajectory["next_latents"],
                    "log_probs": trajectory["log_probs"],
                    "negative_prompt_embeds": trajectory["negative_prompt_embeds"],
                }
            )

        # Convert to numpy array
        hindsight_rewards = np.array(hindsight_rewards)

        # Repack the trajectories
        hindsight_trajectories = self._repack_new_trajectory(hindsight_trajectories)

        return hindsight_trajectories, hindsight_rewards

    def _change_prompt(self, prompt):
        # Modify prompt
        words = prompt.split()

        # Randomly shuffle the words
        random.shuffle(words)
        return " ".join(words)

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
            if self.config.train_cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat(
                    [sample["negative_prompt_embeds"], sample["prompt_embeds"]]
                )
            else:
                embeds = sample["prompt_embeds"]

            for j in range(self.num_train_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    loss, approx_kl, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["log_probs"][:, j],
                        sample["advantages"],
                        embeds,
                    )
                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            (
                                self.trainable_layers.parameters()
                                if not isinstance(self.trainable_layers, list)
                                else self.trainable_layers
                            ),
                            self.config.train_max_grad_norm,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

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
