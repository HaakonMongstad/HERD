# HERD: Hindsight Experience Replay for Diffusion Models
Fine-tuning Text to Image Diffusion Models Using Reinforcement Learning

<div style="display: flex; background-color: white; padding: 100; display: inline-block;">
<img src="img/HERD_diagram.png" alt="Image Description" width="200" />
<img src="img/IR_diagram.png" alt="Image Description" width="300" />
</div>

In this work, we propose utilizing a
hindsight experience replay buffer along with
DDPO and evaluating prior approaches that
combine RL with diffusion models to achieve
better results in image synthesis. Leveraging the Transformer Reinforcement Learning
library, we test various policy gradient reinforcement learning algorithms to evaluate
each performance on an existing open-source
text-to-image model, Stable Diffusion v1-5.

Paper to Code: [HERD: Fine-tuning Diffusion Models with Reinforcement Learning](https://drive.google.com/file/d/1GuugzyKCL5i0Yrl8u2Y43t2xzYv8f2tM/view?usp=sharing)

## Setup

### Create Conda environment
```bash
conda env create -f environment.yml
conda activate rl-train-diffusion
```

### Install Reward model
```bash
bash install_image_reward.sh
```

## Training
To train the model without specifying any additional arguments, run the following `command`:
```bash
python train.py 
```
To train the model with optional arguments, use:
```bash
python train.py [options]
```
### Command Line Arguments
- `--reward_model`: Specify the reward model to use. 
    - Options: `aesthetic`, `imagereward`. 
    - Default: `imagereward`.
- `--algorithm`: Specify the algorithm to use. 
    - Options: `herd`, `ddpg`, `dpok`. 
    - Default: `herd`.
- `--log_with`: Specify the logging platform to use. 
    - Options: `wandb`, `tensorboard`. 
    - Default: `wandb`.

## Examples
Train using HERD algorithm with ImageReward model and logging with wandb
```bash
python train.py --reward_model imagereward --algorithm herd --log_with wandb
```
Train using DDPG algorithm with aesthetic model and logging with tensorboard
```bash
python train.py --reward_model aesthetic --algorithm ddpg --log_with tensorboard
```