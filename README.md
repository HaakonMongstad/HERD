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
To train the model interactively, run the following command:
```bash
python train.py 
```
This will prompt you to enter the necessary arguments.
### Command Line Inputs
- `reward_model`: Specify the reward model to use. 
    - Options: `aesthetic`, `imagereward`. 
- `algorithm`: Specify the algorithm to use. 
    - Options: `herd`, `ddpg`, `dpok`. 
- `log_with`: Specify the logging platform to use. 
    - Options: `wandb`, `tensorboard`. 
- `prompt`: Specify prompt(s) for generating images. Multiple prompts should be seperated by commas.
