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

### Create Conda environment
```
conda env create -f environment.yml
conda activate rl-train-diffusion
```

### Install Reward model
```
bash install_image_reward.sh
```

## To train model
```
python train.py
```
