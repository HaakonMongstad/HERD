# HERD: Hindsight Experience Replay for Diffusion Models
Finetuning Text to Image Diffusion Models using Reinforcement Learning

<div style="display: flex;">
<img src="img/HERD_diagram.png" alt="Image Description" width="200" />
<img src="img/IR_diagram.png" alt="Transparent Image" style="background-color: white; padding: 10px;" width="300" />
</div>


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
