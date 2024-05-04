from diffusers import StableDiffusionPipeline
import torch
from cv2 import cv2
import numpy as np
from PIL import Image
import ImageReward as RM


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
reward_model = RM.load("ImageReward-v1.0")

# Move the model to the device
pipe.to(device)
reward_model.to(device)

# Define the input
text = "Lionel Messi displayed as a sitcom character."

# Run the model
image = pipe(text).images[0]

# Display the image
image.save("test.jpg")

score = reward_model.score(image=image, prompt=text)
print(f"Score: {score}")
