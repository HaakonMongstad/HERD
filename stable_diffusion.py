from diffusers import StableDiffusionPipeline
import torch
from cv2 import cv2
import numpy as np
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Move the model to the device
pipe.to(device)

# Define the input
text = "A green colored rabbit."

# Run the model
image = pipe(text).images[0]

# Display the image
image.save("test.jpg")


