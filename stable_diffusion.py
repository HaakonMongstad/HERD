from diffusers import StableDiffusionPipeline
import torch
from cv2 import cv2


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Define the input
text = "A photo of a cat."

# Run the model
image = pipe(text).images[0]

# display the image
cv2.imshow("image", image)
