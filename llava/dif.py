import torch
from diffusers import PixArtAlphaPipeline
import random

# Load the pipeline
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16, use_safetensors=True)
pipe.enable_model_cpu_offload()

# Define the prompt
prompt = "Visualize an immense, semi-transparent being, composed entirely of flora and fauna native to an alien rainforest, majestically surfacing from a mirror-like lake of liquid mercury. This occurs under a kaleidoscopic sunset that forms infinite fractals in a sky filled not with stars, but with shimmering, multi-dimensional geometrical patterns in a dazzling array of neon colors reminiscent of an unabridged dream. This cosmic event is being observed by the mysterious shadow figures that live within the astral reflections of the lake's surface on the moon."

# Batch generation
for i in range(5):
    # Set a new random seed
    seed = random.randint(0, 1000000)
    generator = torch.manual_seed(seed)

    # Generate image
    image = pipe(prompt, generator=generator).images[0]

    # Save image with a unique filename
    image.save(f"./catcus_{i}.png")
