from diffusers import DiffusionPipeline

import torch

model_id="./db_out/pig"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "a photo of zxw cartoon character"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("test.png")
