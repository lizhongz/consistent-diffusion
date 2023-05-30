import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2", torch_dtype=torch.float16
).to("cuda")
pipe.unet.load_attn_procs(
    "out/robo", weight_name="pytorch_custom_diffusion_weights.bin"
)
pipe.load_textual_inversion("out/robo", weight_name="<c1>.bin")
pipe.load_textual_inversion("out/robo", weight_name="<s1>.bin")


for i in range(0, 2):
    image = pipe(
        "a <c1> cartoon character playing soccer by the lake",
        num_inference_steps=20,
        guidance_scale=6.0,
        eta=1.0,
    ).images[0]
    image.save(str(i)+"porky.png")