import copy
import functools
import torch
import math
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

# Example:
#
# # Custom Diffusion
# cd_pipe = inference.create_custom_diffusion_pipe(
#     "CompVis/stable-diffusion-v1-4",
#     "./saved_models_boy",
#     "pytorch_custom_diffusion_weights.bin",
#     "<new2>.bin")
# images = inference.inference(
#     cd_pipe, 
#     prompt="<new2> cartoon character lived on Mars, the red planet.",
#     size=64,
#     batch_size=8)

def create_custom_diffusion_pipe(base_model, saved_dir, attention_file, embedding_file):
    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16).to("cuda")
    pipe.unet.load_attn_procs(
        saved_dir, weight_name=attention_file)
    pipe.load_textual_inversion(saved_dir, weight_name=embedding_file)
    return pipe

def create_sd_pipe(base_model):
    return DiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    ).to("cuda")
    
def create_sd_img2img_pipe(base_model):
    return StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

def inference(pipe, prompt='', size=4, batch_size=8, steps=100,
              guidance_scale=6.0, eta=1.0, init_image=None, strength=0.8):
    '''Genearate images with the given pipeline.
    
    Params:
      size: # of images to generate
      batch_size: per batch images to generate. too many can cause OOM.
    '''
    if init_image:
        # Only StableDiffusionImg2ImgPipeline requires conditioned image strength.
        pipe = functools.partial(pipe, image=init_image.convert("RGB"), strength=strength)
    
    images = []
    n_batches = math.ceil(1.0 * size / batch_size)
    for i in range(n_batches):
        print(f"batch {i} generating {batch_size} images")
        result = pipe(
            prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps, 
            guidance_scale=guidance_scale, eta=eta,
        )
        images.extend(copy.deepcopy(result.images))
        
        # Release GPU memory
        del result
        torch.cuda.empty_cache()

    return images
