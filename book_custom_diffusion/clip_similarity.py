import numpy as np
import torch
import PIL

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

clip_id='openai/clip-vit-large-patch14'
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to('cuda')
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to('cuda')

class ClipSimilarity:
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder, clip_model='openai/clip-vit-large-patch14'):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.clip_model = clip_model
    
    def encode_image(self, image: PIL.Image.Image):
        preprocessed_img = self.image_processor(image, return_tensors="pt")["pixel_values"]
        preprocessed_img = {"pixel_values": preprocessed_img.to('cuda')}
        features = self.image_encoder(**preprocessed_img).image_embeds
        features = features / features.norm(dim=1, keepdim=True)
        return features

    def image_similarity(self, img1, img2):
        '''Cosine similiary between two images.'''
        e1 = self.encode_image(img1)
        e2 = self.encode_image(img2)
        score = torch.nn.functional.cosine_similarity(e1, e2)
        return round(float(score.item()), 4)
        # return e1@e2.T/(e1.norm()*e2.norm())
        
    def text_image_similiarity(self, prompt, img):
        '''CLIP score.'''
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (255, 255, 3) -> (3, 255, 255)
        score = clip_score(t, prompt, model_name_or_path=self.clip_model).detach()
        return round(float(score.item()), 4)
    
def CreateClipSimilarity(): 
    return ClipSimilarity(tokenizer, text_encoder, image_processor, image_encoder, clip_model=clip_id)