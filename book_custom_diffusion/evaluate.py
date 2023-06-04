import os
import numpy as np
from itertools import product
from PIL import Image
from typing import List

import clip_similarity

# Examples:
#
#    img_sim_score = evaluate.evalulate_image_sim(
#        load_images('data/cartoon_boy/'),
#        load_images('data/cartoon_boy/gen_red_planet'))
#
#    text_sim_score = evaluate.evalulate_text_sim(
#        "cartoon character lived on Mars, the red planet.",
#        load_images('data/cartoon_boy/gen_red_planet'))
#

def evalulate_image_sim(images_a: List[Image.Image], images_b: List[Image.Image]) -> float:
    '''Evaluate image similarity/alignment.
    
    Returns the average of pair-wise similarity scores.
    '''
    cs = clip_similarity.CreateClipSimilarity()
    
    img_sim_scores = []
    for (a, b) in product(images_a, images_b):
        img_sim_scores.append(cs.image_similarity(a, b))
    assert len(img_sim_scores) == (len(images_a)*len(images_b))
    return np.array(img_sim_scores).mean()

def evalulate_text_sim(prompt: str, images: List[Image.Image]) -> float:
    '''Evaluate text similarity/alignment.
    
    Returns the average of (prompt, image) scores for all given images.
    '''
    cs = clip_similarity.CreateClipSimilarity()
    
    text_sim_scores = []
    for img in images:
        text_sim_scores.append(cs.text_image_similiarity(prompt, img))
    return np.array(text_sim_scores).mean()
