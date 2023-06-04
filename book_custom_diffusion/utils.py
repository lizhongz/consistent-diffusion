import os
from PIL import Image

def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def load_images(input_dir):
    imgs = []
    for f in os.listdir(input_dir):
        try:
            img = Image.open(os.path.join(input_dir, f))
            imgs.append(img)
        except IOError:
            # Not an image file or other IO error.
            continue
    return imgs

def save_images(images, out_dir, prefix=''):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(len(images)):
        images[i].save(os.path.join(out_dir, f'{prefix}{i}.png'), format='PNG')

