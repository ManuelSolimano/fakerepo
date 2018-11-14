""" This file will contain all functions and methods of forgery detection.
"""



import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance, ImageFile

def ela_substract(image_path, quality):
    """
    Error Level Analysis. The input image is resaved to a given quality using
    JPEG standard compression. Then the difference between  the original and resaved image
    is calculated. Use this method to detect differences in compression level within an image.
    """

    path_list = image_path.split('/')
    new_path = ''.join(path_list[:-1]) + '/'+ path_list[-1][:path_list[-1].rfind('.')] + \
                   '_{:d}.jpg'.format(quality)

    im = Image.open(image_path)

    # This prevents failure when a non-jpeg image is used.
    if im.mode != "RGB":
        im = im.convert("RGB")

    im.save(new_path,"JPEG", quality=quality)
    reencoded = Image.open(new_path)
    os.remove(new_path)


    diff = ImageChops.difference(im, reencoded)

    extrema = diff.getextrema()

    max_diff = max([ex[1] for ex in extrema])

    # Enhance contrast. Otherwise the image will look very bad.
    scale = 255.0/max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    return np.array(diff)
