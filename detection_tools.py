""" This file will contain all functions and methods of forgery detection.
"""



import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance, ImageFile

def ela_substract(image_path, quality):
    """
    No sé si esto esta bien implementado...
    Revisar https://github.com/Ghirensics/ghiro/blob/master/plugins/processing/ela.py
    """
#     path_list = image_path.split('/')
#     original = imageio.imread(image_path)
#     new_path = ''.join(path_list[:-1]) + '/'+ path_list[-1][:path_list[-1].rfind('.')] + \
#                    '_{:d}.jpg'.format(quality)
#     imageio.imwrite(new_path, original, quality=quality)
#     reencoded = imageio.imread(new_path)
#     os.remove(new_path)
#     diff = np.array(original, np.float64) - np.array(reencoded, np.float64)

    path_list = image_path.split('/')
    new_path = ''.join(path_list[:-1]) + '/'+ path_list[-1][:path_list[-1].rfind('.')] + \
                   '_{:d}.jpg'.format(quality)

    im = Image.open(image_path)


    if im.mode != "RGB":
        im = im.convert("RGB")

    im.save(new_path,"JPEG", quality=quality)
    reencoded = Image.open(new_path)
    os.remove(new_path)


    diff = ImageChops.difference(im, reencoded)

    extrema = diff.getextrema()

    max_diff = max([ex[1] for ex in extrema])

    scale = 255.0/max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    return im, diff
