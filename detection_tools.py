""" This file will contain all functions and methods of forgery detection.
"""



import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance, ImageFile
import pywt
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation

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


def noise_detection_wavelet(img, wave, r, filename, dpi=80):
    # print("Image size:", img.shape)
    # Wavelet transform of image, and plot approximation and diagonal transform
    titles = ['Approximation', 'Diagonal detail']
    coeffs2 = pywt.dwt2(img, wave)  # bior1.3
    LL, (LH, HL, HH) = coeffs2
    x, y = img.shape
    fig = plt.figure(figsize=(2 * y / dpi, x / dpi))
    for j, a in enumerate([LL, HH]):
        ax = fig.add_subplot(1, 2, j + 1)
        ax.imshow(a, interpolation="nearest", cmap='gray')
        ax.set_title(titles[j] + " of " + filename, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('wave_' + wave + '_HH_' + filename)
    plt.show()

    rx, ry = int(x / r), int(y / r)
    regions = []  # The image is segmented in regions of R x R squares
    sigmas = []  # Sigma is the noise level for each region
    for i in range(rx):
        region_list = []
        sigma_list = []
        for j in range(ry):
            region = np.asarray(img[rx + r * i:rx + r * (i + 1), ry + r * j:ry + r * (j + 1)])
            region_list.append(region)

            mad = median_absolute_deviation(region)
            sigma = mad / 0.6745
            sigma_list.append(sigma)
        regions.append(region_list)
        sigmas.append(sigma_list)
    regions = np.asarray(regions)
    sigmas = np.asarray(sigmas)

    plt.imshow(sigmas)
    plt.tight_layout()
    plt.savefig('sigma_' + wave + '_HH_' + filename)
    plt.show()

    return sigmas
