""" This file will contain all functions and methods of forgery detection.
"""



import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance, ImageFile
import pywt
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from skimage.filters import median
import imageio
from numpy.lib.stride_tricks import as_strided


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


def median_filter_residuals(image_path, **kwargs):
    """Computes a local median on the input image and returns the difference.
    Keyword arguments correspond to keyword arguments of skimage.filters.median.
    """

    img = imageio.imread(image_path)
    red, green, blue = img[:,:,0], img[:,:,1] ,img[:,:,2]
    gray_median = median(green, **kwargs).astype(np.int32)
    return np.abs(gray_median - green.astype(np.int32))

def _create_sss_matrix(block, window_size=9):
    """ Creates the self-similarity pixel sampling matrix given a square block.
    """
    assert block.shape[0] == block.shape[1], "The input block needs to be a square array"
    ell, _ = block.shape
    n = (ell - window_size + 1) ** 2
    m = window_size ** 2

    # This was stolen from StackOverflow xd
    #https://stackoverflow.com/questions/19414673/in-numpy-how-to-efficiently-
    #list-all-fixed-size-submatrices
    sub_shape = tuple([window_size, window_size])
    view_shape = tuple(np.subtract(block.shape, sub_shape) + 1) + sub_shape
    arr_view = as_strided(block, view_shape, block.strides * 2)
    arr_view = arr_view.reshape(n, m).T
    return arr_view

def _covariance_matrix(X):
    """ Computes covariance matrix as defined by Zhan et al. (2016).
    """
    _, n = X.shape
    mean = np.average(X, axis=0)
    return  np.dot((X - mean), (X - mean).T) / (n - 1.)

def _get_minimum_eigenvalue(var_matrix):
    """ Solve eigenvalue problem of given matrix and return minimum eigenvalue.
    Assumes input matrix is hermitian.
    """
    eigenvalues = np.linalg.eigvalsh(var_matrix)
    return eigenvalues[0]

def _get_block_at_pos(gray_image, pos, size):
    i, j = pos
    top = i + size // 2 + 1
    bottom = i - size // 2
    left = j - size // 2
    right = j + size // 2 + 1
    return gray_image[bottom:top, left:right]

def lme_transform(gray_image, block_size, window_size):
    """ Perform PCA to get the local minimum eigenvalue at each pixel
    using the method described by Zhan et al. (2016).
    """
    M, N = gray_image.shape
    offset = (block_size  - 1) // 2   #block size has to be an odd integer
    lme_array = np.zeros_like(gray_image).astype(np.float64)

    # I couldn't think of a way to avoid traversing the whole image with for
    # loops. This is likely to be the main bottleneck of the implementation.
    for i in range(offset, M - offset):
        for j in range(offset, N - offset):
            block = _get_block_at_pos(gray_image, (i, j), block_size)
            sss = _create_sss_matrix(block, window_size)
            var = _covariance_matrix(sss)
            lme_array[i][j] = _get_minimum_eigenvalue(var)


    return lme_array



if __name__ == "__main__":
    img = imageio.imread('images/pinera.jpg')[:,:,1]
    lme = lme_transform(img, 25, 15)

