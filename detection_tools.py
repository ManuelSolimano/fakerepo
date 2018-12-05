""" This file will contain all functions and methods of forgery detection.
"""



import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance, ImageFile
import pywt
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from skimage.filters import median
from skimage.transform import resize
import imageio
from numpy.lib.stride_tricks import as_strided
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import gamma, digamma, polygamma
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import newton, minimize_scalar
from scipy.stats import gennorm

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


def noise_detection_wavelet(img, name, wave, r, dpi=80):
    # print("Image size:", img.shape)
    # Wavelet transform of image, and plot approximation and diagonal transform
    titles = ['Approximation', 'Diagonal detail']
    coeffs2 = pywt.dwt2(img, wave)  # bior1.3
    LL, (LH, HL, HH) = coeffs2
    y, x = HH.shape
    fig = plt.figure(figsize=(2 * x / dpi, y / dpi))
    for j, a in enumerate([LL, HH]):
        ax = fig.add_subplot(1, 2, j + 1)
        ax.imshow(a, interpolation="nearest", cmap='gray')
        ax.set_title(titles[j] + " of " + name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('wave_' + wave + '_HH_' + name)
    plt.show()

    rx, ry = int(x / r), int(y / r)
    regions = []  # The image is segmented in regions of R x R squares
    sigmas = []  # Sigma is the noise level for each region
    for j in range(ry):
        region_list = []
        sigma_list = []
        for i in range(rx):
            region = np.asarray(HH[r * j:r * (j + 1), r * i:r * (i + 1)])
            if region.shape[0] == r and region.shape[1] == r:
                region_list.append(region)
                mad = median_absolute_deviation(region)
                sigma = mad / 0.6745
                sigma_list.append(sigma)

        if len(region_list) == 0:
            continue
        regions.append(region_list)
        sigmas.append(sigma_list)
    sigmas = np.asarray(sigmas)

    plt.imshow(sigmas)
    plt.tight_layout()
    plt.savefig('sigma_' + wave + '_HH_' + name)
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

# =============================================================================
# LME noise level inconsistency method
# =============================================================================

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

def _partition_input(gray_image, block_size, d):
    """ Divide input image in LxL blocks with d overlapping pixels.
    Return: an array view of all blocks.
    """
    M, N = gray_image.shape
    offset = block_size - d
    vshape = (M // offset, N // offset, block_size, block_size)
    strides = tuple(np.array(gray_image.strides) * offset) + gray_image.strides
    return as_strided(gray_image, vshape, strides)

def lme_transform(gray_image, block_size, window_size):
    """ Perform PCA to get the local minimum eigenvalue at each pixel
    using the method described by Zhan et al. (2016).
    """
#    M, N = gray_image.shape
#    offset = (block_size  - 1) // 2   #block size has to be an odd integer
#    lme_array = np.zeros_like(gray_image).astype(np.float64)

    # I couldn't think of a way to avoid traversing the whole image with for
    # loops. This is likely to be the main bottleneck of the implementation.
#    for i in range(offset, M - offset):
#        for j in range(offset, N - offset):
#            block = _get_block_at_pos(gray_image, (i, j), block_size)
#            sss = _create_sss_matrix(block, window_size)
#            var = _covariance_matrix(sss)
#            lme_array[i][j] = _get_minimum_eigenvalue(var)

    lme_array = np.zeros_like(gray_image).astype(np.float64)
    M, N = gray_image.shape
    vshape= (M, N, block_size, block_size)
    strides = gray_image.strides * 2
    view = as_strided(gray_image, vshape, strides)
    for i in range(M):
        for j in range(N):
            sss = _create_sss_matrix(view[i,j], window_size)
            var = _covariance_matrix(sss)
            lme_array[i,j] = _get_minimum_eigenvalue(var)
#    view = _partition_input(gray_image, block_size, 1)
#    for i in range(view.shape[0]):
#        for j in range(view.shape[1]):
#            sss = _create_sss_matrix(view[i,j], window_size)
#            var = _covariance_matrix(sss)
#            view[i,j][:,:]= _get_minimum_eigenvalue(var)
#    return gray_image
    return lme_array

# =============================================================================
# Blur type inconsistency detection (Bahrami et al. 2015)
# =============================================================================



def _estimate_kernel(block):
    """ Uses Maximum A Posteriori bayesian inference to estimate the blur
    kernel of a given patch of image.
    Return: kernel array
    """
    pass

def _ggd(x, amp, mu, sigma, beta):
    return amp * (beta / (2 * sigma + gamma(1./beta))) * \
    np.exp(-(np.abs(x - mu)/sigma) ** beta)

def _func(beta, hist):
    mu = hist.mean()
    diff = np.abs(hist - mu)
    g = 1 + digamma(1/beta) / beta \
    - np.sum(np.log(diff) * diff ** beta) / np.sum(diff ** beta) \
    + np.log((beta / hist.size) * np.sum(diff ** beta)) / beta
    return g

def _func_prime(beta, hist):
    mu = hist.mean()
    diff = np.abs(hist - mu)
    g_prime = - digamma(1 / beta) / beta \
    - polygamma(3, 1 / beta) / (beta ** 2) \
    - polygamma(3, 1 / beta) / (beta ** 3) \
    + 1 / (beta ** 2) \
    - np.sum((np.log(diff) ** 2) * diff ** beta) / np.sum(diff ** beta) \
    + (np.sum(np.log(diff) * diff ** beta) / np.sum(diff ** beta)) ** 2 \
    + np.sum(np.log(diff) * diff ** beta) / (beta * np.sum(diff ** beta)) \
    - np.log((beta / hist.size) * np.sum(diff ** beta)) / (beta ** 2)
    return g_prime

def _estimate_parameters(kernel, method='fit2'):
    """ Estimates dispersion (\sigma) and shape parameter (\beta) of the
    gray value normalized histogram of the kernel. The model is a generalized
    gaussian distribution.
    Two methods can be used: 1. Curve fit using least squares, and 2. MLE
    estimator formula from Wikipedia (article on GDD).
    Return: Feature vector [sigma, beta].
    """
    bins = 256 #int(kernel.max())
    kernel = kernel.astype(np.float) / np.sum(kernel)
    hist, _ = np.histogram(kernel.ravel(), bins)
    keep = np.where(hist > 0)[0]
    hist = hist[keep]       # reject zero values
    hist = hist / hist.sum() # normalize
    x_axis = np.linspace(0., kernel.max(), hist.size)

    if method == 'fit':
        filtered = savgol_filter(hist, 11, 1)   # remove noise
        popt, pcov = curve_fit(_ggd, x_axis, filtered, p0=[1, 5e-4, 5e-4, 1])
        return popt[2:], hist, filtered, x_axis

    elif method == 'fit2':
        beta, loc, sigma = gennorm.fit(hist)
        return np.array([sigma, beta])

    elif method == 'mle':
        m1 = np.average(np.abs(hist)) # compute first moment
        m2 = np.average(np.abs(hist) ** 2) # compute second moment
        beta0 = m1/np.sqrt(m2)
        beta = newton(_func, x0=beta0, args=(hist,), fprime=_func_prime,
                      maxiter=9999)
        stat = lambda mu: np.sum(np.abs(hist - mu) ** beta)
        mean = minimize_scalar(stat).x
        sigma = ((beta / hist.size) * np.sum(np.abs(hist - mean) ** beta)) ** (1/beta)
        return np.array([sigma, beta])

def _generate_dataset():
    scatter_plot = imageio.imread('rasterize_dataset.png')
    red = scatter_plot[:,:,0]
    blue = scatter_plot[:,:2]

    mblur = np.where(red < 200)
    mblur = np.array(mblur, dtype=np.float)
    mblur[0] = -1.1696e-6 * mblur[0] + 5.906e-4 # convert vertical indices into
    # the sigma coordinate
    mblur[1] = 4e-3 * mblur[1] - 0.208 # convert horizontal indices into shape
    # parameter (beta or gamma or whatever) coordinates

    oblur = np.where(blue < 200)
    mblur = np.array(mblur, dtype=np.float)
    oblur[0] = -1.1696e-6 * oblur[0] + 5.906e-4
    oblur[1] = 4e-3 * oblur[1] - 0.208

    split = mblur.shape[1]
    data = np.concatenate([mblur.T, oblur.T])
    labels = np.zeros(data.shape[0], dtype=np.int)
    labels[split:] = 1
    return data, labels

def _train_classifier(data, labels, **kwargs):
    """ Trains a Fisher Linear Discriminant model for classifying between
    out of focus or motion blur.
    input:
        -data: A (N,2)-shaped array of feature vectors. E.g. the ith element
        of this array must be [sigma_i, beta_i]
        -labels: A N-long array of zeros or ones indicating the ground truth
        class of each data element. So, labels[i] whether data[i] belongs to
        the 0-class (motion blur) or the 1-class (defocus blur).
        -kwargs(optional): Keyword arguments of LDA constructor. Please don't
        mess with this.
    Return:
        -clf: A Sci-kit learn LinearDiscriminantAnalysis instance.
    """
    clf = LinearDiscriminantAnalysis(**kwargs)
    clf.fit(data, labels)
    return clf

def _smooth_or_not(block, threshold=100):
    """ Decides wether a block is smooth or not based on its power spectrum
    frequency distribution. The threshold may not be well calibrated for every
    image.
    """
    # Compute power spectrum
    fft = np.fft.fftshift(np.fft.fft2(block))
    ps = np.abs(fft) ** 2

    # Compute smoothness metric
    metric = ps.max() / (ps.sum() - ps.max())
    if metric > threshold:
        return 0    # The block is smooth
    elif metric < threshold:
        return 1    # The block is non-smooth



def _smooth_block_analysis(blocks):
    """ Classify each block as smooth or non-smooth and then analyze their
    spatial distribution to further refine the blur type classification.
    Return: binary image separating motion blur from defocus blur blocks.
    """
    pass

def detect_blur_inconsistency(image):
    pass




#if __name__ == "__main__":
#    img = imageio.imread('../clase02oct/cameraman.png')
#    lme = lme_transform(img, 5, 3)
###    lme = np.abs(lme)
###    exp = np.floor(-np.log10(lme.max()))
###    lme *= 10 ** exp
#    plt.imshow(lme)


