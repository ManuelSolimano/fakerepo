import numpy as np
import detection_tools as dt
import imageio
import glob

def iterate_method(method_name='ela', root_path, **kwargs):
    """ Use this function to iterate through all target images
    in a given root directory. Every detection thecnique added
    to this function must return a single output image.
    """

    if method_name == 'ela':
        method = dt.ela_substract

    elif method_name == 'wavelet':
        method = dt.noise_detection_wavelet

    else:
        return

    path_list = glob.glob('{:s}/*'.format(root_path.strip('/'))

    for image_path in path_list:
        result = method(image_path, **kwargs)
        output_path = 'test_results/' + image_path[:-4].split('/')[1] + \
                '_{:s}'.format(method_name) + '.png'
        imageio.imwrite(output_path, result)




if __name__ == "__main__":
    #Change this if you want to use another method.
    iterate_method(method_name='ela', quality=85)
