import numpy as np
import detection_tools as dt
import imageio
import glob

def iterate_method(method_name='ela', **kwargs):

    if method_name == 'ela':
        method = dt.ela_substract

    else:
        return

    path_list = glob.glob('images/*')

    for image_path in path_list:
        result = method(image_path, **kwargs)
        output_path = 'test_results/' + image_path[:-4].split('/')[1] + \
                '_{:s}'.format(method_name) + '.png'
        imageio.imwrite(output_path, result)




if __name__ == "__main__":
    iterate_method(method_name='ela', quality=85)
