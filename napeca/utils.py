import numpy as np
import os
import tifffile as tiff


def uint8_arr(arr):

    # convert data to appropriate type
    # https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
    if np.issubdtype(np.uint16, arr.dtype) | np.issubdtype(np.int16, arr.dtype) | np.issubdtype(np.float32, arr.dtype) | np.issubdtype(np.float64, arr.dtype):

        data_out = arr / float(np.max(arr))  # normalize the data to 0 - 1; need to use float to ensure data can have decimals
        data_out = 255 * data_out  # Now scale by 255

    else:

        data_out = arr

    return data_out.astype('uint8')


def uint16_arr(arr):

    # convert data to appropriate type
    # https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
    if np.issubdtype(np.uint8, arr.dtype) | np.issubdtype(np.int16, arr.dtype) | np.issubdtype(np.float32, arr.dtype) | np.issubdtype(np.float64, arr.dtype):

        data_out = arr / float(np.max(arr))  # normalize the data to 0 - 1; need to use float to ensure data can have decimals
        data_out = 65535 * data_out  # Now scale by 255

    else:

        data_out = arr

    return data_out.astype('uint16')