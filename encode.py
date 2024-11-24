import cv2 as cv
import numpy as np
from utils import *
from huffman import huffman_compress

def encode(Q, input, out):
    # Load the image
    image = cv.imread(input)

    if len(image.shape) == 3 and image.shape[2] == 3:
        qmY = quant_matrix * (50/Q)
        qmC = qmY * 2
        image = image.astype(np.float32)
        image, (h, w), ps = zero_pad(image)

        # Convert from BGR to YCbCr
        b,g,r = image[:,:,0]/256.0,image[:,:,1]/256.0,image[:,:,2]/256.0

        y =     16  + 65.481 * r    + 128.553 * g   + 24.966 * b
        cb =    128 - 37.79 * r     - 74.203 * g    + 112 * b
        cr =    128 + 112 * r       - 93.786 * g    - 18.214 * b

        # Downsample Cb and Cr
        cr = downsampling_channel(cr)
        cb = downsampling_channel(cb)

        # Centre at 0
        y = y - 128
        cr = cr - 128
        cb = cb - 128

        # Compress each channel
        y = jpeg_compress(y, qmY).flatten()
        cr = jpeg_compress(cr, qmC).flatten()
        cb = jpeg_compress(cb, qmC).flatten()

        flat_data = list(y) + list(cr) + list(cb)
        image = remove_equal_padding(image, (h, w))
        return image, huffman_compress(flat_data, [Q, h, w, ps, True] , out)

    elif len(image.shape) == 2:
        qm = quant_matrix * (50/Q)
        # Load grayscale image
        image = cv.imread(input, cv.IMREAD_GRAYSCALE)
        image, (h, w), ps = zero_pad(image)
        # Shifting image pixel intensities from 0:255 to -128:127
        image = image - 128
        print(f"Original image shape: {h}x{w}")
        # Compression
        compressed_blocks = jpeg_compress(image, qm)
        flat_data = compressed_blocks.flatten()
        image = remove_equal_padding(image, (h, w))
        return image + 128, huffman_compress(flat_data, [Q, h, w, ps, False] , out)
    else:
        print("The image format is unknown.")

