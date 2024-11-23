import cv2
import numpy as np
from utils import *
from huffman import huffman_compress

def encode(Q, input, out):
    # Quantization matrix
    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Load the image
    image = cv2.imread(input)

    if len(image.shape) == 3 and image.shape[2] == 3:
        qmY = quant_matrix * (50/Q)
        qmC = qmY * 2
        image, (h, w), ps = zero_pad(image)
        # Convert from RGB to YCbCr
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cb, cr = cv2.split(ycbcr)
        # print(y[0:100])
        # Downsample Cb and Cr
        cb = downsampling_channel(cb)
        cr = downsampling_channel(cr)
        np.savetxt("1.txt", cb, fmt="%d")
        y = y - 128
        cb = cb - 128
        cr = cr - 128
        # Compress each channel
        compressed_y = jpeg_compress(y, qmY)
        compressed_cb = jpeg_compress(cb, qmC)
        compressed_cr = jpeg_compress(cr, qmC)
        # compressed_y = compressed_y.reshape(-1, 64)
        # compressed_cb = compressed_cb.reshape(-1, 64)
        # compressed_cr = compressed_cr.reshape(-1, 64)
        y = jpeg_decompress(compressed_y, qmY, ps[:2])
        cb = jpeg_decompress(compressed_cb, qmC, (ps[0]//2,ps[1]//2))
        cr = jpeg_decompress(compressed_cr, qmC, (ps[0]//2,ps[1]//2))
        # dcb = jpeg_decompress(compressed_cb, qmC, ps[:2])
        # dcr = jpeg_decompress(compressed_cr, qmC, ps[:2])
        cb = cb + 128
        np.savetxt("2.txt", cb, fmt="%d")
        dcb = upsampling_channel(cb)
        dcr = upsampling_channel(cr)
        print(y.shape)
        print(dcb.shape)
        print(dcr.shape)
        dy = y + 128
        dcb = dcb + 128
        dcr = dcr + 128
        ycbcr = np.concatenate((dy.reshape(ps[0], ps[1], 1), dcb.reshape(ps[0], ps[1], 1), dcr.reshape(ps[0], ps[1], 1)), axis=2).astype(np.uint8)
        decompressed_image = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
        plt.imshow(decompressed_image)
        plt.show()
        exit(0)
        flat_data = list(compressed_y) + list(compressed_cb) + list(compressed_cr)
        # print(flat_data[0:100])
        image = remove_equal_padding(image, (h, w))
        return image, huffman_compress(flat_data, [Q, h, w, ps, True] , out)

    elif len(image.shape) == 2:
        qm = quant_matrix * (50/Q)
        # Load grayscale image
        image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
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

