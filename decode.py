from encode import *
from huffman import huffman_decompress
import matplotlib.pyplot as plt

def decode(input, out, show):
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

    # Decompression
    decompressed_blocks, info, sz = huffman_decompress(input)
    decompressed_blocks = np.array(decompressed_blocks)
    Q, height, width = info
    qm = quant_matrix * (50/Q)
    decompressed_blocks = decompressed_blocks.reshape(-1, 64)
    decompressed_image = jpeg_decompress(decompressed_blocks, qm, (height, width))
    decompressed_image = decompressed_image + 128
    cv2.imwrite(f"{out}", decompressed_image)
    print("Size of Compressed image:", sz)
    print("Bits Per Pixel (BPP):", (sz*8)/(height*width))
    if show:
        plt.imshow(decompressed_image, cmap='gray')
        plt.show()
    return decompressed_image