import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from collections import defaultdict, Counter
# import pickle
from huffman import huffman_compress, huffman_decompress

# 2D DCT and IDCT
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Quantization
def quantize(block, quant_matrix):
    return np.int32(np.round(block / quant_matrix))

def dequantize(block, quant_matrix):
    return block * quant_matrix

# JPEG-like compression
def jpeg_compress(image, quant_matrix):
    # Split into 8x8 blocks
    h, w = image.shape
    compressed_blocks = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i+8, j:j+8]
            dct_block = dct2(block)
            quant_block = quantize(dct_block, quant_matrix)
            compressed_blocks.append(quant_block.flatten())
    return np.array(compressed_blocks)

def jpeg_decompress(compressed_blocks, quant_matrix, shape):
    decompressed_image = np.zeros(shape, dtype=np.float32)
    h, w = shape
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            quant_block = compressed_blocks[idx].reshape(8, 8)
            dequant_block = dequantize(quant_block, quant_matrix)
            block = idct2(dequant_block)
            decompressed_image[i:i+8, j:j+8] = block
            idx += 1
    return np.clip(decompressed_image, 0, 255).astype(np.uint8)

def calculate_mse(original, compressed):
    """Calculate the Mean Squared Error between two images."""
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions for MSE calculation.")
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    return mse


# Main function
def main():
    # Load grayscale image
    image = cv2.imread('barbara256.png', cv2.IMREAD_GRAYSCALE)
    image = image - 128
    h, w = image.shape
    print(f"Original image shape: {h}x{w}")

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

    quant_matrix = quant_matrix * 0.5
    # Compress
    compressed_blocks = jpeg_compress(image, quant_matrix)
    print(f"Compressed shape: {compressed_blocks.shape}")

    flat_data = compressed_blocks.flatten()

    huffman_compress(flat_data, [h, w] , 'compressed_data.pkl')

    decompressed_blocks, info = huffman_decompress('compressed_data.pkl')
    decompressed_blocks = np.array(decompressed_blocks)
    height, width = info
    decompressed_blocks = decompressed_blocks.reshape(-1, 64)
    decompressed_image = jpeg_decompress(decompressed_blocks, quant_matrix, (height, width))
    # Compute MSE
    mse = calculate_mse(image, decompressed_image)
    print(f"Mean Squared Error (MSE): {mse}")

    decompressed_image = decompressed_image + 128
    image = image + 128
    # Save the original image
    cv2.imwrite("original_image.jpg", image)

    # Save the decompressed image
    cv2.imwrite("compressed_image.jpg", decompressed_image)

    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(decompressed_image, cmap='gray')
    plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
