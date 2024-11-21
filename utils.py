from scipy.fftpack import dct, idct
import numpy as np

# 2D DCT
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# 2D IDCT
def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Quantization
def quantize(block, quant_matrix):
    return np.int32(np.round(block / quant_matrix))

# Dequantization
def dequantize(block, quant_matrix):
    return block * quant_matrix

# Compression
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

# Decompression
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

# Calculate the Root Mean Squared Error between two images
def calculate_rmse(original, compressed):
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions for MSE calculation.")
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    return np.sqrt(mse)
