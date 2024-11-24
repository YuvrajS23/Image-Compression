from scipy.fftpack import dctn, idctn
import numpy as np
import matplotlib.pyplot as plt
import os

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


# Quantization
def quantize(block, quant_matrix):
    return (np.round(block / quant_matrix)).astype(np.int32)

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
            dct_block = dctn(block, norm='ortho')
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
            block = idctn(dequant_block, norm='ortho')
            decompressed_image[i:i+8, j:j+8] = block
            idx += 1

    return decompressed_image

# Calculate bits per pixel (BPP)
def calculate_bpp(encoded_size, image_shape):
    total_pixels = image_shape[0] * image_shape[1]
    return (encoded_size * 8) / total_pixels

# Calculate the Root Mean Squared Error between two images
def calculate_rmse(original, compressed):
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions for MSE calculation.")
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    return np.sqrt(mse)

# Zero-pads an image equally on all sides to make its height and width multiples of 16 (to account for downsampling)
def zero_pad(image):
    height, width = image.shape[:2]

    # Calculate padding needed
    pad_height = (16 - height % 16) % 16
    pad_width = (16 - width % 16) % 16

    # Split padding equally between start and end
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding
    if image.ndim == 3:  # Color image
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:  # Grayscale image
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

    return padded_image, (height, width), padded_image.shape[:2]

# Removes equal padding from an image to restore it to its original dimensions
def remove_equal_padding(padded_image, original_shape):
    padded_height, padded_width = padded_image.shape[:2]
    original_height, original_width = original_shape

    # Calculate padding values
    pad_height = padded_height - original_height
    pad_width = padded_width - original_width

    pad_top = pad_height // 2
    pad_left = pad_width // 2

    # Remove the padding by slicing
    unpadded_image = padded_image[
        pad_top:pad_top + original_height,
        pad_left:pad_left + original_width
    ]
    return unpadded_image

def downsampling_channel(ch):
    h, w = ch.shape
    ch = ch.reshape(h//2, 2, w//2, 2)
    ch = np.swapaxes(ch, 1, 2)
    ch = ch.reshape(h//2, w//2, 4)
    ch = ch.mean(axis=2,dtype=ch.dtype)
    return ch

def upsampling_channel(ch):
    h,w = ch.shape
    ch = ch.reshape(h, w, 1)
    ch = ch.repeat(4, axis=2)
    ch = ch.reshape(h, w, 2, 2)
    ch = np.swapaxes(ch, 1, 2)
    ch = ch.reshape(2*h, 2*w)
    return ch
