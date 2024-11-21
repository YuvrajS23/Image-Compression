import cv2
import matplotlib.pyplot as plt
from utils import *
from huffman import huffman_compress, huffman_decompress
import argparse

# Parsing the args
def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Compression with a specified quality factor.")

    # Quality factor (integer)
    parser.add_argument(
        "quality_factor",
        type=int,
        help="An integer to decide the quality factor"
    )

    # File name (string)
    parser.add_argument(
        "file_name",
        type=str,
        help="Path to the file to be encrypted or decrypted"
    )

    # Encryption flag
    parser.add_argument(
        "-e",
        action="store_true",
        help="Enable encryption"
    )

    # Decryption flag
    parser.add_argument(
        "-d",
        action="store_true",
        help="Enable decryption"
    )

    # Save result flag
    parser.add_argument(
        "-s",
        action="store_true",
        help="Show the result"
    )

    # Output file name
    parser.add_argument(
        "-o",
        type=str,
        metavar="File",
        help="Output file name (used if -s is set)"
    )

    # Parse arguments
    args = parser.parse_args()
    return args

# Main function
def main(Q, fn, enc, dec, show, ofn):
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

    quant_matrix = quant_matrix * (50/Q)
    # If both encode and decode
    if enc and dec:
        # Load grayscale image
        image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        # Shifting image pixel intensities from 0:255 to -128:127
        image = image - 128
        h, w = image.shape
        print(f"Original image shape: {h}x{w}")

        # Compression
        compressed_blocks = jpeg_compress(image, quant_matrix)
        flat_data = compressed_blocks.flatten()

        huffman_compress(flat_data, [h, w] , 'compressed_data.pkl')
        # Decompression
        decompressed_blocks, info, sz = huffman_decompress('compressed_data.pkl')
        decompressed_blocks = np.array(decompressed_blocks)
        height, width = info
        decompressed_blocks = decompressed_blocks.reshape(-1, 64)
        decompressed_image = jpeg_decompress(decompressed_blocks, quant_matrix, (height, width))
        # Compute RMSE
        rmse = calculate_rmse(image, decompressed_image)
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print("Size of Compressed image:", sz)
        print("Bits Per Pixel (BPP):", (sz*8)/(height*width))
        decompressed_image = decompressed_image + 128
        image = image + 128
        if show:
            plt.imshow(image, cmap='gray')
            plt.show()
            plt.imshow(decompressed_image, cmap='gray')
            plt.show()
        
        if ofn:
            # Save the decompressed image
            cv2.imwrite(f"{ofn}", decompressed_image)
    # if only encode
    elif enc:
        # Load grayscale image
        image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        # Shifting image pixel intensities from 0:255 to -128:127
        image = image - 128
        h, w = image.shape
        print(f"Original image shape: {h}x{w}")

        # Compression
        compressed_blocks = jpeg_compress(image, quant_matrix)
        flat_data = compressed_blocks.flatten()
        fl = fn.split('.')

        huffman_compress(flat_data, [h, w] , fl[0] + '_myjpeg.pkl')
    # If only decode 
    elif dec:
        decompressed_blocks, info, sz = huffman_decompress(fn)
        decompressed_blocks = np.array(decompressed_blocks)
        height, width = info
        decompressed_blocks = decompressed_blocks.reshape(-1, 64)
        decompressed_image = jpeg_decompress(decompressed_blocks, quant_matrix, (height, width))
        print("Size of Compressed image:", sz)
        print("Bits Per Pixel (BPP):", (sz*8)/(height*width))
        decompressed_image = decompressed_image + 128
        if show:
            plt.imshow(decompressed_image, cmap='gray')
            plt.show()
        if ofn:
            # Save the decompressed image
            cv2.imwrite(f"{ofn}", decompressed_image)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.quality_factor, args.file_name, args.e, args.d, args.s, args.o)
