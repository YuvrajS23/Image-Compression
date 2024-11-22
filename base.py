import cv2
from decode import *
from huffman import huffman_compress, huffman_decompress
import argparse

# TODO: Implement the subparser
# Implement for colour images
# Add Encode and Decode Functions
# Write plotting code
# Start the report

# Parsing the args
def parse_arguments():
    parser = argparse.ArgumentParser(description="Encode, decode, or perform both operations on a file.")
    
    # Subparsers for 'encode', 'decode', and 'encodeanddecode'
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operation mode")

    # Encode subparser
    encode_parser = subparsers.add_parser("encode", help="Encode a file")
    encode_parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to encode")
    encode_parser.add_argument("-q", "--quality_factor", type=int, required=True, help="Quality factor for encoding")
    encode_parser.add_argument("-o", "--output_file", type=str, default="jpeg_compression.pkl", help="Output file for encoded data")

    # Decode subparser
    decode_parser = subparsers.add_parser("decode", help="Decode a file")
    decode_parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to decode")
    decode_parser.add_argument("-q", "--quality_factor", type=int, required=True, help="Quality factor for decoding")
    decode_parser.add_argument("-o", "--output_file", type=str, default="compressed_img.png", help="Output file for decoded data")
    decode_parser.add_argument("-s", "--show_file", action="store_true", help="Show the file of decoded data")

    # Encode and Decode subparser
    encode_and_decode_parser = subparsers.add_parser("encodeanddecode", help="Encode and then decode a file")
    encode_and_decode_parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to encode and decode")
    encode_and_decode_parser.add_argument("-q", "--quality_factor", type=int, required=True, help="Quality factor for encoding")
    encode_and_decode_parser.add_argument("-e", "--encoded_file", type=str, default="jpeg_compression.pkl", help="Temporary file for encoded data")
    encode_and_decode_parser.add_argument("-o", "--output_file", type=str, default="compressed_img.png", help="Final output file after decoding")
    encode_and_decode_parser.add_argument("-s", "--show_file", action="store_true", help="Show the file of decoded data")


    return parser.parse_args()


# Main function
def main(mode, Q, input, out, show, encoded_file):
    # If both encode and decode
    if mode == "encodeanddecode":
        image, enc_file = encode(Q, input, encoded_file)
        decompressed_image, info, sz = decode(Q, enc_file, out, show)
        h, w = info
        # Compute RMSE
        rmse = calculate_rmse(image, decompressed_image)
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print("Size of Compressed image:", sz)
        print("Bits Per Pixel (BPP):", (sz*8)/(h*w))
    # if only encode
    elif mode == "encode":
        image, enc_file = encode(Q, input, out)
    # If only decode 
    elif mode == "decode":
        decompressed_image, info, sz = decode(Q, input, out, show)
        h, w = info
        print("Size of Compressed image:", sz)
        print("Bits Per Pixel (BPP):", (sz*8)/(h*w))

if __name__ == "__main__":
    args = parse_arguments()
    main(args.mode, args.quality_factor, args.input_file, args.output_file, args.show_file, args.encoded_file)
