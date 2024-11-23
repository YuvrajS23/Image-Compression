from decode import *
import argparse

# TODO: Implement for colour images
# Write plotting code
# Start the report

# Parsing the args
def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Compression")

    # Subparsers
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operation mode")

    # Encode subparser
    encode_parser = subparsers.add_parser("encode", help="Encode a file")
    encode_parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to encode")
    encode_parser.add_argument("-q", "--quality_factor", type=int, required=True, help="Quality factor for encoding")
    encode_parser.add_argument("-o", "--output_file", type=str, default="jpeg_compression.pkl", help="Output file for encoded data")

    # Decode subparser
    decode_parser = subparsers.add_parser("decode", help="Decode a file")
    decode_parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to decode")
    decode_parser.add_argument("-o", "--output_file", type=str, default="compressed_img.png", help="Output file for decoded data")

    # Encode and Decode subparser
    encode_and_decode_parser = subparsers.add_parser("encodeanddecode", help="Encode and then decode a file")
    encode_and_decode_parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to encode and decode")
    encode_and_decode_parser.add_argument("-q", "--quality_factor", type=int, required=True, help="Quality factor for encoding")
    encode_and_decode_parser.add_argument("-e", "--encoded_file", type=str, default="jpeg_compression.pkl", help="Temporary file for encoded data")
    encode_and_decode_parser.add_argument("-o", "--output_file", type=str, default="compressed_img.png", help="Final output file after decoding")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == "encodeanddecode":
        image, enc_file = encode(args.quality_factor, args.input_file, args.encoded_file)
        decompressed_image = decode(enc_file, args.output_file)
        # Compute RMSE
        rmse = calculate_rmse(image, decompressed_image)
        print(f"Root Mean Squared Error (RMSE): {rmse}")
    elif args.mode == "encode":
        image, enc_file = encode(args.quality_factor, args.input_file, args.output_file)
    elif args.mode == "decode":
        decompressed_image = decode(args.input_file, args.output_file)
