from decode import *
import argparse

# TODO: Implement for colour images
# Write plotting code
# Start the report

# Parsing the args
def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Compression")
    
    #Subparsers
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
        decompressed_image = decode(enc_file, out, show)
        # Compute RMSE
        rmse = calculate_rmse(image, decompressed_image)
        print(f"Root Mean Squared Error (RMSE): {rmse}")
    # if only encode
    elif mode == "encode":
        image, enc_file = encode(Q, input, out)
    # If only decode 
    elif mode == "decode":
        decompressed_image = decode(input, out, show)

if __name__ == "__main__":
    args = parse_arguments()
    encoded_file = ""
    show_file = True
    quality_factor = 100
    if args.mode == "encodeanddecode":
        encoded_file = args.encoded_file
    if args.mode != "encode":
        show_file = args.show_file
    if args.mode != "decode":
        quality_factor = args.quality_factor
    main(args.mode, quality_factor, args.input_file, args.output_file, show_file, encoded_file)
