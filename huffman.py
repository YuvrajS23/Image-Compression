import pickle
import dahuffman
import os

# Huffman Encoding
def huffman_compress(data, info, file_path):
    codec = dahuffman.HuffmanCodec.from_data(data)
    compressed_data = codec.encode(data)
    package = [codec, compressed_data, info]
    with open(file_path, "wb") as file:
        pickle.dump(package, file)
    return file_path
    
# Decoding
def huffman_decompress(file_path):
    with open(file_path, "rb") as file:
        loaded_package = pickle.load(file)
    loaded_codec = loaded_package[0]
    loaded_compressed_data = loaded_package[1]
    info = loaded_package[2]
    decompressed_data = loaded_codec.decode(loaded_compressed_data)
    original_array = list(map(int, decompressed_data))
    return original_array, info, os.path.getsize(file_path)