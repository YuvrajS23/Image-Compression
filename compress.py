import pickle
import dahuffman
import os
import sys

def rl_encode(l):
    out = []
    count = 0
    for i in l:
        if i == 0:
            count += 1
        elif count > 0 and i != 0:
            out.append(0)
            out.append(count)
            count = 0
            out.append(i)
        else:
            out.append(i)
    if count > 0:
        out.append(0)
        out.append(count)
    return out

def rl_decode(o):
    l = []
    i = 0
    while i < len(o):
        if o[i] == 0:
            l += [0] * o[i + 1]
            i = i + 2
        else:
            l.append(o[i])
            i = i + 1
    return l

# Huffman Encoding
def data_compress(data, info, file_path):
    data = rl_encode(data)

    codec = dahuffman.HuffmanCodec.from_data(data)
    compressed_data = codec.encode(data)
    package = [codec, compressed_data, info]

    # codec.print_code_table()
    # print(data)
    # print(sys.getsizeof(compressed_data))
    # print(compressed_data)

    with open(file_path, "wb") as file:
        pickle.dump(package, file)
    return file_path

# Decoding
def data_decompress(file_path):
    with open(file_path, "rb") as file:
        loaded_package = pickle.load(file)
    loaded_codec = loaded_package[0]
    loaded_compressed_data = loaded_package[1]
    info = loaded_package[2]
    decompressed_data = loaded_codec.decode(loaded_compressed_data)

    decompressed_data = rl_decode(decompressed_data)

    return decompressed_data, info, os.path.getsize(file_path)
