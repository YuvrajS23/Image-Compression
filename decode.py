from encode import *
from huffman import huffman_decompress

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
    # print(decompressed_blocks[0:100])
    Q, height, width, padded_shape, isColor = info
    ph, pw = padded_shape
    l = ph*pw
    if isColor:
        qmY = quant_matrix * (50/Q)
        qmC = qmY * 2
        # assert(len(decompressed_blocks) == 2*height*width)
        y = decompressed_blocks[0: l]
        color = decompressed_blocks[l :]
        cb = color[:len(color)//2]
        cr = color[len(color)//2:]
        y = y.reshape(-1, 64)
        cb = cb.reshape(-1, 64)
        cr = cr.reshape(-1, 64)
        decompressed_y = jpeg_decompress(y, qmY, (ph, pw))
        # print(decompressed_y[0:100])
        decompressed_cb = jpeg_decompress(cb, qmC, (ph // 2, pw // 2))
        decompressed_cr = jpeg_decompress(cr, qmC, (ph // 2, pw // 2))
        # Upsample Cb and Cr back to original size
        decompressed_cb = upsampling_channel(decompressed_cb)
        decompressed_cr = upsampling_channel(decompressed_cr)
        # Merge channels and convert back to RGB
        decompressed_y = decompressed_y + 128
        decompressed_cr = decompressed_cr + 128
        decompressed_cb = decompressed_cb + 128
        ycbcr = np.concatenate((decompressed_y.reshape(ph, pw, 1), decompressed_cb.reshape(ph, pw, 1), decompressed_cr.reshape(ph, pw, 1)), axis=2).astype(np.uint8)
        decompressed_image = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
        cv2.imwrite(f"{out}", decompressed_image)
        print("Size of Compressed image:", sz)
        print("Bits Per Pixel (BPP):", (sz*8)/(height*width))
        decompressed_image = remove_equal_padding(decompressed_image, (height, width))
        if show:
            plt.imshow(decompressed_image)
            plt.show()
        return decompressed_image

    else:
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