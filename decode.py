import cv2 as cv
from encode import *
from huffman import huffman_decompress
from utils import quant_matrix

def decode(input, out):
    # Decompression
    decompressed_blocks, info, sz = huffman_decompress(input)
    decompressed_blocks = np.array(decompressed_blocks).astype(np.float32)

    Q, height, width, padded_shape, isColor = info
    ph, pw = padded_shape

    if isColor:
        qmY = quant_matrix * (50/Q)
        qmC = qmY * 2

        dy = (ph, pw)
        dcr = (ph//2, pw//2)
        dcb = (ph//2, pw//2)

        y = decompressed_blocks[:(dy[0]*dy[1])]
        cr = decompressed_blocks[(dy[0]*dy[1]):(dy[0]*dy[1]+dcr[0]*dcr[1])]
        cb = decompressed_blocks[(dy[0]*dy[1]+dcr[0]*dcr[1]):]

        y = y.reshape(-1, 64)
        cr = cr.reshape(-1, 64)
        cb = cb.reshape(-1, 64)

        y = jpeg_decompress(y, qmY, dy)
        cr = jpeg_decompress(cr, qmC, dcr)
        cb = jpeg_decompress(cb, qmC, dcb)

        # Upsample Cb and Cr back to original size
        cr = upsampling_channel(cr)
        cb = upsampling_channel(cb)

        # Merge channels and convert back to RGB
        y = y + 128
        cr = cr + 128
        cb = cb + 128

        ycrcb = np.concatenate((y.reshape(ph, pw, 1), cr.reshape(ph, pw, 1), cb.reshape(ph, pw, 1)), axis=2).astype(np.uint8)
        decompressed_image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)
        decompressed_image = remove_equal_padding(decompressed_image, (height, width))

        cv.imwrite(f"{out}", decompressed_image)
        print("Size of Compressed image:", sz)
        print("Bits Per Pixel (BPP):", (sz*8)/(height*width))

        return decompressed_image

    else:
        qm = quant_matrix * (50/Q)
        decompressed_blocks = decompressed_blocks.reshape(-1, 64)
        decompressed_image = jpeg_decompress(decompressed_blocks, qm, (height, width))
        decompressed_image = decompressed_image + 128
        cv.imwrite(f"{out}", decompressed_image)
        print("Size of Compressed image:", sz)
        print("Bits Per Pixel (BPP):", (sz*8)/(height*width))
        return decompressed_image
