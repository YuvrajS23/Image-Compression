from decode import *

#JPEG Compression
def jpeg_comp(file_path, quality):
    image = cv.imread(file_path)
    cv2.imwrite('jpeg_comp.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    comp = cv.imread('jpeg_comp.jpg')
    sz = os.path.getsize('jpeg_comp.jpg')
    rmse = calculate_rmse(image, comp)
    bpp = calculate_bpp(sz, image.shape)
    return rmse, bpp

# Process images in the folder and compute RMSE and BPP for each Q
def process_images(folder_path, quality_list):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            image_rmse = []
            image_bpp = []
            jpeg_rmse = []
            jpeg_bpp = []

            for quality in quality_list:
                image, enc_file = encode(quality, file_path, "jpeg_compression.pkl")
                decompressed_image, sz = decode(enc_file, "compressed_img.png")
                # Compute RMSE
                rmse = calculate_rmse(image, decompressed_image)
                # Compute BPP
                bpp = calculate_bpp(sz, image.shape)
                rmse_jpeg, bpp_jpeg = jpeg_comp(file_path, quality)
                image_rmse.append(rmse)
                image_bpp.append(bpp)
                jpeg_rmse.append(rmse_jpeg)
                jpeg_bpp.append(bpp_jpeg)

            # Plot RMSE vs BPP
            plot_rmse_vs_bpp(file_name, quality_list, [image_rmse, jpeg_rmse], [image_bpp, jpeg_bpp])


# Plot RMSE vs BPP for each image
def plot_rmse_vs_bpp(image_name, quality_list, rmse_values, bpp_values):
    plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots()
    for i in [0, 1]:
        ax.plot(bpp_values[i], rmse_values[i], marker='o', linestyle='-', label=("My JPEG" if i == 0 else "Existing JPEG"))
        for i, (bpp, rmse) in enumerate(zip(bpp_values[i], rmse_values[i])):
            ax.text(bpp, rmse, f" Q={quality_list[i]}", fontsize=12)

        ax.set_xlabel("Bits Per Pixel (BPP)")
        ax.set_ylabel("Root Mean Squared Error (RMSE)")
        ax.set_title(f"RMSE vs BPP for {image_name}")
        ax.grid(True)
        ax.legend(loc="upper right")
    fig.savefig(f"plot_rmse_vs_bpp_'{image_name}'.png")

# Mean Compression ratio for a given quality factor
def mean_compression_ratio(folder_path, quality):
    sum_original = 0
    sum_compressed = 0
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            original_size = os.path.getsize(file_path)
            image, enc_file = encode(quality, file_path, "jpeg_compression.pkl")
            decompressed_image, sz = decode(enc_file, "compressed_img.png")
            sum_original += original_size
            sum_compressed += sz
    return sum_original / sum_compressed


# Plot MCR vs Quality Factor
def plot_mcr_vs_quality(folder_path, quality_list):
    mcrs = []
    for q in quality_list:
        mcr = mean_compression_ratio(folder_path, q)
        mcrs.append(mcr)
    plt.figure(figsize=(8, 5))
    plt.plot(quality_list, mcrs, marker='o', linestyle='-', label="MCR v Q")
    plt.xlabel("Quality Factor")
    plt.ylabel("Mean Compression Ratio")
    plt.title("Mean Compression Ratio vs Quality Factor")
    plt.savefig("plot_mcr_vs_q_NET.png")

# Input parameters
folder_path = "./images/NET"
quality_list = [1, 5, 10, 25, 50, 100]

# Process images and compute metrics
# process_images(folder_path, quality_list)
plot_mcr_vs_quality(folder_path, quality_list)


