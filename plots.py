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
            # Read image
            original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                print(f"Skipping {file_name}: Could not read image.")
                continue
            
            image_rmse = []
            image_bpp = []
            jpeg_rmse = []
            jpeg_bpp = []

            for quality in quality_list:
                image, enc_file = encode(quality, file_path, "jpeg_compression.pkl")
                decompressed_image, bpp = decode(enc_file, "compressed_img.png")
                # Compute RMSE
                rmse = calculate_rmse(image, decompressed_image)
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
    for i in [0, 1]:
        plt.plot(bpp_values[i], rmse_values[i], marker='o', linestyle='-', label="RMSE vs BPP")
        for i, (bpp, rmse) in enumerate(zip(bpp_values[i], rmse_values[i])):
            plt.text(bpp, rmse, f" Q={quality_list[i]}", fontsize=12)
        
        plt.xlabel("Bits Per Pixel (BPP)")
        plt.ylabel("Root Mean Squared Error (RMSE)")
        plt.title(f"RMSE vs BPP for {image_name}")
        plt.grid(True)
        if i == 0:
            plt.legend("My-JPEG")
        else:
            plt.legend("Existing-JPEG")
    plt.show()

# Input parameters
folder_path = "./images"
quality_list = [1, 5, 10, 25, 50, 100]

# Process images and compute metrics
process_images(folder_path, quality_list)


