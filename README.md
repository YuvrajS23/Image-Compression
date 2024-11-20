# JPEG Image Compression Engine

This project implements a simplified version of the JPEG image compression algorithm. The implementation includes the major steps of the JPEG compression pipeline, designed for grayscale images. It calculates RMSE vs. BPP to analyze compression quality and efficiency across multiple images.

## Features
- **2D DCT and IDCT**: Computes and reconstructs image blocks using the Discrete Cosine Transform.
- **Quantization**: Applies a standard JPEG quantization matrix to achieve compression.
- **Huffman Encoding**: Compresses quantized data using Huffman coding.
- **File I/O**: Saves the compressed data to a file and reconstructs the image from it.
- **Quality Analysis**: Supports analysis of compression quality through RMSE and BPP.

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required libraries: 
  - `numpy`
  - `opencv-python`
  - `scipy`
  - `matplotlib`

Install dependencies using:
```bash
pip install numpy opencv-python scipy matplotlib
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/jpeg-compression.git
   cd jpeg-compression
   ```

2. Place the grayscale images you want to test in the project directory.

---

## Usage

1. **Prepare an Image**:
   Ensure your input image is grayscale. You can convert images to grayscale using any image editing software or Python.

2. **Run the Compression Script**:
   ```bash
   python compression.py
   ```

3. **View Outputs**:
   - **Original Image**: Displayed during execution.
   - **Decompressed Image**: Reconstructed from the compressed data.
   - **Compressed File**: Saved as `compressed_data.pkl`.

4. **Analyze Quality**:
   - RMSE and BPP can be calculated for different quality factors by modifying the quantization matrix in the script.

---

## Code Overview

### Main Components:
1. **DCT and IDCT**: Handles the transformation of image blocks.
2. **Quantization**: Compresses the DCT coefficients using a quantization matrix.
3. **Huffman Encoding**:
   - Builds a frequency tree for the quantized data.
   - Compresses the data using variable-length codes.
4. **File I/O**:
   - Saves the encoded data to a file.
   - Reads and reconstructs the compressed image.

### Key Functions:
- `dct2(block)`: Performs 2D DCT on an 8x8 block.
- `idct2(block)`: Performs 2D inverse DCT.
- `jpeg_compress(image, quant_matrix)`: Compresses the image.
- `jpeg_decompress(compressed_blocks, quant_matrix, shape)`: Decompresses the image.

---

## Input/Output

### Input:
- Grayscale image (e.g., `example.jpg`).

### Output:
- **Compressed Data File**: `compressed_data.pkl`.
- **Decompressed Image**: Displayed in a window.

---

## Evaluation
The compression quality can be assessed using:
- **RMSE (Root Mean Squared Error)**: Quantifies the difference between the original and decompressed images.
- **BPP (Bits Per Pixel)**: Measures the compression ratio.

To compute RMSE vs. BPP for multiple images, modify the script to load and process a batch of images, then plot the results using `matplotlib`.

---

## Future Work
- Extend support for color images.
- Implement a user interface for easier interaction.
- Optimize Huffman encoding for faster performance.
- Experiment with different quantization matrices.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author
[Your Name]  
[Your Contact Information]  
Feel free to reach out for feedback or suggestions!# Image-Compression
