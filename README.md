# Image Compression and Dimensionality Reduction Using SVD

A comprehensive Python project designed to ingest image data, apply linear algebra—specifically Singular Value Decomposition (SVD)—and drastically reduce memory footprint while mathematically preserving structural integrity. This project utilizes Python's core scientific stack to simulate real-world mathematical optimization and data compression workflows.

## 💡 Overview

Real-world machine learning systems operate under strict hardware and memory constraints. This project automates the foundational linear algebra workflow of dimensionality reduction. It reads a standard image dataset, strictly correctly strips unnecessary color channels to format a mathematically valid 2D grayscale matrix, performs Singular Value Decomposition, and outputs a reconstructed image using a truncated number of singular values. This proves that massive storage savings can be achieved with minimal mathematical or visual loss.

## ✨ Features

* **Rigorous Data Hygiene:** Safely slices image arrays to extract strictly RGB channels, utilizing dot-product luminosity weights to convert the data into a 2D grayscale matrix, preventing shape mismatches during calculation.
* **Economy-Size Computation:** Enforces `full_matrices=False` during NumPy's SVD calculation. This bypasses the generation of redundant, empty vectors, preventing catastrophic out-of-memory errors on limited hardware.
* **Mathematical Truncation:** Reconstructs the image matrix using only the top $k$ singular values, effectively filtering out low-variance structural data while retaining the core visual patterns.
* **Objective Verification:** Calculates the exact Mean Squared Error (MSE) and storage compression ratio to factually ground the quality-versus-size trade-off.
* **Visual Comparison:** Generates strictly accurate, side-by-side plots to visually verify the fidelity of the compressed mathematical reconstruction against the original.

## 🛠️ Prerequisites

* Python 3.8 or higher
* A standard Python IDE (VS Code, PyCharm) or Jupyter Notebook
* Core Scientific Libraries: `numpy`, `matplotlib`, `scikit-learn`

## 🚀 Usage

1. Clone this repository to your local machine.
2. Open your terminal or command prompt and install the necessary dependencies:

    ```bash
    pip install numpy matplotlib scikit-learn
    ```

3. Launch your Python environment or execute the script directly:

    ```bash
    python svd_compression.py
    ```

4. Adjust the $k$ parameter in the source code to independently verify the mathematical thresholds of image degradation.

## 📊 Expected Output

Upon successful execution, the script will process the raw image array in memory and output fact-grounded terminal metrics alongside an analytical visualization, including:

1. **Matrix Diagnostics:** Terminal output verifying the dimensions of the original image, and the resulting $U$, $\Sigma$, and $V^T$ matrices to ensure correct factorization.
2. **Objective Metrics:** Terminal output stating the exact calculated Mean Squared Error (MSE) and the multiplier for the compression ratio based on your chosen $k$ value.
3. **Visual Comparison Window:** A side-by-side Matplotlib pop-up displaying the original high-fidelity grayscale matrix next to the truncated reconstruction.

## 🧩 How It Works (Under the Hood)

This script serves as a practical application of foundational linear algebra using NumPy:

1. **File I/O & Preprocessing:** The script loads a standard test image using `sklearn.datasets.load_sample_image()`. It transforms the 3D RGB array into a strictly 2D grayscale matrix using standard mathematical luminosity weights.
2. **Linear Algebra Computation:** `np.linalg.svd()` factorizes the image matrix into orthonormal bases ($U$, $V^T$) and a 1D array of descending singular values ($\Sigma$), strictly bounded by the memory-efficient economy-size optimization.
3. **Matrix Truncation & Reconstruction:** The algorithm dynamically isolates the first $k$ columns of $U$, the top $k \times k$ diagonal block of $\Sigma$, and the top $k$ rows of $V^T$. It then computes the nested dot products (`np.dot()`) to build the compressed approximation matrix.
4. **Synthesis:** Scikit-Learn's metrics module calculates the error rate between the original and reconstructed matrices, while Matplotlib synthesizes the mathematical output into a human-readable visual report.