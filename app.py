import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.metrics import mean_squared_error

#reconstructing image using first k components
def img_compress(U, S, VT, K):
    U_k = U[:, :K]
    S_k = np.diag(S[:K])
    VT_k = VT[:K, :]

    reconstructed_img = np.dot(U_k, np.dot(S_k, VT_k))
    return reconstructed_img

#load image from sklearn
color_img = load_sample_image("china.jpg")

#convert to grey scale by multiplying from the below value
gray_img = np.dot(color_img[...,:3],[0.299,0.587,0.114])

#performing SVD compression
U, S, VT = np.linalg.svd(gray_img, full_matrices = False)

#compressing the image by using first 50 singular values
k = 50
compressed_img = img_compress(U, S, VT, k)

# Calculate storage size
original_size = gray_img.shape[0] * gray_img.shape[1]
compressed_size = (U.shape[0] * k) + k + (k * VT.shape[1])

compression_ratio = original_size / compressed_size

# Calculate Mean Squared Error
mse = mean_squared_error(gray_img, compressed_img)

print(f"Compression Ratio (k={k}): {compression_ratio:.2f}x")
print(f"Mean Squared Error: {mse:.2f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Grayscale")
plt.imshow(gray_img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Compressed (k={k})")
plt.imshow(compressed_img, cmap='gray')
plt.show()