import numpy as np
from PIL import Image, ImageOps
import math


def preprocess_image(image: Image) -> Image:
    """
    Create a square matrix with side length that's a power of 2.
    """
    image = ImageOps.grayscale(image)  # convert to grayscale first so padding is less expensive, might be a large image
    dim = max(image.size)  # Find the largest dimension
    new_dim = 2 ** int(math.ceil(math.log(dim, 2)))  # Find the next power of 2
    return ImageOps.pad(image, (new_dim, new_dim))

def get_haar_step(i: int, k: int) -> np.ndarray:
    transform = np.zeros((2 ** k, 2 ** k))
    # Averages (1)
    for j in range(2 ** (k - i - 1)):
        transform[2 * j, j] = 0.5
        transform[2 * j + 1, j] = 0.5
    # Details (2)
    offset = 2 ** (k - i - 1)
    for j in range(offset):
        transform[2 * j, offset + j] = 0.5
        transform[2 * j + 1, offset + j] = -0.5
    # Identity (3)
    for j in range(2 ** (k - i), 2 ** k):
        transform[j, j] = 1
    return transform

def get_haar_transform(k: int) -> np.ndarray:
    transform = np.eye(2 ** k)
    for i in range(k):
        transform = transform @ get_haar_step(i, k)
    return transform

def haar_encode(a: np.ndarray) -> np.ndarray:
    k = int(np.ceil(np.log2(len(a))))
    #assert a.shape == (k, k)
    row_encoder = get_haar_transform(k)
    return row_encoder.T @ a @ row_encoder

def haar_decode(a: np.ndarray) -> np.ndarray:
    k = int(np.ceil(np.log2(len(a))))
    #assert a.shape == (k, k)
    row_decoder = np.linalg.inv(get_haar_transform(k))
    return row_decoder.T @ a @ row_decoder

def truncate_values(a: np.ndarray, threshold: float):
    return np.where(np.abs(a) < threshold, 0, a)

'''
im = Image.open('Algeo02-22051/test/dataset/11.jpg')
img = preprocess_image(im)
A = np.array(img)
print(f"length: {len(A)}")
print(f"width: {len(A[0])}")
print(f"shape: {A.shape}")
print(int(np.ceil(np.log2(len(A)))))

print(A)
E = haar_encode(A)
print(E)
print("=============")
D = haar_decode(E)
print(D)
'''
