import numpy as np
from HaarWaveletTransform import preprocess_image
import matplotlib.pyplot as plt
from PIL import Image
import math


def create_framework_matrix(a: np.array) -> np.array:
    framework_matrix = [[0 for x in range(256)] for x in range(256)]
    framework_matrix = np.array(framework_matrix)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if j == 0:
                continue
            else:
                prev = a[i][j - 1]
                current = a[i][j]
                framework_matrix[prev][current] += 1

    return framework_matrix


def contrast(a: np.array) -> int:
    c = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            c += a[i][j] * (i - j) * (i - j)

    return c


def homogeneity(a: np.array) -> int:
    h = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            h += a[i][j] / (1 + (i - j) * (i - j))

    return h


def entropy(a: np.array) -> int:
    e = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if (a[i][j] <= 0):
                continue
            else:
                e += a[i][j] * math.log(a[i][j])

    return (-1) * e


img1 = Image.open('Algeo02-22051/test/dataset/99.jpg')
plt.imshow(img1)
plt.show()

img2 = Image.open('Algeo02-22051/test/dataset/100.jpg')
plt.imshow(img2)
plt.show()

processed1 = preprocess_image(img1)
numpydata1 = np.array(processed1)

processed2 = preprocess_image(img2)
numpydata2 = np.array(processed2)


framework_matrix1 = create_framework_matrix(numpydata1)
framework_matrix2 = create_framework_matrix(numpydata2)

symmetric_matrix1 = framework_matrix1 + framework_matrix1.transpose()
symmetric_matrix2 = framework_matrix2 + framework_matrix2.transpose()

symmetric_matrix_normalized1 = symmetric_matrix1 / symmetric_matrix1.sum()
symmetric_matrix_normalized2 = symmetric_matrix2 / symmetric_matrix2.sum()

# Contrast
contrast1 = contrast(symmetric_matrix_normalized1)
contrast2 = contrast(symmetric_matrix_normalized2)

# Homogeneity
homogeneity1 = homogeneity(symmetric_matrix_normalized1)
homogeneity2 = homogeneity(symmetric_matrix_normalized2)

# Entropy
entropy1 = entropy(symmetric_matrix_normalized1)
entropy2 = entropy(symmetric_matrix_normalized2)

# Cosine Similarity
penyebut = (contrast1 * contrast2) + (homogeneity1 * homogeneity2) + (entropy1 * entropy2)
pembilang = math.sqrt(pow(contrast1, 2) + pow(homogeneity1, 2) + pow(entropy1, 2)) * \
            math.sqrt(pow(contrast2, 2) + pow(homogeneity2, 2) + pow(entropy2, 2))
cosine_similarity = penyebut / pembilang

print("=================")
print(f'cosine_similarity: {cosine_similarity * 100} %')
