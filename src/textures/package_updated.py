import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import math
import csv

def loadImages(dir):
    filenames = os.listdir(dir)
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))

    images = []
    for filename in sorted_filenames:
        img = cv2.imread(os.path.join(dir,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images

images = loadImages("../../test/dataset")
def preprocess_image(image: Image) -> Image:
    """
    Create a square matrix with side length that's a power of 2.
    """
    image = ImageOps.grayscale(image)  # convert to grayscale first so padding is less expensive, might be a large image
    dim = max(image.size)  # Find the largest dimension
    new_dim = 2 ** int(math.ceil(math.log(dim, 2)))  # Find the next power of 2
    return ImageOps.pad(image, (new_dim, new_dim))

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

def symmetric_matrix(a: np.array) -> np.array:
    temp = a + a.transpose()
    return temp / temp.sum()

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

def cosine_similarity(a: np.array, b: np.array) -> float:

    # Contrast
    contrast1 = contrast(a)
    contrast2 = contrast(b)

    # Homogeneity
    homogeneity1 = homogeneity(a)
    homogeneity2 = homogeneity(b)

    # Entropy
    entropy1 = entropy(a)
    entropy2 = entropy(b)

    # Cosine Similarity
    penyebut = (contrast1 * contrast2) + (homogeneity1 * homogeneity2) + (entropy1 * entropy2)
    pembilang = math.sqrt(pow(contrast1, 2) + pow(homogeneity1, 2) + pow(entropy1, 2)) * \
                math.sqrt(pow(contrast2, 2) + pow(homogeneity2, 2) + pow(entropy2, 2))

    cosine_similarity = penyebut / pembilang
    return cosine_similarity * 100

def save_csv(images):
    flattened_arrays = [array.flatten() for array in images]

    # Save the flattened arrays to a CSV file with each element on a separate row
    # with open("../../test/dataset.csv", 'w') as file:
    #     for array in flattened_arrays:
    #         # Convert the array elements to strings and join them with commas
    #         line = ','.join(map(str, array))
    #         file.write(line + '\n')

    # Save the flattened arrays to a CSV file with each element on a separate row
    with open("../../test/dataset.csv", 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(flattened_arrays)

def load_csv(filename):
    lines = ""
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Process each line to convert it into a NumPy array of shape (16, 72)
    arrays = []
    for line in lines:
        values = list(map(int, line.strip().split(',')))
        # Reshape the values into arrays of shape (16, 72)
        array = np.array(values).reshape(16, 72)
        arrays.append(array)
    return arrays

def search(img1):
    res60 = {}
    for i in range(len(db)):
        similarity = cosine_similarity(img1, db[i])
        if similarity >= 60:
            res60[i] = similarity

    sorted_res = dict(sorted(res60.items(), key=lambda item: item[1], reverse=True))
    return sorted_res


for i in range(len(images)):
    images[i] = symmetric_matrix(create_framework_matrix(np.array(preprocess_image(images[i]))))

save_csv(images)
db = load_csv("../../test/dataset.csv")

res = search(images[2])

for number in res:
    file_path = "../../test/dataset/" + str(number) + ".jpg"  # Assuming images are named as 0.jpg, 1.jpg, ...
    img = mpimg.imread(file_path)  # Read the image file
    plt.imshow(img)  # Display the image
    plt.title(f"Image {number}")
    plt.axis('off')  # Hide axes
    plt.show()