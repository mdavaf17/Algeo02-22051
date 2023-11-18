import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
import csv


def preprocess_image(image):
    image = ImageOps.grayscale(image)
    dim = max(image.size)
    new_dim = 2 ** int(math.ceil(math.log(dim, 2)))
    
    return ImageOps.pad(image, (new_dim, new_dim))


def create_framework_matrix(a):
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


def contrast(a):
    c = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            c += a[i][j] * (i - j) * (i - j)

    return c


def homogeneity(a):
    h = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            h += a[i][j] / (1 + (i - j) * (i - j))

    return h


def entropy(a):
    e = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if (a[i][j] <= 0):
                continue
            else:
                e += a[i][j] * math.log(a[i][j])

    return (-1) * e


def save_texture_csv():
    dir = "../test/dataset"
    filenames = os.listdir(dir)
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))

    images = []
    for filename in sorted_filenames:
        img = Image.open(os.path.join(dir,filename))
        if img is not None:
            framework_matrix = create_framework_matrix(np.array(preprocess_image(img)))
            symmetric_matrix = framework_matrix + framework_matrix.transpose()
            symmetric_matrix_normalized = symmetric_matrix / symmetric_matrix.sum()

            c, h, e = contrast(symmetric_matrix_normalized), homogeneity(symmetric_matrix_normalized), entropy(symmetric_matrix_normalized)
            images.append([c, h, e])
    
    try:
        with open("../test/db_texture.csv", 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(images)
    except Exception as e:
        print(f"Error: {e}")

def load_texture_csv(filename):
    lines = ""
    with open(filename, 'r') as file:
        lines = file.readlines()

    arrays = []
    for line in lines:
        values = list(map(float, line.strip().split(',')))
        arrays.append(values)
    return arrays


def cosine_similarity(c1, h1, e1, c2, h2, e2):
    cosine = ((c1 * c2) + (h1 * h2) + (e1 * e2)) / (math.sqrt(pow(c1, 2) + pow(h1, 2) + pow(e1, 2)) * math.sqrt(pow(c2, 2) + pow(h2, 2) + pow(e2, 2)))
    
    return cosine * 100


def search_texture(c1, h1, e1):
    db = load_texture_csv("../test/db_texture.csv")
    res60 = {}
    for i in range(len(db)):
        c2, h2, e2 = db[i][0], db[i][1], db[i][2]
        similarity = cosine_similarity(c1, h1, e1, c2, h2, e2)
        if similarity >= 60:
            res60[i] = similarity
    
    sorted_res = dict(sorted(res60.items(), key=lambda item: item[1], reverse=True))
    return sorted_res