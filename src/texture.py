import numpy as np
from PIL import Image, ImageOps
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


def dissimilarity(a: np.array) -> float:
    d = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            d += a[i][j] * abs(i - j)

    return d


def asm_val(a: np.array) -> float:
    asm = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            asm += a[i][j] * a[i][j]

    return asm


def correlation(a: np.array) -> float:
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4153941/
    co = 0
    mu_x = 0
    mu_y = 0
    sigma_x = 0
    sigma_y = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            mu_x += i * a[i][j]
            mu_y += j * a[i][j]

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            sigma_x += (i - mu_x) * (i - mu_x) * a[i][j]
            sigma_y += ((j - mu_y) * (j - mu_y) * a[i][j])

    sigma_x = math.sqrt(sigma_x)
    sigma_y = math.sqrt(sigma_y)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if((sigma_x * sigma_y) <= 0):
                continue
            else:
                co += ((i - mu_x) * (j - mu_y) * a[i][j]) / (sigma_x * sigma_y)

    return co


def inverse_contrast(a: np.array) -> float:
    ic = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if((i - j) <= 0):
                continue
            else:
                ic += a[i][j] / ((i - j) * (i - j))

    return ic


def distortion(a: np.array) -> float:
    dt = 0

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            dt += a[i][j] * ((i - j) * (i - j) * (i - j))

    return dt


def save_texture_csv():
    if os.path.exists("../test/db_texture.csv"):
        os.remove("../test/db_texture.csv")

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

            c, h, e, d, asm, en, co, ic, dt = contrast(symmetric_matrix_normalized), homogeneity(symmetric_matrix_normalized), entropy(symmetric_matrix_normalized), dissimilarity(symmetric_matrix_normalized), asm_val(symmetric_matrix_normalized), math.sqrt(asm_val(symmetric_matrix_normalized)), correlation(symmetric_matrix_normalized), inverse_contrast(symmetric_matrix_normalized), distortion(symmetric_matrix_normalized)
            images.append([c, h, e, d, asm, en, co, ic, dt])
    
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


def cosine_similarity_texture(c1, h1, e1, d1, asm1, en1, co1, ic1, dt1, c2, h2, e2, d2, asm2, en2, co2, ic2, dt2):
    cosine = ((c1 * c2) + (h1 * h2) + (e1 * e2) + (d1 * d2) + (asm1 * asm2) + (en1 * en2) + (co1 * co2) + (ic1 * ic2) + (dt1 * dt2)) / (math.sqrt(pow(c1, 2) + pow(h1, 2) + pow(e1, 2) + pow(d1, 2) + pow(asm1, 2) + pow(en1, 2) + pow(co1, 2) + pow(ic1, 2) + pow(dt1, 2)) * math.sqrt(pow(c2, 2) + pow(h2, 2) + pow(e2, 2)  + pow(d2, 2) + pow(asm2, 2) + pow(en2, 2) + pow(co2, 2) + pow(ic2, 2) + pow(dt2, 2)))

    return cosine * 100


def search_texture(c1, h1, e1, d1, asm1, en1, co1, ic1, dt1):
    db = load_texture_csv("../test/db_texture.csv")
    res60 = {}
    for i in range(len(db)):
        c2, h2, e2, d2, asm2, en2, co2, ic2, dt2 = db[i][0], db[i][1], db[i][2], db[i][3], db[i][4], db[i][5], db[i][6], db[i][7], db[i][8]
        similarity = cosine_similarity_texture(c1, h1, e1, d1, asm1, en1, co1, ic1, dt1, c2, h2, e2, d2, asm2, en2, co2, ic2, dt2)
        if similarity > 60:
            res60[i] = similarity

    sorted_res = dict(sorted(res60.items(), key=lambda item: item[1], reverse=True))
    return sorted_res