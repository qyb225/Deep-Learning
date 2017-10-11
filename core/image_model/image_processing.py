import numpy as np
from PIL import Image

def image_to_matrix(image, width = 0, height = 0):
    if width == 0 or height == 0:
        img = image
    else:
        img = image.resize((width, height))
    mat = np.zeros((width * height, 3))

    for i in range(width):
        for j in range(height):
            pix = img.getpixel((i, j))
            mat[i * height + j] = pix
    return mat

def image_to_vector(image, width = 0, height = 0):
    mat = image_to_matrix(image, width, height)
    return mat.reshape((mat.shape[0] * mat.shape[1], 1))

def cut_image(image_addr, image_name, size):
    print ("loading " + image_addr)
    image = Image.open(image_addr)
    img = image.resize(size)
    img.save(image_name, "JPEG")
    print (image_name + " saved")
