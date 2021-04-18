from PIL import Image
import numpy as np


def main():
    image = Image.open('C:\\eiffel-tower.jpg')
    red, green, blue = get_color_matrices(image)
    redSVD = np.linalg.svd(red)
    blueSVD = np.linalg.svd(blue)
    greenSVD = np.linalg.svd(green)
    errors = dict()
    k_levels = [2, 50, 100]
    for k in k_levels:
        rk = get_best_approx(k, redSVD)
        bk = get_best_approx(k, blueSVD)
        gk = get_best_approx(k, greenSVD)
        create_image(image, rk, bk, gk, k)
        errors[k] = get_error(redSVD[1], k)
    for k in errors:
        print(k, errors[k])


def create_image(image, red, blue, green, k):
    pix = np.array(image)
    new_image = np.zeros((pix.shape[0], pix.shape[1], 3))
    new_image[:, :, 0] = red
    new_image[:, :, 1] = green
    new_image[:, :, 2] = blue
    new_image = Image.fromarray(new_image.astype('uint8'))
    new_image.save(str(k)+".png")


def get_color_matrices(image):
    pix = np.array(image)
    red = pix[:, :, 0]
    blue = pix[:, :, 1]
    green = pix[:, :, 2]
    return red, blue, green


def get_best_approx(k, SVD):
    S = np.zeros((SVD[0].shape[0], SVD[2].shape[0]))
    for i in range(k):
        S[i][i] = SVD[1][i]
    Ak = np.matmul(S, SVD[2])
    Ak = np.matmul(SVD[0], Ak)
    return Ak


def get_error(singular, k):
    singular = square(singular)
    return sum(singular[k+1:])/sum(singular)


def square(list1):
    return [i ** 2 for i in list1]


if __name__ == "__main__":
    main()
