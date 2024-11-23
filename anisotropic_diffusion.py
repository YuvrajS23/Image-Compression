import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# (x,y) -> (x,y)
def grad_x(I):
    return cv.filter2D(I, cv.CV_32F, np.array([[1,0,-1]]))


# (x,y) -> (x,y)
def grad_y(I):
    return cv.filter2D(I, cv.CV_32F, np.array([[1],[0],[-1]]))


# (x,y) -> (x,y,2)
def grad(I):
    return np.stack([grad_x(I), grad_y(I)], axis=2)


def diffusivity_function(s, l):
    return (1 + (s / l)**2)**(-0.5)


# (x,y) -> (x,y,2,2)
def diffusivity_tensor(I, sigma, l):
    smoothed = cv.filter2D(I, cv.CV_32F, cv.getGaussianKernel(5, sigma, cv.CV_32F))
    grad_smoothed = grad(smoothed)
    grad_smoothed_vvT = grad_smoothed[:,:,np.newaxis,:] * grad_smoothed[:,:,:,np.newaxis]

    g_grad_smoothed_vvT = diffusivity_function(grad_smoothed_vvT, l)
    return g_grad_smoothed_vvT


# (x,y) -> (x,y)
def eed(I, sigma, l):
    D = diffusivity_tensor(I, sigma, l)

    D_grad = (D @ (grad(I)[:,:,:,np.newaxis]))[:,:,:,0]
    D_grad_x = D_grad[:,:,0]
    D_grad_y = D_grad[:,:,1]

    div_D_grad = grad_x(D_grad_x) + grad_y(D_grad_y)

    return div_D_grad


def perform_diffusion(shape, values, rmse_cutoff, step):
    sigma = 10.0
    lamb = 0.88

    mask = np.ones(shape, dtype=np.float32)
    I = np.zeros(shape, dtype=np.float32)
    for pos, val in values.items():
        mask[pos] = 0.0
        I[pos] = val

    while True:
        e = step * mask * eed(I, sigma, lamb)
        ne = np.linalg.norm(e) / np.linalg.norm(I)
        if ne < rmse_cutoff:
            break
        I = I + e

    return I


def get_boundary(shape):
    return set([
        (0,0),
        (shape[0]-1,shape[1]-1),
        (0,shape[1]-1),
        (shape[0]-1,0),
        (shape[0]//2-1,0),
        (0,shape[1]//2-1),
        (shape[0]-1,shape[1]//2-1),
        (shape[0]//2-1,shape[1]-1)
    ])


def split(I):
    shape = I.shape
    if shape[0] > shape[1]:
        return I[:(shape[0]//2+1),:], I[(shape[0]//2):,:], 0
    else:
        return I[:,:(shape[1]//2+1)], I[:,(shape[1]//2):], 1


def perform_splits(I, split_cutoff, diffusion_cutoff):
    img_repr = ()

    s = get_boundary(I.shape)
    values = dict([(x,I[x]) for x in s])

    if (I.shape[0] <= 3 or I.shape[1] <= 3):
        return (values,I.shape)

    img = perform_diffusion(I.shape, values, diffusion_cutoff, 0.1)

    err = np.linalg.norm(img - I)
    if err > split_cutoff:
        img1, img2, a = split(I)
        print(f"split {I.shape} {err}")
        return (perform_splits(img1, split_cutoff, diffusion_cutoff), perform_splits(img2, split_cutoff, diffusion_cutoff), a)
    else:
        return (values, I.shape)


def populate(res):
    if len(res) == 2:
        z = np.zeros(res[1])
        for p, v in res[0].items():
            z[p] = v
        return z
    else:
        print(res)
        z1 = populate(res[0])
        z2 = populate(res[1])
        print(z1.shape, z2.shape, res[2])
        return np.concatenate((z1,z2),axis=res[2])


if __name__ == "__main__":
    I = np.float32(cv.imread("barbara256.png", cv.IMREAD_GRAYSCALE) / 256.0)
    # I = 0.9 * np.ones((256, 256), dtype=np.float32)

    split_cutoff = 1.0

    res = perform_splits(I, split_cutoff, 0.01)

    print(res)

    recon = populate(res)

    plt.imshow(np.floor(I * 256), cmap='gray')
    plt.show()
