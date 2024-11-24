import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# (x,y) -> (x,y)
def grad_x(I):
    return cv.filter2D(I, cv.CV_64F, np.array([[1,-1]]))


# (x,y) -> (x,y)
def grad_y(I):
    return cv.filter2D(I, cv.CV_64F, np.array([[1],[-1]]))


# (x,y) -> (x,y,2)
def grad(I):
    return np.stack([grad_x(I), grad_y(I)], axis=2)


def diffusivity_function(s, l):
    return l * (s**2 + l**2)**(-0.5)


# (x,y) -> (x,y,2,2)
def diffusivity_tensor(I, sigma, l):
    smoothed = cv.filter2D(I, cv.CV_64F, cv.getGaussianKernel(3, sigma, cv.CV_64F))

    grad_smoothed = grad(smoothed)

    grad_smoothed = grad_smoothed + np.finfo(np.float64).eps

    grad_smoothed = grad_smoothed * (((grad_smoothed[:,:,0]**2 + grad_smoothed[:,:,1]**2))**(-0.5))[:,:,np.newaxis]

    perp_grad_smoothed = np.stack((-grad_smoothed[:,:,1], grad_smoothed[:,:,0]), axis=2)

    eig1 = diffusivity_function(grad_smoothed[:,:,0]**2 + grad_smoothed[:,:,1]**2, l)[:,:,np.newaxis,np.newaxis]
    eig2 = np.ones((I.shape[0], I.shape[1], 1, 1))

    grad_smoothed_vvT = eig1 * (grad_smoothed[:,:,:,np.newaxis] * grad_smoothed[:,:,np.newaxis,:]) + eig2 * (perp_grad_smoothed[:,:,:,np.newaxis] * perp_grad_smoothed[:,:,np.newaxis,:])

    return grad_smoothed_vvT


# (x,y) -> (x,y)
def eed(I, sigma, l):
    D = diffusivity_tensor(I, sigma, l)

    D_grad = (D @ (grad(I)[:,:,:,np.newaxis]))[:,:,:,0]
    # D_grad = grad(I)
    D_grad_x = D_grad[:,:,0]
    D_grad_y = D_grad[:,:,1]

    div_D_grad = grad_x(D_grad_x) + grad_y(D_grad_y)

    return div_D_grad


def perform_diffusion(values, shape, sigma, lamb, diffusion_cutoff, step):
    mask = np.ones(shape, dtype=np.float64)
    I = np.zeros(shape, dtype=np.float64)
    for pos, val in values.items():
        mask[pos[0],pos[1]] = 0.0
        I[pos[0],pos[1]] = val

    for _ in range(50):
        e = step * eed(I, sigma, lamb)
        ne = rmse(e) / rmse(I)
        if ne < diffusion_cutoff:
            break
        I = I + e
    plt.imshow(np.floor(I * 256), cmap='gray')
    plt.show()

    return I


def rmse(I):
    return (np.mean(I**2))**0.5


def get_boundary(shape):
    s = set()
    for i in range(shape[0]):
        s.add((i, 0))
        s.add((i,shape[1]-1))
    for i in range(shape[1]):
        s.add((0,i))
        s.add((shape[0]-1,i))
    return s

def split(I):
    shape = I.shape
    if shape[0] > shape[1]:
        return I[:(shape[0]//2+1),:], I[(shape[0]//2):,:], 0
    else:
        return I[:,:(shape[1]//2+1)], I[:,(shape[1]//2):], 1


n = 0


def perform_splits(I, split_cutoff, sigma, lamb, diffusion_cutoff, step):
    global n

    if (I.shape[0] <= 3 or I.shape[1] <= 3):
        n = n + I.shape[0]*I.shape[1]
        return (I,)

    s = get_boundary(I.shape)
    values = dict([(x,I[x]) for x in s])

    img = perform_diffusion(values, I.shape, sigma, lamb, diffusion_cutoff, step)

    err = rmse(img - I)
    if err > split_cutoff:
        img1, img2, a = split(I)
        return (perform_splits(img1, split_cutoff, sigma, lamb, diffusion_cutoff, step), perform_splits(img2, split_cutoff, sigma, lamb, diffusion_cutoff, step), a)
    else:
        n = n + s.size()
        # print(values, I.shape)
        return (values, I.shape)


def populate(res, sigma, lamb, diffusion_cutoff, step):
    if len(res) == 1:
        return res[0]
    elif len(res) == 2:
        return perform_diffusion(res[0], res[1], sigma, lamb, diffusion_cutoff, step)
    else:
        z1 = populate(res[0], sigma, lamb, diffusion_cutoff, step)
        z2 = populate(res[1], sigma, lamb, diffusion_cutoff, step)
        if res[2] == 0:
            return np.concatenate((z1[:-1,:],z2),axis=0)
        else:
            return np.concatenate((z1[:,:-1],z2),axis=1)


if __name__ == "__main__":
    I = np.float64(cv.imread("barbara256.png", cv.IMREAD_GRAYSCALE) / 256.0)
    # I = 0.9 * np.ones((256, 256), dtype=np.float64)

    split_cutoff = 2

    sigma = 1.0
    lamb = 0.0018
    diffusion_cutoff = 0.0001
    step = 0.1

    res = perform_splits(I, split_cutoff, sigma, lamb, diffusion_cutoff, step)

    recon = populate(res, sigma, lamb, diffusion_cutoff, step)

    print(n)

    plt.imshow(np.floor(recon * 256), cmap='gray')
    plt.show()

    print(rmse(I - recon))
