import numpy as np

# Differentiation matrix
def differentiation_matrix(n, h=1.0):
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i,i] = -1.0
        D[i, i+1] = 1.0
    return D/h

# Convolutions
def convolve2d(image, kernel):
    H, W = image.shape
    m, n = kernel.shape
    pad_r = m//2
    pad_c = n//2
    padded = np.pad(image, ((pad_r, pad_r), (pad_c, pad_c)), mode='constant')
    output = np.zeros_like(image, dtype=float)
    for i in range(H):
        for j in range(W):
            output[i, j] = np.sum(kernel * padded[i:i+m, j:j+n])
    return output

def convolve2d_separable(image, row_kernel, col_kernel):
    temp = convolve_1d(image, col_kernel.ravel(), axis=0)
    output = convolve_1d(temp, row_kernel.ravel(), axis=1)
    return output

def convolve_1d(image, kernel_1d, axis):
    m = len(kernel_1d)
    pad = m//2
    output = np.zeros_like(image, dtype=float)
    if axis == 0:
        padded = np.pad(image, ((pad, pad), (0, 0)), mode='constant')
        for i in range(image.shape[0]):
            output[i, :] = np.sum(kernel_1d[:, None] * padded[i:i+m, :], axis=0)
    else:
        padded = np.pad(image, ((0, 0), (pad, pad)), mode='constant')
        for j in range(image.shape[1]):
            output[:, j] = np.sum(kernel_1d[None, :] * padded[:, j:j+m], axis=1)
    return output

# Kernels
def gaussian_kernel(size, sigma):
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    return kernel / kernel.sum()

def sobel_kernel():
    Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=float)
    Gy = np.array([[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]], dtype=float)
    return Gx, Gy

def prewitt_kernel():
    Gx = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=float)
    Gy = np.array([[-1,-1,-1],[ 0, 0, 0],[ 1, 1, 1]], dtype=float)
    return Gx, Gy

def scharr_kernel():
    Gx = np.array([[ -3, 0,  3],[-10, 0, 10],[ -3, 0,  3]], dtype=float)
    Gy = np.array([[-3,-10,-3],[ 0,  0,  0],[ 3, 10,  3]], dtype=float)
    return Gx, Gy

def laplacian_kernel():
    return np.array([[0,  1, 0],[1, -4, 1],[0,  1, 0]], dtype=float)

def laplacian_of_gaussian_kernel(size, sigma):
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    kernel = -(1/(np.pi*sigma**4)) * (1 - r2/(2*sigma**2)) * np.exp(-r2/(2*sigma**2))
    kernel -= kernel.mean()
    return kernel

# Gradient and edge magnitude
def gradient(image, kernel="sobel"):
    kernels = {"sobel": sobel_kernel, "prewitt": prewitt_kernel, "scharr": scharr_kernel}
    if kernel not in kernels:
        raise ValueError(f"Unknown kernel: {kernel}. Choose: sobel | prewitt | scharr")
    Kx, Ky = kernels[kernel]()
    Gx = convolve2d(image, Kx)
    Gy = convolve2d(image, Ky)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)
    return magnitude, direction, Gx, Gy

def smooth_then_gradient(image, sigma=1.0, kernel_size=5, deriv_kernel="sobel"):
    G = gaussian_kernel(kernel_size, sigma)
    g_1d = G[kernel_size//2, :]
    smoothed = convolve2d_separable(image, g_1d.reshape(1,-1), g_1d.reshape(-1,1))
    return gradient(smoothed, kernel=deriv_kernel)