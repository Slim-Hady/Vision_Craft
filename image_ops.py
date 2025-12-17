# image_ops.py

import numpy as np
from skimage import io as skio, img_as_float, img_as_ubyte, exposure, color  # color & exposure from lecture.[file:33]
from skimage.util import random_noise
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter

# ---------- Basic helpers ----------

def load_image(file_storage):
    """Load image from Flask FileStorage to float [0,1]."""
    img = skio.imread(file_storage)
    img = img_as_float(img)
    return img

def to_uint8(img):
    """Convert float [0,1] or any numeric array to uint8."""
    img = np.clip(img, 0, 1)
    return img_as_ubyte(img)

# ---------- Grayscale ----------

def to_grayscale(img):
    """
    Convert RGB image to grayscale float [0,1].
    Uses skimage.color.rgb2gray like in the histogram lecture code.
    """
    if img.ndim == 2:
        return img
    gray = color.rgb2gray(img)
    return gray

# ---------- Noise ----------

def add_gaussian_noise(img, mean=0.0, sigma=0.05):
    """
    Add Gaussian noise with given mean and sigma (variance = sigma^2).
    Similar concept to random_noise in skimage.
    """
    noisy = random_noise(img, mode='gaussian', mean=mean, var=sigma**2)
    return noisy

def add_salt_pepper_noise(img, amount=0.02, s_vs_p=0.5):
    """
    Add salt & pepper noise with given amount and salt/pepper ratio.
    """
    noisy = random_noise(img, mode='s&p', amount=amount, salt_vs_pepper=s_vs_p)
    return noisy

# ---------- Filters ----------

def apply_median_filter(img, size=3):
    """
    Apply median filter of given kernel size.
    """
    if img.ndim == 3:
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:, :, c] = median_filter(img[:, :, c], size=size)
    else:
        result = median_filter(img, size=size)
    return result

def apply_average_filter(img, size=3):
    """
    Apply average (mean) filter using uniform_filter.
    """
    if img.ndim == 3:
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:, :, c] = uniform_filter(img[:, :, c], size=size)
    else:
        result = uniform_filter(img, size=size)
    return result

# ---------- Gamma correction ----------

def apply_gamma_correction(img, gamma=1.0):
    """
    Power-law (gamma) transformation s = c * r^gamma as in lecture 5.
    """
    img_f = img_as_float(img)
    c = 1.0
    out = c * (img_f ** gamma)
    out = np.clip(out, 0, 1)
    return out

# ---------- Histogram equalization ----------

def apply_hist_equalization(img):
    """
    Histogram equalization on grayscale version of image using skimage.exposure.equalize_hist.
    """
    gray = to_grayscale(img)
    equalized = exposure.equalize_hist(gray)
    return equalized

def compute_histogram(img):
    """
    Return histogram (counts) and bin edges for grayscale image (0..1).
    Uses np.histogram similar to lab/lecture histogram examples.
    """
    gray = to_grayscale(img)
    hist, bins = np.histogram(gray.ravel(), bins=256, range=(0.0, 1.0))
    return hist.tolist(), bins.tolist()

# ---------- Simple fusion (if you used it) ----------

def simple_average_fusion(img1, img2):
    """
    Simple pixel-wise average fusion of two same-size images.
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    fused = 0.5 * img1 + 0.5 * img2
    fused = np.clip(fused, 0, 1)
    return fused

def weighted_fusion(img1, img2, alpha=0.5):
    """
    Weighted fusion alpha*img1 + (1-alpha)*img2.
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    fused = alpha * img1 + (1 - alpha) * img2
    fused = np.clip(fused, 0, 1)
    return fused
