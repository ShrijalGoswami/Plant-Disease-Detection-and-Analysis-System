"""
Experiment 1 : Image preprocessing and edge detection.
Input  : a PlantVillage grape leaf with black-rot lesions and the classical lena image.
Steps  : colour conversion, resizing, histogram equalisation / CLAHE, smoothing filters
         on noisy images (with PSNR), unsharp-mask sharpening, morphological opening and
         closing, then Sobel / Prewitt / Roberts / Laplacian / LoG / Canny edge detectors.
Output : outputs/exp01/fig01 ... fig05 and printed statistics.
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "outputs/exp01"
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)

LEAF = "../data/plantvillage/Grape___Black_rot/00905d40-bddf-460e-b348-1dbb6a34653b___FAM_B.Rot 0664.JPG"


def show_grid(images, titles, rows, cols, fname, size=3.2, cmap="gray"):
    """Save a grid of images with titles (RGB or single channel)."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    for ax, im, t in zip(axes.ravel(), images, titles):
        ax.imshow(im, cmap=None if im.ndim == 3 else cmap)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 1. preprocessing
bgr = cv2.imread(LEAF)                                  # OpenCV loads as BGR
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)              # convert for display / models
print("Original leaf image shape:", bgr.shape, "dtype:", bgr.dtype)
rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
print("Resized to:", rgb.shape, " grayscale mean = %.1f, std = %.1f" % (gray.mean(), gray.std()))

equ = cv2.equalizeHist(gray)                            # global histogram equalisation
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
unsharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)     # sharpen = g + k (g - blur(g))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
for name, im in [("gray", gray), ("equalised", equ), ("CLAHE", clahe)]:
    print("%-10s mean = %6.1f  std = %5.1f  contrast (p95-p5) = %3d"
          % (name, im.mean(), im.std(), np.percentile(im, 95) - np.percentile(im, 5)))

show_grid([rgb, gray, equ, clahe, blur, unsharp, opening, closing],
          ["Original (RGB, 512x512)", "Grayscale", "Histogram equalised", "CLAHE (clip 2, 8x8)",
           "Gaussian blur 7x7", "Unsharp mask sharpening", "Morphological opening", "Morphological closing"],
          2, 4, "fig01_preprocessing_grid.png")

# ---------------------------------------------------------------- 2. histograms
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for j, (im, t) in enumerate([(gray, "Grayscale"), (equ, "Histogram equalised"), (clahe, "CLAHE")]):
    axes[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
    axes[0, j].set_title(t)
    axes[0, j].axis("off")
    axes[1, j].hist(im.ravel(), bins=256, range=(0, 255), color="steelblue")
    axes[1, j].set_title("Histogram: " + t)
    axes[1, j].set_xlabel("Intensity")
    axes[1, j].set_ylabel("Pixel count")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig02_histograms.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 3. noise and smoothing
gauss_noisy = np.clip(gray.astype(np.float32) + np.random.normal(0, 20, gray.shape), 0, 255).astype(np.uint8)
sp_noisy = gray.copy()
mask = np.random.rand(*gray.shape)
sp_noisy[mask < 0.03] = 0                               # pepper
sp_noisy[mask > 0.97] = 255                             # salt
rows_img, rows_title = [], []
for noisy, label in [(gauss_noisy, "Gaussian noise (sigma 20)"), (sp_noisy, "Salt & pepper (6%)")]:
    filtered = {"Gaussian 5x5": cv2.GaussianBlur(noisy, (5, 5), 1.2),
                "Median 5x5": cv2.medianBlur(noisy, 5),
                "Bilateral d=9": cv2.bilateralFilter(noisy, 9, 60, 7)}
    rows_img.append(noisy)
    rows_title.append("%s\nPSNR = %.2f dB" % (label, cv2.PSNR(gray, noisy)))
    for k, v in filtered.items():
        p = cv2.PSNR(gray, v)
        print("%-26s -> %-14s PSNR = %.2f dB" % (label, k, p))
        rows_img.append(v)
        rows_title.append("%s\nPSNR = %.2f dB" % (k, p))
show_grid(rows_img, rows_title, 2, 4, "fig03_denoising_psnr.png")

# ---------------------------------------------------------------- 4. edge detectors
def to_u8(x):
    """Scale |response| so that its 99.5th percentile maps to 255 (robust to outliers)."""
    x = np.abs(x)
    return np.uint8(np.clip(255 * x / (np.percentile(x, 99.5) + 1e-9), 0, 255))


smooth = cv2.GaussianBlur(gray, (7, 7), 1.5)                      # suppress noise before differentiation
sx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.hypot(sx, sy)
kpx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)   # Prewitt kernels
kpy = kpx.T
prewitt = np.hypot(cv2.filter2D(smooth.astype(np.float32), -1, kpx),
                   cv2.filter2D(smooth.astype(np.float32), -1, kpy))
krx = np.array([[1, 0], [0, -1]], np.float32)                      # Roberts cross kernels
kry = np.array([[0, 1], [-1, 0]], np.float32)
roberts = np.hypot(cv2.filter2D(smooth.astype(np.float32), -1, krx),
                   cv2.filter2D(smooth.astype(np.float32), -1, kry))
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
log = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 2.0), cv2.CV_64F, ksize=3)   # Laplacian of Gaussian
canny = cv2.Canny(smooth, 60, 160)

responses = {"Sobel X": sx, "Sobel Y": sy, "Sobel magnitude": sobel_mag, "Prewitt": prewitt,
             "Roberts": roberts, "Laplacian": laplacian, "LoG (sigma 2)": log}
print("\nEdge pixel percentage (a pixel is an edge if |response| > 25 % of the maximum response):")
for name, r in responses.items():
    print("  %-16s %5.2f %%" % (name, 100.0 * np.mean(np.abs(r) > 0.25 * np.abs(r).max())))
print("  %-16s %5.2f %%" % ("Canny (60,160)", 100.0 * np.mean(canny > 0)))
show_grid([gray] + [to_u8(r) for r in responses.values()] + [canny],
          ["Grayscale input"] + list(responses.keys()) + ["Canny (60, 160)"],
          3, 3, "fig04_edge_detectors.png", size=3.4)

# ---------------------------------------------------------------- 5. Canny thresholds
lena = cv2.cvtColor(cv2.imread("data/lena.jpg"), cv2.COLOR_BGR2GRAY)
pairs = [(30, 90), (60, 160), (120, 260)]
imgs, titles = [], []
for name, im in [("Leaf", gray), ("Lena", lena)]:
    imgs.append(im)
    titles.append(name + " input")
    for lo, hi in pairs:
        e = cv2.Canny(cv2.GaussianBlur(im, (5, 5), 1.0), lo, hi)
        pct = 100.0 * np.mean(e > 0)
        print("Canny %-5s thresholds (%3d,%3d): edge pixels = %5.2f %%" % (name, lo, hi, pct))
        imgs.append(e)
        titles.append("Canny (%d, %d): %.2f%% edges" % (lo, hi, pct))
show_grid(imgs, titles, 2, 4, "fig05_canny_thresholds.png")
print("Saved figures to", OUT)
