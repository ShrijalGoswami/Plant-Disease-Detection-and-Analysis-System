"""
Experiment 6 : Image segmentation methods.
Inputs : PlantVillage leaves (grape black rot for lesion segmentation, potato late blight
         for GrabCut) and the OpenCV smarties image for watershed object counting.
Methods: global / Otsu / adaptive thresholding, HSV colour thresholding (leaf and lesion
         masks), K-means colour clustering, mean-shift filtering, marker based watershed,
         GrabCut foreground extraction and a seeded region-growing routine written in numpy.
Output : outputs/exp06/fig01 ... fig05 and printed statistics.
"""
import os
from collections import deque
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "outputs/exp06"
os.makedirs(OUT, exist_ok=True)
PV = "../data/plantvillage/"
LEAF = PV + "Grape___Black_rot/00905d40-bddf-460e-b348-1dbb6a34653b___FAM_B.Rot 0664.JPG"
LEAF2 = PV + "Potato___Late_blight/0085ef03-aec3-431a-99a1-de286e10c0cf___RS_LB 2949.JPG"


def show_grid(images, titles, rows, cols, fname, size=3.2):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    for ax, im, t in zip(np.ravel(axes), images, titles):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    for ax in np.ravel(axes)[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def overlay(rgb, mask, colour=(255, 0, 0), alpha=0.45):
    out = rgb.copy()
    out[mask > 0] = (alpha * np.array(colour) + (1 - alpha) * out[mask > 0]).astype(np.uint8)
    return out


bgr = cv2.resize(cv2.imread(LEAF), (512, 512), interpolation=cv2.INTER_CUBIC)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# ------------------------------------------------ 1. threshold based segmentation
_, glob = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
otsu_t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ad_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
ad_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
print("Otsu threshold selected automatically = %.0f" % otsu_t)
for n, m in [("global T=100", glob), ("Otsu", otsu), ("adaptive mean", ad_mean), ("adaptive Gaussian", ad_gauss)]:
    print("  %-18s foreground (dark) pixels = %5.1f %%" % (n, 100 * np.mean(m == 0)))
show_grid([rgb, gray, glob, otsu, ad_mean, ad_gauss],
          ["Input leaf (RGB)", "Grayscale", "Global threshold T = 100", "Otsu threshold T = %.0f" % otsu_t,
           "Adaptive mean (31x31, C=5)", "Adaptive Gaussian (31x31, C=5)"], 2, 3, "fig01_thresholding.png")

# ------------------------------------------------ 2. HSV colour segmentation of leaf and lesions
leaf_mask = cv2.inRange(hsv, (15, 40, 20), (95, 255, 255))            # green-yellow hues
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
leaf_mask = cv2.morphologyEx(cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
leaf_mask = np.zeros_like(leaf_mask)
cv2.drawContours(leaf_mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)   # largest blob = leaf, holes filled
interior = cv2.erode(leaf_mask, kernel, iterations=1)                          # drop the shadowed leaf border
lesion = cv2.inRange(hsv, (0, 60, 45), (25, 255, 220)) & interior              # brown hues, not too dark (shadow)
lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
n_les, _, stats, _ = cv2.connectedComponentsWithStats(lesion)
n_les = int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 40))
leaf_area, lesion_area = np.count_nonzero(leaf_mask), np.count_nonzero(lesion)
print("Leaf area = %d px (%.1f %% of image), lesion area = %d px = %.2f %% of leaf, lesions counted = %d"
      % (leaf_area, 100 * leaf_area / leaf_mask.size, lesion_area, 100 * lesion_area / leaf_area, n_les))
show_grid([rgb, hsv[:, :, 0], hsv[:, :, 1], leaf_mask, lesion, overlay(overlay(rgb, 255 - leaf_mask, (0, 0, 255), 0.5), lesion)],
          ["Input leaf", "Hue channel", "Saturation channel", "Leaf mask (HSV range + morphology)",
           "Lesion mask (%.2f%% of leaf)" % (100 * lesion_area / leaf_area),
           "Overlay: background blue, lesions red"], 2, 3, "fig02_hsv_lesion_segmentation.png")

# ------------------------------------------------ 3. K-means colour clustering and mean shift
data = rgb.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
km_imgs, km_titles = [rgb], ["Input"]
for k in (2, 3, 4):
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    seg = centers.astype(np.uint8)[labels.ravel()].reshape(rgb.shape)
    print("K-means k=%d : compactness = %.3e, cluster sizes = %s" % (k, compactness, np.bincount(labels.ravel()).tolist()))
    km_imgs.append(seg)
    km_titles.append("K-means, k = %d" % k)
ms = cv2.pyrMeanShiftFiltering(bgr, sp=15, sr=30)
ms_rgb = cv2.cvtColor(ms, cv2.COLOR_BGR2RGB)
print("Mean shift (sp=15, sr=30): unique colours %d -> %d" % (len(np.unique(rgb.reshape(-1, 3), axis=0)), len(np.unique(ms_rgb.reshape(-1, 3), axis=0))))
km_imgs.append(ms_rgb)
km_titles.append("Mean shift filtering (sp=15, sr=30)")
_, ms_lab, ms_c = cv2.kmeans(ms_rgb.reshape(-1, 3).astype(np.float32), 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
km_imgs.append(ms_c.astype(np.uint8)[ms_lab.ravel()].reshape(rgb.shape))
km_titles.append("Mean shift + K-means (k = 3)")
show_grid(km_imgs, km_titles, 2, 3, "fig03_kmeans_meanshift.png")

# ------------------------------------------------ 4. marker based watershed on smarties
sm = cv2.imread("data/smarties.png")
sm_rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
sm_hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
fg = cv2.inRange(sm_hsv[:, :, 1], 70, 255)                                     # coloured sweets are saturated
fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.55 * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
sure_bg = cv2.dilate(fg, kernel, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)
n_markers, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1                                                          # background label 1, unknown 0
markers[unknown == 255] = 0
markers = cv2.watershed(sm, markers)
boundary = markers == -1
boundary[[0, -1], :] = False                                                   # ignore the image frame
boundary[:, [0, -1]] = False
ws_vis = sm_rgb.copy()
ws_vis[boundary] = [255, 0, 0]
labels_vis = np.uint8(255 * (markers - markers.min()) / (markers.max() - markers.min()))
print("Watershed on smarties: %d objects segmented" % (n_markers - 1))
show_grid([sm_rgb, fg, dist / dist.max(), sure_fg, cv2.applyColorMap(labels_vis, cv2.COLORMAP_JET)[:, :, ::-1], ws_vis],
          ["Smarties input", "Saturation threshold mask", "Distance transform", "Sure foreground (markers)",
           "Watershed labels", "Watershed boundaries: %d objects" % (n_markers - 1)], 2, 3, "fig04_watershed.png")

# ------------------------------------------------ 5. GrabCut and region growing
bgr2 = cv2.resize(cv2.imread(LEAF2), (512, 512), interpolation=cv2.INTER_CUBIC)
rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
gc_mask = np.zeros(bgr2.shape[:2], np.uint8)
rect = (20, 20, 472, 472)
bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(bgr2, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
print("GrabCut foreground = %.1f %% of image" % (100 * np.mean(fg_mask > 0)))


def region_grow(img, seed, tol):
    """Seeded region growing: add 4-neighbours whose intensity differs from the seed mean by < tol."""
    h, w = img.shape
    grown = np.zeros((h, w), np.uint8)
    q = deque([seed])
    grown[seed] = 255
    seed_val = float(img[seed])
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and grown[ny, nx] == 0 and abs(float(img[ny, nx]) - seed_val) < tol:
                grown[ny, nx] = 255
                q.append((ny, nx))
    return grown


# Region growing works on the Lab 'a' channel (green is negative), where leaf and grey background differ
a_chan = cv2.GaussianBlur(cv2.cvtColor(bgr2, cv2.COLOR_BGR2LAB)[:, :, 1], (5, 5), 1.5)
seed = (256, 200)                                              # a pixel inside the leaf (row, col)
rg = region_grow(a_chan, seed, tol=14)
rg = cv2.morphologyEx(rg, cv2.MORPH_CLOSE, kernel)
iou = np.count_nonzero(rg & fg_mask) / np.count_nonzero(rg | fg_mask)
print("Region growing from seed %s on Lab a-channel (tol 14): region = %.1f %% of image, IoU with GrabCut = %.3f"
      % (str(seed), 100 * np.mean(rg > 0), iou))
seg_vis = rgb2.copy()
seg_vis[fg_mask == 0] = 0
rg_vis = overlay(rgb2, rg, (255, 255, 0), 0.45)
cv2.circle(rg_vis, seed[::-1], 7, (255, 0, 0), -1)
show_grid([rgb2, fg_mask, seg_vis, a_chan, rg, rg_vis],
          ["Potato leaf input", "GrabCut mask (rect init, 5 iters)", "GrabCut foreground",
           "Lab a-channel (smoothed)", "Region growing (seed inside leaf, tol 14)", "Region grown overlay, IoU = %.2f" % iou],
          2, 3, "fig05_grabcut_region_growing.png")
print("Saved figures to", OUT)
