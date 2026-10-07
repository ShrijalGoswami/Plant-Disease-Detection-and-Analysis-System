"""
Experiment 4 : Determine depth map from a stereo pair
------------------------------------------------------
Input  : Middlebury "Aloe" rectified stereo pair (data/aloeL.jpg, data/aloeR.jpg)
Method : Block Matching (StereoBM) and Semi-Global Block Matching (StereoSGBM)
Output : disparity maps, parameter sweep, depth map Z = f*B/d, figures in outputs/exp04
"""
import os, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(LAB, "outputs", "exp04")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. load pair
SCALE = 0.5
imgL = cv2.imread(os.path.join(LAB, "data", "aloeL.jpg"))
imgR = cv2.imread(os.path.join(LAB, "data", "aloeR.jpg"))
imgL = cv2.resize(imgL, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
imgR = cv2.resize(imgR, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
H, W = grayL.shape
print(f"Stereo pair loaded and scaled by {SCALE}: {W} x {H} pixels")

# Camera geometry of the Middlebury 2006 data set (full resolution): f = 3740 px, B = 160 mm.
# The published pair is cropped, so the true disparity is d + dmin with dmin = 270 px (dmin.txt), scaled here.
F_PX = 3740.0 * SCALE          # focal length in pixels at the working resolution
BASELINE_MM = 160.0
DMIN = 270.0 * SCALE           # disparity offset at the working resolution

# ---------------------------------------------------------------- 2. matchers
def run_bm(num_disp=112, block=15):
    bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
    bm.setPreFilterCap(31); bm.setTextureThreshold(10); bm.setUniquenessRatio(10)
    bm.setSpeckleWindowSize(100); bm.setSpeckleRange(2)
    return bm.compute(grayL, grayR).astype(np.float32) / 16.0

def run_sgbm(num_disp=112, block=5):
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block,
                                 P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,
                                 disp12MaxDiff=1, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return sgbm.compute(imgL, imgR).astype(np.float32) / 16.0

def clean(disp):
    """Mark invalid disparities (<= 0) as NaN and median-filter the valid ones."""
    d = disp.copy()
    d[d <= 0] = np.nan
    filled = np.where(np.isnan(d), 0, d).astype(np.float32)
    filled = cv2.medianBlur(filled, 5)
    d = np.where(np.isnan(d) | (filled <= 0), np.nan, filled)
    return d

t0 = time.time(); disp_bm = run_bm(); t_bm = time.time() - t0
t0 = time.time(); disp_sgbm = run_sgbm(); t_sgbm = time.time() - t0
dbm, dsg = clean(disp_bm), clean(disp_sgbm)
for name, d, t in [("StereoBM", dbm, t_bm), ("StereoSGBM", dsg, t_sgbm)]:
    valid = np.isfinite(d)
    print(f"{name:10s}: time {t*1000:6.1f} ms, valid pixels {100*valid.mean():5.1f} %, "
          f"disparity min {np.nanmin(d):5.1f}, max {np.nanmax(d):5.1f}, mean {np.nanmean(d):5.1f} px")

# ---------------------------------------------------------------- 3. depth
depth = F_PX * BASELINE_MM / (dsg + DMIN)   # mm, NaN where disparity invalid
valid = np.isfinite(depth)
print(f"Depth from SGBM (f = {F_PX:.0f} px, B = {BASELINE_MM:.0f} mm, dmin = {DMIN:.0f} px): "
      f"near (1st percentile) {np.nanpercentile(depth, 1):.0f} mm, far (99th percentile) {np.nanpercentile(depth, 99):.0f} mm, "
      f"median {np.nanmedian(depth):.0f} mm")
np.save(os.path.join(OUT, "disparity_sgbm.npy"), dsg)
cv2.imwrite(os.path.join(OUT, "left_half.png"), imgL)

# ---------------------------------------------------------------- 4. figures
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].imshow(rgb(imgL)); ax[0].set_title("Left image (aloeL)")
ax[1].imshow(rgb(imgR)); ax[1].set_title("Right image (aloeR)")
for a in ax: a.axis("off")
fig.suptitle("Rectified stereo pair (Middlebury Aloe, half resolution)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_stereo_pair.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
for a, d, name in zip(ax, [dbm, dsg], ["StereoBM (block 15)", "StereoSGBM (block 5, 3-way)"]):
    im = a.imshow(d, cmap="jet", vmin=0, vmax=112); a.set_title(name + " disparity"); a.axis("off")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="disparity (px)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig02_bm_vs_sgbm.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(2, 3, figsize=(12, 7))
for a, b in zip(ax[0], [3, 7, 11]):
    a.imshow(clean(run_sgbm(112, b)), cmap="jet", vmin=0, vmax=112)
    a.set_title(f"SGBM blockSize = {b}, numDisparities = 112"); a.axis("off")
for a, nd in zip(ax[1], [48, 80, 112]):
    a.imshow(clean(run_sgbm(nd, 5)), cmap="jet", vmin=0, vmax=112)
    a.set_title(f"SGBM blockSize = 5, numDisparities = {nd}"); a.axis("off")
fig.suptitle("Effect of matcher parameters on the disparity map")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig03_parameter_sweep.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
im = ax[0].imshow(depth, cmap="viridis_r", vmin=np.nanpercentile(depth, 1), vmax=np.nanpercentile(depth, 99))
ax[0].set_title("Depth map Z = f B / (d + dmin) in mm, SGBM"); ax[0].axis("off")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02, label="depth (mm)")
ax[1].hist(dsg[valid].ravel(), bins=100, color="steelblue")
ax[1].set_xlabel("disparity (px)"); ax[1].set_ylabel("pixel count"); ax[1].set_title("Histogram of valid SGBM disparities")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig04_depth_and_histogram.png"), dpi=150); plt.close(fig)
print("Figures written to", OUT)
