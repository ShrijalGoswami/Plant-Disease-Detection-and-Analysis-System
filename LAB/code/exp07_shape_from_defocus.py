"""
Experiment 7 : Construct a 3D model from defocus (shape from focus / depth from defocus)
----------------------------------------------------------------------------------------
A real focal stack is not available, so a synthetic stack is rendered from a textured
image (PlantVillage leaf) and a known ground-truth depth map. Each slice k is focused
at depth k; every pixel is blurred with sigma = ALPHA * |depth - k| (thin-lens model).
Depth is recovered with two focus measures (Sum-Modified-Laplacian, variance of the
Laplacian), sub-slice Gaussian interpolation, and compared with the ground truth.
Output : outputs/exp07 figures, all-in-focus image, error statistics
"""
import os, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(LAB, "outputs", "exp07"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. texture image + ground-truth depth
leaves = sorted(glob.glob(os.path.join(LAB, "..", "data", "plantvillage", "Tomato___Septoria_leaf_spot", "*.JPG")))
src = leaves[0] if leaves else os.path.join(LAB, "data", "baboon.jpg")
img = cv2.resize(cv2.imread(src), (256, 256), interpolation=cv2.INTER_AREA)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
H, W = rgb.shape[:2]
N, ALPHA = 15, 0.7                                   # number of focus settings, blur growth (px per slice)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
d_true = 1.0 + 8.0 * xx / (W - 1) + 6.0 * np.exp(-((xx - 150) ** 2 + (yy - 110) ** 2) / (2 * 45.0 ** 2))
d_true = np.clip(d_true, 0, N - 1)                   # depth in "slice units" (0 = nearest focus plane)
print(f"Texture: {os.path.basename(src)}; stack of {N} slices; ground-truth depth range {d_true.min():.2f} .. {d_true.max():.2f}")

# ---------------------------------------------------------------- 2. render the focal stack
step = 0.25
sigmas = np.arange(0, ALPHA * (N - 1) + step, step)
blurred = np.stack([rgb if s == 0 else cv2.GaussianBlur(rgb, (0, 0), s) for s in sigmas])   # (levels, H, W, 3)
stack = np.empty((N, H, W, 3), np.float32)
for k in range(N):
    idx = np.rint(ALPHA * np.abs(d_true - k) / step).astype(int)          # blur level of each pixel
    stack[k] = np.take_along_axis(blurred, idx[None, :, :, None], axis=0)[0]
stack = np.clip(stack + rng.normal(0, 0.01, stack.shape).astype(np.float32), 0, 1)      # sensor noise (about 2.5 grey levels)
gray = stack.mean(axis=3)

# ---------------------------------------------------------------- 3. focus measures
WIN = 7
def sml(I):                                         # Sum-Modified-Laplacian (Nayar and Nakagawa)
    kx = np.array([[-1, 2, -1]], np.float32); ky = kx.T
    ml = np.abs(cv2.filter2D(I, -1, kx)) + np.abs(cv2.filter2D(I, -1, ky))
    return cv2.boxFilter(ml, -1, (WIN, WIN))
def var_lap(I):                                     # local variance of the Laplacian response
    L = cv2.Laplacian(I, cv2.CV_32F, ksize=3)
    return cv2.boxFilter(L * L, -1, (WIN, WIN)) - cv2.boxFilter(L, -1, (WIN, WIN)) ** 2

def depth_from_stack(fm):
    """argmax over the stack + 3-point Gaussian (parabolic in log) interpolation + median smoothing"""
    k = np.argmax(fm, axis=0)
    kc = np.clip(k, 1, N - 2)
    lf = np.log(fm + 1e-9)
    f0 = np.take_along_axis(lf, kc[None], 0)[0]
    fm1 = np.take_along_axis(lf, (kc - 1)[None], 0)[0]
    fp1 = np.take_along_axis(lf, (kc + 1)[None], 0)[0]
    denom = fm1 + fp1 - 2 * f0
    delta = np.where(np.abs(denom) > 1e-9, 0.5 * (fm1 - fp1) / np.where(np.abs(denom) > 1e-9, denom, 1), 0)
    d = kc + np.clip(delta, -1, 1)
    return cv2.medianBlur(d.astype(np.float32), 5), k

fm_sml = np.stack([sml(g) for g in gray])
fm_vl = np.stack([var_lap(g) for g in gray])
d_sml, k_sml = depth_from_stack(fm_sml)
d_vl, k_vl = depth_from_stack(fm_vl)
for name, d in [("SML", d_sml), ("Var-Laplacian", d_vl)]:
    err = d - d_true
    print(f"{name:14s}: RMSE {np.sqrt(np.mean(err ** 2)):.3f} slices, MAE {np.mean(np.abs(err)):.3f} slices, "
          f"pixels within 1 slice {100 * np.mean(np.abs(err) <= 1):.1f} %")
aif = np.take_along_axis(stack, k_sml[None, :, :, None], axis=0)[0]        # all-in-focus composite
sharp_gain = sml(aif.mean(2)).mean() / sml(gray[N // 2]).mean()
print(f"All-in-focus image: mean SML {sharp_gain:.2f} x that of the middle slice")

# ---------------------------------------------------------------- 4. two-image depth from defocus
def lap_energy(I):                                  # noise-robust local Laplacian energy (pre-smoothing + 15 x 15 window)
    return cv2.boxFilter(np.abs(cv2.Laplacian(cv2.GaussianBlur(I, (0, 0), 1.0), cv2.CV_32F)), -1, (15, 15))
e_near, e_far = lap_energy(gray[0]), lap_energy(gray[N - 1])
ratio = e_near / (e_near + e_far + 1e-6)                                    # relative blur measure
corr = np.corrcoef(ratio.ravel(), d_true.ravel())[0, 1]
print(f"Two-image defocus ratio vs true depth: correlation coefficient {corr:.3f}")

# ---------------------------------------------------------------- 5. figures
show = [0, 3, 7, 10, 14]
fig, ax = plt.subplots(1, len(show) + 1, figsize=(15, 3.2))
for a, k in zip(ax, show):
    a.imshow(stack[k]); a.set_title(f"slice {k}: focus at depth {k}"); a.axis("off")
im = ax[-1].imshow(d_true, cmap="viridis", vmin=0, vmax=N - 1); ax[-1].set_title("ground-truth depth"); ax[-1].axis("off")
fig.colorbar(im, ax=ax[-1], fraction=0.046, pad=0.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_focal_stack.png"), dpi=150); plt.close(fig)

pix = [(60, 40), (110, 150), (200, 230)]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, (r, c) in zip(ax, pix):
    curve = fm_sml[:, r, c] / fm_sml[:, r, c].max()
    a.plot(range(N), curve, "o-", label="SML (normalised)")
    a.axvline(d_true[r, c], color="k", ls="--", label=f"true depth {d_true[r, c]:.2f}")
    a.axvline(d_sml[r, c], color="r", ls=":", label=f"estimated {d_sml[r, c]:.2f}")
    a.set_xlabel("focus setting (slice index)"); a.set_ylabel("focus measure"); a.set_title(f"pixel (row {r}, col {c})"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig02_focus_curves.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
for a, d, t in zip(ax[0], [d_true, d_sml, d_vl], ["Ground-truth depth", "Recovered depth: SML", "Recovered depth: variance of Laplacian"]):
    im = a.imshow(d, cmap="viridis", vmin=0, vmax=N - 1); a.set_title(t); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
for a, d, t in zip(ax[1][:2], [d_sml, d_vl], ["|error| SML (slices)", "|error| variance of Laplacian (slices)"]):
    im = a.imshow(np.abs(d - d_true), cmap="magma", vmin=0, vmax=3); a.set_title(t); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
ax[1][2].hist((d_sml - d_true).ravel(), bins=80, range=(-4, 4), alpha=0.7, label="SML")
ax[1][2].hist((d_vl - d_true).ravel(), bins=80, range=(-4, 4), alpha=0.7, label="Var-Laplacian")
ax[1][2].set_xlabel("depth error (slices)"); ax[1][2].set_ylabel("pixels"); ax[1][2].set_title("Error histogram"); ax[1][2].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig03_depth_comparison.png"), dpi=150); plt.close(fig)

fig = plt.figure(figsize=(13, 5.5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(xx, yy, d_true, cmap="viridis", rstride=4, cstride=4, linewidth=0, antialiased=False)
ax1.set_title("Ground-truth surface"); ax1.view_init(elev=45, azim=-60); ax1.invert_yaxis()
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(xx, yy, d_sml, facecolors=aif, rstride=2, cstride=2, linewidth=0, antialiased=False, shade=False)
ax2.set_title("Recovered surface (SML) textured with the all-in-focus image"); ax2.view_init(elev=45, azim=-60); ax2.invert_yaxis()
for a in (ax1, ax2):
    a.set_xlabel("x (px)"); a.set_ylabel("y (px)"); a.set_zlabel("depth (slices)"); a.set_zlim(0, N - 1)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig04_3d_surface.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
ax[0].imshow(stack[N // 2]); ax[0].set_title(f"single slice {N // 2}"); ax[0].axis("off")
ax[1].imshow(aif); ax[1].set_title("all-in-focus composite"); ax[1].axis("off")
im = ax[2].imshow(ratio, cmap="coolwarm"); ax[2].set_title("two-image blur ratio E0 / (E0 + E14)"); ax[2].axis("off")
fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02)
sel = rng.choice(H * W, 3000, replace=False)
ax[3].scatter(d_true.ravel()[sel], ratio.ravel()[sel], s=3, alpha=0.4)
ax[3].set_xlabel("true depth (slices)"); ax[3].set_ylabel("blur ratio"); ax[3].set_title(f"ratio vs depth, r = {corr:.2f}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig05_allinfocus_and_dfd.png"), dpi=150); plt.close(fig)
cv2.imwrite(os.path.join(OUT, "all_in_focus.png"), cv2.cvtColor((aif * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print("Figures written to", OUT)
