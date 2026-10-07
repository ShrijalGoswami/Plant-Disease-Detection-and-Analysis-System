"""
Experiment 14 : Construct a 3D model from a single image
---------------------------------------------------------
Part A : learning-based monocular depth (MiDaS v2.1 small, relative inverse depth) ->
         back-projection with an assumed pinhole camera -> textured 3D mesh + PLY point clouds
Part B : classical shape from shading on a synthetic Lambertian sphere lit from the viewing
         direction: the image irradiance equation becomes the eikonal equation
         |grad Z| = sqrt(1 / E^2 - 1), solved by fast sweeping from the brightest point.
Output : outputs/exp14 figures and PLY files
"""
import os, glob, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(LAB, "outputs", "exp14"); os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- Part A : MiDaS monocular depth
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False, trust_repo=True, verbose=False)
model.load_state_dict(torch.load(os.path.join(LAB, "data", "models", "midas_v21_small_256.pt"), map_location="cpu"))
model.to(device).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, verbose=False).small_transform
print(f"MiDaS v2.1 small loaded on {device}; parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

def predict_inverse_depth(rgb):
    x = transform(rgb).to(device)                       # resize to 256 (multiple of 32), normalise, NCHW
    with torch.no_grad():
        t0 = time.time()
        pred = model(x)
        if device == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0
        pred = F.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False)
    return pred.squeeze().cpu().numpy(), dt

def to_depth(inv):
    """MiDaS gives relative inverse depth (up to scale and shift); map it to a relative depth in [1, 5]."""
    n = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
    return 1.0 / (0.2 + 0.8 * n)

def backproject(depth, f_scale=1.2):
    """Pinhole back-projection with an assumed focal length f = 1.2 * width and centred principal point."""
    H, W = depth.shape
    f, cx, cy = f_scale * W, W / 2.0, H / 2.0
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return (u - cx) * depth / f, -(v - cy) * depth / f, depth        # X right, Y up, Z forward

def write_ply(path, X, Y, Z, rgb, stride=2):
    P = np.stack([X[::stride, ::stride].ravel(), Y[::stride, ::stride].ravel(), Z[::stride, ::stride].ravel()], 1)
    C = rgb[::stride, ::stride].reshape(-1, 3)
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(P))
        np.savetxt(fh, np.hstack([P, C]), fmt="%.4f %.4f %.4f %d %d %d")
    return len(P)

def grid_mesh(X, Y, Z, rgb, stride, jump=1.10, z_far=4.0):
    """Quads on the (sub-sampled) pixel grid. A cell is dropped when the full-resolution depth inside it varies by
    more than the factor 'jump' (occlusion edge) or when it lies on the far background / sky (depth > z_far)."""
    Xs, Ys, Zs, C = X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride], rgb[::stride, ::stride] / 255.0
    k = np.ones((stride + 1, stride + 1), np.uint8)
    zmax = cv2.dilate(Z.astype(np.float32), k)[stride // 2::stride, stride // 2::stride][:Zs.shape[0] - 1, :Zs.shape[1] - 1]
    zmin = cv2.erode(Z.astype(np.float32), k)[stride // 2::stride, stride // 2::stride][:Zs.shape[0] - 1, :Zs.shape[1] - 1]
    keep = ((zmax / zmin) < jump) & (zmax < z_far)
    P = np.stack([Xs, Zs, Ys], axis=-1)                                  # plot axes: X, Z (depth), Y
    quads = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], axis=2)[keep]
    cols = ((C[:-1, :-1] + C[:-1, 1:] + C[1:, 1:] + C[1:, :-1]) / 4.0)[keep]
    return quads, cols

leaf = sorted(glob.glob(os.path.join(LAB, "..", "data", "plantvillage", "Potato___Late_blight", "*.JPG")))
inputs = [("building", os.path.join(LAB, "data", "building.jpg")), ("home", os.path.join(LAB, "data", "home.jpg")),
          ("leaf", leaf[0] if leaf else os.path.join(LAB, "data", "fruits.jpg"))]
results = []
for name, path in inputs:
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    predict_inverse_depth(rgb)                          # warm-up for this image size
    inv, dt = predict_inverse_depth(rgb)
    depth = to_depth(inv)
    X, Y, Z = backproject(depth)
    n_pts = write_ply(os.path.join(OUT, f"{name}.ply"), X, Y, Z, rgb)
    print(f"{name:9s}: {rgb.shape[1]}x{rgb.shape[0]} px, inference {dt * 1000:.1f} ms, inverse depth {inv.min():.1f}..{inv.max():.1f}, "
          f"relative depth 1.0..{depth.max():.1f}, PLY points {n_pts}")
    results.append((name, rgb, inv, depth, (X, Y, Z)))

# ---------------------------------------------------------------- Part B : classical shape from shading
S, r = 128, 45.0
ys, xs = np.mgrid[0:S, 0:S] - S / 2 + 0.5
inside = xs ** 2 + ys ** 2 < r ** 2
Z_true = np.where(inside, np.sqrt(np.maximum(r ** 2 - xs ** 2 - ys ** 2, 0)), 0.0)
E = np.where(inside, Z_true / r, 0.0)                    # Lambertian sphere, light along the optical axis: E = n . l = n_z

def eikonal_sfs(E, mask, sweeps=6):
    """Solve |grad T| = sqrt(1/E^2 - 1) (Rouy-Tourin / fast sweeping) with T = 0 at the brightest point."""
    Fm = np.minimum(np.sqrt(np.maximum(1.0 / np.maximum(E, 1e-3) ** 2 - 1.0, 0.0)), 50.0)
    T = np.full(E.shape, np.inf); T[E >= E.max() - 1e-6] = 0.0
    H, W = E.shape
    orders = [(range(1, H - 1), range(1, W - 1)), (range(1, H - 1), range(W - 2, 0, -1)),
              (range(H - 2, 0, -1), range(1, W - 1)), (range(H - 2, 0, -1), range(W - 2, 0, -1))]
    for _ in range(sweeps):
        for rows, cols in orders:
            for i in rows:
                for j in cols:
                    if not mask[i, j]: continue
                    a, b, f = min(T[i - 1, j], T[i + 1, j]), min(T[i, j - 1], T[i, j + 1]), Fm[i, j]
                    if not np.isfinite(min(a, b)): continue           # no known neighbour yet
                    t = min(a, b) + f if abs(a - b) >= f else 0.5 * (a + b + np.sqrt(2 * f * f - (a - b) ** 2))
                    if t < T[i, j]: T[i, j] = t
    return T

t0 = time.time(); T = eikonal_sfs(E, inside); t_sfs = time.time() - t0
Z_sfs = np.where(inside & np.isfinite(T), r - T, 0.0)   # height falls away from the brightest point (convex solution)
ev = inside & (E > 0.1)
rmse = np.sqrt(np.mean((Z_sfs[ev] - Z_true[ev]) ** 2))
print(f"Shape from shading (eikonal fast sweeping, {t_sfs:.1f} s): sphere radius {r:.0f} px, "
      f"height RMSE {rmse:.2f} px ({100 * rmse / r:.1f} % of the radius) over {ev.sum()} pixels with E > 0.1")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))
for col, (name, rgb, inv, depth, _) in enumerate(results):
    ax[0, col].imshow(rgb); ax[0, col].set_title(f"input: {name} ({rgb.shape[1]}x{rgb.shape[0]})"); ax[0, col].axis("off")
    im = ax[1, col].imshow(inv, cmap="inferno"); ax[1, col].set_title("MiDaS inverse depth (bright = near)"); ax[1, col].axis("off")
    fig.colorbar(im, ax=ax[1, col], fraction=0.046, pad=0.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_midas_depth.png"), dpi=150); plt.close(fig)

def set_axes(ax, X, Y, Z, title):
    ax.set_xlabel("X"); ax.set_ylabel("Z (depth)"); ax.set_zlabel("Y"); ax.set_title(title)
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Z.min(), Z.max()); ax.set_zlim(Y.min(), Y.max())
    ax.set_box_aspect((np.ptp(X), np.ptp(Z), np.ptp(Y)))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis.set_major_locator(plt.MaxNLocator(4))

name, rgb, inv, depth, (X, Y, Z) = results[0]
quads, cols = grid_mesh(X, Y, Z, rgb, stride=max(1, rgb.shape[1] // 170))
print(f"Textured mesh of '{name}': {len(quads)} quads kept after removing depth discontinuities")
fig = plt.figure(figsize=(13, 6))
for k, (el, az) in enumerate([(22, -55), (10, -115)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(quads, facecolors=cols, edgecolors="none"))
    ax.view_init(elev=el, azim=az); set_axes(ax, X, Y, Z, f"{name}: textured mesh (elev {el}, azim {az})")
fig.suptitle("3D model from a single photograph: MiDaS depth back-projected with an assumed pinhole camera")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig02_textured_surface.png"), dpi=150); plt.close(fig)

rng = np.random.default_rng(0)
fig = plt.figure(figsize=(13, 5.5))
for k, (name, rgb, inv, depth, (X, Y, Z)) in enumerate(results[1:]):
    idx = rng.choice(X.size, 30000, replace=False)
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(X.ravel()[idx], Z.ravel()[idx], Y.ravel()[idx], c=rgb.reshape(-1, 3)[idx] / 255.0, s=1, depthshade=False)
    ax.view_init(elev=18, azim=-70); set_axes(ax, X, Y, Z, f"{name}: point cloud (30 000 of {X.size} points)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig03_point_clouds.png"), dpi=150); plt.close(fig)

fig = plt.figure(figsize=(14, 3.8))
ax = fig.add_subplot(1, 4, 1); ax.imshow(E, cmap="gray"); ax.set_title("rendered Lambertian sphere (input)"); ax.axis("off")
ax = fig.add_subplot(1, 4, 2); im = ax.imshow(Z_true, cmap="viridis", vmin=0, vmax=r); ax.set_title("true height map"); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046)
ax = fig.add_subplot(1, 4, 3); im = ax.imshow(Z_sfs, cmap="viridis", vmin=0, vmax=r); ax.set_title(f"recovered height (RMSE {rmse:.1f} px)"); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046)
ax = fig.add_subplot(1, 4, 4, projection="3d")
ax.plot_surface(xs[::2, ::2], ys[::2, ::2], Z_sfs[::2, ::2], cmap="viridis", linewidth=0, antialiased=False)
ax.set_title("recovered surface (3D)"); ax.set_zlim(0, r); ax.view_init(elev=35, azim=-50)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig04_shape_from_shading.png"), dpi=150); plt.close(fig)
print("Figures written to", OUT)
