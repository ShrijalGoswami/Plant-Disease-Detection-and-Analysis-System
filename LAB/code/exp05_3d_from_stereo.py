"""
Experiment 5 : Construct a 3D model from a stereo pair
-------------------------------------------------------
Input  : SGBM disparity of the Aloe pair (from experiment 4, recomputed here if missing)
Method : disparity -> 3D points with the reprojection matrix Q (cv2.reprojectImageTo3D),
         coloured point cloud (PLY file), triangulated mesh from the image grid, views.
Output : outputs/exp05/aloe.ply and figures
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(LAB, "outputs", "exp05"); os.makedirs(OUT, exist_ok=True)
SCALE, F_PX, BASELINE_MM = 0.5, 3740.0 * 0.5, 160.0     # Middlebury 2006 geometry at half resolution
DMIN = 270.0 * SCALE                                     # Middlebury disparity offset (dmin.txt) at half resolution

# ---------------------------------------------------------------- 1. disparity + left image
def compute_sgbm():
    L = cv2.resize(cv2.imread(os.path.join(LAB, "data", "aloeL.jpg")), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    R = cv2.resize(cv2.imread(os.path.join(LAB, "data", "aloeR.jpg")), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    b = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=112, blockSize=b, P1=8 * 3 * b * b, P2=32 * 3 * b * b,
                                 disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(L, R).astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d, L

disp_file = os.path.join(LAB, "outputs", "exp04", "disparity_sgbm.npy")
left_file = os.path.join(LAB, "outputs", "exp04", "left_half.png")
if os.path.exists(disp_file) and os.path.exists(left_file):
    disp, left = np.load(disp_file), cv2.imread(left_file)
else:
    disp, left = compute_sgbm()
H, W = disp.shape
rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------- 2. reprojection to 3D
cx, cy = W / 2.0, H / 2.0
Q = np.array([[1, 0, 0, -cx],
              [0, 1, 0, -cy],
              [0, 0, 0, F_PX],
              [0, 0, 1.0 / BASELINE_MM, 0]], dtype=np.float64)   # Q maps [x, y, d, 1] -> [X, Y, Z, W]
disp0 = np.where(np.isfinite(disp), disp + DMIN, 0).astype(np.float32)   # true disparity = measured + dmin
pts = cv2.reprojectImageTo3D(disp0, Q)                       # shape (H, W, 3), units mm
X, Y, Z = pts[..., 0], -pts[..., 1], pts[..., 2]             # flip Y so that up is positive
valid = np.isfinite(disp) & (disp > 0) & np.isfinite(Z)
z_far = np.percentile(Z[valid], 99.0)                         # drop far outliers (tiny disparities)
valid &= Z < z_far
print(f"Image {W} x {H}; valid 3D points: {valid.sum()} ({100 * valid.mean():.1f} % of pixels)")
print(f"Depth range kept: {Z[valid].min():.0f} .. {Z[valid].max():.0f} mm, width span {X[valid].min():.0f} .. {X[valid].max():.0f} mm")

# ---------------------------------------------------------------- 3. PLY point cloud
P = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
C = rgb[valid]
with open(os.path.join(OUT, "aloe.ply"), "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(P))
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    np.savetxt(f, np.hstack([P, C]), fmt="%.2f %.2f %.2f %d %d %d")
print(f"Point cloud written: aloe.ply with {len(P)} coloured vertices")

# ---------------------------------------------------------------- 4. triangle mesh from the image grid
step = 4
ii, jj = np.arange(0, H, step), np.arange(0, W, step)
Xg, Yg, Zg, Vg = X[np.ix_(ii, jj)], Y[np.ix_(ii, jj)], Z[np.ix_(ii, jj)], valid[np.ix_(ii, jj)]
Cg = rgb[np.ix_(ii, jj)] / 255.0
tris, cols = [], []
z_jump = 50.0                                                 # mm; do not connect across depth discontinuities
for r in range(len(ii) - 1):
    for c in range(len(jj) - 1):
        quad = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)]
        for tri in ([quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]):
            if all(Vg[i, j] for i, j in tri):
                zs = [Zg[i, j] for i, j in tri]
                if max(zs) - min(zs) < z_jump:
                    tris.append([(Xg[i, j], Zg[i, j], Yg[i, j]) for i, j in tri])
                    cols.append(np.mean([Cg[i, j] for i, j in tri], axis=0))
tris, cols = np.array(tris), np.array(cols)
normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
light = np.array([0.3, -0.8, 0.5]); light /= np.linalg.norm(light)
shade = 0.35 + 0.65 * np.abs(normals @ light)
print(f"Mesh: {len(tris)} triangles on a {len(ii)} x {len(jj)} vertex grid (step {step})")

# ---------------------------------------------------------------- 5. figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
im = ax[0].imshow(disp, cmap="jet"); ax[0].set_title("SGBM disparity (px)"); ax[0].axis("off")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)
ax[1].imshow(valid, cmap="gray"); ax[1].set_title(f"Valid mask used for reconstruction ({100 * valid.mean():.1f} %)"); ax[1].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_disparity_mask.png"), dpi=150); plt.close(fig)

rng = np.random.default_rng(0)
sel = rng.choice(len(P), size=min(20000, len(P)), replace=False)
def set_axes(ax, title):
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Z depth (mm)"); ax.set_zlabel("Y (mm)"); ax.set_title(title)
    ax.set_box_aspect((np.ptp(P[:, 0]), np.ptp(P[:, 2]), np.ptp(P[:, 1])))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis.set_major_locator(plt.MaxNLocator(4))
fig = plt.figure(figsize=(12, 5.5))
for k, (el, az) in enumerate([(18, -65), (25, -120)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(P[sel, 0], P[sel, 2], P[sel, 1], c=C[sel] / 255.0, s=1.5, depthshade=False)
    ax.view_init(elev=el, azim=az); set_axes(ax, f"Coloured point cloud (elev {el}, azim {az})")
fig.suptitle("3D point cloud reconstructed from the stereo pair (20 000 of %d points shown)" % len(P))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig02_point_cloud.png"), dpi=150); plt.close(fig)

fig = plt.figure(figsize=(12, 5.5))
for k, (fc, title) in enumerate([(cols, "Textured triangle mesh"), (np.repeat(shade[:, None], 3, 1) * 0.85, "Shaded mesh (Lambertian)")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors="none"))
    ax.set_xlim(P[:, 0].min(), P[:, 0].max()); ax.set_ylim(P[:, 2].min(), P[:, 2].max()); ax.set_zlim(P[:, 1].min(), P[:, 1].max())
    ax.view_init(elev=20, azim=-75); set_axes(ax, title)
fig.suptitle(f"Surface mesh with {len(tris)} triangles (depth jumps > {z_jump:.0f} mm not connected)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig03_mesh.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(P[sel, 0], P[sel, 2], c=C[sel] / 255.0, s=1); ax[0].set_xlabel("X (mm)"); ax[0].set_ylabel("Z depth (mm)")
ax[0].set_title("Top view (X-Z): depth layering of leaves and background"); ax[0].set_aspect("equal")
ax[1].scatter(P[sel, 2], P[sel, 1], c=C[sel] / 255.0, s=1); ax[1].set_xlabel("Z depth (mm)"); ax[1].set_ylabel("Y (mm)")
ax[1].set_title("Side view (Z-Y)"); ax[1].set_aspect("equal")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig04_top_side_views.png"), dpi=150); plt.close(fig)
print("Figures written to", OUT)
