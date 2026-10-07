"""
Experiment 2: Camera calibration (Zhang's planar-target method).

Thirteen views of a 9 x 6 inner-corner chessboard (data/left*.jpg, 640 x 480)
are used to estimate the intrinsic matrix K, the lens distortion coefficients
and the extrinsic pose of every view with cv2.calibrateCamera.  The result is
checked through the reprojection error, the calibration is saved for later
experiments and one image is undistorted.
"""
import os, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(LAB)
OUT = "outputs/exp02"
os.makedirs(OUT, exist_ok=True)

PATTERN = (9, 6)        # inner corners along width and height
SQUARE = 25.0           # size of one square in mm (assumed)
SENSOR = (4.8, 3.6)     # assumed sensor size in mm, for the field of view only

# ---------------------------------------------------------------- 1. corners
# Object points of the board in its own coordinate frame (plane Z = 0).
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE

files = sorted(glob.glob("data/left*.jpg"))
obj_pts, img_pts, names, corner_vis = [], [], [], []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for f in files:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    if not ok:
        print("corners not found in", f)
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    obj_pts.append(objp)
    img_pts.append(corners)
    names.append(os.path.basename(f))
    corner_vis.append(cv2.drawChessboardCorners(img.copy(), PATTERN, corners, ok))
h, w = gray.shape
print(f"Images used: {len(obj_pts)} of {len(files)}, image size {w}x{h}, "
      f"pattern {PATTERN[0]}x{PATTERN[1]}, square {SQUARE} mm")

# ------------------------------------------------------------ 2. calibration
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
np.set_printoptions(precision=4, suppress=True)
print(f"RMS reprojection error: {rms:.4f} px")
print("Camera matrix K:\n", K)
print("Distortion coefficients [k1 k2 p1 p2 k3]:", dist.ravel())
fovx, fovy, focal_mm, pp, ar = cv2.calibrationMatrixValues(K, (w, h), *SENSOR)
print(f"fx = {K[0,0]:.2f} px, fy = {K[1,1]:.2f} px, cx = {K[0,2]:.2f} px, cy = {K[1,2]:.2f} px")
print(f"Field of view: {fovx:.2f} deg (horizontal) x {fovy:.2f} deg (vertical); "
      f"focal length = {focal_mm:.2f} mm for a {SENSOR[0]} x {SENSOR[1]} mm sensor; "
      f"aspect ratio fy/fx = {ar:.4f}")

# ------------------------------------------------ 3. per-image reprojection
errs = []
for i in range(len(obj_pts)):
    proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
    d = img_pts[i].reshape(-1, 2) - proj.reshape(-1, 2)
    e = np.sqrt(np.mean(np.sum(d ** 2, axis=1)))          # RMS error in px
    errs.append(e)
    print(f"  {names[i]}: reprojection error = {e:.4f} px")
print(f"Mean per-image error = {np.mean(errs):.4f} px, worst = {np.max(errs):.4f} px")

# ------------------------------------------------- 4. extrinsics of view 1
R0, _ = cv2.Rodrigues(rvecs[0])
C0 = -R0.T @ tvecs[0]
print(f"Extrinsics of {names[0]}:")
print("R =\n", R0)
print("t (mm) =", tvecs[0].ravel())
print("Camera centre in board frame (mm) =", C0.ravel(),
      f"| distance to board origin = {np.linalg.norm(tvecs[0]):.1f} mm")

np.savez(f"{OUT}/calibration.npz", K=K, dist=dist, rvecs=np.array(rvecs),
         tvecs=np.array(tvecs), image_size=np.array([w, h]), rms=rms,
         per_image_error=np.array(errs))

# ------------------------------------------------------------ 5. undistort
img = cv2.imread(files[0])
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
und_a = cv2.undistort(img, K, dist, None, newK)
mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_32FC1)
und_b = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
print("Valid ROI after undistortion (x, y, w, h):", roi)
print("Max pixel difference undistort() vs remap():",
      int(np.abs(und_a.astype(int) - und_b.astype(int)).max()))
gx, gy = np.meshgrid(np.arange(w), np.arange(h))
disp = np.hypot(mapx - gx, mapy - gy)          # how far each pixel moves
print(f"Distortion displacement: max {disp.max():.2f} px at the corners, "
      f"mean {disp.mean():.2f} px")

# -------------------------------------------------------------- figures
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(2, 3, figsize=(12, 6.9))
for ax, vis, nm in zip(axes.ravel(), corner_vis[:6], names[:6]):
    ax.imshow(rgb(vis)); ax.set_title(f"{nm}: {PATTERN[0]}x{PATTERN[1]} corners"); ax.axis("off")
fig.suptitle("Detected and sub-pixel refined chessboard corners", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/fig01_corners.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(errs)), errs, color="steelblue")
ax.axhline(rms, color="red", ls="--", label=f"overall RMS = {rms:.3f} px")
ax.set_xticks(range(len(errs))); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylabel("reprojection error (px)"); ax.set_title("Per-image reprojection error")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig02_reprojection_error.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
axes[0].imshow(rgb(img)); axes[0].set_title("original " + names[0])
axes[1].imshow(rgb(und_a)); axes[1].set_title("undistorted (alpha = 1)")
x, y, rw, rh = roi
axes[1].add_patch(plt.Rectangle((x, y), rw, rh, fill=False, ec="lime", lw=1.5))
axes[2].imshow(cv2.absdiff(img, und_a).max(axis=2), cmap="gray"); axes[2].set_title("|original - undistorted|")
im = axes[3].imshow(disp, cmap="jet"); axes[3].set_title("pixel displacement (px)")
fig.colorbar(im, ax=axes[3], fraction=0.046)
for ax in axes: ax.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/fig03_undistortion.png", dpi=150); plt.close(fig)

# camera poses in the board frame: centre C = -R^T t; each camera is drawn as a
# small frustum whose apex is C and whose base is the image rectangle at 60 mm.
fig = plt.figure(figsize=(11, 7.5))
ax = fig.add_subplot(111, projection="3d")
bw, bh = (PATTERN[0] - 1) * SQUARE, (PATTERN[1] - 1) * SQUARE
ax.plot([0, bw, bw, 0, 0], [0, 0, bh, bh, 0], [0, 0, 0, 0, 0], "k-", lw=2, label="chessboard (Z = 0)")
ax.scatter(objp[:, 0], objp[:, 1], -objp[:, 2], c="k", s=6)
Kinv = np.linalg.inv(K)
corners_px = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
    R, _ = cv2.Rodrigues(rv)
    C = (-R.T @ tv).ravel()
    base = np.array([C + R.T @ (60.0 * (Kinv @ c) / (Kinv @ c)[2]) for c in corners_px])
    col = plt.cm.tab20(i / 13)
    for b in base:
        ax.plot([C[0], b[0]], [C[1], b[1]], [-C[2], -b[2]], color=col, lw=1)
    loop = np.vstack([base, base[:1]])
    ax.plot(loop[:, 0], loop[:, 1], -loop[:, 2], color=col, lw=1.2)
    ax.text(C[0], C[1], -C[2] + 12, names[i][4:6], fontsize=8, ha="center")
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("-Z (mm)  (towards the cameras)")
ax.set_title("Estimated camera poses (extrinsics) of the 13 views relative to the board")
ax.view_init(elev=28, azim=-55); ax.set_box_aspect((1.3, 1, 1)); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig04_camera_poses.png", dpi=150); plt.close(fig)
print("Saved figures and calibration.npz to", OUT)
