"""
Experiment 8: 3D model from ordinary images (structure from motion).

Three photographs of Wadham College (Oxford VGG multi-view data set) are
matched with SIFT.  The essential matrix of the first pair gives the relative
camera pose, the matches are triangulated into 3D points, and the third view
is registered incrementally with PnP so that new points can be added.  The
projection matrices shipped with the data set are used only to obtain the
intrinsic matrix K and to check the estimated pose.
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(LAB)
OUT, DATA = "outputs/exp08", "data/mview/wadham"
os.makedirs(OUT, exist_ok=True)
cv2.setRNGSeed(0); np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)
NAMES = ["001", "002", "003"]

imgs = [cv2.imread(f"{DATA}/{n}.jpg") for n in NAMES]
grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
h, w = grays[0].shape
# K and the ground-truth poses come from the 3x4 matrices P = K [R | -R C].
Ks, R_gt, C_gt = [], [], []
for n in NAMES:
    Kd, Rd, Cd = cv2.decomposeProjectionMatrix(np.loadtxt(f"{DATA}/{n}.P"))[:3]
    Ks.append(Kd / Kd[2, 2]); R_gt.append(Rd); C_gt.append((Cd[:3] / Cd[3]).ravel())
K = np.mean(Ks, axis=0); K[0, 1] = 0.0
print(f"Images: {len(NAMES)} views of {w}x{h} px. Shared intrinsic matrix K:\n{K}")

# ------------------------------------------------ 1. features and matching
sift = cv2.SIFT_create(nfeatures=8000)
kps, des = zip(*[sift.detectAndCompute(g, None) for g in grays])
print("SIFT keypoints per image:", [len(k) for k in kps])
bf = cv2.BFMatcher(cv2.NORM_L2)

def match(i, j, ratio=0.8):
    good = [m for m, n in bf.knnMatch(des[i], des[j], k=2) if m.distance < ratio * n.distance]
    qi = np.array([m.queryIdx for m in good]); ti = np.array([m.trainIdx for m in good])
    return good, qi, ti, np.float32([kps[i][q].pt for q in qi]), np.float32([kps[j][t].pt for t in ti])

def reproj(X, R, t, x):
    """Pixel error of 3D points X projected with pose (R, t) against pixels x."""
    xp = cv2.projectPoints(X, cv2.Rodrigues(R)[0], t, K, None)[0].reshape(-1, 2)
    return np.linalg.norm(xp - x, axis=1)

# ------------------------------------ 2. two-view geometry: E, pose, points
good, q0, t1, p0, p1 = match(0, 1)
E, mE = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
n_pose, R1, t1v, mP = cv2.recoverPose(E, p0, p1, K, mask=mE.copy())
inl = mP.ravel() > 0
print(f"View 1-2: {len(good)} ratio-test matches, {int(mE.sum())} essential-matrix inliers, "
      f"{n_pose} pass the cheirality check")
# scale ambiguity: fix the baseline to the ground-truth distance between the cameras
base_gt = np.linalg.norm(C_gt[1] - C_gt[0]); t1v = t1v * base_gt
poses = [(np.eye(3), np.zeros((3, 1))), (R1, t1v)]
P0 = K @ np.hstack(poses[0]); P1 = K @ np.hstack(poses[1])
Xh = cv2.triangulatePoints(P0, P1, p0[inl].T, p1[inl].T)
X = (Xh[:3] / Xh[3]).T
ok = (X[:, 2] > 0) & ((R1 @ X.T + t1v)[2] > 0)
err = np.maximum(reproj(X, *poses[0], p0[inl]), reproj(X, *poses[1], p1[inl]))
ok &= err < 2.0
pts, cols = X[ok], imgs[0][p0[inl][ok][:, 1].astype(int), p0[inl][ok][:, 0].astype(int)][:, ::-1]
obs = {(1, int(t)): k for k, t in enumerate(t1[inl][ok])}      # (view, keypoint) -> point id
stage_counts = [len(pts)]
print(f"Triangulated {ok.sum()} points from 2 views (mean reprojection error {err[ok].mean():.3f} px)")

def pose_error(R_est, t_est, i, j):
    """Angle errors of the estimated relative pose w.r.t. the data-set cameras."""
    R_rel = R_gt[j] @ R_gt[i].T; t_rel = R_gt[j] @ (C_gt[i] - C_gt[j])
    dR = np.degrees(np.arccos(np.clip((np.trace(R_est.T @ R_rel) - 1) / 2, -1, 1)))
    dt = np.degrees(np.arccos(np.clip(np.dot(t_est.ravel(), t_rel) / np.linalg.norm(t_est) / np.linalg.norm(t_rel), -1, 1)))
    return dR, dt
dR, dt = pose_error(R1, t1v, 0, 1)
print(f"Pose 1->2 vs ground truth: rotation error {dR:.2f} deg, translation direction error {dt:.2f} deg")

# ------------------------------ 3. incremental view: PnP + new triangulation
for v in range(2, len(NAMES)):
    good, qi, ti, pa, pb = match(v - 1, v)
    known = np.array([(k, obs[(v - 1, int(q))]) for k, q in enumerate(qi) if (v - 1, int(q)) in obs])
    ok3, rvec, tvec, inl3 = cv2.solvePnPRansac(pts[known[:, 1]], pb[known[:, 0]], K, None,
                                               reprojectionError=2.0, iterationsCount=500, confidence=0.999)
    Rv, tv = cv2.Rodrigues(rvec)[0], tvec
    poses.append((Rv, tv))
    dR, dt = pose_error(Rv @ poses[v - 1][0].T, tv - Rv @ poses[v - 1][0].T @ poses[v - 1][1], v - 1, v)
    print(f"View {v+1}: {len(good)} matches, {len(known)} have known 3D points, PnP inliers {len(inl3)}; "
          f"pose {v}->{v+1} error: rotation {dR:.2f} deg, translation direction {dt:.2f} deg")
    new = np.array([k for k, q in enumerate(qi) if (v - 1, int(q)) not in obs])
    Pa, Pb = K @ np.hstack(poses[v - 1]), K @ np.hstack(poses[v])
    Xh = cv2.triangulatePoints(Pa, Pb, pa[new].T, pb[new].T); Xn = (Xh[:3] / Xh[3]).T
    e = np.maximum(reproj(Xn, *poses[v - 1], pa[new]), reproj(Xn, *poses[v], pb[new]))
    okn = ((poses[v - 1][0] @ Xn.T + poses[v - 1][1])[2] > 0) & ((Rv @ Xn.T + tv)[2] > 0) & (e < 2.0)
    for k, t in zip(new[okn], ti[new[okn]]):
        obs[(v, int(t))] = len(pts) + int(np.sum(okn[:np.where(new == k)[0][0]]))
    pts = np.vstack([pts, Xn[okn]])
    cols = np.vstack([cols, imgs[v - 1][pa[new][okn][:, 1].astype(int), pa[new][okn][:, 0].astype(int)][:, ::-1]])
    err = np.concatenate([err[ok] if v == 2 else err, e[okn]])
    stage_counts.append(len(pts))
    print(f"  added {okn.sum()} new points -> total {len(pts)}")
# Points seen under a tiny parallax angle (nearly parallel rays) are poorly
# conditioned and lie far away; drop them together with any point behind the
# cameras' median depth range.
cams = np.array([(-Rv.T @ tv).ravel() for Rv, tv in poses])
r1 = pts - cams[0]; r2 = pts - cams[1]
parallax = np.degrees(np.arccos(np.clip(np.sum(r1 * r2, 1) / np.linalg.norm(r1, axis=1) / np.linalg.norm(r2, axis=1), -1, 1)))
lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)      # robust bounding box
good_pt = (parallax > 1.0) & np.all((pts > lo) & (pts < hi), axis=1)
print(f"Removed {(~good_pt).sum()} outlier points (parallax < 1 deg or outside the 1-99 percentile box)")
pts, cols, err = pts[good_pt], cols[good_pt], err[good_pt]
stage_counts.append(len(pts))
print(f"Final model: {len(pts)} coloured 3D points, mean reprojection error {err.mean():.3f} px, "
      f"median {np.median(err):.3f} px, median parallax {np.median(parallax[good_pt]):.2f} deg")
with open(f"{OUT}/points.ply", "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, c in zip(pts, cols):
        f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

# ------------------------------------------------------------- figures
sel = np.where(inl)[0][::max(1, inl.sum() // 120)]
vis = cv2.drawMatches(imgs[0], kps[0], imgs[1], kps[1], [good_ for k, good_ in enumerate(match(0, 1)[0]) if k in set(sel)],
                      None, matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax.axis("off")
ax.set_title(f"SIFT matches between view 1 and view 2 after RANSAC ({inl.sum()} inliers, {len(sel)} drawn)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig01_sift_matches.png", dpi=150); plt.close(fig)

F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
idx = np.where(inl)[0][:: max(1, inl.sum() // 12)][:12]
la, lb = imgs[0].copy(), imgs[1].copy()
lines_b = cv2.computeCorrespondEpilines(p0[idx].reshape(-1, 1, 2), 1, F).reshape(-1, 3)
lines_a = cv2.computeCorrespondEpilines(p1[idx].reshape(-1, 1, 2), 2, F).reshape(-1, 3)
for k, (ra, rb) in enumerate(zip(lines_a, lines_b)):
    col = tuple(int(c) for c in np.random.RandomState(k).randint(60, 255, 3))
    for im, r, p in [(la, ra, p0[idx][k]), (lb, rb, p1[idx][k])]:
        x0, y0 = 0, int(-r[2] / r[1]); x1, y1 = w, int(-(r[2] + r[0] * w) / r[1])
        cv2.line(im, (x0, y0), (x1, y1), col, 2); cv2.circle(im, tuple(p.astype(int)), 8, col, -1)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, im, ttl in zip(axes, [la, lb], ["view 1: epipolar lines of points in view 2", "view 2: epipolar lines of points in view 1"]):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(ttl); ax.axis("off")
fig.suptitle("Epipolar geometry from the estimated essential matrix (F = K^-T E K^-1)", fontsize=12, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{OUT}/fig02_epipolar_lines.png", dpi=150); plt.close(fig)

fig = plt.figure(figsize=(13, 6.5))
allp = np.vstack([pts, cams]); span = allp.max(0) - allp.min(0)
for k, (el, az, ttl) in enumerate([(12, -75, "view from behind the cameras"), (88, -90, "top view (plan of the courtyard)")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols / 255.0, s=1.5)
    for v, Cv in enumerate(cams):
        ax.scatter(Cv[0], Cv[2], -Cv[1], marker="^", s=70, c="red", edgecolors="k")
        ax.text(Cv[0] + 1.5 * (v - 1), Cv[2] - 2, -Cv[1] + 1.5, f"cam {v+1}", color="red", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Z (depth)"); ax.set_title(ttl)
    ax.set_zlabel("-Y (up)") if k == 0 else ax.set_zticks([])
    ax.set_box_aspect((span[0], span[2], span[1])); ax.view_init(elev=el, azim=az)
fig.suptitle(f"Sparse 3D model: {len(pts)} points from {len(NAMES)} views (red triangles: estimated cameras)", fontsize=12, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/fig03_point_cloud.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(err, bins=40, color="steelblue"); axes[0].set_xlabel("reprojection error (px)"); axes[0].set_ylabel("points")
axes[0].set_title(f"Reprojection error, mean {err.mean():.2f} px")
axes[1].bar(["2 views", "3 views", "after outlier removal"][:len(stage_counts)], stage_counts, color="darkorange")
axes[1].set_ylabel("3D points"); axes[1].set_title("Model size after each stage")
fig.tight_layout(); fig.savefig(f"{OUT}/fig04_errors_and_counts.png", dpi=150); plt.close(fig)
print("Saved points.ply and figures to", OUT)
