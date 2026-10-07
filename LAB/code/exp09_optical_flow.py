"""
Experiment 9 : Implement optical flow method
---------------------------------------------
Part A  sparse optical flow  : Shi-Tomasi corners + pyramidal Lucas-Kanade tracker
Part B  dense optical flow   : Farneback and DIS, colour-wheel and quiver visualisation
Part C  Lucas-Kanade written from scratch in NumPy, compared with OpenCV
Inputs : data/vtest.avi (static CCTV camera, walking people), data/Megamind.avi (large motion)
"""
import os, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(LAB, "outputs", "exp09")
VTEST, MEGA = os.path.join(LAB, "data", "vtest.avi"), os.path.join(LAB, "data", "Megamind.avi")
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)


def read_frames(path, start, count):
    """Read `count` consecutive BGR frames starting at frame index `start`."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def flow_to_hsv(flow):
    """Colour-wheel image: hue = flow direction, brightness = flow magnitude."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def quiver(ax, img, flow, step=16, scale=0.25, title=""):
    """Arrows of the flow on a coarse grid (arrow length = magnitude / scale)."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    ax.imshow(rgb(img))
    ax.quiver(xs, ys, flow[ys, xs, 0], flow[ys, xs, 1], color="yellow",
              angles="xy", scale_units="xy", scale=scale, width=0.0025)
    ax.set_title(title)
    ax.axis("off")


# ------------------------------------------------------------- Part A : sparse Lucas-Kanade
frames = read_frames(VTEST, 0, 61)
grays = [gray(f) for f in frames]
feature_params = dict(maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
colors = np.random.randint(0, 255, (5000, 3))
p0 = cv2.goodFeaturesToTrack(grays[0], mask=None, **feature_params)
ids, next_id = np.arange(len(p0)), len(p0)
canvas = np.zeros_like(frames[0])              # accumulated track lines
snapshots, stats = {}, []
print("Part A : sparse LK on vtest.avi frames 0-60 (initial corners = %d)" % len(p0))
for i in range(1, len(frames)):
    p1, st, err = cv2.calcOpticalFlowPyrLK(grays[i - 1], grays[i], p0, None, **lk_params)
    ok = st.ravel() == 1
    p0, p1, ids = p0[ok], p1[ok], ids[ok]
    disp = np.linalg.norm((p1 - p0).reshape(-1, 2), axis=1)
    moving = disp > 0.5                        # static background corners barely move
    stats.append([ok.sum(), disp.mean(), moving.sum(), disp[moving].mean() if moving.any() else 0.0])
    for (a, b), (c, d), k in zip(p1.reshape(-1, 2)[moving], p0.reshape(-1, 2)[moving], ids[moving]):
        cv2.line(canvas, (int(c), int(d)), (int(a), int(b)), colors[k % 5000].tolist(), 2)
    if i % 15 == 0:                            # snapshot and re-detect corners in empty areas
        vis = cv2.add(frames[i], canvas)
        for (a, b) in p1.reshape(-1, 2):
            cv2.circle(vis, (int(a), int(b)), 3, (0, 255, 255), -1)
        snapshots[i] = vis
        free = np.full(grays[i].shape, 255, np.uint8)
        for (a, b) in p1.reshape(-1, 2):
            cv2.circle(free, (int(a), int(b)), 7, 0, -1)
        new = cv2.goodFeaturesToTrack(grays[i], mask=free, **feature_params)
        if new is not None:
            p1 = np.vstack([p1, new])
            ids = np.concatenate([ids, np.arange(next_id, next_id + len(new))])
            next_id += len(new)
    p0 = p1.reshape(-1, 1, 2)
stats = np.array(stats)
for i in (0, 14, 29, 44, 59):
    print("  frame %2d->%2d : tracked=%3d  mean disp=%.2f px  moving(>0.5px)=%2d  mean disp of moving=%.2f px"
          % (i, i + 1, stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]))
print("  average over 60 frames : tracked=%.1f  mean disp=%.3f px  moving=%.1f  moving disp=%.2f px"
      % tuple(stats.mean(0)))
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, k in zip(axes.ravel(), sorted(snapshots)):
    ax.imshow(rgb(snapshots[k]))
    ax.set_title("Frame %d : LK tracks (lines) and tracked corners (dots)" % k)
    ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_lk_tracks.png"), dpi=150); plt.close()

# ------------------------------------------------------------- Part B : dense optical flow
fb = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
methods = {"Farneback": lambda a, b: cv2.calcOpticalFlowFarneback(a, b, None, **fb),
           "DIS": lambda a, b: dis.calc(a, b, None)}
seq = [gray(f) for f in read_frames(VTEST, 100, 21)]
times = {m: [] for m in methods}
mags = {m: [] for m in methods}
for g0, g1 in zip(seq[:-1], seq[1:]):
    for name, fn in methods.items():
        t = time.perf_counter(); fl = fn(g0, g1); times[name].append(time.perf_counter() - t)
        m = np.linalg.norm(fl, axis=2); mags[name].append([m.mean(), m.max()])
print("Part B : dense flow on 20 consecutive vtest pairs (frames 100-120, %dx%d)" % seq[0].shape[::-1])
for name in methods:
    print("  %-9s : %.1f ms/frame   mean |flow| = %.3f px   max |flow| = %.2f px"
          % (name, 1000 * np.mean(times[name]), np.mean(mags[name], 0)[0], np.mean(mags[name], 0)[1]))
f0, f1 = read_frames(VTEST, 100, 2)
flow_fb, flow_dis = methods["Farneback"](gray(f0), gray(f1)), methods["DIS"](gray(f0), gray(f1))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
axes[0].imshow(rgb(f0)); axes[0].set_title("vtest.avi frame 100"); axes[0].axis("off")
axes[1].imshow(flow_to_hsv(flow_fb)); axes[1].set_title("Farneback flow 100->101 (hue = direction, brightness = magnitude)"); axes[1].axis("off")
quiver(axes[2], f0, flow_fb, step=16, scale=0.25, title="Farneback flow vectors (x4 magnified)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_farneback_vtest.png"), dpi=150); plt.close()
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
axes[0].imshow(flow_to_hsv(flow_fb)); axes[0].set_title("Farneback : %.1f ms/frame" % (1000 * np.mean(times["Farneback"]))); axes[0].axis("off")
axes[1].imshow(flow_to_hsv(flow_dis)); axes[1].set_title("DIS (medium preset) : %.1f ms/frame" % (1000 * np.mean(times["DIS"]))); axes[1].axis("off")
plt.savefig(os.path.join(OUT, "fig03_farneback_vs_dis.png"), dpi=150); plt.close()
m0, m1 = read_frames(MEGA, 121, 2)                      # camera + character motion
flow_m = methods["Farneback"](gray(m0), gray(m1))
mag_m = np.linalg.norm(flow_m, axis=2)
print("  Megamind pair 121->122 : mean |flow| = %.2f px, max |flow| = %.2f px" % (mag_m.mean(), mag_m.max()))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].imshow(rgb(m0)); axes[0].set_title("Megamind.avi frame 121"); axes[0].axis("off")
axes[1].imshow(flow_to_hsv(flow_m)); axes[1].set_title("Farneback dense flow 121->122"); axes[1].axis("off")
quiver(axes[2], m0, flow_m, step=20, scale=0.5, title="Flow vectors (x2 magnified)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_megamind_dense_flow.png"), dpi=150); plt.close()


# ------------------------------------------------------------- Part C : Lucas-Kanade from scratch
def lk_numpy(I1, I2, pts, win=21, iters=20):
    """Iterative single-level LK: for each point solve  [Ix Iy] v = -It  by least squares."""
    I1f, I2f = I1.astype(np.float32), I2.astype(np.float32)
    Ix = cv2.Sobel(I1f, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(I1f, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    out = []
    for x, y in pts:
        A = np.stack([cv2.getRectSubPix(Ix, (win, win), (x, y)).ravel(),
                      cv2.getRectSubPix(Iy, (win, win), (x, y)).ravel()], axis=1)
        T = cv2.getRectSubPix(I1f, (win, win), (x, y)).ravel()
        ATA = A.T @ A                                   # 2x2 structure tensor of the window
        v = np.zeros(2, np.float32)
        for _ in range(iters):                          # Newton-style refinement of v
            It = cv2.getRectSubPix(I2f, (win, win), (x + v[0], y + v[1])).ravel() - T
            dv = np.linalg.solve(ATA, -A.T @ It)
            v += dv
            if np.linalg.norm(dv) < 1e-3:
                break
        out.append(v)
    return np.array(out)


g0, g1 = gray(f0), gray(f1)
cand = cv2.goodFeaturesToTrack(g0, maxCorners=300, qualityLevel=0.01, minDistance=7)
p_cv, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, cand, None, winSize=(21, 21), maxLevel=0,
                                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
d_cv, xy = (p_cv - cand).reshape(-1, 2), cand.reshape(-1, 2)
n = np.linalg.norm(d_cv, axis=1)
mov = np.where((n > 1.0) & (n < 6) & (st.ravel() == 1))[0][:6]           # 6 corners on walking people
x0, y0 = int(max(xy[mov, 0].min() - 50, 0)), int(max(xy[mov, 1].min() - 50, 0))
x1, y1 = int(min(xy[mov, 0].max() + 50, g0.shape[1])), int(min(xy[mov, 1].max() + 50, g0.shape[0]))
inside = (xy[:, 0] > x0 + 10) & (xy[:, 0] < x1 - 10) & (xy[:, 1] > y0 + 10) & (xy[:, 1] < y1 - 10)
sel = np.concatenate([mov, np.where((n < 0.2) & (st.ravel() == 1) & inside)[0][:4]])  # + 4 static corners
pts = xy[sel]
d_np = lk_numpy(g0, g1, pts)
print("Part C : NumPy LK vs OpenCV LK (single level, 21x21 window) on %d points of the pair 100->101" % len(sel))
print("   x      y   |  NumPy (dx, dy)   |  OpenCV (dx, dy)  | diff px")
for (x, y), a, b in zip(pts, d_np, d_cv[sel]):
    print("  %5.1f %5.1f | %6.2f %6.2f     | %6.2f %6.2f     | %.3f" % (x, y, a[0], a[1], b[0], b[1], np.linalg.norm(a - b)))
print("  mean absolute difference NumPy vs OpenCV = %.3f px" % np.linalg.norm(d_np - d_cv[sel], axis=1).mean())
crop = rgb(f0)[y0:y1, x0:x1]
fig, ax = plt.subplots(figsize=(11, 11 * crop.shape[0] / crop.shape[1] + 0.6), constrained_layout=True)
ax.imshow(crop)
K = 6
ax.scatter(pts[:, 0] - x0, pts[:, 1] - y0, s=60, facecolors="none", edgecolors="yellow", linewidths=1.5, label="corner")
ax.quiver(pts[:, 0] - x0, pts[:, 1] - y0, K * d_np[:, 0], K * d_np[:, 1], color="red", angles="xy",
          scale_units="xy", scale=1, width=0.008, label="NumPy LK")
ax.quiver(pts[:, 0] - x0, pts[:, 1] - y0, K * d_cv[sel, 0], K * d_cv[sel, 1], color="cyan", angles="xy",
          scale_units="xy", scale=1, width=0.003, label="OpenCV LK")
ax.set_xlim(0, crop.shape[1]); ax.set_ylim(crop.shape[0], 0)
ax.legend(loc="upper right"); ax.axis("off")
ax.set_title("Lucas-Kanade from scratch (red) vs OpenCV (cyan) on 6 moving + 4 static corners, arrows x%d, frames 100->101" % K)
plt.savefig(os.path.join(OUT, "fig05_numpy_vs_opencv_lk.png"), dpi=150); plt.close()
print("Figures written to", OUT)
