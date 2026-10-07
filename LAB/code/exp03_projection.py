"""
Experiment 3: Projection.

Part A implements the pinhole camera model x = K [R | t] X by hand and checks it
against cv2.projectPoints; perspective foreshortening and orthographic projection
are compared.  Part B projects a virtual cube onto real chessboard images from the
pose returned by cv2.solvePnP (augmented reality).  Part C treats a planar scene:
a homography rectifies a photographed sudoku grid and a rotation-only homography
synthesises the view of a rotated camera.
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(LAB)
OUT = "outputs/exp03"
os.makedirs(OUT, exist_ok=True)
np.set_printoptions(precision=4, suppress=True)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Intrinsics from experiment 2 (fallback: a synthetic 640x480 camera).
calib = "outputs/exp02/calibration.npz"
if os.path.exists(calib):
    c = np.load(calib); K, dist = c["K"], c["dist"]; W, H = (int(v) for v in c["image_size"])
else:
    W, H = 640, 480; K = np.array([[536.0, 0, 320], [0, 536.0, 240], [0, 0, 1]]); dist = np.zeros(5)
print("Intrinsic matrix K:\n", K)

# ------------------------------------------ Part A: pinhole model by hand
def rot_euler(rx, ry, rz):
    """Rotation matrix R = Rz(rz) Ry(ry) Rx(rx), angles in degrees."""
    a, b, g = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def project_perspective(X, K, R, t):
    """Homogeneous pinhole projection  x ~ K [R | t] X  ->  N x 2 pixels."""
    Xc = R @ X.T + t.reshape(3, 1)          # world -> camera coordinates
    x = K @ Xc                              # camera -> image (homogeneous)
    return (x[:2] / x[2]).T                 # dehomogenise

def project_orthographic(X, R, t, s, cx, cy):
    """Orthographic (parallel) projection: depth is simply dropped."""
    Xc = R @ X.T + t.reshape(3, 1)
    return np.c_[s * Xc[0] + cx, s * Xc[1] + cy]

# A small house (metres): cube base, ridge roof.  Edges as vertex-index pairs.
V = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 2], [1, -1, 2],
              [1, 1, 2], [-1, 1, 2], [0, -1, 3], [0, 1, 3]], float)
E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
     (2, 6), (3, 7), (4, 8), (5, 8), (7, 9), (6, 9), (8, 9)]
# Camera orientation from Euler angles: world Z is up, the camera looks along
# its own +Z axis, so Rx(90) turns +Y(world) into the viewing direction; an
# extra 25 deg tilt looks down on the house and Rz(-35) turns the house.
R = rot_euler(90 + 25, 0, 0) @ rot_euler(0, 0, -35)
print("Rotation from Euler angles (rx=115, rz=-35):\n", R)

def draw_wire(ax, pts, color, lw=1.5, marker=None, label=None):
    for a, b in E:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=color, lw=lw)
    if marker:
        ax.plot(pts[:, 0], pts[:, 1], marker, color="red", ms=5, label=label)

CENTRE = np.array([0, 0, 1.2])                       # middle of the house (metres)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
d0 = 6.0
for ax, d in zip(axes, [4.0, 8.0, 20.0]):
    t = np.array([0, 0, d]) - R @ CENTRE             # house centre on the optical axis, depth d
    print(f"camera centre in world coordinates for d = {d} m: {(-R.T @ t).round(2)}")
    Kd = K.copy(); Kd[0, 0] *= d / d0; Kd[1, 1] *= d / d0   # zoom so the house keeps its size
    x_manual = project_perspective(V, Kd, R, t)
    x_cv, _ = cv2.projectPoints(V, cv2.Rodrigues(R)[0], t, Kd, None)
    diff = np.abs(x_manual - x_cv.reshape(-1, 2)).max()
    print(f"distance {d:4.1f} m, f = {Kd[0,0]:7.1f} px: max |manual - cv2.projectPoints| = {diff:.2e} px")
    draw_wire(ax, x_manual, "tab:blue", marker="o", label="cv2.projectPoints")
    ax.plot(x_cv[:, 0, 0], x_cv[:, 0, 1], "o", color="red", ms=5)
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, ec="gray", ls="--"))
    ax.set_xlim(-20, W + 20); ax.set_ylim(H + 20, -20); ax.set_aspect("equal")
    ax.set_title(f"camera distance {d:.0f} m, f = {Kd[0,0]:.0f} px"); ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
axes[0].legend(loc="lower right")
fig.suptitle("Pinhole projection x = K[R|t]X (lines: NumPy, dots: cv2.projectPoints). Zooming keeps the size "
             "constant while foreshortening decreases with distance", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig01_pinhole_projection.png", dpi=150); plt.close(fig)

t = np.array([0, 0, 4.0]) - R @ CENTRE
x_persp = project_perspective(V, K, R, t)
x_orth = project_orthographic(V, R, t, K[0, 0] / 4.0, K[0, 2], K[1, 2])
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, pts, name in zip(axes, [x_persp, x_orth], ["perspective (pinhole)", "orthographic (parallel)"]):
    draw_wire(ax, pts, "tab:green" if "ortho" in name else "tab:blue", lw=2)
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, ec="gray", ls="--"))
    ax.set_xlim(-20, W + 20); ax.set_ylim(H + 20, -20); ax.set_aspect("equal"); ax.set_title(name)
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
fig.suptitle("Same camera pose: perspective vs orthographic projection of the house", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig02_perspective_vs_orthographic.png", dpi=150); plt.close(fig)
front = np.linalg.norm(x_persp[1] - x_persp[0]); back = np.linalg.norm(x_persp[2] - x_persp[3])
print(f"Perspective: front edge {front:.1f} px, back edge {back:.1f} px (ratio {front/back:.2f}); "
      f"orthographic: {np.linalg.norm(x_orth[1]-x_orth[0]):.1f} px and {np.linalg.norm(x_orth[2]-x_orth[3]):.1f} px")

# ---------------------------- Part B: pose from PnP and augmented reality
PATTERN, SQ = (9, 6), 25.0
objp = np.zeros((54, 3), np.float32); objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQ
axis3d = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]) * SQ
cube3d = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                     [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]]) * SQ
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, name in zip(axes.ravel(), ["left01", "left03", "left05", "left08"]):
    img = cv2.imread(f"data/{name}.jpg"); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)        # extrinsics of this view
    ax_px = cv2.projectPoints(axis3d, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(int)
    cb_px = cv2.projectPoints(cube3d, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(int)
    o = tuple(ax_px[0])
    cv2.drawContours(img, [cb_px[:4]], -1, (0, 255, 0), -3)                 # base face
    for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(cb_px[i]), tuple(cb_px[j]), (255, 0, 0), 3)   # vertical edges
    cv2.drawContours(img, [cb_px[4:]], -1, (0, 0, 255), 3)                 # top face
    for k, col in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.arrowedLine(img, o, tuple(ax_px[k]), col, 4, tipLength=0.1)   # X red, Y green, Z blue
    ax.imshow(rgb(img)); ax.axis("off")
    ax.set_title(f"{name}.jpg  |t| = {np.linalg.norm(tvec):.0f} mm, rvec = {rvec.ravel().round(2)}")
    print(f"solvePnP {name}: rvec = {rvec.ravel()}, tvec (mm) = {tvec.ravel()}")
fig.suptitle("Augmented reality: 3D axes and a 75 mm cube projected with the solvePnP pose", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig03_ar_projection.png", dpi=150); plt.close(fig)

# --------------------------------- Part C: planar projective transformation
sud = cv2.imread("data/sudoku.png")
src = np.float32([[73, 84], [492, 69], [520, 522], [34, 516]])         # TL, TR, BR, BL grid corners
S = 450
dst = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
Hm, _ = cv2.findHomography(src, dst)
rect = cv2.warpPerspective(sud, Hm, (S, S))
print("Homography (image -> fronto-parallel grid):\n", Hm / Hm[2, 2])
# grid of the rectified square mapped back with H^-1 to check the alignment
grid = np.float32([[[k * S / 9, 0], [k * S / 9, S]] for k in range(10)] +
                  [[[0, k * S / 9], [S, k * S / 9]] for k in range(10)])
back = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), np.linalg.inv(Hm)).reshape(-1, 2, 2)
overlay = sud.copy()
for (a, b) in back.astype(int):
    cv2.line(overlay, tuple(a), tuple(b), (0, 0, 255), 1)
# rotation-only homography H = K R K^-1 (camera panned by 25 deg, tilted by 10 deg)
Ks = np.array([[S, 0, S / 2], [0, S, S / 2], [0, 0, 1]], float)
Hrot = Ks @ rot_euler(10, 25, 0) @ np.linalg.inv(Ks)
crn = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), Hrot).reshape(-1, 2)
shift = np.array([[1, 0, -crn[:, 0].min()], [0, 1, -crn[:, 1].min()], [0, 0, 1]])
size = tuple(np.ceil(crn.max(0) - crn.min(0)).astype(int))
rotated = cv2.warpPerspective(rect, shift @ Hrot, size)
print("Rotation-only homography K R K^-1:\n", Hrot / Hrot[2, 2])
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(rgb(sud)); axes[0].plot(*np.vstack([src, src[:1]]).T, "r-o", ms=4); axes[0].set_title("input with 4 grid corners")
axes[1].imshow(rgb(rect)); axes[1].set_title(f"rectified with H ({S}x{S})")
axes[2].imshow(rgb(overlay)); axes[2].set_title("9x9 grid mapped back with H^-1")
axes[3].imshow(rgb(rotated)); axes[3].set_title("rotation-only H = K R K^-1")
for ax in axes: ax.axis("off")
fig.suptitle("Planar projective transformation (homography)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig04_homography.png", dpi=150); plt.close(fig)
print("Saved figures to", OUT)
