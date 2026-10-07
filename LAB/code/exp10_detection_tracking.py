"""
Experiment 10 : Implement object detection and tracking from video
--------------------------------------------------------------------
Detector : Faster R-CNN (ResNet-50 FPN v2, COCO weights, torchvision) applied to every frame
Tracker  : SORT-style multi-object tracker written here: one constant-velocity Kalman filter
           per object + Hungarian (linear_sum_assignment) data association on IoU
Extra    : OpenCV single-object MIL tracker compared with the detector on one person
Input    : data/vtest.avi (CCTV camera, 768 x 576, 10 fps)
"""
import os, time
import numpy as np
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(LAB, "outputs", "exp10")
os.makedirs(OUT, exist_ok=True)
N_FRAMES, SCORE_TH, IOU_TH, MIN_HITS, MAX_MISSES = 151, 0.6, 0.3, 2, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                   box_score_thresh=SCORE_TH).to(device).eval()


def detect_people(frame):
    """Run the detector on one BGR frame; return boxes (x1,y1,x2,y2) and scores of class person."""
    x = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        out = model([x])[0]
    keep = out["labels"] == 1                                    # COCO label 1 = person
    return out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()


def iou_matrix(a, b):
    """IoU between every box of a (N,4) and every box of b (M,4)."""
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class Track:
    """One tracked person: Kalman filter with state (cx, cy, w, h, vx, vy), measurement (cx, cy, w, h)."""
    next_id = 1

    def __init__(self, box):
        self.id, Track.next_id = Track.next_id, Track.next_id + 1
        kf = cv2.KalmanFilter(6, 4)
        kf.transitionMatrix = np.eye(6, dtype=np.float32)
        kf.transitionMatrix[0, 4] = kf.transitionMatrix[1, 5] = 1.0       # cx += vx, cy += vy
        kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)
        kf.processNoiseCov = np.diag([1, 1, 1, 1, 4, 4]).astype(np.float32) * 0.05
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 2.0
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 10.0
        kf.statePost = np.r_[self.to_z(box), 0.0, 0.0].astype(np.float32).reshape(6, 1)
        self.kf, self.streak, self.misses, self.confirmed = kf, 1, 0, False
        self.trail = [self.centre(box)]

    @staticmethod
    def to_z(b):
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], np.float32)

    @staticmethod
    def centre(b):
        return (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))

    def predict(self):
        s = self.kf.predict().ravel()
        return np.array([s[0] - s[2] / 2, s[1] - s[3] / 2, s[0] + s[2] / 2, s[1] + s[3] / 2])

    def update(self, box):
        self.kf.correct(self.to_z(box).reshape(4, 1))
        self.streak, self.misses = self.streak + 1, 0
        self.confirmed = self.confirmed or self.streak >= MIN_HITS
        self.trail.append(self.centre(self.box))

    @property
    def box(self):
        s = self.kf.statePost.ravel()
        return np.array([s[0] - s[2] / 2, s[1] - s[3] / 2, s[0] + s[2] / 2, s[1] + s[3] / 2])


def track_step(tracks, dets):
    """Predict all tracks, assign detections by Hungarian algorithm on IoU, create and delete tracks."""
    preds = np.array([t.predict() for t in tracks]).reshape(-1, 4)
    un_t, un_d = set(range(len(tracks))), set(range(len(dets)))
    if len(tracks) and len(dets):
        iou = iou_matrix(preds, dets)
        for i, j in zip(*linear_sum_assignment(-iou)):
            if iou[i, j] >= IOU_TH:
                tracks[i].update(dets[j]); un_t.discard(i); un_d.discard(j)
    for i in un_t:
        tracks[i].misses += 1; tracks[i].streak = 0
    for j in un_d:
        tracks.append(Track(dets[j]))
    tracks[:] = [t for t in tracks if t.misses <= MAX_MISSES]
    return [t for t in tracks if t.confirmed and t.misses == 0]


# ------------------------------------------------------------ detection + multi-object tracking
cap = cv2.VideoCapture(os.path.join(LAB, "data", "vtest.avi"))
writer = cv2.VideoWriter(os.path.join(OUT, "tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (768, 576))
tracks, all_dets, frames_keep, n_active, n_det, times, seen_ids = [], [], {}, [], [], [], set()
palette = np.random.RandomState(1).randint(60, 255, (200, 3)).tolist()
for i in range(N_FRAMES):
    ok, frame = cap.read()
    if not ok:
        break
    t0 = time.perf_counter(); boxes, scores = detect_people(frame); torch.cuda.synchronize() if device.type == "cuda" else None
    times.append(time.perf_counter() - t0); all_dets.append((boxes, scores)); n_det.append(len(boxes))
    active = track_step(tracks, boxes)
    n_active.append(len(active)); seen_ids.update(t.id for t in active)
    vis = frame.copy()
    for t in active:
        x1, y1, x2, y2 = t.box.astype(int); col = palette[t.id % 200]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, "ID %d" % t.id, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.polylines(vis, [np.array(t.trail[-40:], np.int32)], False, col, 2)
    cv2.putText(vis, "frame %d  active tracks %d" % (i, len(active)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.write(vis)
    if i in (0, 25, 50, 75, 100, 125, 150):
        frames_keep[i] = (frame, vis)
cap.release(); writer.release()
print("Detector : Faster R-CNN ResNet-50 FPN v2 (COCO) on %s, %d frames of vtest.avi" % (device, len(times)))
print("  mean inference time = %.1f ms/frame (excluding warm-up frame), mean persons detected per frame = %.2f"
      % (1000 * np.mean(times[1:]), np.mean(n_det)))
print("Tracker  : Kalman + Hungarian (IoU >= %.1f, confirm after %d hits, delete after %d misses)" % (IOU_TH, MIN_HITS, MAX_MISSES))
print("  unique confirmed tracks = %d, mean active tracks per frame = %.2f, max = %d" % (len(seen_ids), np.mean(n_active), max(n_active)))

# ------------------------------------------------------------ single-object MIL tracker vs detector
cap = cv2.VideoCapture(os.path.join(LAB, "data", "vtest.avi"))
ok, frame0 = cap.read()
b0, s0 = all_dets[0]
ref = b0[np.argmax((b0[:, 2] - b0[:, 0]) * (b0[:, 3] - b0[:, 1]))]      # largest person in frame 0
mil = cv2.TrackerMIL_create()
mil.init(frame0, tuple(int(v) for v in (ref[0], ref[1], ref[2] - ref[0], ref[3] - ref[1])))
ious, mil_frames = [], {}
for i in range(1, 61):
    ok, frame = cap.read()
    ok, (x, y, w, h) = mil.update(frame)
    mil_box = np.array([x, y, x + w, y + h], float)
    dets = all_dets[i][0]
    if len(dets):                                                 # follow the same person in the detections
        ov = iou_matrix(ref[None], dets)[0]                       # (the box overlapping the previous one)
        if ov.max() > 0.05:
            ref = dets[np.argmax(ov)]
    ious.append(iou_matrix(mil_box[None], ref[None])[0, 0])
    if i in (20, 40, 60):
        vis = frame.copy()
        cv2.rectangle(vis, tuple(mil_box[:2].astype(int)), tuple(mil_box[2:].astype(int)), (255, 255, 0), 2)
        cv2.rectangle(vis, tuple(ref[:2].astype(int)), tuple(ref[2:].astype(int)), (0, 255, 0), 2)
        mil_frames[i] = vis
cap.release()
print("MIL tracker vs detector on one person over 60 frames : mean IoU = %.3f, min IoU = %.3f, frames with IoU < 0.5 = %d"
      % (np.mean(ious), np.min(ious), int(np.sum(np.array(ious) < 0.5))))

# ------------------------------------------------------------ figures
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for ax, i in zip(axes.ravel(), (0, 50, 100, 150)):
    vis = frames_keep[i][0].copy()
    for b, s in zip(*all_dets[i]):
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.putText(vis, "%.2f" % s, (int(b[0]), int(b[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    ax.imshow(rgb(vis)); ax.set_title("Frame %d : %d persons detected (score >= %.1f)" % (i, len(all_dets[i][0]), SCORE_TH)); ax.axis("off")
plt.savefig(os.path.join(OUT, "fig01_detections.png"), dpi=150); plt.close()
fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), constrained_layout=True)
for ax, i in zip(axes.ravel(), (25, 50, 75, 100, 125, 150)):
    ax.imshow(rgb(frames_keep[i][1])); ax.set_title("Frame %d : tracks with IDs and trails" % i); ax.axis("off")
plt.savefig(os.path.join(OUT, "fig02_tracking_grid.png"), dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.plot(n_det, label="persons detected per frame", lw=1.2)
ax.plot(n_active, label="active confirmed tracks", lw=1.8)
ax.set_xlabel("frame"); ax.set_ylabel("count"); ax.grid(alpha=0.3); ax.legend()
ax.set_title("Detections and active tracks over %d frames (unique tracks = %d)" % (len(n_det), len(seen_ids)))
plt.savefig(os.path.join(OUT, "fig03_tracks_per_frame.png"), dpi=150); plt.close()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
axes[0].plot(range(1, 61), ious, marker=".", color="tab:red"); axes[0].axhline(0.5, ls="--", color="gray")
axes[0].set_xlabel("frame"); axes[0].set_ylabel("IoU (MIL box vs detector box)"); axes[0].set_ylim(0, 1); axes[0].grid(alpha=0.3)
axes[0].set_title("MIL tracker vs Faster R-CNN, mean IoU = %.2f" % np.mean(ious))
for ax, i in zip(axes[1:], (20, 60)):
    ax.imshow(rgb(mil_frames[i])); ax.set_title("Frame %d : MIL (cyan) vs detector (green)" % i); ax.axis("off")
plt.savefig(os.path.join(OUT, "fig04_mil_vs_detector.png"), dpi=150); plt.close()
print("Outputs written to", OUT)
