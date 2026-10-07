"""
Experiment 12 : Object detection from dynamic background for surveillance
-------------------------------------------------------------------------
Moving-object detection with three background models: running-average frame differencing,
Gaussian mixture (MOG2) and K-nearest-neighbour (KNN) background subtraction, followed by
morphological clean-up and contour bounding boxes. Includes a dynamic-background test
(swaying tree) and a zone-intrusion alarm as a surveillance application.
Inputs : data/vtest.avi (CCTV, walking people), data/tree.avi (swaying tree, a hand enters at the end)
"""
import os, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(LAB, "outputs", "exp12")
os.makedirs(OUT, exist_ok=True)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


class RunningAverage:
    """Frame differencing against an exponentially updated background B = (1-a) B + a I."""

    def __init__(self, alpha=0.02, thresh=30):
        self.alpha, self.thresh, self.bg = alpha, thresh, None

    def apply(self, frame):
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        if self.bg is None:
            self.bg = g.copy()
        mask = (cv2.absdiff(g, self.bg) > self.thresh).astype(np.uint8) * 255
        cv2.accumulateWeighted(g, self.bg, self.alpha)
        return mask

    def getBackgroundImage(self):
        return cv2.cvtColor(self.bg.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def make_models():
    return {"Differencing": RunningAverage(),
            "MOG2": cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True),
            "KNN": cv2.createBackgroundSubtractorKNN(history=300, dist2Threshold=400.0, detectShadows=True)}


def clean(mask, min_area):
    """Shadow removal (127 -> 0), median filter, opening, closing, then contour bounding boxes."""
    fg = (mask == 255).astype(np.uint8) * 255
    fg = cv2.medianBlur(fg, 5)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.dilate(fg, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= min_area]
    return fg, boxes


def draw_boxes(frame, boxes, color=(0, 255, 0)):
    vis = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    return vis


def run(video, n_frames, min_area, keep_frames, bg_at):
    """Run the three models over a video; collect masks, boxes, timings and background images."""
    models, cap = make_models(), cv2.VideoCapture(video)
    stats = {m: {"fg": [], "n": [], "t": []} for m in models}
    kept, bgs, boxes_all = {}, {}, {m: [] for m in models}
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        for name, model in models.items():
            t0 = time.perf_counter()
            raw = model.apply(frame)
            fg, boxes = clean(raw, min_area)
            stats[name]["t"].append(time.perf_counter() - t0)
            stats[name]["fg"].append((raw == 255).mean()); stats[name]["n"].append(len(boxes))
            boxes_all[name].append(boxes)
            if i in keep_frames:
                kept[(i, name)] = (frame, raw, fg, boxes)
            if i in bg_at:
                bgs[(i, name)] = model.getBackgroundImage()
    cap.release()
    return stats, kept, bgs, boxes_all


# ------------------------------------------------------------ vtest.avi : pedestrians
VT = os.path.join(LAB, "data", "vtest.avi")
keep = (60, 120, 180, 240, 299)
stats, kept, bgs, boxes_all = run(VT, 300, 400, keep, (30, 299))
print("vtest.avi (768x576), 300 frames, min blob area 400 px")
for name in stats:
    s = stats[name]
    print("  %-12s : mean fg fraction = %.4f   mean objects/frame = %.2f   %.1f fps (apply + clean-up)"
          % (name, np.mean(s["fg"][20:]), np.mean(s["n"][20:]), 1 / np.mean(s["t"])))
f = 120
frame, raw_m, fg_m, bx_m = kept[(f, "MOG2")]
fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), constrained_layout=True)
axes[0, 0].imshow(rgb(frame)); axes[0, 0].set_title("vtest.avi frame %d" % f)
axes[0, 1].imshow(raw_m, cmap="gray", vmin=0, vmax=255); axes[0, 1].set_title("MOG2 raw output (grey = shadow, white = foreground)")
axes[0, 2].imshow(fg_m, cmap="gray"); axes[0, 2].set_title("MOG2 after shadow removal + morphology")
axes[1, 0].imshow(kept[(f, "Differencing")][2], cmap="gray"); axes[1, 0].set_title("Running-average differencing mask (cleaned)")
axes[1, 1].imshow(kept[(f, "KNN")][2], cmap="gray"); axes[1, 1].set_title("KNN mask (cleaned)")
axes[1, 2].imshow(rgb(draw_boxes(frame, bx_m))); axes[1, 2].set_title("Moving objects from MOG2 : %d boxes" % len(bx_m))
for ax in axes.ravel():
    ax.axis("off")
plt.savefig(os.path.join(OUT, "fig01_masks_comparison.png"), dpi=150); plt.close()
fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), constrained_layout=True)
for ax, i in zip(axes.ravel(), (60, 120, 180, 240, 299)):
    fr, _, _, bx = kept[(i, "MOG2")]
    ax.imshow(rgb(draw_boxes(fr, bx))); ax.set_title("Frame %d : %d moving objects (MOG2)" % (i, len(bx))); ax.axis("off")
axes[1, 2].plot(stats["MOG2"]["n"], color="tab:green"); axes[1, 2].set_xlabel("frame"); axes[1, 2].set_ylabel("objects")
axes[1, 2].set_title("Objects per frame (MOG2)"); axes[1, 2].grid(alpha=0.3)
plt.savefig(os.path.join(OUT, "fig02_detections_grid.png"), dpi=150); plt.close()
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
axes[0, 0].imshow(rgb(bgs[(30, "MOG2")])); axes[0, 0].set_title("MOG2 background model after 30 frames"); axes[0, 0].axis("off")
axes[0, 1].imshow(rgb(bgs[(299, "MOG2")])); axes[0, 1].set_title("MOG2 background model after 300 frames"); axes[0, 1].axis("off")
axes[1, 0].imshow(rgb(bgs[(299, "Differencing")])); axes[1, 0].set_title("Running-average background after 300 frames"); axes[1, 0].axis("off")
for name in stats:
    axes[1, 1].plot(stats[name]["fg"], label=name, lw=1.2)
axes[1, 1].set_xlabel("frame"); axes[1, 1].set_ylabel("foreground pixel fraction"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim(0, 0.06); axes[1, 1].set_title("Foreground fraction per frame (raw masks, y-axis clipped at 0.06)")
plt.savefig(os.path.join(OUT, "fig03_background_models.png"), dpi=150); plt.close()

# ------------------------------------------------------------ tree.avi : dynamic background
TR = os.path.join(LAB, "data", "tree.avi")
t_stats, t_kept, _, t_boxes = run(TR, 68, 150, (40, 62), ())
print("tree.avi (320x240), 68 frames: swaying leaves, a hand enters at about frame 53")
for name in t_stats:
    s = t_stats[name]
    print("  %-12s : mean fg fraction frames 20-52 (no object) = %.4f   frames 55-67 (hand) = %.4f   false boxes/frame (20-52) = %.2f"
          % (name, np.mean(s["fg"][20:53]), np.mean(s["fg"][55:]), np.mean(s["n"][20:53])))
fig, axes = plt.subplots(2, 4, figsize=(16, 6.6), constrained_layout=True)
for r, i in enumerate((40, 62)):
    axes[r, 0].imshow(rgb(t_kept[(i, "MOG2")][0])); axes[r, 0].set_title("tree.avi frame %d" % i)
    for c, name in enumerate(("Differencing", "MOG2", "KNN"), start=1):
        fr, raw, fg, bx = t_kept[(i, name)]
        axes[r, c].imshow(rgb(draw_boxes(cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR), bx, (0, 0, 255))))
        axes[r, c].set_title("%s : fg %.1f%%, %d boxes" % (name, 100 * t_stats[name]["fg"][i], len(bx)))
for ax in axes.ravel():
    ax.axis("off")
plt.savefig(os.path.join(OUT, "fig04_dynamic_background_tree.png"), dpi=150); plt.close()

# ------------------------------------------------------------ zone intrusion alarm (MOG2 boxes on vtest)
zone = np.array([[100, 40], [330, 40], [330, 135], [40, 135]], np.int32)      # lawn in front of the building
alarm = [any(cv2.pointPolygonTest(zone, (x + w / 2.0, y + h), False) >= 0 for x, y, w, h in bx) for bx in boxes_all["MOG2"]]
first = int(np.argmax(alarm)) if any(alarm) else -1
print("Zone intrusion (MOG2 boxes, foot point inside polygon): %d of 300 frames raised an alarm, first alarm at frame %d"
      % (sum(alarm), first))
cap, demo = cv2.VideoCapture(VT), {}
for i in range(300):
    ok, fr = cap.read()
    if i in (30, 100, 130, 240):
        col = (0, 0, 255) if alarm[i] else (0, 200, 0)
        vis = draw_boxes(fr, boxes_all["MOG2"][i], (255, 200, 0))
        cv2.polylines(vis, [zone], True, col, 3)
        cv2.putText(vis, "ALARM: intrusion" if alarm[i] else "zone clear", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
        demo[i] = vis
cap.release()
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for ax, i in zip(axes.ravel(), sorted(demo)):
    ax.imshow(rgb(demo[i])); ax.set_title("Frame %d : %s" % (i, "ALARM" if alarm[i] else "clear")); ax.axis("off")
plt.savefig(os.path.join(OUT, "fig05_zone_intrusion.png"), dpi=150); plt.close()
print("Outputs written to", OUT)
