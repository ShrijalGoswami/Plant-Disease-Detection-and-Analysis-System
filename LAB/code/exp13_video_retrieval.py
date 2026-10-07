"""
Experiment 13 : Content based video retrieval (CBVR)
----------------------------------------------------
1. Shot boundary detection with HSV colour-histogram differences, key-frame selection
2. Feature extraction per shot: HSV colour histogram, edge-orientation histogram and a
   ResNet-18 deep embedding of the key frame
3. Similarity search: transformed query clips are matched against the shot database and
   ranked; evaluation with rank of the true shot, mean reciprocal rank (MRR) and P@1
Inputs : data/Megamind.avi, data/Megamind_bugy.avi, data/vtest.avi, data/tree.avi
"""
import os
import numpy as np
import cv2
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(LAB, "outputs", "exp13")
os.makedirs(OUT, exist_ok=True)
VIDEOS = ["Megamind.avi", "Megamind_bugy.avi", "vtest.avi", "tree.avi"]
SIZE, CUT_TH, MIN_SHOT, MAX_SEG = (320, 240), 0.12, 8, 60
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()


def hsv_hist(frame):
    """Normalised 8x8x4 HSV colour histogram (256 bins)."""
    h = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    return (h / h.sum()).ravel()


def edge_hist(frame):
    """Gradient-orientation histogram (16 bins, weighted by gradient magnitude) - a texture/shape cue."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    h, _ = np.histogram(ang, bins=16, range=(0, 2 * np.pi), weights=mag)
    return h / (h.sum() + 1e-9)


def deep_feat(frame):
    """512-d global-average-pooled ResNet-18 embedding of the frame, L2 normalised."""
    x = cv2.resize(rgb(frame), (224, 224)).astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        f = resnet(x)[0].cpu().numpy()
    return f / np.linalg.norm(f)


def read_video(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA))
    return frames


def detect_shots(hists):
    """Cut when the L1 histogram distance between consecutive frames exceeds CUT_TH."""
    d = np.array([0.0] + [0.5 * np.abs(hists[i] - hists[i - 1]).sum() for i in range(1, len(hists))])
    cuts, last = [0], 0
    for i in range(1, len(d)):
        if d[i] > CUT_TH and i - last >= MIN_SHOT:
            cuts.append(i); last = i
    shots = [(a, b) for a, b in zip(cuts, cuts[1:] + [len(hists)]) if b - a >= MIN_SHOT]
    segs = []                                     # long static shots are split into <= MAX_SEG frame clips
    for a, b in shots:
        n = int(np.ceil((b - a) / MAX_SEG))
        segs += [(a + k * (b - a) // n, a + (k + 1) * (b - a) // n) for k in range(n)]
    return d, cuts, shots, segs


def clip_features(frames, a, b):
    """Features of a clip = mean HSV histogram of 5 sampled frames + edge and deep features of the key frame."""
    idx = np.linspace(a, b - 1, 5).astype(int)
    key = frames[(a + b) // 2]
    return {"hist": np.mean([hsv_hist(frames[i]) for i in idx], 0), "edge": edge_hist(key), "deep": deep_feat(key)}


# ------------------------------------------------------------ build the shot database
db, videos, curves = [], {}, {}
for name in VIDEOS:
    frames = read_video(os.path.join(LAB, "data", name))
    videos[name] = frames
    d, cuts, shots, segs = detect_shots([hsv_hist(f) for f in frames])
    curves[name] = (d, cuts)
    print("%-18s : %4d frames, %2d shots detected, %2d database clips after splitting long shots" % (name, len(frames), len(shots), len(segs)))
    for k, (a, b) in enumerate(segs):
        db.append(dict(video=name, clip=k, start=a, end=b, key=frames[(a + b) // 2], **clip_features(frames, a, b)))
print("Database size = %d clips" % len(db))

# ------------------------------------------------------------ queries (transformed clips)
def crop80(f):
    h, w = f.shape[:2]; return cv2.resize(f[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)], SIZE)
def bright(f):
    return cv2.convertScaleAbs(f, alpha=1.0, beta=40)
def flip(f):
    return cv2.flip(f, 1)
def jpeg_small(f):
    small = cv2.resize(f, (80, 60), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 20])
    return cv2.resize(cv2.imdecode(buf, 1), SIZE, interpolation=cv2.INTER_LINEAR)

def find(video, clip):
    return next(i for i, e in enumerate(db) if e["video"] == video and e["clip"] == clip)

queries = []
for (video, clip), (label, T) in zip([("Megamind.avi", 2), ("Megamind_bugy.avi", 3), ("vtest.avi", 4), ("tree.avi", 0)],
                                     [("central 80% crop", crop80), ("brightness +40", bright), ("horizontal flip", flip), ("80x60 + JPEG q20", jpeg_small)]):
    gt = find(video, clip); e = db[gt]
    frames = [T(f) for f in videos[video][e["start"]:e["end"]]]
    queries.append(dict(label=label, gt=gt, key=frames[len(frames) // 2], **clip_features(frames, 0, len(frames))))
gt = find("Megamind.avi", 4); e = db[gt]                  # untransformed clip from the second half of a shot
a, b = e["start"] + (e["end"] - e["start"]) // 2, e["end"]
queries.append(dict(label="later frames of same shot", gt=gt, key=videos["Megamind.avi"][(a + b) // 2], **clip_features(videos["Megamind.avi"], a, b)))

# ------------------------------------------------------------ retrieval and evaluation
chi2 = lambda p, q: 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-9))
dist_fns = {"hist": chi2, "edge": chi2, "deep": lambda p, q: 1.0 - float(p @ q)}
ranks = {k: [] for k in ["hist", "edge", "deep", "combined"]}
top5 = []
print("\nRetrieval results (rank of the true clip out of %d, combined distance = 0.4 hist + 0.2 edge + 0.4 deep, each scaled by its mean)" % len(db))
for q in queries:
    D = {k: np.array([fn(q[k], e[k]) for e in db]) for k, fn in dist_fns.items()}
    D["combined"] = 0.4 * D["hist"] / D["hist"].mean() + 0.2 * D["edge"] / D["edge"].mean() + 0.4 * D["deep"] / D["deep"].mean()
    line = "  %-26s (true = %s clip %d) :" % (q["label"], db[q["gt"]]["video"], db[q["gt"]]["clip"])
    for k in ranks:
        r = 1 + int((D[k] < D[k][q["gt"]]).sum()); ranks[k].append(r); line += "  %s rank %d" % (k, r)
    print(line)
    top5.append(np.argsort(D["combined"])[:5].tolist() + [D["combined"]])
print("\n  %-10s   MRR    P@1   mean rank" % "feature")
metrics = {}
for k, r in ranks.items():
    r = np.array(r); metrics[k] = (np.mean(1.0 / r), np.mean(r == 1), r.mean())
    print("  %-10s  %.3f  %.2f   %.2f" % (k, *metrics[k]))

# ------------------------------------------------------------ figures
d, cuts = curves["Megamind.avi"]
fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
ax.plot(d, lw=1, label="HSV histogram L1 distance between consecutive frames")
ax.axhline(CUT_TH, color="gray", ls="--", label="cut threshold %.2f" % CUT_TH)
for c in cuts[1:]:
    ax.axvline(c, color="tab:red", alpha=0.6, lw=1)
ax.set_xlabel("frame"); ax.set_ylabel("distance"); ax.set_title("Shot boundary detection on Megamind.avi : %d cuts (red lines)" % (len(cuts) - 1))
ax.legend(loc="upper right"); ax.grid(alpha=0.3)
plt.savefig(os.path.join(OUT, "fig01_shot_boundaries.png"), dpi=150); plt.close()
cols = 8; rows = int(np.ceil(len(db) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 1.9 * rows), constrained_layout=True)
for ax in axes.ravel():
    ax.axis("off")
for ax, e in zip(axes.ravel(), db):
    ax.imshow(rgb(e["key"])); ax.set_title("%s clip %d\nframes %d-%d" % (e["video"].replace(".avi", ""), e["clip"], e["start"], e["end"] - 1), fontsize=8)
fig.suptitle("Key frames of the %d database clips" % len(db))
plt.savefig(os.path.join(OUT, "fig02_keyframe_gallery.png"), dpi=150); plt.close()
fig, axes = plt.subplots(len(queries), 6, figsize=(15, 2.6 * len(queries)), constrained_layout=True)
for r, (q, res) in enumerate(zip(queries, top5)):
    idx, D = res[:5], res[5]
    axes[r, 0].imshow(rgb(q["key"])); axes[r, 0].set_title("Query: %s\n(true clip rank %d)" % (q["label"], ranks["combined"][r]), fontsize=9)
    for c, i in enumerate(idx, start=1):
        e = db[i]; ax = axes[r, c]
        ax.imshow(rgb(e["key"])); ax.set_title("#%d %s clip %d\nd = %.2f" % (c, e["video"].replace(".avi", ""), e["clip"], D[i]), fontsize=8)
        if i == q["gt"]:
            for s in ax.spines.values():
                s.set_edgecolor("lime"); s.set_linewidth(4)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Top-5 retrieved clips per query (combined distance); green frame = correct source clip")
plt.savefig(os.path.join(OUT, "fig03_retrieval_results.png"), dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
x = np.arange(len(metrics)); w = 0.38
ax.bar(x - w / 2, [m[0] for m in metrics.values()], w, label="MRR")
ax.bar(x + w / 2, [m[1] for m in metrics.values()], w, label="P@1")
ax.set_xticks(x); ax.set_xticklabels(list(metrics)); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title("Retrieval quality of each feature type over %d queries" % len(queries))
plt.savefig(os.path.join(OUT, "fig04_retrieval_metrics.png"), dpi=150); plt.close()
print("Outputs written to", OUT)
