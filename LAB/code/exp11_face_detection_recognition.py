"""
Experiment 11 : Face detection and recognition.
Detection   : Haar cascade (Viola-Jones) and the YuNet CNN detector on lena, messi5 and a
              collage of Olivetti faces.
Recognition : Olivetti faces (40 people x 10 images) with Eigenfaces (PCA + SVM, PCA + 1-NN)
              and Local Binary Pattern histograms (chi-square 1-NN); face verification with
              the SFace deep embedding model (cosine similarity).
Output      : outputs/exp11/fig01 ... fig05 and printed accuracies.
"""
import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.abspath("data/cv2contrib"))   # OpenCV 4.11 contrib build (has CascadeClassifier)
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

OUT = "outputs/exp11"
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)
YUNET = "data/models/face_detection_yunet_2023mar.onnx"
SFACE = "data/models/face_recognition_sface_2021dec.onnx"

haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
yunet = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), score_threshold=0.7)


def detect_haar(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


def detect_yunet(bgr):
    yunet.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = yunet.detect(bgr)
    return np.empty((0, 15)) if faces is None else faces


def draw(bgr, haar_boxes, yunet_faces):
    a, b = bgr.copy(), bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in haar_boxes:
        cv2.rectangle(a, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes.detectMultiScale(gray[y:y + h // 2, x:x + w], 1.1, 6):
            cv2.rectangle(a, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
    for f in yunet_faces:
        x, y, w, h = f[:4].astype(int)
        cv2.rectangle(b, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for k in range(5):                                          # eyes, nose, mouth corners
            cv2.circle(b, (int(f[4 + 2 * k]), int(f[5 + 2 * k])), 3, (0, 0, 255), -1)
        cv2.putText(b, "%.2f" % f[14], (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


# ------------------------------------------------ 1. detection on natural images
data = fetch_olivetti_faces(shuffle=False)
X, y = data.images, data.target                                     # 400 x 64 x 64 in [0,1], 40 classes
imgs, titles = [], []
for name in ["lena.jpg", "messi5.jpg"]:
    bgr = cv2.imread("data/" + name)
    hb, yf = detect_haar(bgr), detect_yunet(bgr)
    print("%-10s Haar faces = %d, YuNet faces = %d" % (name, len(hb), len(yf)))
    a, b = draw(bgr, hb, yf)
    imgs += [a, b]
    titles += ["%s: Haar cascade (%d face, eyes in blue)" % (name, len(hb)), "%s: YuNet CNN (%d face, 5 landmarks)" % (name, len(yf))]
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, im, t in zip(axes.ravel(), imgs, titles):
    ax.imshow(im), ax.set_title(t, fontsize=10), ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig01_haar_vs_yunet.png"), dpi=150)
plt.close(fig)

# collage of 12 different Olivetti subjects (first image of subjects 0..11)
tile, margin = 128, 24
collage = np.full((3 * (tile + margin) + margin, 4 * (tile + margin) + margin), 200, np.uint8)
for i in range(12):
    r, c = divmod(i, 4)
    face = cv2.resize((X[10 * i] * 255).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_CUBIC)
    collage[margin + r * (tile + margin):margin + r * (tile + margin) + tile,
            margin + c * (tile + margin):margin + c * (tile + margin) + tile] = face
collage_bgr = cv2.cvtColor(collage, cv2.COLOR_GRAY2BGR)
hb, yf = detect_haar(collage_bgr), detect_yunet(collage_bgr)
print("collage    Haar faces = %d / 12, YuNet faces = %d / 12" % (len(hb), len(yf)))
a, b = draw(collage_bgr, hb, yf)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(a), axes[0].set_title("Haar cascade on 12-face collage: %d detected" % len(hb)), axes[0].axis("off")
axes[1].imshow(b), axes[1].set_title("YuNet on 12-face collage: %d detected" % len(yf)), axes[1].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig02_collage_detection.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------ 2. recognition: stratified 7 / 3 split per subject
idx = np.arange(400).reshape(40, 10)
train_idx = idx[:, :7].ravel()
test_idx = idx[:, 7:].ravel()
Xf = X.reshape(400, -1)
Xtr, ytr, Xte, yte = Xf[train_idx], y[train_idx], Xf[test_idx], y[test_idx]
pca = PCA(n_components=120, whiten=True, random_state=0).fit(Xtr)
Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
print("PCA: 120 components explain %.1f %% of the variance" % (100 * pca.explained_variance_ratio_.sum()))
svm = SVC(kernel="rbf", C=10, gamma="scale").fit(Ztr, ytr)
knn = KNeighborsClassifier(n_neighbors=1).fit(Ztr, ytr)
pred_svm, pred_knn = svm.predict(Zte), knn.predict(Zte)
acc_svm, acc_knn = accuracy_score(yte, pred_svm), accuracy_score(yte, pred_knn)
print("Eigenfaces + SVM accuracy = %.2f %%  (%d / %d)" % (100 * acc_svm, (pred_svm == yte).sum(), len(yte)))
print("Eigenfaces + 1-NN accuracy = %.2f %%" % (100 * acc_knn))

fig, axes = plt.subplots(2, 7, figsize=(12, 4))
axes[0, 0].imshow(pca.mean_.reshape(64, 64), cmap="gray"), axes[0, 0].set_title("Mean face", fontsize=9)
for k in range(13):
    ax = axes.ravel()[k + 1]
    ax.imshow(pca.components_[k].reshape(64, 64), cmap="gray"), ax.set_title("Eigenface %d" % (k + 1), fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig03_mean_face_eigenfaces.png"), dpi=150)
plt.close(fig)


def lbp_hist(img, grid=8):
    """8-neighbour LBP codes (256 patterns), then a histogram per grid cell (concatenated, L1 normalised)."""
    im = (img * 255).astype(np.int32) if img.max() <= 1 else img.astype(np.int32)
    c = im[1:-1, 1:-1]
    nb = [im[:-2, :-2], im[:-2, 1:-1], im[:-2, 2:], im[1:-1, 2:], im[2:, 2:], im[2:, 1:-1], im[2:, :-2], im[1:-1, :-2]]
    code = np.zeros_like(c)
    for bit, n in enumerate(nb):
        code |= (n >= c).astype(np.int32) << bit
    h, w = code.shape
    feats = []
    for i in range(grid):
        for j in range(grid):
            cell = code[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            feats.append(np.bincount(cell.ravel(), minlength=256))
    f = np.concatenate(feats).astype(np.float64)
    return f / f.sum()


Ltr = np.array([lbp_hist(im) for im in X[train_idx]])
Lte = np.array([lbp_hist(im) for im in X[test_idx]])
chi2 = ((Lte[:, None, :] - Ltr[None, :, :]) ** 2 / (Lte[:, None, :] + Ltr[None, :, :] + 1e-10)).sum(-1)
pred_lbp = ytr[chi2.argmin(1)]
acc_lbp = accuracy_score(yte, pred_lbp)
print("LBP histogram + chi-square 1-NN accuracy = %.2f %%" % (100 * acc_lbp))

sel = np.random.choice(len(yte), 10, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(11, 5.2))
for ax, i in zip(axes.ravel(), sel):
    ok = pred_svm[i] == yte[i]
    ax.imshow(Xte[i].reshape(64, 64), cmap="gray")
    ax.set_title("true %d / pred %d" % (yte[i], pred_svm[i]), color="green" if ok else "red", fontsize=10)
    ax.axis("off")
fig.suptitle("Eigenfaces + SVM predictions on test faces (red = wrong)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig04_recognition_predictions.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------ 3. SFace verification (deep embeddings)
sface = cv2.FaceRecognizerSF.create(SFACE, "")


def olivetti_bgr(subject, k):
    """Return Olivetti image k of a subject as a 256 x 256 BGR image."""
    g = cv2.resize((X[10 * subject + k] * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_CUBIC)
    g = cv2.copyMakeBorder(g, 96, 96, 96, 96, cv2.BORDER_CONSTANT, value=128)   # grey margin around the crop
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def embed(bgr):
    f = detect_yunet(bgr)
    if len(f) == 0:                                                  # retry with a grey border around the crop
        bgr = cv2.copyMakeBorder(bgr, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        f = detect_yunet(bgr)
    aligned = sface.alignCrop(bgr, f[0])                             # 112 x 112 face aligned on the 5 landmarks
    return sface.feature(aligned), cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)


pairs = [(cv2.imread("data/lena.jpg"), "lena.jpg", cv2.imread("data/messi5.jpg"), "messi5.jpg", "different people"),
         (olivetti_bgr(3, 0), "subject 3, image 0", olivetti_bgr(3, 7), "subject 3, image 7", "same person"),
         (olivetti_bgr(3, 0), "subject 3, image 0", olivetti_bgr(12, 0), "subject 12, image 0", "different people")]
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.3, left=0.06, right=0.98, top=0.94, bottom=0.07)
ax_bar = fig.add_subplot(gs[:, :2])
accs = [100 * acc_svm, 100 * acc_knn, 100 * acc_lbp]
bars = ax_bar.bar(["PCA + SVM", "PCA + 1-NN", "LBP + chi2 1-NN"], accs, color=["#2b7bba", "#5fa8d3", "#e07b39"])
ax_bar.bar_label(bars, fmt="%.1f%%")
ax_bar.set_ylim(0, 105), ax_bar.set_ylabel("Test accuracy (%)"), ax_bar.set_title("Olivetti recognition (120 test faces)")
pair_axes = []
for r, (im1, n1, im2, n2, label) in enumerate(pairs):
    e1, a1 = embed(im1)
    e2, a2 = embed(im2)
    cos = sface.match(e1, e2, cv2.FaceRecognizerSF_FR_COSINE)
    verdict = "MATCH" if cos >= 0.363 else "NO MATCH"
    print("SFace %-35s cosine = %.3f -> %s (threshold 0.363)" % ("%s vs %s:" % (n1, n2), cos, verdict))
    axes_pair = []
    for c, (im, n) in enumerate([(a1, n1), (a2, n2)]):
        ax = fig.add_subplot(gs[r, 2 + c])
        ax.imshow(im), ax.axis("off"), ax.set_title(n, fontsize=9)
        axes_pair.append(ax)
    pair_axes.append((axes_pair, "%s: cosine = %.3f -> %s" % (label, cos, verdict),
                      "green" if (verdict == "MATCH") == (label == "same person") else "red"))
for (axl, axr), text, colour in pair_axes:                          # caption under each aligned pair
    pl, pr = axl.get_position(), axr.get_position()
    fig.text((pl.x0 + pr.x1) / 2, pl.y0 - 0.02, text, ha="center", va="top", fontsize=9.5, color=colour)
fig.savefig(os.path.join(OUT, "fig05_accuracy_and_sface_verification.png"), dpi=150)
plt.close(fig)
cm = confusion_matrix(yte, pred_svm)
print("Confusion matrix (SVM): %d subjects perfectly recognised, %d misclassified test faces" % ((cm.diagonal() == 3).sum(), len(yte) - cm.trace()))
print("Saved figures to", OUT)
