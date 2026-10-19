"""Sensor Fusion Lab - demo klasifikasi penyebab deforestasi dengan citra SINTETIS.
Jalankan:  pip install -r requirements.txt  &&  python app.py   (buka http://127.0.0.1:5000)
"""
import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

N = 16
P = N * N
CLASSES = ["Pertanian", "Pertambangan", "Penebangan", "Kebakaran", "Banjir"]
MODES = ["Optik saja", "Radar saja", "Gabungan"]
# Nilai rata-rata per kelas. Optik: (merah, hijau, NIR). Radar: (VV, VH).
OPT_FOREST = np.array([.08, .12, .55])
RAD_FOREST = np.array([.45, .25])
OPT_DIST = np.array([[.30, .28, .42], [.50, .45, .38], [.18, .18, .45], [.10, .09, .12], [.07, .10, .07]])
RAD_DIST = np.array([[.30, .12], [.60, .30], [.40, .22], [.38, .20], [.08, .04]])
CLOUD = np.array([.75, .78, .80])

rng = np.random.default_rng(42)
yy, xx = np.mgrid[:N, :N]


def make_mask(c):
    m = np.zeros((N, N), bool)
    if c == 0:  # persegi panjang = pembukaan lahan
        w, h = rng.integers(8, 12), rng.integers(7, 11)
        x0, y0 = rng.integers(0, N - w + 1), rng.integers(0, N - h + 1)
        m[y0:y0 + h, x0:x0 + w] = True
    elif c in (1, 3):  # gumpalan tak beraturan
        cx, cy = rng.uniform(5, 11, 2)
        rx, ry = rng.uniform(4, 6, 2) if c == 1 else rng.uniform(3.5, 6.5, 2)
        jag = 0.7 if c == 1 else 1.0
        m = np.hypot((xx - cx) / rx, (yy - cy) / ry) + jag * (rng.random((N, N)) - .5) < 1
    elif c == 2:  # petak kecil tersebar
        for _ in range(7):
            s = rng.integers(2, 4)
            x, y = rng.integers(0, N - s + 1, 2)
            m[y:y + s, x:x + s] = True
    else:  # pita berkelok di sepanjang sungai
        ph, w, off = rng.uniform(0, 6.28), rng.uniform(2, 3.5), rng.uniform(6, 10)
        m = np.abs(yy - (off + 3 * np.sin(xx / 3 + ph))) < w
    return m


def make_sample(c):
    m = make_mask(c)
    opt = np.where(m, OPT_DIST[c][:, None, None], OPT_FOREST[:, None, None]) \
        + .03 * rng.normal() + .04 * rng.normal(size=(3, N, N))
    rad = np.where(m, RAD_DIST[c][:, None, None], RAD_FOREST[:, None, None]) \
        * np.exp(.3 * rng.normal(size=(2, N, N)))  # bintik (speckle) radar
    field = zoom(rng.random((4, 4)), N / 4, order=1).ravel()
    rank = np.empty(P)
    rank[np.argsort(field + 1e-9 * rng.random(P))] = np.arange(P) / P  # urutan: piksel awan duluan
    return dict(c=c, opt=opt, rad=rad, rank=rank.reshape(N, N), cn=.03 * rng.normal(size=(3, N, N)))


def apply_cloud(s, cov):
    cl = s["rank"] < cov
    return np.where(cl, CLOUD[:, None, None] + s["cn"], s["opt"])


def stats(a):
    return [a.mean(), a.std(), a.min(), a.max()]


def feats(s, mode, cov):
    f = []
    if mode in (0, 2):
        o = apply_cloud(s, cov)
        for b in range(3):
            f += stats(o[b])
    if mode in (1, 2):
        for b in range(2):
            f += stats(s["rad"][b])
    return f


def dataset(n_per_class):
    return [make_sample(c) for c in range(5) for _ in range(n_per_class)]


TRAIN, TEST = dataset(150), dataset(40)
TRAIN_COV = rng.uniform(0, .9, len(TRAIN))  # latih dengan tutupan awan acak
MODELS = []
for mode in range(3):
    X = [feats(s, mode, cv) for s, cv in zip(TRAIN, TRAIN_COV)]
    MODELS.append(RandomForestClassifier(80, random_state=0).fit(X, [s["c"] for s in TRAIN]))
Y_TEST = np.array([s["c"] for s in TEST])


def evaluate(mode, cov):
    pred = MODELS[mode].predict([feats(s, mode, cov) for s in TEST])
    return float((pred == Y_TEST).mean()), confusion_matrix(Y_TEST, pred, labels=range(5)).tolist()


CURVE_X = list(range(0, 101, 5))
CURVE = [[evaluate(m, x / 100)[0] for x in CURVE_X] for m in range(3)]


def to_rgb(arr):
    return (np.clip(arr, 0, 1) * 255).astype(int).transpose(1, 2, 0).reshape(-1).tolist()


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", classes=CLASSES, modes=MODES)


@app.get("/api/run")
def run():
    cov = min(max(request.args.get("cloud", 0, type=int), 0), 100) / 100
    k = min(max(request.args.get("sample", 0, type=int), 0), 39)
    results = []
    for m in range(3):
        acc, cm = evaluate(m, cov)
        results.append(dict(mode=MODES[m], acc=acc, cm=cm))
    previews = []
    for c in range(5):
        s = TEST[c * 40 + k]
        o = apply_cloud(s, cov)[[2, 0, 1]] * 1.4  # false-color: NIR, merah, hijau
        r = np.repeat(np.clip(s["rad"][0:1] * 1.4, 0, 1), 3, axis=0)
        previews.append(dict(cls=CLASSES[c], optical=to_rgb(o), radar=to_rgb(r)))
    return jsonify(cloud=int(cov * 100), n_test=len(TEST), results=results, previews=previews,
                   curve=dict(x=CURVE_X, y=CURVE), classes=CLASSES)


if __name__ == "__main__":
    app.run(debug=False)
