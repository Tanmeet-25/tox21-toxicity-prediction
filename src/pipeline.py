"""Featurise Tox21, scaffold-split, train a masked multi-task MLP, export app data.

The network mirrors the notebook's shared trunk (512 -> 256 -> 128 -> 12 sigmoid heads)
and its masked, class-weighted binary cross-entropy: untested assays contribute no gradient.
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BUILD = ROOT / "src"

import json, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, rdFingerprintGenerator as rfg
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, average_precision_score

RDLogger.DisableLog("rdApp.*")
rng = np.random.default_rng(17)
CSV = str(DATA / "tox21_cleaned_final.csv")
EPS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
       "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

# ---------------------------------------------------------------- load + parse
df = pd.read_csv(CSV)
mols, keep = [], []
for i, s in enumerate(df["canonical_smiles"].tolist()):
    if not isinstance(s, str) or not s.strip():
        continue
    m = Chem.MolFromSmiles(s)
    if m is not None and m.GetNumHeavyAtoms() >= 2:
        mols.append(m); keep.append(i)
df = df.iloc[keep].reset_index(drop=True)
print(f"parsed {len(df)} / {len(keep)} compounds")

# ---------------------------------------------------------------- featurise
gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
DESC = [("MolWt", Descriptors.MolWt), ("LogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA), ("HBD", Descriptors.NumHDonors),
        ("HBA", Descriptors.NumHAcceptors), ("RotB", Descriptors.NumRotatableBonds),
        ("Rings", Descriptors.RingCount), ("CSP3", Descriptors.FractionCSP3),
        ("HeavyAtoms", Descriptors.HeavyAtomCount)]

fp = np.zeros((len(mols), 2048), dtype=np.float32)
dsc = np.zeros((len(mols), len(DESC)), dtype=np.float32)
for i, m in enumerate(mols):
    fp[i] = gen.GetFingerprintAsNumPy(m)
    for j, (_, f) in enumerate(DESC):
        try: dsc[i, j] = f(m)
        except Exception: dsc[i, j] = 0.0
dsc = np.clip(dsc, np.percentile(dsc, 1, axis=0), np.percentile(dsc, 99, axis=0))
dsc = (dsc - dsc.mean(0)) / (dsc.std(0) + 1e-8)
X = np.hstack([fp, dsc]).astype(np.float32)
print("features", X.shape)

Y = df[EPS].to_numpy(dtype=np.float32)
M = (~np.isnan(Y)).astype(np.float32)          # 1 = tested, 0 = missing
Y = np.nan_to_num(Y)

# ---------------------------------------------------------------- scaffold split
scaf = []
for m in mols:
    try: scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False))
    except Exception: scaf.append("")
groups = {}
for i, s in enumerate(scaf):
    groups.setdefault(s, []).append(i)
order = sorted(groups.values(), key=len, reverse=True)
n = len(df); want_v, want_t = int(.15 * n), int(.15 * n)
tr, va, te = [], [], []
for gi, idx in enumerate(order):
    if len(idx) > 40 or gi < 3:                 # giant scaffolds anchor train
        tr += idx
    elif len(va) < want_v and gi % 3 == 1: va += idx
    elif len(te) < want_t and gi % 3 == 2: te += idx
    else: tr += idx
tr, va, te = np.array(tr), np.array(va), np.array(te)
assert not (set(scaf[i] for i in tr) & set(scaf[i] for i in te)), "scaffold leak"
print(f"split  train {len(tr)}  val {len(va)}  test {len(te)}  "
      f"scaffolds {len(groups)}  leak 0")

# ---------------------------------------------------------------- masked multi-task MLP
def init(a, b): return [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32),
                        np.zeros(b, dtype=np.float32)]
SIZES = [X.shape[1], 512, 256, 128, 12]
W = [init(SIZES[i], SIZES[i + 1]) for i in range(4)]
mW = [[np.zeros_like(w), np.zeros_like(b)] for w, b in W]
vW = [[np.zeros_like(w), np.zeros_like(b)] for w, b in W]
DROP = [0.30, 0.30, 0.20]

pos = (Y * M)[tr].sum(0); neg = M[tr].sum(0) - pos
cw = np.sqrt(np.clip(neg / np.maximum(pos, 1), 1, None)).astype(np.float32)  # tempered
print("class weights", np.round(cw, 2))

def forward(xb, train=True):
    acts, masks, h = [xb], [], xb
    for li in range(3):
        h = np.maximum(h @ W[li][0] + W[li][1], 0)
        if train:
            dm = (rng.random(h.shape) > DROP[li]).astype(np.float32) / (1 - DROP[li])
            h = h * dm; masks.append(dm)
        else: masks.append(None)
        acts.append(h)
    z = h @ W[3][0] + W[3][1]
    return acts, masks, z

def adam(GRAD, step, lr=1.2e-3, b1=.9, b2=.999, eps=1e-8):
    for li in range(4):
        for pi in range(2):
            g = GRAD[li][pi]
            mW[li][pi] = b1 * mW[li][pi] + (1 - b1) * g
            vW[li][pi] = b2 * vW[li][pi] + (1 - b2) * g * g
            mh = mW[li][pi] / (1 - b1 ** step); vh = vW[li][pi] / (1 - b2 ** step)
            W[li][pi] -= lr * mh / (np.sqrt(vh) + eps)

def macro_auc(idx):
    _, _, z = forward(X[idx], train=False)
    p = 1 / (1 + np.exp(-z))
    out = []
    for t in range(12):
        s = M[idx][:, t] > 0
        yt = Y[idx][s, t]
        if s.sum() > 20 and 0 < yt.sum() < s.sum():
            out.append(roc_auc_score(yt, p[s, t]))
    return float(np.mean(out)), p

best, best_W, bad, step = -1, None, 0, 0
BS = 128
for epoch in range(1, 41):
    perm = rng.permutation(tr)
    for s in range(0, len(perm), BS):
        b = perm[s:s + BS]
        if len(b) < 8: continue
        xb, yb, mb = X[b], Y[b], M[b]
        acts, masks, z = forward(xb, True)
        p = 1 / (1 + np.exp(-z))
        wt = yb * cw + (1 - yb)                      # weight positives only
        denom = max(mb.sum(), 1)
        dz = (p - yb) * wt * mb / denom
        GRAD = [[None, None] for _ in range(4)]
        g = dz
        for li in range(3, -1, -1):
            GRAD[li][0] = acts[li].T @ g
            GRAD[li][1] = g.sum(0)
            if li > 0:
                g = g @ W[li][0].T
                g = g * masks[li - 1] * (acts[li] > 0)
        step += 1; adam(GRAD, step)
    vauc, _ = macro_auc(va)
    print(f"  epoch {epoch:2d}  val macro ROC-AUC {vauc:.4f}")
    if vauc > best + 1e-4:
        best, bad = vauc, 0
        best_W = [[w.copy(), b.copy()] for w, b in W]
    else:
        bad += 1
        if bad >= 6: print("  early stop"); break
W = best_W
test_auc, P_test = macro_auc(te)
_, P_all = macro_auc(np.arange(n))
print(f"\nVAL  macro ROC-AUC {best:.4f}")
print(f"TEST macro ROC-AUC {test_auc:.4f}")

per = {}
for t, e in enumerate(EPS):
    s = M[te][:, t] > 0
    yt = Y[te][s, t]
    if s.sum() > 20 and 0 < yt.sum() < s.sum():
        per[e] = {"roc": float(roc_auc_score(yt, P_test[s, t])),
                  "pr": float(average_precision_score(yt, P_test[s, t])),
                  "tested": int(s.sum()), "active": int(yt.sum())}
        print(f"  {e:15s} ROC {per[e]['roc']:.3f}  PR {per[e]['pr']:.3f}")

np.save(str(BUILD / "P_all.npy"), P_all)
json.dump({"test_macro": test_auc, "val_macro": best, "per_endpoint": per,
           "split": {"train": len(tr), "val": len(va), "test": len(te),
                     "scaffolds": len(groups)},
           "test_idx": te.tolist()},
          open(str(BUILD / "model_report.json"), "w"), indent=1)
df.to_csv(str(BUILD / "parsed.csv"), index=False)
print("saved")
