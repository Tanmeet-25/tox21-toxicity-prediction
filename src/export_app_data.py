"""Build the JSON payload the web app ships with: real structures, labels, predictions, stats."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BUILD = ROOT / "src"

import json, numpy as np, pandas as pd, csv
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
EPS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
       "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
FULL = {
    "NR-AR": "Androgen receptor, full", "NR-AR-LBD": "Androgen receptor, ligand-binding domain",
    "NR-AhR": "Aryl hydrocarbon receptor", "NR-Aromatase": "Aromatase (CYP19A1)",
    "NR-ER": "Estrogen receptor, full", "NR-ER-LBD": "Estrogen receptor, ligand-binding domain",
    "NR-PPAR-gamma": "Peroxisome proliferator-activated receptor γ",
    "SR-ARE": "Antioxidant response element", "SR-ATAD5": "ATAD5 genotoxicity reporter",
    "SR-HSE": "Heat-shock response element", "SR-MMP": "Mitochondrial membrane potential",
    "SR-p53": "p53 tumour-suppressor response"}

df = pd.read_csv(str(BUILD / "parsed.csv"))
P = np.load(str(BUILD / "P_all.npy"))
rep = json.load(open(str(BUILD / "model_report.json")))
test_idx = set(rep["test_idx"])

Y = df[EPS].to_numpy(dtype=float)

# ---------------------------------------------------------------- dataset stats
stats = []
for t, e in enumerate(EPS):
    col = Y[:, t]; tested = ~np.isnan(col); act = int(np.nansum(col))
    stats.append({"id": e, "name": FULL[e], "family": "NR" if e.startswith("NR") else "SR",
                  "tested": int(tested.sum()), "active": act,
                  "activeRate": round(act / tested.sum() * 100, 2),
                  "missingRate": round((len(df) - tested.sum()) / len(df) * 100, 2),
                  "heldOutRoc": round(rep["per_endpoint"][e]["roc"], 3),
                  "heldOutPr": round(rep["per_endpoint"][e]["pr"], 3)})

# ---------------------------------------------------------------- benchmarks
bench = []
for r in csv.DictReader(open(str(DATA / "Tox21_v2_results_with_CI.csv"))):
    bench.append({"model": r["Model"], "endpoint": r["Endpoint"],
                  "roc": float(r["ROC_AUC"]), "rocLo": float(r["ROC_lo"]), "rocHi": float(r["ROC_hi"]),
                  "pr": float(r["PR_AUC"]), "prLo": float(r["PR_lo"]), "prHi": float(r["PR_hi"]),
                  "tested": int(r["Tested"]), "active": int(r["Active"])})

# ---------------------------------------------------------------- pick compounds
cand = []
for i in range(len(df)):
    if i not in test_idx:
        continue
    s = df["canonical_smiles"].iloc[i]
    if not isinstance(s, str): continue
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    na = m.GetNumAtoms()
    if not (6 <= na <= 38): continue
    tested = int((~np.isnan(Y[i])).sum())
    if tested < 9: continue
    cand.append((i, m, tested, int(np.nansum(Y[i]))))

cand.sort(key=lambda c: (-c[2], -c[3]))
# keep a spread: every risk level represented
buckets = {}
for c in cand:
    buckets.setdefault(min(c[3], 4), []).append(c)
picked = []
for k in sorted(buckets, reverse=True):
    picked += buckets[k][:70]
picked = picked[:300]
print("compounds selected:", len(picked),
      "| active-count spread:", {k: min(len(v), 70) for k, v in sorted(buckets.items())})

compounds = []
for i, m, tested, nact in picked:
    mol = Chem.Mol(m)
    try:
        # kekulise so aromatic rings draw as alternating single/double bonds
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        continue
    conf = mol.GetConformer()
    pts = np.array([[conf.GetAtomPosition(a).x, conf.GetAtomPosition(a).y]
                    for a in range(mol.GetNumAtoms())])
    pts -= pts.min(0)
    span = max(pts.max(), 1e-6)
    pts = np.round(pts / span * 1000).astype(int)
    atoms = [[a.GetSymbol(), int(pts[k][0]), int(pts[k][1]), int(a.GetFormalCharge())]
             for k, a in enumerate(mol.GetAtoms())]
    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
              {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
               Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}.get(b.GetBondType(), 1)]
             for b in mol.GetBonds()]
    labels = [None if np.isnan(Y[i][t]) else int(Y[i][t]) for t in range(12)]
    props = {}
    for key, fn in [("mw", Descriptors.MolWt), ("logp", Descriptors.MolLogP),
                    ("tpsa", Descriptors.TPSA), ("hbd", Descriptors.NumHDonors),
                    ("hba", Descriptors.NumHAcceptors),
                    ("rotb", Descriptors.NumRotatableBonds),
                    ("rings", Descriptors.RingCount),
                    ("heavy", Descriptors.HeavyAtomCount)]:
        try: props[key] = round(float(fn(m)), 2)
        except Exception: props[key] = None
    try: props["formula"] = rdMolDescriptors.CalcMolFormula(m)
    except Exception: props["formula"] = ""
    compounds.append({
        "id": str(df["mol_id"].iloc[i]),
        "smiles": df["canonical_smiles"].iloc[i],
        "atoms": atoms, "bonds": bonds,
        "labels": labels,
        "pred": [round(float(P[i][t]), 4) for t in range(12)],
        "nTested": tested, "nActive": nact, "props": props})

payload = {
    "meta": {
        "compounds": int(len(df)),
        "endpoints": 12,
        "scaffolds": rep["split"]["scaffolds"],
        "split": rep["split"],
        "testMacroRoc": round(rep["test_macro"], 4),
        "valMacroRoc": round(rep["val_macro"], 4),
        "featureDim": 2057,
        "shown": len(compounds)},
    "endpoints": stats,
    "benchmarks": bench,
    "compounds": compounds}

out = str(BUILD / "app_data.json")
json.dump(payload, open(out, "w"), separators=(",", ":"))
import os
print("wrote", out, round(os.path.getsize(out) / 1024, 1), "KB")
print("macro test ROC", payload["meta"]["testMacroRoc"])
