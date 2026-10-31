#!/usr/bin/env python3
"""Independent verification of the committed PAR2 CPU de novo production artifacts.

This script does NOT import ``par2_denovo_cpu.py``. Every constant below is
re-declared from the specification written in ``HANDOFF_TO_CLAUDE.md`` sections
11 and 12, so that a disagreement between the documented method and the
committed CSVs is surfaced rather than hidden by shared code.

For every row of ``DENOVO_15000.csv`` and ``DENOVO_TOP3000_FOR_DOCKING.csv`` it
re-parses the SMILES with RDKit and recomputes the eleven descriptors, the
Morgan reference Tanimoto, and the four-term ``design_score``, then compares
against the stored columns. It also re-checks the family hard filters, the PAINS
A/B/C catalogues, the narrow reactive-group screen, canonical-SMILES
uniqueness, family allocation, scaffold caps, and the TOP3000-subset relation.

Outputs (written under ``04_VERIFICATION/``):
  DENOVO_VERIFICATION.json  full machine-readable result
  DENOVO_VERIFICATION.md    human-readable summary
  denovo_row_mismatches.csv rows whose recomputed values disagree (if any)

Read-only with respect to every production artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# --- Specification re-declared from HANDOFF_TO_CLAUDE.md sections 11 and 12 ---

FAMILY_CFG = {
    "orthosteric": {
        "target": dict(mw=360, logp=3.0, tpsa=65, hba=3.5, hbd=1.0,
                       rotb=4.0, fsp3=0.35, arom=2.0, refsim=0.38),
        "sigma": dict(mw=85, logp=1.0, tpsa=25, hba=2.0, hbd=1.3,
                      rotb=3.0, fsp3=0.22, arom=1.2, refsim=0.20),
        "hard": dict(mw=(230, 500), logp=(1.0, 5.0), tpsa=(25, 115),
                     hba=(1, 9), hbd=(0, 3), rotb=(0, 9), charge=(-1, 1)),
    },
    "allosteric": {
        "target": dict(mw=375, logp=3.7, tpsa=52, hba=3.0, hbd=0.7,
                       rotb=3.5, fsp3=0.30, arom=2.0, refsim=0.35),
        "sigma": dict(mw=90, logp=1.0, tpsa=24, hba=1.8, hbd=1.1,
                      rotb=2.8, fsp3=0.22, arom=1.2, refsim=0.20),
        "hard": dict(mw=(240, 510), logp=(1.8, 5.5), tpsa=(18, 100),
                     hba=(1, 8), hbd=(0, 3), rotb=(0, 8), charge=(-1, 1)),
    },
    "exploratory": {
        "target": dict(mw=350, logp=3.1, tpsa=62, hba=3.5, hbd=1.0,
                       rotb=4.0, fsp3=0.32, arom=2.0, refsim=0.25),
        "sigma": dict(mw=110, logp=1.4, tpsa=34, hba=2.5, hbd=1.7,
                      rotb=4.0, fsp3=0.28, arom=1.5, refsim=0.25),
        "hard": dict(mw=(200, 520), logp=(0.5, 5.7), tpsa=(15, 125),
                     hba=(0, 10), hbd=(0, 4), rotb=(0, 10), charge=(-1, 1)),
    },
}

REACTIVE_SMARTS = {
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "sulfonyl_halide": "S(=O)(=O)[F,Cl,Br,I]",
    "isocyanate": "N=C=O",
    "isothiocyanate": "N=C=S",
    "peroxide": "[OX2]-[OX2]",
    "diazo": "[N-]=[N+]=C",
}

# Documented campaign expectations (handoff sections 9 and 15).
EXPECTED_FAMILY_ALLOCATION = {"orthosteric": 10000, "allosteric": 3500, "exploratory": 1500}
EXPECTED_TOTAL = 15000
EXPECTED_TOP_DOCKING = 3000
TOP_DOCKING_SCAFFOLD_CAP = 12
HEAVY_ATOM_RANGE = (12, 44)

# Routing contract from handoff section 17.
FAMILY_ROUTING = {
    "orthosteric": ["5NDD"],
    "allosteric": ["5NDZ"],
    "exploratory": ["5NDD", "5NDZ"],
}

# Numeric agreement tolerances. Descriptors are float32-ish through CSV text, so
# compare with a small absolute tolerance rather than exact equality.
TOL = {
    "design_score": 1e-6,
    "ref_tanimoto": 1e-6,
    "mw": 1e-3,
    "logp": 1e-3,
    "tpsa": 1e-3,
    "fsp3": 1e-6,
    "qed": 1e-6,
}
INT_FIELDS = ("hba", "hbd", "rotb", "rings", "arom", "charge")

# Per-worker globals.
MORGAN = None
REF_FP: Dict[str, object] = {}
PAINS: Optional[FilterCatalog] = None
REACTIVE: Dict[str, object] = {}


def gaussian(x: float, target: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - target) / sigma) ** 2)


def init_worker(ref_smiles: Dict[str, str]) -> None:
    global MORGAN, REF_FP, PAINS, REACTIVE
    MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    REF_FP = {}
    for family, smi in ref_smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            REF_FP[family] = MORGAN.GetFingerprint(mol)
    params = FilterCatalogParams()
    for name in ("PAINS_A", "PAINS_B", "PAINS_C"):
        enum = getattr(FilterCatalogParams.FilterCatalogs, name, None)
        if enum is not None:
            params.AddCatalog(enum)
    PAINS = FilterCatalog(params)
    REACTIVE = {k: Chem.MolFromSmarts(v) for k, v in REACTIVE_SMARTS.items()}


def descriptors(m: Chem.Mol) -> Dict[str, float]:
    return {
        "mw": float(Descriptors.MolWt(m)),
        "logp": float(Crippen.MolLogP(m)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
        "hba": int(Lipinski.NumHAcceptors(m)),
        "hbd": int(Lipinski.NumHDonors(m)),
        "rotb": int(Lipinski.NumRotatableBonds(m)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(m)),
        "rings": int(rdMolDescriptors.CalcNumRings(m)),
        "arom": int(rdMolDescriptors.CalcNumAromaticRings(m)),
        "charge": int(Chem.GetFormalCharge(m)),
        "qed": float(QED.qed(m)),
    }


def hard_pass(d: Dict[str, float], family: str) -> bool:
    h = FAMILY_CFG[family]["hard"]
    for key in ("mw", "logp", "tpsa", "hba", "hbd", "rotb", "charge"):
        lo, hi = h[key]
        if not lo <= d[key] <= hi:
            return False
    return d["rings"] <= 6


def feature_score(d: Dict[str, float], family: str) -> float:
    if family == "orthosteric":
        return float(np.mean([
            gaussian(d["hba"], 3.5, 2.0),
            gaussian(d["arom"], 2.0, 1.2),
            gaussian(d["fsp3"], 0.35, 0.22),
        ]))
    if family == "allosteric":
        return float(np.mean([
            gaussian(d["hba"], 3.0, 1.8),
            gaussian(d["arom"], 2.0, 1.2),
            gaussian(d["logp"], 3.7, 1.0),
            gaussian(d["fsp3"], 0.30, 0.22),
        ]))
    return float(np.mean([
        gaussian(d["hba"], 3.5, 2.5),
        gaussian(d["fsp3"], 0.32, 0.28),
    ]))


def recompute_row(task: Tuple[int, str, str]) -> Dict:
    """Recompute one row from its SMILES alone. Returns a findings dict."""
    idx, smi, family = task
    out: Dict[str, object] = {"index": idx, "family": family, "parse_ok": False}
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return out
    out["parse_ok"] = True
    out["canonical_smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    out["heavy_atoms"] = int(mol.GetNumHeavyAtoms())

    out["reactive_hits"] = [k for k, p in REACTIVE.items()
                            if p is not None and mol.HasSubstructMatch(p)]
    out["pains_hit"] = bool(PAINS is not None and PAINS.HasMatch(mol))

    d = descriptors(mol)
    out.update({f"calc_{k}": v for k, v in d.items()})
    out["hard_pass"] = hard_pass(d, family) if family in FAMILY_CFG else False

    fp = MORGAN.GetFingerprint(mol)
    if family == "exploratory":
        refsim = max(DataStructs.TanimotoSimilarity(fp, REF_FP["orthosteric"]),
                     DataStructs.TanimotoSimilarity(fp, REF_FP["allosteric"]))
    elif family in REF_FP:
        refsim = DataStructs.TanimotoSimilarity(fp, REF_FP[family])
    else:
        return out
    out["calc_ref_tanimoto"] = float(refsim)

    cfg = FAMILY_CFG[family]
    terms = [gaussian(d[k], cfg["target"][k], cfg["sigma"][k])
             for k in ("mw", "logp", "tpsa", "hba", "hbd", "rotb", "fsp3", "arom")]
    property_score = float(np.mean(terms))
    novelty = gaussian(refsim, cfg["target"]["refsim"], cfg["sigma"]["refsim"])
    feature = feature_score(d, family)
    out["calc_design_score"] = (0.50 * property_score + 0.20 * d["qed"]
                               + 0.20 * novelty + 0.10 * feature)
    out["calc_scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=True) or out["canonical_smiles"]
    return out


def load_reference_smiles(v2_root: Path) -> Dict[str, str]:
    lig = v2_root / "01_REFERENCE_POCKETS" / "ligands"
    paths = {
        "orthosteric": lig / "5NDD_AZ8838_crystallographic_clean.sdf",
        "allosteric": lig / "5NDZ_AZ3451_crystallographic_clean.sdf",
    }
    ref: Dict[str, str] = {}
    for family, path in paths.items():
        if not path.exists():
            raise SystemExit(f"Missing reference ligand: {path}")
        mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]
        if not mols:
            raise SystemExit(f"No valid molecule in {path}")
        mol = mols[0]
        Chem.SanitizeMol(mol)
        ref[family] = Chem.MolToSmiles(Chem.RemoveHs(mol), True)
    return ref


def verify_table(df: pd.DataFrame, ref_smiles: Dict[str, str], workers: int,
                 label: str) -> Tuple[Dict, pd.DataFrame]:
    """Recompute every row of df and compare against the stored columns."""
    tasks = [(i, str(r.smiles), str(r.family))
             for i, r in enumerate(df.itertuples(index=False))]
    ctx = mp.get_context("fork" if platform.system() == "Linux" else "spawn")
    with ctx.Pool(workers, initializer=init_worker, initargs=(ref_smiles,)) as pool:
        results = pool.map(recompute_row, tasks, chunksize=128)

    mismatches: List[Dict] = []
    unparsed, pains_hits, reactive_hits, hard_fails, heavy_out = [], [], [], [], []

    for res in results:
        i = res["index"]
        stored = df.iloc[i]
        if not res["parse_ok"]:
            unparsed.append(int(i))
            continue
        if res["pains_hit"]:
            pains_hits.append(int(i))
        if res["reactive_hits"]:
            reactive_hits.append({"index": int(i), "groups": res["reactive_hits"]})
        if not res["hard_pass"]:
            hard_fails.append(int(i))
        lo, hi = HEAVY_ATOM_RANGE
        if not lo <= res["heavy_atoms"] <= hi:
            heavy_out.append({"index": int(i), "heavy_atoms": res["heavy_atoms"]})

        bad: Dict[str, object] = {}
        for field, tol in TOL.items():
            key = f"calc_{field}"
            if key not in res or field not in stored.index:
                continue
            delta = abs(float(res[key]) - float(stored[field]))
            if delta > tol:
                bad[field] = {"stored": float(stored[field]),
                              "recomputed": float(res[key]), "abs_delta": delta}
        for field in INT_FIELDS:
            key = f"calc_{field}"
            if key not in res or field not in stored.index:
                continue
            if int(res[key]) != int(stored[field]):
                bad[field] = {"stored": int(stored[field]), "recomputed": int(res[key])}
        if "scaffold" in stored.index and res.get("calc_scaffold") != str(stored["scaffold"]):
            bad["scaffold"] = {"stored": str(stored["scaffold"]),
                               "recomputed": res.get("calc_scaffold")}
        if bad:
            mismatches.append({"index": int(i), "smiles": str(stored.smiles),
                               "family": str(stored.family), "fields": bad})

    canon = [r.get("canonical_smiles") for r in results if r["parse_ok"]]
    stored_smiles = df["smiles"].astype(str).tolist()

    summary = {
        "label": label,
        "rows": int(len(df)),
        "unparsable_smiles": len(unparsed),
        "unparsable_indices": unparsed[:50],
        "stored_smiles_unique": int(len(set(stored_smiles))),
        "recomputed_canonical_unique": int(len(set(canon))),
        "smiles_already_canonical": bool(set(canon) == set(stored_smiles)),
        "family_counts": {k: int(v) for k, v in
                          Counter(df["family"].astype(str)).items()},
        "rows_failing_family_hard_filters": len(hard_fails),
        "rows_failing_hard_filter_indices": hard_fails[:50],
        "pains_positive_rows": len(pains_hits),
        "pains_positive_indices": pains_hits[:50],
        "reactive_group_rows": len(reactive_hits),
        "reactive_group_detail": reactive_hits[:50],
        "heavy_atom_range_violations": len(heavy_out),
        "heavy_atom_violation_detail": heavy_out[:50],
        "rows_with_value_mismatch": len(mismatches),
        "mismatch_fields": dict(Counter(
            f for m in mismatches for f in m["fields"])),
        "unique_scaffolds": int(df["scaffold"].astype(str).nunique())
                            if "scaffold" in df.columns else None,
        "max_scaffold_repeat": int(Counter(df["scaffold"].astype(str))
                                   .most_common(1)[0][1])
                               if "scaffold" in df.columns and len(df) else None,
    }
    mism_df = pd.DataFrame([
        {"table": label, "index": m["index"], "family": m["family"],
         "smiles": m["smiles"], "field": f,
         "stored": v.get("stored"), "recomputed": v.get("recomputed"),
         "abs_delta": v.get("abs_delta")}
        for m in mismatches for f, v in m["fields"].items()
    ])
    return summary, mism_df


def descriptor_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    fields = ["design_score", "ref_tanimoto", "mw", "logp", "tpsa",
              "hba", "hbd", "rotb", "fsp3", "qed"]
    out = {}
    for f in fields:
        if f in df.columns:
            s = pd.to_numeric(df[f], errors="coerce")
            out[f] = {"min": float(s.min()), "median": float(s.median()),
                      "mean": float(s.mean()), "max": float(s.max())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-root", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="YONSEI_FINAL10_V2 directory (default: parent of scripts/)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <v2-root>/04_VERIFICATION)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    v2 = args.v2_root.resolve()
    out_dir = (args.out_dir or (v2 / "04_VERIFICATION")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = v2 / "02_DENOVO_CPU" / "DENOVO_15000.csv"
    top_path = v2 / "02_DENOVO_CPU" / "DENOVO_TOP3000_FOR_DOCKING.csv"
    for p in (full_path, top_path):
        if not p.exists():
            raise SystemExit(f"Missing production artifact: {p}")

    ref_smiles = load_reference_smiles(v2)
    print(f"Reference SMILES:\n  orthosteric (AZ8838): {ref_smiles['orthosteric']}"
          f"\n  allosteric  (AZ3451): {ref_smiles['allosteric']}", flush=True)

    full = pd.read_csv(full_path)
    top = pd.read_csv(top_path)

    print(f"Verifying {len(full)} rows of DENOVO_15000.csv "
          f"with {args.workers} workers ...", flush=True)
    full_sum, full_mism = verify_table(full, ref_smiles, args.workers, "DENOVO_15000")
    print(f"Verifying {len(top)} rows of DENOVO_TOP3000_FOR_DOCKING.csv ...", flush=True)
    top_sum, top_mism = verify_table(top, ref_smiles, args.workers, "DENOVO_TOP3000")

    # Cross-table relations.
    full_smiles = set(full["smiles"].astype(str))
    top_smiles = set(top["smiles"].astype(str))
    top_scaffold_counts = Counter(top["scaffold"].astype(str))
    routing = {fam: FAMILY_ROUTING.get(fam, []) for fam in top["family"].astype(str).unique()}
    assignments = int(sum(len(FAMILY_ROUTING.get(str(f), []))
                          for f in top["family"]))

    cross = {
        "top3000_is_subset_of_15000": bool(top_smiles <= full_smiles),
        "top3000_rows_absent_from_15000": int(len(top_smiles - full_smiles)),
        "top3000_scaffold_cap": TOP_DOCKING_SCAFFOLD_CAP,
        "top3000_max_scaffold_repeat": int(top_scaffold_counts.most_common(1)[0][1])
                                       if top_scaffold_counts else 0,
        "top3000_scaffold_cap_respected":
            bool(top_scaffold_counts and
                 top_scaffold_counts.most_common(1)[0][1] <= TOP_DOCKING_SCAFFOLD_CAP),
        "top3000_family_counts": {k: int(v) for k, v in
                                  Counter(top["family"].astype(str)).items()},
        "top3000_routing": routing,
        "planned_ligand_pocket_assignments": assignments,
        "family_allocation_matches_spec":
            full_sum["family_counts"] == EXPECTED_FAMILY_ALLOCATION,
        "expected_family_allocation": EXPECTED_FAMILY_ALLOCATION,
        "total_matches_spec": bool(len(full) == EXPECTED_TOTAL),
        "top_docking_count_matches_spec": bool(len(top) == EXPECTED_TOP_DOCKING),
    }

    def table_ok(s: Dict) -> bool:
        return (s["unparsable_smiles"] == 0
                and s["rows_with_value_mismatch"] == 0
                and s["rows_failing_family_hard_filters"] == 0
                and s["pains_positive_rows"] == 0
                and s["reactive_group_rows"] == 0
                and s["heavy_atom_range_violations"] == 0
                and s["stored_smiles_unique"] == s["rows"])

    overall = bool(table_ok(full_sum) and table_ok(top_sum)
                   and cross["top3000_is_subset_of_15000"]
                   and cross["top3000_scaffold_cap_respected"]
                   and cross["family_allocation_matches_spec"]
                   and cross["total_matches_spec"]
                   and cross["top_docking_count_matches_spec"])

    result = {
        "verification_passed": overall,
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "workers": args.workers,
        },
        "reference_smiles": ref_smiles,
        "inputs": {"DENOVO_15000.csv": str(full_path),
                   "DENOVO_TOP3000_FOR_DOCKING.csv": str(top_path)},
        "full_set": full_sum,
        "docking_set": top_sum,
        "cross_table": cross,
        "descriptor_summary_full_set": descriptor_summary(full),
        "descriptor_summary_docking_set": descriptor_summary(top),
    }

    (out_dir / "DENOVO_VERIFICATION.json").write_text(json.dumps(result, indent=2))

    mism = pd.concat([d for d in (full_mism, top_mism) if not d.empty], ignore_index=True) \
        if (not full_mism.empty or not top_mism.empty) else pd.DataFrame()
    if not mism.empty:
        mism.to_csv(out_dir / "denovo_row_mismatches.csv", index=False)

    write_markdown(out_dir / "DENOVO_VERIFICATION.md", result)

    print(f"\nVerification {'PASSED' if overall else 'FAILED'}")
    print(f"  wrote {out_dir/'DENOVO_VERIFICATION.json'}")
    print(f"  wrote {out_dir/'DENOVO_VERIFICATION.md'}")
    if not mism.empty:
        print(f"  wrote {out_dir/'denovo_row_mismatches.csv'} ({len(mism)} field mismatches)")
    return 0 if overall else 1


def write_markdown(path: Path, r: Dict) -> None:
    env, full, top, cross = r["environment"], r["full_set"], r["docking_set"], r["cross_table"]
    status = "PASSED" if r["verification_passed"] else "FAILED"
    L = [
        "# Independent verification of the CPU de novo production artifacts",
        "",
        f"**Result: {status}**",
        "",
        "This report was produced by `scripts/verify_denovo_outputs.py`, which "
        "re-derives every descriptor, the Morgan reference Tanimoto, and the "
        "`design_score` from each stored SMILES using the specification in "
        "`HANDOFF_TO_CLAUDE.md` sections 11-12. It does not import the "
        "generator, so documentation/output drift is detectable.",
        "",
        "Verification is a reproducibility and internal-consistency check on the "
        "committed CSVs. It says nothing about PAR2 biological activity.",
        "",
        "## Environment",
        "",
        "| Component | Version |",
        "|---|---|",
        f"| Python | {env['python']} |",
        f"| RDKit | {env['rdkit']} |",
        f"| pandas | {env['pandas']} |",
        f"| NumPy | {env['numpy']} |",
        f"| Workers | {env['workers']} |",
        "",
        "## Reference ligands used for `ref_tanimoto`",
        "",
        "| Family | Canonical SMILES |",
        "|---|---|",
        f"| orthosteric (5NDD / AZ8838 / 8TZ) | `{r['reference_smiles']['orthosteric']}` |",
        f"| allosteric (5NDZ / AZ3451 / 8UN) | `{r['reference_smiles']['allosteric']}` |",
        "",
        "## Per-table checks",
        "",
        "| Check | DENOVO_15000 | DENOVO_TOP3000 |",
        "|---|---:|---:|",
        f"| Rows | {full['rows']} | {top['rows']} |",
        f"| Unparsable SMILES | {full['unparsable_smiles']} | {top['unparsable_smiles']} |",
        f"| Unique stored SMILES | {full['stored_smiles_unique']} | {top['stored_smiles_unique']} |",
        f"| Stored SMILES already canonical | {full['smiles_already_canonical']} | {top['smiles_already_canonical']} |",
        f"| Rows with recomputed-value mismatch | {full['rows_with_value_mismatch']} | {top['rows_with_value_mismatch']} |",
        f"| Rows failing family hard filters | {full['rows_failing_family_hard_filters']} | {top['rows_failing_family_hard_filters']} |",
        f"| PAINS A/B/C positive rows | {full['pains_positive_rows']} | {top['pains_positive_rows']} |",
        f"| Reactive-group rows | {full['reactive_group_rows']} | {top['reactive_group_rows']} |",
        f"| Heavy-atom range violations (12-44) | {full['heavy_atom_range_violations']} | {top['heavy_atom_range_violations']} |",
        f"| Unique Murcko scaffolds | {full['unique_scaffolds']} | {top['unique_scaffolds']} |",
        f"| Max scaffold repeat | {full['max_scaffold_repeat']} | {top['max_scaffold_repeat']} |",
        "",
        "## Family allocation",
        "",
        "| Family | DENOVO_15000 | Specified | DENOVO_TOP3000 | Routes to |",
        "|---|---:|---:|---:|---|",
    ]
    for fam in ("orthosteric", "allosteric", "exploratory"):
        L.append(f"| {fam} | {full['family_counts'].get(fam, 0)} | "
                 f"{cross['expected_family_allocation'][fam]} | "
                 f"{cross['top3000_family_counts'].get(fam, 0)} | "
                 f"{', '.join(FAMILY_ROUTING[fam])} |")
    L += [
        "",
        "## Cross-table relations",
        "",
        "| Check | Value |",
        "|---|---|",
        f"| TOP3000 is a subset of DENOVO_15000 | {cross['top3000_is_subset_of_15000']} |",
        f"| TOP3000 rows absent from the 15,000 | {cross['top3000_rows_absent_from_15000']} |",
        f"| TOP3000 max scaffold repeat (cap {cross['top3000_scaffold_cap']}) | {cross['top3000_max_scaffold_repeat']} |",
        f"| Scaffold cap respected | {cross['top3000_scaffold_cap_respected']} |",
        f"| Planned ligand-pocket docking assignments | {cross['planned_ligand_pocket_assignments']} |",
        f"| Family allocation matches specification | {cross['family_allocation_matches_spec']} |",
        "",
        "## Descriptor summary (recomputation-verified stored values)",
        "",
        "| Field | Min | Median | Mean | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for f, s in r["descriptor_summary_full_set"].items():
        L.append(f"| {f} | {s['min']:.4f} | {s['median']:.4f} | "
                 f"{s['mean']:.4f} | {s['max']:.4f} |")
    L += [
        "",
        "## Interpretation limits",
        "",
        "- A pass means the committed CSVs are internally consistent and "
        "reproducible from their own SMILES under the documented method.",
        "- `design_score`, `ref_tanimoto` and QED remain two-dimensional design "
        "priors. They are not probabilities of PAR2 inhibition.",
        "- No pose, GNINA score, interaction fingerprint, or procurement fact is "
        "established by this report.",
        "",
    ]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
