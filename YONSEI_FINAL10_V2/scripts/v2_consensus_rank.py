#!/usr/bin/env python3
"""Consensus ranking of the full de novo library against the PAR2 design profile.

Merges the EX8 evidence batches into one universe and ranks WITHIN each pocket.
5NDD and 5NDZ are separate distributions and are never pooled on raw values.

Three separable layers are reported, never silently blended:
  1. structural evidence  - GNINA affinity, CNN features, ligand efficiency,
     repaired reference-antagonist IFP recovery, optional PIGNet2
  2. design-profile fit   - the physicochemical envelope and required pocket
     contacts for a silent orthosteric PAR2 antagonist
  3. risk flags           - agonist-mimicry geometry, reactive chemistry,
     cationic centres, and metric disagreement

Correlated metrics are detected and down-weighted so two views of the same
signal do not count as independent evidence.

No score here is a probability of antagonism. Static docking cannot distinguish
an antagonist from a biased agonist; that requires the functional assays.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ---------------------------------------------------------------------------
# PAR2 design profile. Sources: the 5NDD/AZ8838 and 5NDZ/AZ3451 structural
# literature and the project's target-profile brief. The orthosteric lane is
# primary because competitive orthosteric antagonists give complete monophasic
# trypsin blockade, whereas the extrahelical NAM leaks biphasically.
# ---------------------------------------------------------------------------
PROFILE = {
    "5NDD_orthosteric": dict(
        mw=(350, 480), mw_ideal=(380, 420),
        clogp=(1.5, 3.0), tpsa=(70, 95),
        hbd_max=2, hba_max=5, aromatic_rings_max=2,
        fsp3_min=0.35, rotb_max=5,
        # Protomer- and tautomer-robust anchor set. Exactly these three residues
        # place a polar atom within hydrogen-bonding distance of the
        # crystallographic ligand (2.475, 2.630, 2.929 A); the next nearest polar
        # contact is 4.067 A. WHICH of them ProLIF calls a hydrogen bond depends
        # on the ligand imidazole protomer and the receptor histidine tautomer,
        # so they are scored by count satisfied, not as a fixed pair. The former
        # ["ASP:228","HIS:135"] pair is retired: it omitted TYR:82, the shortest
        # polar contact in the pocket, and its population was 95% disjoint from
        # the crystallographic one (Jaccard 0.051).
        anchor_residues=["TYR:82", "HIS:135", "ASP:228"],
        # Pocket lining verified by heavy-atom distance to the crystal ligand.
        # TYR:311 (8.7 A) and LEU:230 (10.6 A) were removed: they do not line
        # this pocket and were never satisfiable.
        contact_residues=["LYS:131", "ILE:327", "PHE:155", "CYS:226", "ILE:152",
                          "HIS:227", "TRP:141", "TYR:134", "TYR:156", "LEU:330",
                          "LEU:78"],
        # TYR:323 removed at 6.7 A; TYR:326 retained at 4.81 A.
        switch_residues=["TYR:326"],
    ),
    "5NDZ_allosteric": dict(
        mw=(300, 460), mw_ideal=(320, 400),
        clogp=(3.0, 5.0), tpsa=(40, 75),
        hbd_max=2, hba_max=6, aromatic_rings_max=4,
        fsp3_min=0.15, rotb_max=6,
        # This site forms no hydrogen bond to its own crystallographic ligand,
        # so it has no anchor set. Closest polar approach is TYR:210 at 3.393 A.
        anchor_residues=[],
        # GLY:157 removed: residue 157 is ALA, so that key matched 0 of 3,275
        # designs and silently capped this pocket's contact fraction at 0.8.
        contact_residues=["PHE:154", "ALA:157", "ALA:120", "LEU:123",
                          "CYS:161", "TRP:199", "ILE:202", "LEU:203",
                          "TYR:210", "VAL:211"],
        switch_residues=[],
    ),
}
POCKET_WEIGHT = {"5NDD_orthosteric": 1.00, "5NDZ_allosteric": 0.80}

# The agonist-mimicry proxy has been RETIRED. It required LEU:230 and TYR:311,
# which sit 10.6 A and 8.7 A from the crystallographic ligand and therefore do
# not line this pocket at all. It fired on 9 of 12,341 rows because it was very
# nearly unsatisfiable, so it measured nothing. Antagonist versus biased agonist
# cannot be settled by static docking; it needs the functional assays, above all
# G12/13 -> RhoA.
AGONIST_PATTERN: set[str] = set()

METRICS = [
    ("best_vina_affinity", False, "physics"),
    ("ligand_efficiency", True, "physics"),
    ("pignet2_best", False, "physics"),
    ("best_CNNscore", True, "cnn"),
    ("best_CNNaffinity", True, "cnn"),
    ("best_CNN_VS", True, "cnn"),
    ("max_ifp_recovery", True, "ifp"),
]


def window(value, low, high) -> float:
    """1.0 inside the window, tapering linearly to 0 one half-width outside."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if low <= value <= high:
        return 1.0
    span = max((high - low) / 2.0, 1e-9)
    return float(max(0.0, 1.0 - (low - value if value < low else value - high) / span))


def profile_fit(row) -> tuple[float, list[str]]:
    spec = PROFILE.get(row.pocket)
    if spec is None:
        return 0.0, ["unknown pocket"]
    terms, misses = [], []
    terms.append(window(row.mw, *spec["mw"]))
    if not (spec["mw_ideal"][0] <= (row.mw or 0) <= spec["mw_ideal"][1]):
        misses.append("MW outside ideal band")
    terms.append(window(row.clogp, *spec["clogp"]))
    if not (spec["clogp"][0] <= (row.clogp or 0) <= spec["clogp"][1]):
        misses.append("cLogP outside band")
    terms.append(window(row.tpsa, *spec["tpsa"]))
    terms.append(1.0 if (row.hbd or 0) <= spec["hbd_max"] else 0.0)
    terms.append(1.0 if (row.hba or 0) <= spec["hba_max"] else 0.0)
    terms.append(1.0 if (row.rotb or 0) <= spec["rotb_max"] else 0.0)
    terms.append(min(1.0, (row.fsp3 or 0) / spec["fsp3_min"]) if spec["fsp3_min"] else 1.0)
    if (row.fsp3 or 0) < spec["fsp3_min"]:
        misses.append("fsp3 below target (too flat)")
    arom = getattr(row, "aromatic_rings", None)
    if arom is not None and not pd.isna(arom):
        terms.append(1.0 if arom <= spec["aromatic_rings_max"] else 0.0)
        if arom > spec["aromatic_rings_max"]:
            misses.append("too many aromatic rings")
    if (row.formal_charge or 0) > 0:
        misses.append("cationic centre (MRGPRX1 itch risk)")
    return float(np.mean(terms)), misses


def contact_scores(keys: str, pocket: str) -> dict:
    spec = PROFILE.get(pocket, {})
    present = set()
    hbond_residues = set()
    for key in (keys or "").split(";"):
        if not key:
            continue
        residue, _, interaction = key.partition("|")
        base = ":".join(residue.split(":")[:2])
        present.add(base)
        if interaction in ("HBDonor", "HBAcceptor"):
            hbond_residues.add(base)
    anchors = spec.get("anchor_residues", [])
    contacts = spec.get("contact_residues", [])
    switches = spec.get("switch_residues", [])
    anchor_hits = [r for r in anchors if r in hbond_residues]
    anchor_any = [r for r in anchors if r in present]
    return dict(
        anchor_hbond_fraction=(len(anchor_hits) / len(anchors)) if anchors else np.nan,
        anchor_contact_fraction=(len(anchor_any) / len(anchors)) if anchors else np.nan,
        anchor_hbond_residues=";".join(anchor_hits),
        contact_fraction=(len([r for r in contacts if r in present]) / len(contacts))
        if contacts else np.nan,
        switch_contact=bool([r for r in switches if r in present]),
        agonist_pattern_flag=bool(AGONIST_PATTERN) and AGONIST_PATTERN.issubset(present),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, nargs="+", required=True,
                        help="one or more EVIDENCE_*.csv tables to merge")
    parser.add_argument("--ifp-keys", type=Path, nargs="*", default=[],
                        help="V2_POSE_INTERACTIONS_SUMMARY.csv files carrying ifp_keys")
    parser.add_argument("--pignet", type=Path, default=None,
                        help="optional PIGNET2 molecule summary")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(p) for p in args.evidence]
    e = pd.concat(frames, ignore_index=True)
    before = len(e)
    e = e.drop_duplicates(["cand_id", "pocket"], keep="first")
    print(f"[merge] {before} rows from {len(frames)} batches -> {len(e)} unique ligand-pocket rows")
    print(f"[merge] unique designs: {e.cand_id.nunique()}; by batch: "
          f"{e.batch.value_counts().to_dict() if 'batch' in e else 'n/a'}")

    if args.ifp_keys:
        keys = pd.concat([pd.read_csv(p, usecols=["cand_id", "pocket", "ifp_keys"])
                          for p in args.ifp_keys], ignore_index=True)
        keys = keys.drop_duplicates(["cand_id", "pocket"], keep="first")
        e = e.merge(keys, on=["cand_id", "pocket"], how="left")
    elif "ifp_keys" not in e.columns:
        e["ifp_keys"] = ""

    if args.pignet and args.pignet.exists():
        pig = pd.read_csv(args.pignet)
        col = next((c for c in pig.columns if "best" in c.lower() and "ens" in c.lower()),
                   next((c for c in pig.columns if c.lower().startswith("best")), None))
        if col:
            pig = pig.rename(columns={col: "pignet2_best"})
            e = e.merge(pig[["cand_id", "pocket", "pignet2_best"]],
                        on=["cand_id", "pocket"], how="left")
    if "pignet2_best" not in e:
        e["pignet2_best"] = np.nan

    if "aromatic_rings" not in e:
        e["aromatic_rings"] = [
            rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(s))
            if isinstance(s, str) and Chem.MolFromSmiles(s) else np.nan
            for s in e.smiles_prepared]

    contact = pd.DataFrame([contact_scores(k, p)
                            for k, p in zip(e.ifp_keys.fillna(""), e.pocket)])
    e = pd.concat([e.reset_index(drop=True), contact.reset_index(drop=True)], axis=1)
    fits = [profile_fit(r) for r in e.itertuples()]
    e["profile_fit"] = [f[0] for f in fits]
    e["profile_misses"] = [";".join(f[1]) for f in fits]

    # Percentiles within pocket, and correlation-aware weighting so that four
    # views of the same CNN signal do not outvote independent evidence.
    available = [(m, hb, fam) for m, hb, fam in METRICS
                 if m in e.columns and e[m].notna().any()]
    report_corr = {}
    for pocket, sub in e.groupby("pocket"):
        cols = [m for m, _, _ in available]
        corr = sub[cols].corr(method="spearman").round(3)
        report_corr[pocket] = corr.fillna("").to_dict()
    for m, higher, _ in available:
        e[f"pct_{m}"] = e.groupby("pocket")[m].rank(pct=True, ascending=higher)

    families = {}
    for m, _, fam in available:
        families.setdefault(fam, []).append(m)
    fam_weight = {fam: 1.0 / len(cols) for fam, cols in families.items()}
    e["structural_evidence"] = sum(
        e[f"pct_{m}"].fillna(0.5) * fam_weight[fam] for m, _, fam in available
    ) / sum(fam_weight[fam] for _, _, fam in available)

    pct_cols = [f"pct_{m}" for m, _, _ in available]
    e["metric_disagreement"] = e[pct_cols].std(axis=1).round(4)

    e["reactive_or_pains"] = (e.get("pains_flag", False).fillna(False)
                              if "pains_flag" in e else False)
    e["consensus_score"] = (
        0.60 * e.structural_evidence
        + 0.25 * e.profile_fit
        + 0.15 * e.contact_fraction.fillna(0)
    ) * e.pocket.map(POCKET_WEIGHT).fillna(0.8)
    e.loc[e.agonist_pattern_flag, "consensus_score"] -= 0.05

    # Pareto front within pocket over the independent evidence axes.
    e["pareto_front"] = False
    for pocket, sub in e.groupby("pocket"):
        axes = [c for c in ["pct_best_vina_affinity", "pct_max_ifp_recovery",
                            "pct_ligand_efficiency", "pct_best_CNN_VS"] if c in sub]
        vals = sub[axes].fillna(0).to_numpy()
        order = np.argsort(-vals.sum(axis=1))
        front, best = [], np.full(vals.shape[1], -np.inf)
        for i in order:
            if not np.any(np.all(vals[order] >= vals[i], axis=1)
                          & np.any(vals[order] > vals[i], axis=1)):
                front.append(sub.index[i])
        e.loc[front, "pareto_front"] = True

    # Similarity to the correct crystallographic antagonist per pocket, using the
    # CORRECTED AZ8838 tautomer. Reported for novelty context; never a penalty.
    refs = {}
    for pocket, smi in (("5NDD_orthosteric", "CCCc1cc(F)ccc1[C@H](O)c1[nH]ccn1"),
                        ("5NDZ_allosteric",
                         "C[C@@H](C1CCCCC1)n1c(-c2cc3c(cc2Br)OCO3)nc2cc"
                         "(C(=O)Nc3ccc(C#N)cc3)ccc21")):
        m = Chem.MolFromSmiles(smi)
        refs[pocket] = (MORGAN.GetFingerprint(m), "AZ8838" if "5NDD" in pocket else "AZ3451")
    tan, refname = [], []
    for smi, pocket in zip(e.smiles_prepared, e.pocket):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None or pocket not in refs:
            tan.append(np.nan); refname.append("")
            continue
        tan.append(DataStructs.TanimotoSimilarity(MORGAN.GetFingerprint(mol), refs[pocket][0]))
        refname.append(refs[pocket][1])
    e["tanimoto_to_reference"] = tan
    e["reference_antagonist"] = refname

    e["rank_in_pocket"] = e.groupby("pocket").consensus_score.rank(
        ascending=False, method="first").astype(int)
    e = e.sort_values("consensus_score", ascending=False).reset_index(drop=True)
    e.insert(0, "consensus_rank", range(1, len(e) + 1))

    # Aliases so the clustering/seed stage can consume this table directly.
    e["postdock_rank"] = e.consensus_rank
    e["postdock_score"] = e.consensus_score
    e["ifp_recovery"] = e.max_ifp_recovery
    e["best_affinity"] = e.best_vina_affinity

    args.out.mkdir(parents=True, exist_ok=True)
    e.to_csv(args.out / "DENOVO_EX8_CONSENSUS_ALL.csv", index=False)
    e.head(2000).to_csv(args.out / "DENOVO_EX8_CONSENSUS_TOP2000.csv", index=False)
    e.head(1000).to_csv(args.out / "DENOVO_EX8_CONSENSUS_TOP1000.csv", index=False)
    for pocket, sub in e.groupby("pocket"):
        sub.head(500).to_csv(args.out / f"CONSENSUS_TOP500_{pocket}.csv", index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        rows=int(len(e)), unique_designs=int(e.cand_id.nunique()),
        by_pocket={k: int(v) for k, v in e.pocket.value_counts().items()},
        by_batch={k: int(v) for k, v in e.batch.value_counts().items()} if "batch" in e else {},
        metrics_used=[m for m, _, _ in available],
        pignet2_available=bool(e.pignet2_best.notna().any()),
        spearman_within_pocket=report_corr,
        agonist_pattern_flagged=int(e.agonist_pattern_flag.sum()),
        pareto_front_size=int(e.pareto_front.sum()),
        anchor_hbond_hits_5NDD=int(
            (e[e.pocket == "5NDD_orthosteric"].anchor_hbond_fraction > 0).sum()),
        both_anchors_5NDD=int(
            (e[e.pocket == "5NDD_orthosteric"].anchor_hbond_fraction >= 0.999).sum()),
        profile_fit_median={p: round(float(s.profile_fit.median()), 4)
                            for p, s in e.groupby("pocket")},
        caveat=("Ranking evidence only. Static docking cannot distinguish an antagonist "
                "from a biased agonist; agonist_pattern_flag marks poses whose contacts "
                "resemble the tethered-ligand trajectory and must be tested functionally."),
    )
    (args.out / "CONSENSUS_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "spearman_within_pocket"},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
