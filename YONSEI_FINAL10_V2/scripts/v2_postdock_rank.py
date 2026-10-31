#!/usr/bin/env python3
"""Transparent post-docking ranking of the de novo PAR2 designs.

Each design is ranked inside the pocket it was designed for. 5NDD and 5NDZ
evidence is never pooled into one z-score: scores are standardised within a
pocket, and an exploratory design docked to both pockets keeps both rows while
the summary table reports the pocket where its evidence is stronger.

Ranking terms, all reported per molecule so the ranking can be audited:
  intended-pocket occupancy, reference-antagonist interaction-fingerprint
  recovery, GNINA evidence, ligand efficiency, medicinal-chemistry
  plausibility, lipophilicity moderation, sp3 character, and a one-sided
  novelty bonus.

Novelty is a bonus only. A design is never penalised for having modest
Tanimoto similarity to AZ8838 or AZ3451.

None of these numbers is a probability of PAR2 antagonism. CNNscore and
CNNaffinity are ranking features, not calibrated predictions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

POCKET_REFERENCE_SMILES = {
    # Correct AZ8838 (wwPDB 8TZ, InChIKey IDFPQEHZYBXIFO-LBPRGKRZSA-N). The
# repository SDF stores a wrong 4H-imidazole tautomer; see
# 04_VERIFICATION/REFERENCE_MODEL_DEFECTS.md defect 1.
    "5NDD_orthosteric": ("AZ8838", "CCCc1cc(F)ccc1[C@H](O)c1[nH]ccn1"),
    "5NDZ_allosteric": ("AZ3451", "C[C@@H](C1CCCCC1)n1c(-c2cc3c(cc2Br)OCO3)nc2cc(C(=O)Nc3ccc(C#N)cc3)ccc21"),
}
FAMILY_INTENDED_POCKET = {
    "orthosteric": {"5NDD_orthosteric"},
    "allosteric": {"5NDZ_allosteric"},
    "exploratory": {"5NDD_orthosteric", "5NDZ_allosteric"},
}

WEIGHTS = {
    "ifp_recovery": 0.26,
    "gnina_evidence": 0.24,
    "ligand_efficiency": 0.16,
    "pocket_occupancy": 0.10,
    "med_chem": 0.10,
    "sp3_character": 0.06,
    "lipophilicity_moderation": 0.08,
}
NOVELTY_BONUS_MAX = 0.10

REACTIVE_SMARTS = {
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "michael_acceptor_aldehyde": "[CX3H1](=O)[CX3]=[CX3]",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "azide": "[N-]=[N+]=[N-]",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "peroxide": "[OX2][OX2]",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "thiol": "[SX2H1]",
    "quinone": "O=C1C=CC(=O)C=C1",
    "nitro_aromatic": "[a][NX3](=O)=O",
}

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def robust_z(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if mad and mad > 1e-9:
        scale = mad * 1.4826
    else:
        std = numeric.std()
        scale = std if std and std > 1e-9 else 1.0
    z = (numeric - median) / scale
    if not higher_is_better:
        z = -z
    return z.clip(-4, 4).fillna(0.0)


def unit(values: pd.Series) -> pd.Series:
    """Map a robust z-score onto 0..1 so weighted terms stay comparable."""
    return ((values + 3.0) / 6.0).clip(0.0, 1.0)


def alert_table(smiles_list: list[str]) -> pd.DataFrame:
    def build(entry):
        params = FilterCatalogParams()
        params.AddCatalog(entry)
        return FilterCatalog(params)

    pains = build(FilterCatalogParams.FilterCatalogs.PAINS)
    brenk = build(FilterCatalogParams.FilterCatalogs.BRENK)
    patterns = {k: Chem.MolFromSmarts(v) for k, v in REACTIVE_SMARTS.items()}
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            rows.append(dict(pains_flag=True, brenk_flag=True,
                             structural_alerts="UNPARSABLE", reactive_groups="UNPARSABLE"))
            continue
        p = [e.GetDescription() for e in pains.GetMatches(mol)]
        b = [e.GetDescription() for e in brenk.GetMatches(mol)]
        r = [n for n, patt in patterns.items() if patt is not None and mol.HasSubstructMatch(patt)]
        rows.append(dict(pains_flag=bool(p), brenk_flag=bool(b),
                         structural_alerts=";".join(sorted(set(p + b))[:8]),
                         reactive_groups=";".join(r)))
    return pd.DataFrame(rows)


def reference_similarity(smiles_list: list[str], pockets: list[str]) -> list[float]:
    refs = {}
    for pocket, (_name, smiles) in POCKET_REFERENCE_SMILES.items():
        mol = Chem.MolFromSmiles(smiles)
        refs[pocket] = MORGAN.GetFingerprint(mol)
    out = []
    for smiles, pocket in zip(smiles_list, pockets):
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None or pocket not in refs:
            out.append(float("nan"))
            continue
        out.append(DataStructs.TanimotoSimilarity(MORGAN.GetFingerprint(mol), refs[pocket]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligand-table", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-clogp", type=float, default=5.0)
    parser.add_argument("--max-mw", type=float, default=520.0)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ligands = pd.read_csv(args.ligand_table)
    ligands = ligands[ligands.source == "DENOVO"].copy()
    scores = pd.read_csv(args.scores)
    frame = scores.merge(ligands, on="cand_id", how="inner", suffixes=("", "_lig"))
    if frame.empty:
        raise SystemExit("no de novo rows after merging scores with the ligand table")
    if "heavy_atoms_lig" in frame:
        frame["heavy_atoms"] = frame["heavy_atoms"].fillna(frame["heavy_atoms_lig"])

    if args.interactions and Path(args.interactions).exists():
        ifp = pd.read_csv(args.interactions)
        keep = [c for c in ["cand_id", "pocket", "pose_rank", "ifp_recovery",
                            "ifp_n_interactions", "ifp_shared_keys", "ifp_reference_keys"] if c in ifp]
        ifp = ifp[keep].rename(columns={"pose_rank": "ifp_pose_rank"})
        frame = frame.merge(ifp, on=["cand_id", "pocket"], how="left")
        ifp_available = frame.ifp_recovery.notna().any()
    else:
        frame["ifp_recovery"] = np.nan
        frame["ifp_n_interactions"] = np.nan
        frame["ifp_pose_rank"] = np.nan
        ifp_available = False

    frame["intended_pocket"] = frame.apply(
        lambda r: r.pocket in FAMILY_INTENDED_POCKET.get(str(r.denovo_family), set()), axis=1)
    frame["reference_antagonist"] = frame.pocket.map(lambda p: POCKET_REFERENCE_SMILES[p][0]
                                                     if p in POCKET_REFERENCE_SMILES else "")
    frame["tanimoto_to_reference"] = reference_similarity(
        frame.smiles_prepared.tolist(), frame.pocket.tolist())
    frame = pd.concat([frame.reset_index(drop=True),
                       alert_table(frame.smiles_prepared.tolist()).reset_index(drop=True)], axis=1)

    parts = []
    for pocket, sub in frame.groupby("pocket"):
        sub = sub.copy()
        gnina_terms = [
            unit(robust_z(sub.best_affinity, False)),
            unit(robust_z(sub.best_CNN_VS, True)),
            unit(robust_z(sub.best_CNNscore, True)),
            unit(robust_z(sub.best_CNNaffinity, True)),
        ]
        sub["term_gnina_evidence"] = sum(gnina_terms) / len(gnina_terms)
        sub["term_ligand_efficiency"] = unit(robust_z(sub.ligand_efficiency, True))
        sub["term_med_chem"] = unit(robust_z(sub.qed, True))
        sub["term_sp3_character"] = pd.to_numeric(sub.fsp3, errors="coerce").fillna(0.0).clip(0, 1)
        clogp = pd.to_numeric(sub.clogp, errors="coerce")
        # Peak plausibility around cLogP 2.5, tapering to 0 outside roughly 0-5.
        sub["term_lipophilicity_moderation"] = (1.0 - ((clogp - 2.5).abs() / 2.5)).clip(0.0, 1.0).fillna(0.0)
        if ifp_available:
            sub["term_ifp_recovery"] = pd.to_numeric(sub.ifp_recovery, errors="coerce").fillna(0.0).clip(0, 1)
        else:
            sub["term_ifp_recovery"] = np.nan
        occupancy = pd.to_numeric(sub.ifp_n_interactions, errors="coerce")
        sub["pocket_occupancy_ok"] = occupancy.fillna(0) > 0 if ifp_available else np.nan
        sub["term_pocket_occupancy"] = (
            (occupancy.fillna(0).clip(0, 12) / 12.0) if ifp_available else np.nan)
        sub["novelty_bonus"] = (
            NOVELTY_BONUS_MAX
            * (1.0 - pd.to_numeric(sub.tanimoto_to_reference, errors="coerce").fillna(0.5))
        ).clip(0.0, NOVELTY_BONUS_MAX)

        weights = dict(WEIGHTS)
        if not ifp_available:
            weights.pop("ifp_recovery")
            weights.pop("pocket_occupancy")
        total_weight = sum(weights.values())
        evidence = sum(weights[name] * sub[f"term_{name}"] for name in weights) / total_weight
        sub["evidence_score"] = evidence

        penalty = np.zeros(len(sub))
        penalty += 0.20 * (clogp > args.max_clogp).fillna(False).to_numpy()
        penalty += 0.15 * (pd.to_numeric(sub.mw, errors="coerce") > args.max_mw).fillna(False).to_numpy()
        penalty += 0.25 * sub.pains_flag.fillna(False).to_numpy()
        penalty += 0.15 * (sub.reactive_groups.fillna("") != "").to_numpy()
        penalty += 0.10 * (~sub.intended_pocket).to_numpy()
        sub["penalty"] = penalty
        sub["postdock_score"] = sub.evidence_score + sub.novelty_bonus - sub.penalty
        sub["pocket_rank"] = sub.postdock_score.rank(ascending=False, method="first").astype(int)
        parts.append(sub)

    ranked = pd.concat(parts, ignore_index=True).sort_values("postdock_score", ascending=False)

    # One row per design: keep its stronger pocket, carry the other pocket's numbers.
    ordered = ranked.sort_values("postdock_score", ascending=False)
    primary = ordered.drop_duplicates("cand_id", keep="first").copy()
    secondary = (ordered[ordered.duplicated("cand_id", keep="first")]
                 .drop_duplicates("cand_id", keep="first")
                 [["cand_id", "pocket", "postdock_score", "best_affinity", "ifp_recovery"]]
                 .rename(columns={"pocket": "other_pocket",
                                  "postdock_score": "other_pocket_postdock_score",
                                  "best_affinity": "other_pocket_best_affinity",
                                  "ifp_recovery": "other_pocket_ifp_recovery"}))
    primary = primary.merge(secondary, on="cand_id", how="left")
    primary["pocket_preference_margin"] = (
        primary.postdock_score - primary.other_pocket_postdock_score)
    primary = primary.sort_values("postdock_score", ascending=False).reset_index(drop=True)
    primary.insert(0, "postdock_rank", range(1, len(primary) + 1))

    columns = [
        "postdock_rank", "cand_id", "denovo_family", "denovo_rank", "design_score",
        "pocket", "intended_pocket", "pocket_occupancy_ok", "smiles_prepared", "inchikey",
        "murcko", "best_affinity", "best_affinity_pose_rank", "best_CNNscore",
        "best_CNNaffinity", "best_CNN_VS", "toppose_affinity", "toppose_CNN_VS",
        "n_poses", "ligand_efficiency", "heavy_atoms", "ifp_recovery", "ifp_pose_rank",
        "ifp_n_interactions", "reference_antagonist", "tanimoto_to_reference",
        "mw", "clogp", "tpsa", "hba", "hbd", "rotb", "fsp3", "qed", "formal_charge",
        "pains_flag", "brenk_flag", "structural_alerts", "reactive_groups",
        "term_ifp_recovery", "term_gnina_evidence", "term_ligand_efficiency",
        "term_pocket_occupancy", "term_med_chem", "term_sp3_character",
        "term_lipophilicity_moderation", "novelty_bonus", "penalty",
        "evidence_score", "postdock_score", "pocket_rank",
        "other_pocket", "other_pocket_postdock_score", "other_pocket_best_affinity",
        "other_pocket_ifp_recovery", "pocket_preference_margin",
    ]
    columns = [c for c in columns if c in primary.columns]
    table = primary[columns]
    table.to_csv(out / "DENOVO_POSTDOCK_ALL.csv", index=False)
    table.head(500).to_csv(out / "DENOVO_POSTDOCK_TOP500.csv", index=False)
    table.head(300).to_csv(out / "DENOVO_POSTDOCK_TOP300.csv", index=False)
    ranked.to_csv(out / "DENOVO_POSTDOCK_BY_POCKET.csv", index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        designs_ranked=int(len(table)),
        ligand_pocket_rows=int(len(ranked)),
        interaction_fingerprints_used=bool(ifp_available),
        weights=WEIGHTS,
        novelty_bonus_max=NOVELTY_BONUS_MAX,
        novelty_is_bonus_only=True,
        by_family={k: int(v) for k, v in table.denovo_family.value_counts().items()},
        selected_pocket_counts={k: int(v) for k, v in table.pocket.value_counts().items()},
        designs_scored_in_both_pockets=int(table.other_pocket.notna().sum()),
        top20_preview=table.head(20)[
            ["postdock_rank", "cand_id", "denovo_family", "pocket", "best_affinity",
             "best_CNN_VS", "ifp_recovery", "postdock_score"]].to_dict("records"),
        caveat=("Ranking evidence only. CNNscore and CNNaffinity are not calibrated "
                "probabilities of PAR2 antagonism and no molecule here is shown to bind."),
    )
    (out / "DENOVO_POSTDOCK_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "top20_preview"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
