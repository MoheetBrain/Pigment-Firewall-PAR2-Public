#!/usr/bin/env python3
"""Detect azole/heteroaromatic H-placement artifacts in the reference ligands.

Ligands extracted from a PDB entry carry no hydrogens. When bond orders are
perceived from heavy-atom coordinates alone, an azole (imidazole, pyrazole,
triazole, ...) can be assigned a localised ``N=C-C=N`` pattern with the
implicit hydrogen placed on a ring *carbon* instead of a ring *nitrogen*. The
result is a non-aromatic 2H/4H-azole tautomer that has the same molecular
formula and mass as the real aromatic heterocycle, so mass- and formula-based
checks do not catch it.

This matters here because the reference ligand's Morgan fingerprint defines
``ref_tanimoto``, which drives the ``novelty_score`` term (weight 0.20) of
``design_score``, and because BRICS fragments of the reference seed the
family-anchored generation.

RDKit's ``TautomerEnumerator`` does not enumerate this particular C->N 1,3-shift,
so this script implements a targeted detector: for every non-aromatic ring it
looks for a single sp3 ring carbon bearing hydrogen alongside a ring nitrogen
with no hydrogen, relocates one hydrogen from that carbon to that nitrogen, and
re-perceives aromaticity. A ring that becomes aromatic under this transform is
reported as an H-placement artifact together with the corrected SMILES.

With ``--impact`` it additionally quantifies what the correction would do to the
committed de novo scores: recomputed ``ref_tanimoto``, the resulting
``design_score`` shift (novelty term only), rank churn, and how much of the
docking-priority queue would change membership.

Read-only. Never modifies a production artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

REFERENCE_LIGANDS = {
    "orthosteric": ("5NDD / AZ8838 / 8TZ",
                    "01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.sdf"),
    "allosteric": ("5NDZ / AZ3451 / 8UN",
                   "01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.sdf"),
}

# Authoritative wwPDB chemical-component definitions, retrieved from
# https://data.rcsb.org/rest/v1/core/chemcomp/<comp_id>. These are the ground
# truth for what each reference ligand must be. An InChIKey mismatch means the
# repository's reference model is not the deposited crystallographic ligand.
PDB_COMPONENTS = {
    "orthosteric": {
        "comp_id": "8TZ",
        "name": "(S)-(4-fluoranyl-2-propyl-phenyl)-(1H-imidazol-2-yl)methanol",
        "formula": "C13 H15 F N2 O",
        "inchikey": "IDFPQEHZYBXIFO-LBPRGKRZSA-N",
        "smiles": "CCCc1cc(F)ccc1[C@H](O)c2[nH]ccn2",
        "aromatic_bonds": 11,
    },
    "allosteric": {
        "comp_id": "8UN",
        "name": ("2-(6-bromanyl-1,3-benzodioxol-5-yl)-N-(4-cyanophenyl)-"
                 "1-[(1S)-1-cyclohexylethyl]benzimidazole-5-carboxamide"),
        "formula": "C30 H27 Br N4 O3",
        "inchikey": "FJAOGFGHTPYADT-SFHVURJKSA-N",
        "smiles": ("C[C@@H](C1CCCCC1)n1c(-c2cc3c(cc2Br)OCO3)nc2cc"
                   "(C(=O)Nc3ccc(C#N)cc3)ccc21"),
        "aromatic_bonds": 22,
    },
}

# design_score novelty term, per HANDOFF_TO_CLAUDE.md section 12.
NOVELTY_WEIGHT = 0.20
REFSIM_TARGET = {"orthosteric": 0.38, "allosteric": 0.35, "exploratory": 0.25}
REFSIM_SIGMA = {"orthosteric": 0.20, "allosteric": 0.20, "exploratory": 0.25}


def gaussian(x: float, target: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - target) / sigma) ** 2)


def load_sdf_first(path: Path) -> Chem.Mol:
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]
    if not mols:
        raise SystemExit(f"No valid molecule in {path}")
    mol = mols[0]
    Chem.SanitizeMol(mol)
    return mol


def find_h_placement_artifacts(mol: Chem.Mol) -> List[Dict]:
    """Return one record per ring that aromatises after a C->N hydrogen relocation."""
    findings: List[Dict] = []
    ring_info = mol.GetRingInfo()

    for ring in ring_info.AtomRings():
        if len(ring) not in (5, 6):
            continue
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue

        # Candidate donor: sp3 ring carbon carrying at least one hydrogen.
        donors = [i for i in ring
                  if mol.GetAtomWithIdx(i).GetSymbol() == "C"
                  and mol.GetAtomWithIdx(i).GetTotalNumHs() > 0
                  and str(mol.GetAtomWithIdx(i).GetHybridization()) == "SP3"]
        # Candidate acceptor: ring nitrogen with no hydrogen.
        acceptors = [i for i in ring
                     if mol.GetAtomWithIdx(i).GetSymbol() == "N"
                     and mol.GetAtomWithIdx(i).GetTotalNumHs() == 0]
        if not donors or not acceptors:
            continue

        for d in donors:
            for a in acceptors:
                fixed = _relocate_hydrogen(mol, d, a)
                if fixed is None:
                    continue
                if not _ring_is_aromatic(fixed, ring):
                    continue
                findings.append({
                    "ring_atom_indices": list(ring),
                    "ring_size": len(ring),
                    "donor_carbon_idx": int(d),
                    "acceptor_nitrogen_idx": int(a),
                    "stored_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                    "corrected_smiles": Chem.MolToSmiles(fixed, isomericSmiles=True),
                    "stored_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
                    "corrected_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(fixed)),
                    "stored_formula": rdMolDescriptors.CalcMolFormula(mol),
                    "corrected_formula": rdMolDescriptors.CalcMolFormula(fixed),
                    "formula_unchanged": (rdMolDescriptors.CalcMolFormula(mol)
                                          == rdMolDescriptors.CalcMolFormula(fixed)),
                    "stored_hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
                    "corrected_hbd": int(rdMolDescriptors.CalcNumHBD(fixed)),
                    "stored_hba": int(rdMolDescriptors.CalcNumHBA(mol)),
                    "corrected_hba": int(rdMolDescriptors.CalcNumHBA(fixed)),
                })
                return findings  # one canonical correction per molecule is enough
    return findings


def _relocate_hydrogen(mol: Chem.Mol, c_idx: int, n_idx: int) -> Optional[Chem.Mol]:
    """Move one H from ring carbon c_idx to ring nitrogen n_idx, then re-perceive."""
    rw = Chem.RWMol(mol)
    c, n = rw.GetAtomWithIdx(c_idx), rw.GetAtomWithIdx(n_idx)
    if c.GetTotalNumHs() < 1:
        return None
    c.SetNoImplicit(True)
    c.SetNumExplicitHs(max(0, c.GetTotalNumHs() - 1))
    n.SetNoImplicit(True)
    n.SetNumExplicitHs(n.GetTotalNumHs() + 1)

    # Re-perceive the ring as a delocalised azole: alternate the ring bonds.
    ring = next((r for r in mol.GetRingInfo().AtomRings() if c_idx in r and n_idx in r), None)
    if ring is None:
        return None
    for i in range(len(ring)):
        b = rw.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])
        if b is not None:
            b.SetBondType(Chem.BondType.AROMATIC)
            b.SetIsAromatic(True)
    for i in ring:
        rw.GetAtomWithIdx(i).SetIsAromatic(True)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def _ring_is_aromatic(mol: Chem.Mol, ring: Tuple[int, ...]) -> bool:
    try:
        return all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    except Exception:
        return False


def pdb_identity_check(mol: Chem.Mol, family: str) -> Dict:
    """Compare a reference ligand against its authoritative wwPDB component."""
    from rdkit.Chem.inchi import MolToInchiKey

    expected = PDB_COMPONENTS.get(family)
    if expected is None:
        return {"checked": False}
    actual_key = MolToInchiKey(mol)
    actual_arom = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
    return {
        "checked": True,
        "comp_id": expected["comp_id"],
        "pdb_name": expected["name"],
        "expected_inchikey": expected["inchikey"],
        "actual_inchikey": actual_key,
        "inchikey_match": actual_key == expected["inchikey"],
        "expected_aromatic_bonds": expected["aromatic_bonds"],
        "actual_aromatic_bonds": actual_arom,
        "aromatic_bond_match": actual_arom == expected["aromatic_bonds"],
        "expected_smiles": expected["smiles"],
    }


def describe(mol: Chem.Mol) -> Dict:
    return {
        "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 4),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "total_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
    }


def impact_analysis(v2: Path, family: str, corrected_smiles: str) -> Dict:
    """Quantify what the corrected reference would do to the committed scores."""
    import pandas as pd

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fixed_fp = gen.GetFingerprint(Chem.MolFromSmiles(corrected_smiles))

    full = pd.read_csv(v2 / "02_DENOVO_CPU" / "DENOVO_15000.csv")
    fam = full[full.family == family].copy().reset_index(drop=True)
    if fam.empty:
        return {"family": family, "rows": 0}

    target, sigma = REFSIM_TARGET[family], REFSIM_SIGMA[family]
    new_sim, new_score = [], []
    for r in fam.itertuples(index=False):
        m = Chem.MolFromSmiles(r.smiles)
        if m is None:
            new_sim.append(float("nan")); new_score.append(float(r.design_score)); continue
        s = DataStructs.TanimotoSimilarity(gen.GetFingerprint(m), fixed_fp)
        new_sim.append(s)
        new_score.append(float(r.design_score)
                         - NOVELTY_WEIGHT * gaussian(float(r.ref_tanimoto), target, sigma)
                         + NOVELTY_WEIGHT * gaussian(s, target, sigma))
    fam["rt_fix"], fam["ds_fix"] = new_sim, new_score
    d = (fam.ds_fix - fam.design_score).abs()

    old_rank = fam.design_score.rank(ascending=False, method="first")
    new_rank = fam.ds_fix.rank(ascending=False, method="first")

    top = pd.read_csv(v2 / "02_DENOVO_CPU" / "DENOVO_TOP3000_FOR_DOCKING.csv")
    n_slots = int((top.family == family).sum())
    current = set(top[top.family == family].smiles.astype(str))
    would = set(fam.nlargest(n_slots, "ds_fix").smiles.astype(str)) if n_slots else set()

    return {
        "family": family,
        "rows": int(len(fam)),
        "ref_tanimoto_stored": {
            "min": float(fam.ref_tanimoto.min()), "median": float(fam.ref_tanimoto.median()),
            "mean": float(fam.ref_tanimoto.mean()), "max": float(fam.ref_tanimoto.max())},
        "ref_tanimoto_corrected": {
            "min": float(fam.rt_fix.min()), "median": float(fam.rt_fix.median()),
            "mean": float(fam.rt_fix.mean()), "max": float(fam.rt_fix.max())},
        "mean_abs_ref_tanimoto_change": float((fam.rt_fix - fam.ref_tanimoto).abs().mean()),
        "design_score_mean_abs_change": float(d.mean()),
        "design_score_max_abs_change": float(d.max()),
        "rows_changing_gt_0.01": int((d > 0.01).sum()),
        "rows_changing_gt_0.05": int((d > 0.05).sum()),
        "spearman_old_vs_new_design_score":
            float(fam.design_score.corr(fam.ds_fix, method="spearman")),
        "mean_abs_rank_shift": float((old_rank - new_rank).abs().mean()),
        "max_abs_rank_shift": int((old_rank - new_rank).abs().max()),
        "docking_queue_slots": n_slots,
        "docking_queue_retained": int(len(current & would)),
        "docking_queue_changed": int(len(current - would)),
        "docking_queue_retained_pct":
            round(100.0 * len(current & would) / n_slots, 2) if n_slots else None,
        "docking_queue_note": ("Membership comparison by score only; it ignores the "
                               "scaffold-cap-12 reselection, so it is an upper bound "
                               "on real churn."),
    }


def scan_designs(v2: Path, out_dir: Path) -> Dict:
    """Find committed designs that carry an azole in the wrong (non-aromatic) tautomer.

    These molecules inherited the defective motif from the reference seed
    fragment during BRICS assembly. Unlike a scoring error, this cannot be
    repaired by rescoring: the stored structure itself is wrong.
    """
    import pandas as pd
    from collections import Counter

    tables = {
        "DENOVO_15000": v2 / "02_DENOVO_CPU" / "DENOVO_15000.csv",
        "DENOVO_TOP3000_FOR_DOCKING": v2 / "02_DENOVO_CPU" / "DENOVO_TOP3000_FOR_DOCKING.csv",
    }
    summary: Dict = {}
    affected_rows: List[Dict] = []

    for label, path in tables.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        hits = []
        for r in df.itertuples(index=False):
            mol = Chem.MolFromSmiles(str(r.smiles))
            if mol is None:
                continue
            found = find_h_placement_artifacts(mol)
            if found:
                hits.append(r)
                affected_rows.append({
                    "table": label,
                    "denovo_rank": getattr(r, "denovo_rank", None),
                    "family": r.family,
                    "stored_smiles": r.smiles,
                    "corrected_smiles": found[0]["corrected_smiles"],
                    "design_score": getattr(r, "design_score", None),
                })
        summary[label] = {
            "rows": int(len(df)),
            "affected": len(hits),
            "affected_pct": round(100.0 * len(hits) / len(df), 3) if len(df) else 0.0,
            "by_family": {k: int(v) for k, v in
                          Counter(str(h.family) for h in hits).items()},
        }
        print(f"{label}: {len(hits)}/{len(df)} designs carry an azole in the wrong "
              f"tautomer ({summary[label]['affected_pct']}%)")
        print(f"    by family: {summary[label]['by_family']}")

    if affected_rows:
        out = out_dir / "affected_designs_wrong_tautomer.csv"
        pd.DataFrame(affected_rows).to_csv(out, index=False)
        print(f"    wrote {out}")
        summary["output_csv"] = str(out)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--impact", action="store_true",
                    help="Also quantify the effect on the committed de novo scores")
    ap.add_argument("--scan-designs", action="store_true",
                    help="Also scan the committed design CSVs for the same "
                         "wrong-tautomer motif and write the affected rows")
    args = ap.parse_args()

    v2 = args.v2_root.resolve()
    out_dir = (args.out_dir or (v2 / "04_VERIFICATION")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict = {"reference_ligands": {}, "artifacts_found": 0, "identity_mismatches": 0, "impact": {}}

    for family, (label, rel) in REFERENCE_LIGANDS.items():
        path = v2 / rel
        if not path.exists():
            raise SystemExit(f"Missing reference ligand: {path}")
        mol = load_sdf_first(path)
        entry = {"label": label, "path": rel, "stored": describe(mol)}
        entry["pdb_identity"] = pdb_identity_check(mol, family)
        findings = find_h_placement_artifacts(mol)
        entry["h_placement_artifacts"] = findings
        entry["clean"] = (not findings) and entry["pdb_identity"].get("inchikey_match", True)
        if not entry["pdb_identity"].get("inchikey_match", True):
            report["identity_mismatches"] += 1
        if findings:
            report["artifacts_found"] += 1
            corrected = Chem.MolFromSmiles(findings[0]["corrected_smiles"])
            entry["corrected"] = describe(corrected)
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            entry["tanimoto_stored_vs_corrected"] = float(DataStructs.TanimotoSimilarity(
                gen.GetFingerprint(mol), gen.GetFingerprint(corrected)))
            if args.impact:
                report["impact"][family] = impact_analysis(
                    v2, family, findings[0]["corrected_smiles"])
        report["reference_ligands"][family] = entry

        status = "CLEAN" if entry["clean"] else "H-PLACEMENT ARTIFACT"
        print(f"[{status}] {label}")
        pid = entry["pdb_identity"]
        if pid.get("checked"):
            verdict = "MATCH" if pid["inchikey_match"] else "MISMATCH"
            print(f"    wwPDB {pid['comp_id']}  : {pid['pdb_name'][:70]}")
            print(f"    identity  : InChIKey {verdict}  "
                  f"(aromatic bonds {pid['actual_aromatic_bonds']}/"
                  f"{pid['expected_aromatic_bonds']})")
            if not pid["inchikey_match"]:
                print(f"                expected {pid['expected_inchikey']}")
                print(f"                stored   {pid['actual_inchikey']}")
        print(f"    stored    : {entry['stored']['smiles']}")
        print(f"                {entry['stored']['formula']}  MW={entry['stored']['molecular_weight']}  "
              f"aromatic rings={entry['stored']['aromatic_rings']}")
        if findings:
            print(f"    corrected : {entry['corrected']['smiles']}")
            print(f"                {entry['corrected']['formula']}  MW={entry['corrected']['molecular_weight']}  "
                  f"aromatic rings={entry['corrected']['aromatic_rings']}")
            print(f"    Tanimoto(stored, corrected) = {entry['tanimoto_stored_vs_corrected']:.4f}")
            print(f"    formula unchanged by the correction: {findings[0]['formula_unchanged']}")
        print()

    if args.impact and report["impact"]:
        for fam, imp in report["impact"].items():
            print(f"--- impact on the {fam} family ({imp['rows']} designs) ---")
            print(f"    ref_tanimoto max  : {imp['ref_tanimoto_stored']['max']:.4f} "
                  f"-> {imp['ref_tanimoto_corrected']['max']:.4f}")
            print(f"    design_score      : mean |delta|={imp['design_score_mean_abs_change']:.5f}  "
                  f"max |delta|={imp['design_score_max_abs_change']:.5f}")
            print(f"    Spearman(old,new) : {imp['spearman_old_vs_new_design_score']:.5f}")
            print(f"    docking queue     : {imp['docking_queue_retained']}/{imp['docking_queue_slots']} "
                  f"retained ({imp['docking_queue_retained_pct']}%), "
                  f"{imp['docking_queue_changed']} would change")
            print()

    if args.scan_designs:
        print("\n--- scanning committed designs for the same motif ---")
        report["design_scan"] = scan_designs(v2, out_dir)
        print()

    out = out_dir / "REFERENCE_TAUTOMER_CHECK.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 1 if (report["artifacts_found"] or report["identity_mismatches"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
