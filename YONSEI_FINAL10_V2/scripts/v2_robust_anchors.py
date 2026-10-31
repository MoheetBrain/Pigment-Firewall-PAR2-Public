#!/usr/bin/env python3
"""Preparation-independent polar-anchor scoring for docked PAR2 poses.

Why this exists. Two independent audits showed that WHICH hydrogen bonds ProLIF
reports for the 5NDD/AZ8838 reference depends on choices made during structure
preparation rather than on the crystallography:

  * the ligand imidazole protomer. All three accessible protomers share one
    InChIKey, so no integrity check catches the swap, yet the as-built protomer
    gives TYR:82 + HIS:135 while another gives ASP:228 + HIS:135.
  * the receptor histidine tautomers. Re-protonating the receptor correctly at
    pH 7.4 removed the HIS:135 hydrogen bond and added a HIS:227 pi-cation,
    without moving a single heavy atom.

What does NOT move under either choice is the heavy-atom geometry. Exactly three
residues place a polar atom within hydrogen-bonding distance of the
crystallographic ligand - TYR:82 at 2.475 A, HIS:135 at 2.630 A and ASP:228 at
2.929 A - and the next nearest polar contact is 4.067 A away. That gap is the
robust signal, so anchors are scored by heavy-atom polar distance.

This does not claim these are hydrogen bonds. It claims a ligand polar atom sits
close enough to that residue's polar atoms for one to be possible.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Residue -> polar atom names, with the crystallographic reference distance.
ANCHORS = {
    "5NDD_orthosteric": {
        "TYR82": (("TYR", "82"), ("OH",), 2.475),
        "HIS135": (("HIS", "135"), ("ND1", "NE2"), 2.630),
        "ASP228": (("ASP", "228"), ("OD1", "OD2"), 2.929),
    },
    "5NDZ_allosteric": {
        "TYR210": (("TYR", "210"), ("OH",), 3.393),
        "ASN158": (("ASN", "158"), ("OD1", "ND2"), 3.515),
        "CYS161": (("CYS", "161"), ("SG",), 3.773),
    },
}
POLAR_CUTOFF = 3.5      # heavy-atom polar-polar distance admitting a hydrogen bond
CONTACT_CUTOFF = 4.5    # any-heavy-atom contact


def receptor_polar(pdb: Path, spec: dict):
    """Coordinates of the named polar atoms for each anchor residue."""
    want = {}
    for tag, ((resname, resseq), atoms, _ref) in spec.items():
        want[tag] = (resname, resseq, set(atoms))
    found = {tag: [] for tag in want}
    allheavy = []
    for line in pdb.read_text().splitlines():
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if element == "H":
            continue
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        allheavy.append(xyz)
        rn, rs, an = line[17:20].strip(), line[22:26].strip(), line[12:16].strip()
        for tag, (wrn, wrs, wan) in want.items():
            if rn == wrn and rs == wrs and an in wan:
                found[tag].append(xyz)
    missing = [t for t, v in found.items() if not v]
    if missing:
        raise SystemExit(f"anchor atoms not found in {pdb}: {missing}")
    return {t: np.array(v) for t, v in found.items()}, np.array(allheavy)


def score_file(path: Path, pocket: str, anchors, receptor_heavy) -> list[dict]:
    spec = ANCHORS[pocket]
    rows, seen = [], {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for mol in Chem.ForwardSDMolSupplier(handle, removeHs=True, sanitize=False):
            if mol is None:
                continue
            cand = mol.GetProp("cand_id") if mol.HasProp("cand_id") else (
                mol.GetProp("_Name") if mol.HasProp("_Name") else "")
            if not cand:
                continue
            seen[cand] = seen.get(cand, 0) + 1
            xyz = mol.GetConformer().GetPositions()
            polar_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("N", "O", "S")]
            row = dict(cand_id=cand, pocket=pocket, pose_rank=seen[cand],
                       n_ligand_polar_atoms=len(polar_idx))
            hits = 0
            for tag in spec:
                if not polar_idx:
                    row[f"dist_{tag}"] = np.nan
                    row[f"anchor_{tag}"] = False
                    continue
                d = float(np.linalg.norm(
                    anchors[tag][:, None, :] - xyz[polar_idx][None, :, :], axis=2).min())
                row[f"dist_{tag}"] = round(d, 3)
                row[f"anchor_{tag}"] = bool(d <= POLAR_CUTOFF)
                hits += int(d <= POLAR_CUTOFF)
            row["polar_anchor_count"] = hits
            row["polar_anchor_fraction"] = hits / len(spec)
            dmin = np.linalg.norm(receptor_heavy[:, None, :] - xyz[None, :, :], axis=2)
            row["min_receptor_contact_A"] = round(float(dmin.min()), 3)
            row["n_receptor_atoms_within_4_5A"] = int((dmin.min(axis=1) <= CONTACT_CUTOFF).sum())
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, nargs="+", required=True,
                        help="directories holding *_out.sdf.gz pose files")
    parser.add_argument("--receptor-5ndd", type=Path,
                        default=Path("YONSEI_FINAL10_V2/02_REFERENCE_VALIDATION_REPAIRED/"
                                     "5NDD_orthosteric_receptor_pH74_FIXED.pdb"))
    parser.add_argument("--receptor-5ndz", type=Path,
                        default=Path("YONSEI_FINAL10_V2/02_REFERENCE_VALIDATION_REPAIRED/"
                                     "5NDZ_allosteric_receptor_pH74_FIXED.pdb"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rec = {}
    for pocket, path in (("5NDD_orthosteric", args.receptor_5ndd),
                         ("5NDZ_allosteric", args.receptor_5ndz)):
        rec[pocket] = receptor_polar(path, ANCHORS[pocket])
        print(f"[anchors] {pocket}: loaded from {path.name}")

    files = []
    for d in args.poses:
        files += [p for p in sorted(d.rglob("*_out.sdf*")) if not p.name.startswith(".")]
    print(f"[anchors] {len(files)} pose files")

    rows = []
    for n, path in enumerate(files, 1):
        pocket = path.name.split("_shard")[0]
        if pocket not in ANCHORS:
            continue
        rows += score_file(path, pocket, rec[pocket][0], rec[pocket][1])
        if n % 50 == 0:
            print(f"[anchors] {n}/{len(files)}", flush=True)
    poses = pd.DataFrame(rows)
    if poses.empty:
        raise SystemExit("no poses scored")

    best = (poses.sort_values(["polar_anchor_count", "min_receptor_contact_A"],
                              ascending=[False, True])
                 .drop_duplicates(["cand_id", "pocket"], keep="first"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    poses.to_csv(args.out, index=False)
    best_path = args.out.with_name(args.out.stem + "_BEST.csv")
    best.to_csv(best_path, index=False)

    summary = dict(
        poses_scored=int(len(poses)), ligand_pocket_pairs=int(len(best)),
        polar_cutoff_A=POLAR_CUTOFF,
        anchors={k: {t: v[2] for t, v in s.items()} for k, s in ANCHORS.items()},
        anchor_count_distribution={
            p: best[best.pocket == p].polar_anchor_count.value_counts().sort_index().to_dict()
            for p in best.pocket.unique()},
        note=("Heavy-atom polar proximity, independent of ligand protomer and receptor "
              "histidine tautomer. Not a claim that a hydrogen bond exists."),
    )
    args.out.with_name(args.out.stem + "_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
