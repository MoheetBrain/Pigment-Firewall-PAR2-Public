#!/usr/bin/env python3
"""Diagnose why the reference ProLIF fingerprints contain no hydrogen bonds.

``scripts/reference_interactions.py`` loads the reference ligand SDF and the
receptor PDB with ``removeHs=False`` and never calls ``Chem.AddHs``. Neither
input carries explicit hydrogens: the crystallographic ligand SDFs were written
from PDB HETATM records, which have none. ProLIF's ``HBDonor``/``HBAcceptor``
detectors require explicit hydrogens, so no hydrogen bond can be detected at
all, and the committed reference fingerprints consist only of ``VdWContact``
and ``Hydrophobic`` terms.

This is independent of, and additional to, the 8TZ tautomer defect reported by
``check_reference_tautomers.py``. This script separates the two contributions by
generating the reference fingerprint under three conditions:

  A. exactly as committed              - stored tautomer, no explicit hydrogens
  B. stored tautomer + AddHs           - isolates the missing-hydrogen defect
  C. corrected tautomer + AddHs        - adds the imidazole N-H donor that only
                                         exists in the correct aromatic form

Condition A must reproduce the committed CSV; that is the control which shows
the diagnosis is faithful to the production run.

Read-only. Never modifies a production artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SYSTEMS = {
    "5NDD_AZ8838": {
        "ligand": "01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.sdf",
        "receptor": "01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdb",
        "committed_csv": "02_REFERENCE_VALIDATION/5NDD_AZ8838_ProLIF_interactions.csv",
    },
    "5NDZ_AZ3451": {
        "ligand": "01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.sdf",
        "receptor": "01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdb",
        "committed_csv": "02_REFERENCE_VALIDATION/5NDZ_AZ3451_ProLIF_interactions.csv",
    },
}

HBOND_TYPES = ("HBDonor", "HBAcceptor", "XBDonor", "XBAcceptor")


def load_ligand(path: Path) -> Chem.Mol:
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]
    if not mols:
        raise SystemExit(f"No valid molecule in {path}")
    mol = mols[0]
    Chem.SanitizeMol(mol)
    return mol


def fingerprint(ligand: Chem.Mol, receptor: Chem.Mol) -> Optional[Counter]:
    import prolif as plf
    try:
        ifp = plf.Fingerprint().generate(
            plf.Molecule.from_rdkit(ligand, resname="LIG", resnumber=1, chain="A"),
            plf.Molecule.from_rdkit(receptor), metadata=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"      ProLIF failed: {exc}")
        return None
    types: Counter = Counter()
    for _pair, interactions in ifp.items():
        for itype in interactions:
            types[itype] += 1
    return types


def report_condition(label: str, types: Optional[Counter], n_h: int) -> Dict:
    if types is None:
        return {"label": label, "failed": True}
    total = int(sum(types.values()))
    hbonds = int(sum(v for k, v in types.items() if k in HBOND_TYPES))
    print(f"   {label}")
    print(f"      explicit ligand H = {n_h:3d}   interactions = {total:3d}   "
          f"hydrogen bonds = {hbonds}")
    print(f"      breakdown: {dict(types)}")
    return {"label": label, "failed": False, "explicit_ligand_hydrogens": n_h,
            "total_interactions": total, "hydrogen_bonds": hbonds,
            "breakdown": {k: int(v) for k, v in types.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    v2 = args.v2_root.resolve()
    out_dir = (args.out_dir or (v2 / "04_VERIFICATION")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(v2 / "scripts"))
    from check_reference_tautomers import _relocate_hydrogen, find_h_placement_artifacts

    report: Dict = {"systems": {}}

    for name, spec in SYSTEMS.items():
        lig_path, rec_path = v2 / spec["ligand"], v2 / spec["receptor"]
        if not lig_path.exists() or not rec_path.exists():
            print(f"[skip] {name}: missing input")
            continue
        print(f"\n=== {name} ===")

        stored = load_ligand(lig_path)
        receptor = Chem.MolFromPDBFile(str(rec_path), removeHs=False, sanitize=False)

        committed = v2 / spec["committed_csv"]
        n_committed = (sum(1 for _ in committed.open()) - 1) if committed.exists() else None

        entry: Dict = {"committed_interaction_rows": n_committed, "conditions": {}}

        entry["conditions"]["A_as_committed"] = report_condition(
            "A) as committed (stored tautomer, no explicit H)",
            fingerprint(stored, receptor),
            sum(1 for a in stored.GetAtoms() if a.GetSymbol() == "H"))

        a = entry["conditions"]["A_as_committed"]
        if n_committed is not None and not a.get("failed"):
            a["reproduces_committed_csv"] = (a["total_interactions"] == n_committed)
            print(f"      control: committed CSV has {n_committed} rows -> "
                  f"reproduced = {a['reproduces_committed_csv']}")

        with_h = Chem.AddHs(stored, addCoords=True)
        entry["conditions"]["B_stored_plus_hydrogens"] = report_condition(
            "B) stored tautomer + AddHs (isolates the missing-hydrogen defect)",
            fingerprint(with_h, receptor),
            sum(1 for at in with_h.GetAtoms() if at.GetSymbol() == "H"))

        found = find_h_placement_artifacts(stored)
        if found:
            corrected = _relocate_hydrogen(stored, found[0]["donor_carbon_idx"],
                                           found[0]["acceptor_nitrogen_idx"])
            if corrected is not None:
                corr_h = Chem.AddHs(corrected, addCoords=True)
                entry["conditions"]["C_corrected_plus_hydrogens"] = report_condition(
                    "C) corrected tautomer + AddHs (adds the azole N-H donor)",
                    fingerprint(corr_h, receptor),
                    sum(1 for at in corr_h.GetAtoms() if at.GetSymbol() == "H"))
                entry["corrected_retains_conformer"] = bool(corrected.GetNumConformers() > 0)
        else:
            print("   C) not applicable: no tautomer defect in this reference")

        report["systems"][name] = entry

    out = out_dir / "REFERENCE_IFP_DIAGNOSIS.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    any_missing = any(
        c.get("hydrogen_bonds") == 0
        for s in report["systems"].values()
        for k, c in s.get("conditions", {}).items()
        if k == "A_as_committed" and not c.get("failed"))
    return 1 if any_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
