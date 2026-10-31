#!/usr/bin/env python3
"""Generate ProLIF interaction fingerprints for the clean crystal references."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path

from rdkit import Chem

warnings.filterwarnings("ignore", category=DeprecationWarning)
import prolif as plf


SYSTEMS = {
    "5NDD_AZ8838": {
        "ligand": "01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.sdf",
        "receptor": "01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdb",
        "ligand_residue": "8TZ",
    },
    "5NDZ_AZ3451": {
        "ligand": "01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.sdf",
        "receptor": "01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdb",
        "ligand_residue": "8UN",
    },
}


def residue_string(residue) -> str:
    return f"{residue.name}:{residue.number}:{residue.chain or '-'}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "YONSEI_FINAL10_V2/02_REFERENCE_VALIDATION"
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summary: dict = {"tool": "ProLIF", "prolif_version": plf.__version__, "systems": {}}

    for system_name, system in SYSTEMS.items():
        ligand = Chem.MolFromMolFile(str(root / "YONSEI_FINAL10_V2" / system["ligand"]), removeHs=False, sanitize=True)
        receptor = Chem.MolFromPDBFile(str(root / "YONSEI_FINAL10_V2" / system["receptor"]), removeHs=False, sanitize=False)
        if ligand is None or receptor is None:
            raise RuntimeError(f"Could not parse clean reference molecules for {system_name}")
        ligand_plf = plf.Molecule.from_rdkit(ligand, resname="LIG", resnumber=1, chain="A")
        receptor_plf = plf.Molecule.from_rdkit(receptor)
        fingerprint = plf.Fingerprint()
        ifp = fingerprint.generate(ligand_plf, receptor_plf, metadata=True)
        fingerprint_rows: list[dict] = []
        fingerprint_keys: list[str] = []
        interaction_type_counts = Counter()
        for pair, interactions in ifp.items():
            ligand_residue, protein_residue = pair
            protein_residue_string = residue_string(protein_residue)
            for interaction_type, metadata_values in interactions.items():
                interaction_type_counts[interaction_type] += len(metadata_values)
                fingerprint_keys.append(f"{protein_residue_string}|{interaction_type}")
                for metadata in metadata_values:
                    distance = metadata.get("distance")
                    row = {
                        "system": system_name,
                        "ligand_residue": residue_string(ligand_residue),
                        "protein_residue": protein_residue_string,
                        "interaction_type": interaction_type,
                        "distance_angstrom": distance,
                        "ligand_atom_indices": ";".join(str(index) for index in metadata.get("indices", {}).get("ligand", ())),
                        "protein_atom_indices": ";".join(str(index) for index in metadata.get("indices", {}).get("protein", ())),
                        "ligand_parent_atom_indices": ";".join(str(index) for index in metadata.get("parent_indices", {}).get("ligand", ())),
                        "protein_parent_atom_indices": ";".join(str(index) for index in metadata.get("parent_indices", {}).get("protein", ())),
                    }
                    fingerprint_rows.append(row)
        fingerprint_rows.sort(key=lambda row: (row["protein_residue"], row["interaction_type"], row["distance_angstrom"] or 999.0))
        output_csv = out / f"{system_name}_ProLIF_interactions.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fingerprint_rows[0]) if fingerprint_rows else ["system"])
            writer.writeheader()
            writer.writerows(fingerprint_rows)
        system_summary = {
            "system": system_name,
            "ligand_file": str(root / "YONSEI_FINAL10_V2" / system["ligand"]),
            "receptor_file": str(root / "YONSEI_FINAL10_V2" / system["receptor"]),
            "ligand_atom_count": ligand.GetNumAtoms(),
            "receptor_atom_count": receptor.GetNumAtoms(),
            "protein_residue_pair_count": len(ifp),
            "interaction_count": len(fingerprint_rows),
            "interaction_type_counts": dict(sorted(interaction_type_counts.items())),
            "reference_fingerprint_keys": sorted(fingerprint_keys),
            "reference_fingerprint_key_count": len(fingerprint_keys),
            "status": "CRYSTAL_REFERENCE_FINGERPRINT_GENERATED",
        }
        (out / f"{system_name}_ProLIF_interactions.json").write_text(json.dumps({"summary": system_summary, "interactions": fingerprint_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["systems"][system_name] = system_summary
        all_rows.extend(fingerprint_rows)

    if all_rows:
        with (out / "reference_interactions_all.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    (out / "reference_interactions_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
