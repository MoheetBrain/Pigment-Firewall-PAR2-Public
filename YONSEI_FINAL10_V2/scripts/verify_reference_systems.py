#!/usr/bin/env python3
"""Verify the cleaned crystallographic reference systems before GNINA runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem


SYSTEMS = {
    "5NDD_AZ8838": {
        "residue": "8TZ",
        "ligand_pdb": "01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.pdb",
        "ligand_sdf": "01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.sdf",
        "receptor_pdb": "01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdb",
        "receptor_pdbqt": "01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdbqt",
        "config": "01_REFERENCE_POCKETS/configs/BoxA_orthosteric_5NDD_ligand_centered.conf",
    },
    "5NDZ_AZ3451": {
        "residue": "8UN",
        "ligand_pdb": "01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.pdb",
        "ligand_sdf": "01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.sdf",
        "receptor_pdb": "01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdb",
        "receptor_pdbqt": "01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdbqt",
        "config": "01_REFERENCE_POCKETS/configs/BoxB_allosteric_5NDZ_ligand_centered.conf",
    },
}


def pdb_atoms(path: Path):
    atoms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = line[0:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        element = line[76:78].strip().upper() or line[12:16].strip()[0].upper()
        atoms.append(
            {
                "record": record,
                "resname": line[17:20].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": element,
            }
        )
    return atoms


def config_values(path: Path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            try:
                values[key] = float(value)
            except ValueError:
                values[key] = value
    return values


def rounded_coords(atoms):
    return sorted((a["element"], round(a["x"], 3), round(a["y"], 3), round(a["z"], 3)) for a in atoms)


def rounded_xyz(atoms):
    return sorted((round(a["x"], 3), round(a["y"], 3), round(a["z"], 3)) for a in atoms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve() / "YONSEI_FINAL10_V2"
    output = root / "01_BENCHMARK_REPAIR/reference_system_check.json"
    report = {"status": "PASS", "systems": {}}

    for name, system in SYSTEMS.items():
        ligand_pdb_atoms = pdb_atoms(root / system["ligand_pdb"])
        receptor_pdb_atoms = pdb_atoms(root / system["receptor_pdb"])
        receptor_pdbqt_atoms = pdb_atoms(root / system["receptor_pdbqt"])
        ligand_mols = [m for m in Chem.SDMolSupplier(str(root / system["ligand_sdf"]), removeHs=False, sanitize=False) if m is not None]
        if len(ligand_mols) != 1:
            raise SystemExit(f"{name}: expected one ligand SDF molecule, got {len(ligand_mols)}")
        ligand = ligand_mols[0]
        ligand_heavy = [atom for atom in ligand.GetAtoms() if atom.GetAtomicNum() > 1]
        ligand_heavy_coords = []
        conformer = ligand.GetConformer()
        for atom in ligand_heavy:
            point = conformer.GetAtomPosition(atom.GetIdx())
            ligand_heavy_coords.append((atom.GetSymbol().upper(), round(point.x, 3), round(point.y, 3), round(point.z, 3)))

        clean_ligand_ok = (
            bool(ligand_pdb_atoms)
            and all(atom["record"] == "HETATM" and atom["resname"] == system["residue"] for atom in ligand_pdb_atoms)
            and len(ligand_pdb_atoms) == len(ligand_heavy)
            and rounded_coords(ligand_pdb_atoms) == sorted(ligand_heavy_coords)
        )
        receptor_ok = bool(receptor_pdb_atoms) and all(atom["record"] == "ATOM" for atom in receptor_pdb_atoms)
        # PDBQT stores AutoDock atom types (e.g. ``A`` for aromatic carbon)
        # rather than elemental symbols, so compare the coordinate frame only.
        receptor_frame_ok = rounded_xyz(receptor_pdb_atoms) == rounded_xyz(receptor_pdbqt_atoms)
        xs = [atom["x"] for atom in ligand_pdb_atoms]
        ys = [atom["y"] for atom in ligand_pdb_atoms]
        zs = [atom["z"] for atom in ligand_pdb_atoms]
        center = {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys), "z": sum(zs) / len(zs)}
        spans = {"x": max(xs) - min(xs), "y": max(ys) - min(ys), "z": max(zs) - min(zs)}
        config = config_values(root / system["config"])
        config_center = {axis: config[f"center_{axis}"] for axis in "xyz"}
        config_size = {axis: config[f"size_{axis}"] for axis in "xyz"}
        center_error = max(abs(center[axis] - config_center[axis]) for axis in "xyz")
        size_error = max(abs(config_size[axis] - (spans[axis] + 10.0)) for axis in "xyz")
        system_report = {
            "expected_ligand_residue": system["residue"],
            "clean_ligand_only_expected_residue": clean_ligand_ok,
            "clean_ligand_atom_count": len(ligand_pdb_atoms),
            "clean_ligand_heavy_atom_count": len(ligand_heavy),
            "receptor_atom_only_no_hetero_records": receptor_ok,
            "receptor_pdb_pdbqt_coordinate_frame_match": receptor_frame_ok,
            "ligand_heavy_atom_centroid": center,
            "ligand_heavy_atom_span": spans,
            "config_center": config_center,
            "config_size": config_size,
            "center_max_absolute_error_angstrom": center_error,
            "size_vs_span_plus_5A_each_side_max_absolute_error_angstrom": size_error,
            "coordinate_frame_and_box_check_passed": clean_ligand_ok and receptor_ok and receptor_frame_ok and center_error < 0.001 and size_error < 0.001,
        }
        report["systems"][name] = system_report
        report["status"] = "PASS" if report["status"] == "PASS" and system_report["coordinate_frame_and_box_check_passed"] else "FAIL"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
