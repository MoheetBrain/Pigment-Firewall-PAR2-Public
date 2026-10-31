#!/usr/bin/env python3
"""Build clean, ligand-centred PAR2 reference-pocket artifacts.

Inputs are the repository's historical PDBQT receptor coordinates and the
crystallographic ligand PDB files.  The historical files are never modified.
Only ATOM records are copied into the new receptor files; only the expected
8TZ/8UN ligand residue and its CONECT records are copied into the new ligand
files.  Open Babel is used to create SDF/PDB derivatives while preserving the
experimental coordinate frame.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


SYSTEMS = {
    "5NDD_AZ8838": {
        "pdb_id": "5NDD",
        "residue": "8TZ",
        "source_ligand": "BENCHMARK_LIGS/lig_AZ8838_orthosteric_boxA.pdb",
        "source_receptor": "PROTEINS/PAR2_inactive_orthosteric_receptor.pdbqt",
        "box_name": "BoxA_orthosteric",
        "clean_ligand_stem": "5NDD_AZ8838_crystallographic_clean",
        "clean_receptor_stem": "5NDD_inactive_orthosteric_receptor_clean",
    },
    "5NDZ_AZ3451": {
        "pdb_id": "5NDZ",
        "residue": "8UN",
        "source_ligand": "BENCHMARK_LIGS/lig_AZ3451_allosteric_boxB.pdb",
        "source_receptor": "PROTEINS/PAR2_inactive_allosteric_receptor.pdbqt",
        "box_name": "BoxB_allosteric",
        "clean_ligand_stem": "5NDZ_AZ3451_crystallographic_clean",
        "clean_receptor_stem": "5NDZ_inactive_allosteric_receptor_clean",
    },
}


def pdb_atom_serial(line: str) -> int | None:
    try:
        return int(line[6:11])
    except ValueError:
        return None


def parse_xyz(line: str) -> tuple[float, float, float]:
    return tuple(float(line[start : start + 8]) for start in (30, 38, 46))


def write_clean_ligand(source: Path, destination: Path, expected_residue: str) -> dict:
    kept_atoms: list[str] = []
    old_to_new: dict[int, int] = {}
    coords: list[tuple[float, float, float]] = []
    elements: list[str] = []
    next_serial = 1
    all_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in all_lines:
        if line.startswith("HETATM") and line[17:20].strip() == expected_residue:
            old_serial = pdb_atom_serial(line)
            if old_serial is None:
                continue
            new_line = f"{line[:6]}{next_serial:5d}{line[11:]}"
            kept_atoms.append(new_line)
            old_to_new[old_serial] = next_serial
            coords.append(parse_xyz(line))
            element = (line[76:78].strip() or re.sub(r"[^A-Za-z]", "", line[12:16])).upper()
            elements.append(element)
            next_serial += 1

    conect: list[str] = []
    for line in all_lines:
        if not line.startswith("CONECT"):
            continue
        old_serials = []
        for start in range(6, len(line), 5):
            token = line[start : start + 5].strip()
            if token:
                try:
                    old_serials.append(int(token))
                except ValueError:
                    pass
        if len(old_serials) < 2 or old_serials[0] not in old_to_new:
            continue
        new_serials = [old_to_new[old_serials[0]]]
        new_serials.extend(old_to_new[value] for value in old_serials[1:] if value in old_to_new)
        if len(new_serials) > 1:
            conect.append("CONECT" + "".join(f"{value:5d}" for value in new_serials))

    destination.parent.mkdir(parents=True, exist_ok=True)
    crystallographic_header = next((line for line in all_lines if line.startswith("CRYST1")),
                                   "CRYST1".ljust(80))
    # Open Babel is strict about this minimal ligand-only PDB; keep only the
    # crystallographic header plus valid coordinate/connectivity records.
    lines = [crystallographic_header]
    lines.extend(kept_atoms)
    lines.extend(conect)
    lines.append("END")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"atom_count": len(coords), "coords": coords, "elements": elements, "source_serials": old_to_new}


def write_clean_receptor(source: Path, destination: Path) -> int:
    atom_lines = [line for line in source.read_text(encoding="utf-8", errors="replace").splitlines()
                  if line.startswith("ATOM")]
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["REMARK YONSEI_FINAL10_V2 receptor derived from historical PDBQT ATOM records",
             "REMARK No crystallographic HETATM records retained"]
    lines.extend(atom_lines)
    lines.append("TER")
    lines.append("END")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(atom_lines)


def geometry(coords: list[tuple[float, float, float]]) -> dict:
    centroid = [sum(coord[i] for coord in coords) / len(coords) for i in range(3)]
    minimum = [min(coord[i] for coord in coords) for i in range(3)]
    maximum = [max(coord[i] for coord in coords) for i in range(3)]
    span = [maximum[i] - minimum[i] for i in range(3)]
    return {
        "atom_count": len(coords),
        "centroid": {axis: centroid[i] for i, axis in enumerate(("x", "y", "z"))},
        "min": {axis: minimum[i] for i, axis in enumerate(("x", "y", "z"))},
        "max": {axis: maximum[i] for i, axis in enumerate(("x", "y", "z"))},
        "span_angstrom": {axis: span[i] for i, axis in enumerate(("x", "y", "z"))},
    }


def run_obabel(obabel: str, input_path: Path, input_format: str, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [obabel, f"-i{input_format}", str(input_path), f"-o{output_format}", "-O", str(output_path)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Open Babel failed: {' '.join(command)}\n{result.stderr}")


def write_config(path: Path, receptor_relative: str, geometry_data: dict, padding: float) -> dict:
    centroid = geometry_data["centroid"]
    span = geometry_data["span_angstrom"]
    axes = ("x", "y", "z")
    sizes = {axis: span[axis] + 2.0 * padding for axis in axes}
    lines = [
        f"receptor = {receptor_relative}",
        f"center_x = {centroid['x']:.3f}",
        f"center_y = {centroid['y']:.3f}",
        f"center_z = {centroid['z']:.3f}",
        f"size_x = {sizes['x']:.3f}",
        f"size_y = {sizes['y']:.3f}",
        f"size_z = {sizes['z']:.3f}",
        "num_modes = 9",
        "exhaustiveness = 32",
        "cnn_scoring = rescore",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"padding_each_side_angstrom": padding, "size_angstrom": sizes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--obabel", default=shutil.which("obabel") or "obabel")
    parser.add_argument("--padding", type=float, default=5.0)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "YONSEI_FINAL10_V2/01_REFERENCE_POCKETS"
    ligand_dir = base / "ligands"
    receptor_dir = base / "receptors"
    config_dir = base / "configs"
    metadata_dir = base / "metadata"
    for directory in (ligand_dir, receptor_dir, config_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_metadata = {"padding_each_side_angstrom": args.padding, "systems": {}}
    for system_name, system in SYSTEMS.items():
        source_ligand = root / system["source_ligand"]
        source_receptor = root / system["source_receptor"]
        clean_ligand_pdb = ligand_dir / f"{system['clean_ligand_stem']}.pdb"
        clean_ligand_sdf = ligand_dir / f"{system['clean_ligand_stem']}.sdf"
        clean_receptor_pdbqt = receptor_dir / f"{system['clean_receptor_stem']}.pdbqt"
        clean_receptor_pdb = receptor_dir / f"{system['clean_receptor_stem']}.pdb"
        config = config_dir / f"{system['box_name']}_{system['pdb_id']}_ligand_centered.conf"

        ligand_info = write_clean_ligand(source_ligand, clean_ligand_pdb, system["residue"])
        receptor_atom_count = write_clean_receptor(source_receptor, clean_receptor_pdbqt)
        run_obabel(args.obabel, clean_ligand_pdb, "pdb", clean_ligand_sdf, "sdf")
        run_obabel(args.obabel, clean_receptor_pdbqt, "pdbqt", clean_receptor_pdb, "pdb")
        geom = geometry(ligand_info["coords"])
        config_data = write_config(config, f"../receptors/{clean_receptor_pdbqt.name}", geom, args.padding)

        system_metadata = {
            "system": system_name,
            "pdb_id": system["pdb_id"],
            "expected_ligand_residue": system["residue"],
            "source_ligand": str(source_ligand),
            "source_receptor": str(source_receptor),
            "clean_ligand_pdb": str(clean_ligand_pdb),
            "clean_ligand_sdf": str(clean_ligand_sdf),
            "clean_receptor_pdbqt": str(clean_receptor_pdbqt),
            "clean_receptor_pdb": str(clean_receptor_pdb),
            "config": str(config),
            "ligand_geometry_heavy_atom_frame": geom,
            "ligand_element_counts": {element: ligand_info["elements"].count(element) for element in sorted(set(ligand_info["elements"]))},
            "clean_receptor_atom_count": receptor_atom_count,
            "config_box": config_data,
            "coordinate_frame_policy": "copy source coordinates unchanged; filter only by expected ligand residue and ATOM receptor records",
            "validation_status": "STRUCTURE_AND_CONFIG_BUILT; GNINA_REDOCK_PENDING",
        }
        (metadata_dir / f"{system_name}.json").write_text(json.dumps(system_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        all_metadata["systems"][system_name] = system_metadata

    (metadata_dir / "reference_geometry_and_configs.json").write_text(json.dumps(all_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: data["config_box"] for name, data in all_metadata["systems"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
