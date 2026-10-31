#!/usr/bin/env python3
"""Run corrected GNINA reference redocking when a compatible binary exists.

This script intentionally refuses to substitute historical redock outputs and
does not run candidate docking.  It writes only small reference-pose outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


SYSTEMS = {
    "5NDD_AZ8838": {
        "ligand": "01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.sdf",
        "receptor": "01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdbqt",
        "config": "01_REFERENCE_POCKETS/configs/BoxA_orthosteric_5NDD_ligand_centered.conf",
        "output": "01_BENCHMARK_REPAIR/corrected_redock/5NDD_AZ8838_corrected_gnina_redock.sdf",
        "log": "01_BENCHMARK_REPAIR/corrected_redock/5NDD_AZ8838_corrected_gnina_redock.log",
    },
    "5NDZ_AZ3451": {
        "ligand": "01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.sdf",
        "receptor": "01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdbqt",
        "config": "01_REFERENCE_POCKETS/configs/BoxB_allosteric_5NDZ_ligand_centered.conf",
        "output": "01_BENCHMARK_REPAIR/corrected_redock/5NDZ_AZ3451_corrected_gnina_redock.sdf",
        "log": "01_BENCHMARK_REPAIR/corrected_redock/5NDZ_AZ3451_corrected_gnina_redock.log",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=sorted(SYSTEMS))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--gnina", default=shutil.which("gnina"))
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--num-modes", type=int, default=9)
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.gnina:
        raise SystemExit("GNINA executable not found; corrected reference redocking is blocked and candidate docking must not run.")
    system = SYSTEMS[args.system]
    output = root / "YONSEI_FINAL10_V2" / system["output"]
    log = root / "YONSEI_FINAL10_V2" / system["log"]
    output.parent.mkdir(parents=True, exist_ok=True)
    version_result = subprocess.run([args.gnina, "--version"], capture_output=True, text=True, check=False)
    gnina_version = (version_result.stdout or version_result.stderr).strip()
    command = [
        args.gnina,
        "--receptor", str(root / "YONSEI_FINAL10_V2" / system["receptor"]),
        "--ligand", str(root / "YONSEI_FINAL10_V2" / system["ligand"]),
        "--config", str(root / "YONSEI_FINAL10_V2" / system["config"]),
        "--out", str(output),
        "--log", str(log),
        "--exhaustiveness", str(args.exhaustiveness),
        "--num_modes", str(args.num_modes),
        "--seed", str(args.seed),
        "--cnn_scoring", "rescore",
        "--no_gpu",
        "--min_rmsd_filter", "2.0",
        "--addH", "true",
        "--cpu", str(max(1, args.cpu)),
    ]
    print("Running:", " ".join(command))
    result = subprocess.run(command, cwd=str(root), text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    metadata = {
        "system": args.system,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "gnina_executable": str(Path(args.gnina).resolve()),
        "gnina_version": gnina_version,
        "command": command,
        "random_seed": args.seed,
        "exhaustiveness": args.exhaustiveness,
        "num_modes": args.num_modes,
        "cnn_scoring": "rescore",
        "min_rmsd_filter": 2.0,
        "cpu_threads": max(1, args.cpu),
        "force_cpu": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "receptor": str(root / "YONSEI_FINAL10_V2" / system["receptor"]),
        "ligand": str(root / "YONSEI_FINAL10_V2" / system["ligand"]),
        "config": str(root / "YONSEI_FINAL10_V2" / system["config"]),
        "output": str(output),
        "log": str(log),
    }
    metadata_path = output.with_name(output.stem + "_run_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
