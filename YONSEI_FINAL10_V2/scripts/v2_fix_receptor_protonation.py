#!/usr/bin/env python3
"""Repair the receptor protonation state used for interaction fingerprinting.

Two defects are corrected here.

1. The campaign receptors were built with `obabel -h -p 7.4`. The `-h` flag adds
   every hydrogen neutrally and overrides the pH model, so all 39 lysines were
   left neutral (2 H on NZ) and 17 of 29 aspartates and 10 of 23 glutamates were
   left as neutral acids. At pH 7.4 lysine should carry +1 and the acids should
   be deprotonated. Consequence: no salt bridge or receptor-as-donor hydrogen
   bond could ever be detected anywhere in the campaign.

2. Running `obabel -p 7.4` alone fixes the chemistry but RENUMBERS every residue
   (PHE 59 becomes PHE 167), which would invalidate every ProLIF residue key.

The fix keeps the correct chemistry and restores the original numbering by
matching heavy atoms on exact coordinates, which are unchanged by protonation.
Each added hydrogen inherits the residue identity of its nearest heavy atom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_atoms(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        rows.append(dict(line=line, element=element, xyz=np.array(xyz),
                         resname=line[17:20], chain=line[21], resseq=line[22:26],
                         icode=line[26], name=line[12:16]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True,
                        help="receptor with the authoritative residue numbering")
    parser.add_argument("--protonated", type=Path, required=True,
                        help="obabel -p 7.4 output (correct chemistry, renumbered)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    original = read_atoms(args.original)
    protonated = read_atoms(args.protonated)
    orig_heavy = [a for a in original if a["element"] != "H"]
    prot_heavy = [a for a in protonated if a["element"] != "H"]
    if len(orig_heavy) != len(prot_heavy):
        raise SystemExit(f"heavy-atom count differs: {len(orig_heavy)} vs {len(prot_heavy)}")

    # Match heavy atoms on coordinates; protonation must not move them.
    coords = np.array([a["xyz"] for a in orig_heavy])
    mapping, worst = {}, 0.0
    for i, a in enumerate(prot_heavy):
        d = np.linalg.norm(coords - a["xyz"], axis=1)
        j = int(d.argmin())
        worst = max(worst, float(d[j]))
        mapping[id(a)] = orig_heavy[j]
    if worst > args.tolerance:
        raise SystemExit(f"heavy atoms moved during protonation (max {worst:.4f} A)")

    heavy_xyz = np.array([a["xyz"] for a in prot_heavy])
    out_lines, serial = [], 0
    for a in protonated:
        if a["element"] == "H":
            j = int(np.linalg.norm(heavy_xyz - a["xyz"], axis=1).argmin())
            src = mapping[id(prot_heavy[j])]
        else:
            src = mapping[id(a)]
        serial += 1
        line = a["line"]
        line = (f"{line[:6]}{serial:5d}{line[11:17]}{src['resname']}{line[20]}"
                f"{src['chain']}{src['resseq']}{src['icode']}{line[27:]}")
        out_lines.append(line)
    args.out.write_text("\n".join(out_lines) + "\nEND\n")

    # Verify the chemistry we actually care about, geometrically.
    written = read_atoms(args.out)
    hydro = np.array([a["xyz"] for a in written if a["element"] == "H"])
    def attached(resname, name):
        return [int((np.linalg.norm(hydro - a["xyz"], axis=1) < 1.25).sum())
                for a in written if a["resname"].strip() == resname and a["name"].strip() == name]
    lys, asp, glu = attached("LYS", "NZ"), attached("ASP", "OD2"), attached("GLU", "OE2")
    report = dict(
        original=str(args.original), protonated=str(args.protonated), out=str(args.out),
        heavy_atoms=len(orig_heavy), hydrogens=int(len(hydro)),
        max_heavy_atom_shift_A=round(worst, 6),
        lys_total=len(lys), lys_charged_NZ_3H=sum(1 for x in lys if x == 3),
        asp_total=len(asp), asp_protonated=sum(1 for x in asp if x >= 1),
        glu_total=len(glu), glu_protonated=sum(1 for x in glu if x >= 1),
        residue_numbering="restored from the original receptor",
    )
    ok = (report["lys_charged_NZ_3H"] == report["lys_total"]
          and report["asp_protonated"] == 0 and report["glu_protonated"] == 0)
    report["pH74_state_correct"] = bool(ok)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
