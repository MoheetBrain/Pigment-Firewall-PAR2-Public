#!/usr/bin/env python3
"""Build a sourceability-aware V2 candidate universe without docking.

This is intentionally a triage artifact.  It combines V1 final-500 rows,
all FDA_DRUGS records present in the repository, and identifiable non-FDA V1
hits.  Procurement fields remain PENDING because no live vendor lookup was
performed.  No V1 file is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def read_table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mol_and_key(smiles: str, fallback: str = ""):
    if not smiles or not smiles.strip():
        return None, fallback.strip()
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, fallback.strip() or smiles.strip()
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        key = ""
    if not key:
        key = Chem.MolToSmiles(mol, canonical=True)
    return mol, key


def numeric(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def best_by_affinity(rows: list[dict]) -> dict | None:
    valid = []
    for row in rows:
        score = numeric(row.get("minimizedAffinity", ""))
        if score is not None:
            valid.append((score, row))
    return min(valid, key=lambda item: item[0])[1] if valid else None


def merge_value(values) -> str:
    return ";".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def descriptors(mol) -> dict[str, object]:
    if mol is None:
        return {"rdkit_parse_status": "FAILED", "MW": "", "cLogP": "", "TPSA": "", "HBD": "", "HBA": "", "RotBonds": "", "heavy_atoms": ""}
    try:
        return {
            "rdkit_parse_status": "OK",
            "MW": round(Descriptors.MolWt(mol), 3),
            "cLogP": round(Crippen.MolLogP(mol), 3),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotBonds": Lipinski.NumRotatableBonds(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
        }
    except Exception:
        return {"rdkit_parse_status": "DESCRIPTOR_ERROR", "MW": "", "cLogP": "", "TPSA": "", "HBD": "", "HBA": "", "RotBonds": "", "heavy_atoms": ""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "YONSEI_FINAL10_V2/03_SOURCEABILITY"
    out.mkdir(parents=True, exist_ok=True)

    pass1 = read_table(root / "GNINA_SCREEN_EX32/pass1_all_hits.csv")
    final = read_table(root / "RESULTS/CNN_FINAL_SCORES.csv")
    fda = read_table(root / "SCREENING/FDA_DRUGS.csv")

    pass1_groups: dict[str, list[dict]] = defaultdict(list)
    for row in pass1:
        smiles = row.get("SMILES", "") or row.get("compound_smiles", "")
        existing_key = row.get("inchikey", "").strip()
        _, key = mol_and_key(smiles, existing_key or row.get("mol_name", ""))
        pass1_groups[key].append({**row, "_key": key})

    final_groups: dict[str, list[dict]] = defaultdict(list)
    for row in final:
        _, key = mol_and_key(row.get("smiles", ""), row.get("inchikey", "") or row.get("mol_name", ""))
        final_groups[key].append({**row, "_key": key})

    fda_groups: dict[str, list[dict]] = defaultdict(list)
    for row in fda:
        mol, key = mol_and_key(row.get("SMILES", ""), row.get("ChEMBL_ID", "") or row.get("Name", ""))
        fda_groups[key].append({**row, "_key": key, "_mol": mol})

    # Final-500 keys are retained exactly, all FDA records are included, and
    # identifiable non-FDA V1 hits are included if they carry stable database
    # or structure identifiers.
    selected: dict[str, list[str]] = defaultdict(list)
    for key in final_groups:
        selected[key].append("V1_FINAL500")
    for key in fda_groups:
        selected[key].append("FDA_APPROVED_LIBRARY")
    for key, rows in pass1_groups.items():
        identifiable = any(row.get(field, "").strip() for row in rows for field in (
            "inchikey", "inchi", "lotus_id", "npaid", "compound_id", "ChEMBL_ID", "wikidata_id", "compound_inchikey", "compound_inchi"
        ))
        if identifiable:
            selected[key].append("IDENTIFIABLE_OTHER_V1_HIT")

    rows: list[dict] = []
    for key, triage_classes in selected.items():
        p1_rows = pass1_groups.get(key, [])
        final_rows = final_groups.get(key, [])
        fda_rows = fda_groups.get(key, [])
        representative = final_rows[0] if final_rows else (p1_rows[0] if p1_rows else (fda_rows[0] if fda_rows else {}))
        smiles = representative.get("smiles", "") or representative.get("SMILES", "") or representative.get("compound_smiles", "")
        if not smiles and p1_rows:
            smiles = p1_rows[0].get("SMILES", "") or p1_rows[0].get("compound_smiles", "")
        mol, computed_key = mol_and_key(smiles, key)
        key = computed_key or key
        if mol is not None:
            desc = descriptors(mol)
        else:
            desc = descriptors(None)

        best_a = best_by_affinity([row for row in p1_rows if row.get("box") == "BoxA_orthosteric"])
        best_b = best_by_affinity([row for row in p1_rows if row.get("box") == "BoxB_allosteric"])
        best_pass1 = best_by_affinity(p1_rows)
        final_row = min(final_rows, key=lambda row: numeric(row.get("rank", "")) or 10**9) if final_rows else {}
        fda_row = fda_rows[0] if fda_rows else {}
        source_libraries = merge_value(
            [row.get("library", "") for row in p1_rows] +
            ["FDA_DRUGS" for row in fda_rows] +
            [final_row.get("source_library", "")]
        )
        original_identifiers = merge_value(
            [row.get(field, "") for row in p1_rows for field in ("inchikey", "lotus_id", "npaid", "compound_id", "ChEMBL_ID", "wikidata_id", "compound_inchikey")] +
            [fda_row.get("ChEMBL_ID", ""), fda_row.get("Name", "")]
        )
        selected_site = final_row.get("binding_site", "")
        if not selected_site and best_pass1:
            selected_site = best_pass1.get("box", "")
        approval_status = "YES" if fda_row and (numeric(fda_row.get("MaxPhase", "")) or 0) >= 4 else ("KNOWN_HUMAN_EXPOSURE_OR_APPROVAL_PENDING_VERIFICATION" if "FDA_APPROVED_LIBRARY" in triage_classes else "NOT_ESTABLISHED")
        final_rank = final_row.get("rank", "")
        row = {
            "candidate_key": key,
            "SMILES": smiles,
            "InChIKey": key if key.count("-") == 2 and len(key) >= 20 else (fda_row.get("ChEMBL_ID", "") or ""),
            "candidate_name": final_row.get("mol_name", "") or fda_row.get("Name", "") or (p1_rows[0].get("compound_name", "") if p1_rows else "") or (p1_rows[0].get("mol_name", "") if p1_rows else ""),
            "original_library": source_libraries,
            "triage_class": merge_value(triage_classes),
            "original_v1_rank": final_rank,
            "original_v1_status": "IN_V1_FINAL500" if final_row else ("IN_PASS1_ONLY" if p1_rows else "FDA_LIBRARY_NOT_IN_PASS1"),
            "original_selected_binding_site": selected_site,
            "original_boxA_score": best_a.get("minimizedAffinity", "") if best_a else "",
            "original_boxB_score": best_b.get("minimizedAffinity", "") if best_b else "",
            "original_boxA_CNNscore": best_a.get("CNNscore", "") if best_a else "",
            "original_boxB_CNNscore": best_b.get("CNNscore", "") if best_b else "",
            "original_boxA_CNNaffinity": best_a.get("CNNaffinity", "") if best_a else "",
            "original_boxB_CNNaffinity": best_b.get("CNNaffinity", "") if best_b else "",
            "original_boxA_CNN_VS": best_a.get("CNN_VS", "") if best_a else "",
            "original_boxB_CNN_VS": best_b.get("CNN_VS", "") if best_b else "",
            "original_v1_vina_affinity": final_row.get("vina_affinity", ""),
            "original_v1_rmsd": final_row.get("rmsd", ""),
            "original_v1_cnn_score": final_row.get("cnn_score", ""),
            "original_v1_cnn_affinity": final_row.get("cnn_affinity", ""),
            "original_v1_cnn_vs": final_row.get("cnn_vs", ""),
            "original_identifiers": original_identifiers,
            "ChEMBL_ID": fda_row.get("ChEMBL_ID", "") or (p1_rows[0].get("ChEMBL_ID", "") if p1_rows else ""),
            "FDA_or_approved_known_human_exposure": approval_status,
            "commercial_availability": "PENDING_LIVE_VENDOR_LOOKUP",
            "vendor": "PENDING",
            "catalogue_ID": "PENDING",
            "approximate_price": "PENDING",
            "stock_status": "PENDING",
            "procurement_note": "No vendor lookup performed; do not treat this row as confirmed purchasable.",
            **desc,
        }
        rows.append(row)

    rows.sort(key=lambda row: (numeric(row.get("original_v1_rank", "")) is None, numeric(row.get("original_v1_rank", "")) or 10**9, row.get("candidate_name", "")))
    fields = list(rows[0])
    write_csv(out / "V2_CANDIDATE_UNIVERSE.csv", rows, fields)
    universe_by_key = {row["candidate_key"]: row for row in rows}
    top100_rows = []
    for final_row in final[:100]:
        _, final_key = mol_and_key(final_row.get("smiles", ""), final_row.get("inchikey", "") or final_row.get("mol_name", ""))
        triage_row = universe_by_key.get(final_key, {})
        top100_rows.append({
            "v1_rank": final_row.get("rank", ""),
            "mol_name": final_row.get("mol_name", ""),
            "v1_source_library": final_row.get("source_library", ""),
            "v1_binding_site": final_row.get("binding_site", ""),
            "SMILES": final_row.get("smiles", ""),
            "InChIKey": final_row.get("inchikey", "") or triage_row.get("InChIKey", ""),
            "universe_match": bool(triage_row),
            "triage_class": triage_row.get("triage_class", ""),
            "FDA_or_approved_known_human_exposure": triage_row.get("FDA_or_approved_known_human_exposure", ""),
            "commercial_availability": triage_row.get("commercial_availability", "PENDING_LIVE_VENDOR_LOOKUP"),
            "vendor": triage_row.get("vendor", "PENDING"),
            "catalogue_ID": triage_row.get("catalogue_ID", "PENDING"),
            "approximate_price": triage_row.get("approximate_price", "PENDING"),
            "stock_status": triage_row.get("stock_status", "PENDING"),
        })
    write_csv(out / "V1_TOP100_SOURCEABILITY.csv", top100_rows, list(top100_rows[0]))
    summary = {
        "candidate_count": len(rows),
        "triage_class_counts": dict(sorted(Counter(row["triage_class"] for row in rows).items())),
        "original_v1_status_counts": dict(sorted(Counter(row["original_v1_status"] for row in rows).items())),
        "source_libraries": dict(sorted(Counter(row["original_library"] for row in rows).items())),
        "selected_binding_sites": dict(sorted(Counter(row["original_selected_binding_site"] for row in rows).items())),
        "rdkit_parse_status": dict(sorted(Counter(row["rdkit_parse_status"] for row in rows).items())),
        "commercial_availability_status": "PENDING_LIVE_VENDOR_LOOKUP_FOR_ALL_ROWS",
        "vendor_lookup_performed": False,
        "v1_top100_sourceability": {
            "rows": len(top100_rows),
            "universe_matches": sum(row["universe_match"] for row in top100_rows),
            "procurement_status_counts": dict(sorted(Counter(row["commercial_availability"] for row in top100_rows).items())),
            "known_human_exposure_or_approval_counts": dict(sorted(Counter(row["FDA_or_approved_known_human_exposure"] for row in top100_rows).items())),
        },
        "approval_status_counts": dict(sorted(Counter(row["FDA_or_approved_known_human_exposure"] for row in rows).items())),
        "v1_final500_rows_in_universe": sum(row["original_v1_status"] == "IN_V1_FINAL500" for row in rows),
        "all_fda_library_rows_in_universe": sum("FDA_APPROVED_LIBRARY" in row["triage_class"] for row in rows),
        "other_identifiable_v1_hits_in_universe": sum("IDENTIFIABLE_OTHER_V1_HIT" in row["triage_class"] for row in rows),
        "notes": [
            "This file is a sourceability triage universe, not a biological ranking.",
            "No candidate was redocked because the corrected GNINA reference benchmark is blocked.",
            "Procurement fields are intentionally PENDING rather than inferred or invented.",
        ],
    }
    (out / "V2_CANDIDATE_UNIVERSE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# V2 sourceability triage",
        "",
        f"Candidate universe size: **{len(rows):,} unique structure keys**.",
        "",
        "The universe combines the V1 final 500, all FDA_DRUGS records in the repository, and identifiable non-FDA Pass-1 hits. Procurement fields are all `PENDING` because no live vendor lookup was performed.",
        "",
        "## Counts",
        "",
        "| Measure | Count |",
        "|---|---:|",
        f"| V1 final-500 rows represented | {summary['v1_final500_rows_in_universe']} |",
        f"| FDA/approved-library records represented | {summary['all_fda_library_rows_in_universe']} |",
        f"| Other identifiable V1 hits represented | {summary['other_identifiable_v1_hits_in_universe']} |",
        f"| RDKit-parseable structures | {summary['rdkit_parse_status'].get('OK', 0)} |",
        "",
        "No `V2_TOP100.csv`, `V2_TOP50.csv`, or `V2_TOP20.csv` is generated in this phase because the corrected reference redocking gate is blocked.",
        "",
        "`V1_TOP100_SOURCEABILITY.csv` contains the V1 top-100 rows with procurement status. All live procurement fields are `PENDING`.",
    ]
    (out / "SOURCEABILITY_TRIAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
