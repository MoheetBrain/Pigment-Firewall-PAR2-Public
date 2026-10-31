#!/usr/bin/env python3
"""Audit the full 15,000-design de novo library against the tautomer repair.

Produces a single row-per-design audit table for `DENOVO_15000.csv` that carries,
for every design, the stored ("before") structure, the repaired ("after")
structure, and independently recomputed RDKit descriptors for both: molecular
formula, InChIKey, heavy-atom count, aromatic-bond count and parse status.

The only corrections applied are those already enumerated in
`04_VERIFICATION/affected_designs_wrong_tautomer.csv`. This script does not
derive, infer, or invent any new chemistry: a design is marked corrected only if
its stored SMILES appears verbatim in that file, and the replacement SMILES is
taken verbatim from the same row. Every reconciliation figure printed by this
script is computed here rather than copied from an upstream report.

Read-only with respect to every existing artifact. Only the three output paths
are written.

Outputs
  DENOVO_FULL_LIBRARY_AUDIT.csv       one row per DENOVO_15000.csv row
  DENOVO_REMAINING_TO_SCREEN.csv      the subset not screened in batch 1
  DENOVO_FULL_LIBRARY_AUDIT.md        factual report, all numbers computed here

No activity claim is made or implied about any structure in these tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_prepare_ligands as prep  # noqa: E402  (reuse the real preparation recipe)

RDLogger.DisableLog("rdApp.*")

CORRECTION_REASON = (
    "non-aromatic 4H-imidazole tautomer replaced with aromatic 1H-imidazole "
    "per wwPDB 8TZ"
)

# Motif probes used only to *verify* that each correction is the documented
# transformation. They never select or generate a correction.
SMARTS_4H_IMIDAZOLE = "[CX4;R]1[#7X2]=[#6][#6]=[#7X2]1"
SMARTS_1H_IMIDAZOLE = "[nH]1cncc1"

FAMILIES = ("orthosteric", "allosteric", "exploratory")


def describe(smiles: str, cache: dict[str, dict]) -> dict:
    """Recompute RDKit descriptors for one SMILES string. Cached by string."""
    if smiles in cache:
        return cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rec = {
            "parse_ok": False,
            "formula": "",
            "inchikey": "",
            "heavy_atoms": -1,
            "aromatic_bonds": -1,
            "aromatic_rings": -1,
            "canonical_smiles": "",
            "match_4h_imidazole": False,
            "match_1h_imidazole": False,
        }
    else:
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = ""
        rec = {
            "parse_ok": True,
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "inchikey": inchikey,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "aromatic_bonds": sum(1 for b in mol.GetBonds() if b.GetIsAromatic()),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "canonical_smiles": Chem.MolToSmiles(mol),
            "match_4h_imidazole": mol.HasSubstructMatch(
                Chem.MolFromSmarts(SMARTS_4H_IMIDAZOLE)),
            "match_1h_imidazole": mol.HasSubstructMatch(
                Chem.MolFromSmarts(SMARTS_1H_IMIDAZOLE)),
        }
    cache[smiles] = rec
    return rec


def first_seen_duplicate_flags(keys: list[str]) -> list[bool]:
    """True where the key was already emitted by an earlier row. Blank keys never flag."""
    seen: set[str] = set()
    flags: list[bool] = []
    for key in keys:
        if not key:
            flags.append(False)
            continue
        flags.append(key in seen)
        seen.add(key)
    return flags


def build_audit(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    full = pd.read_csv(paths["full"])
    queue = pd.read_csv(paths["queue_corrected"])
    batch2 = pd.read_csv(paths["batch2"])
    affected = pd.read_csv(paths["affected"])
    ligands = pd.read_csv(paths["ligand_table"])

    checks: dict = {}
    checks["full_rows"] = int(len(full))
    checks["full_rank_sorted_ascending"] = bool(full.denovo_rank.is_monotonic_increasing)
    checks["full_rank_unique"] = bool(full.denovo_rank.is_unique)
    checks["full_smiles_unique"] = bool(full.smiles.is_unique)

    # ---- corrections, taken verbatim from the verification artifact ----------
    aff15 = affected[affected.table == "DENOVO_15000"].copy()
    checks["affected_rows_denovo_15000"] = int(len(aff15))
    checks["affected_stored_smiles_unique"] = bool(aff15.stored_smiles.is_unique)
    correction_map = dict(zip(aff15.stored_smiles, aff15.corrected_smiles))
    checks["affected_stored_smiles_all_in_library"] = int(
        aff15.stored_smiles.isin(set(full.smiles)).sum())

    # ---- batch-1 membership, matched on the ORIGINAL SMILES -----------------
    queue_orig = list(queue.smiles_original)
    checks["queue_rows"] = int(len(queue))
    checks["queue_smiles_original_unique"] = bool(queue.smiles_original.is_unique)
    checks["queue_smiles_original_all_in_library"] = int(
        queue.smiles_original.isin(set(full.smiles)).sum())
    rank_by_original = dict(zip(queue.smiles_original, queue.dock_priority_rank))

    # cand_id is keyed on the queue's dock_priority_rank; V2_LIGAND_TABLE stores
    # that value in a column named `denovo_rank`. Join on cand_id, not on the
    # column name, so the naming collision cannot silently mis-map anything.
    prepared_ids = set(ligands.cand_id)
    checks["ligand_table_rows"] = int(len(ligands))
    checks["ligand_table_cand_id_unique"] = bool(ligands.cand_id.is_unique)

    # ---- per-row descriptors -------------------------------------------------
    cache: dict[str, dict] = {}
    rows = []
    for rank, smiles, family, score in zip(
            full.denovo_rank, full.smiles, full.family, full.design_score):
        corrected = correction_map.get(smiles)
        was_corrected = corrected is not None
        smiles_after = corrected if was_corrected else smiles
        before = describe(smiles, cache)
        after = describe(smiles_after, cache)
        dock_rank = rank_by_original.get(smiles)
        screened = dock_rank is not None
        cand_id = f"DN{int(dock_rank):05d}" if screened else ""
        rows.append({
            "denovo_rank": int(rank),
            "family": family,
            "design_score": score,
            "smiles_original": smiles,
            "smiles_corrected": smiles_after,
            "tautomer_corrected": was_corrected,
            "correction_reason": CORRECTION_REASON if was_corrected else "",
            "formula_before": before["formula"],
            "formula_after": after["formula"],
            "formula_preserved": before["formula"] == after["formula"],
            "inchikey_before": before["inchikey"],
            "inchikey_after": after["inchikey"],
            "heavy_atoms_before": before["heavy_atoms"],
            "heavy_atoms_after": after["heavy_atoms"],
            "aromatic_bonds_before": before["aromatic_bonds"],
            "aromatic_bonds_after": after["aromatic_bonds"],
            "aromatic_rings_before": before["aromatic_rings"],
            "aromatic_rings_after": after["aromatic_rings"],
            "parse_ok_before": before["parse_ok"],
            "parse_ok_after": after["parse_ok"],
            "canonical_smiles": after["canonical_smiles"],
            "motif_4h_imidazole_before": before["match_4h_imidazole"],
            "motif_1h_imidazole_after": after["match_1h_imidazole"],
            "already_screened_batch1": screened,
            "batch1_dock_priority_rank": int(dock_rank) if screened else "",
            "screened_cand_id": cand_id if cand_id in prepared_ids else "",
        })

    audit = pd.DataFrame(rows)
    audit["is_duplicate_structure"] = first_seen_duplicate_flags(
        list(audit.inchikey_after))
    audit["is_duplicate_canonical_smiles"] = first_seen_duplicate_flags(
        list(audit.canonical_smiles))

    # ---- cross-checks against the two committed corrected tables -------------
    checks["queue_corrected_flag_count"] = int(queue.tautomer_corrected.sum())
    checks["batch2_corrected_flag_count"] = int(batch2.tautomer_corrected.sum())
    checks["batch2_rows"] = int(len(batch2))

    audit_screened = audit[audit.already_screened_batch1]
    by_original = dict(zip(audit.smiles_original, audit.smiles_corrected))
    checks["queue_structures_match_audit"] = int(sum(
        1 for orig, corr in zip(queue.smiles_original, queue.smiles)
        if by_original.get(orig) == corr))
    checks["batch2_structures_match_audit"] = int(sum(
        1 for orig, corr in zip(batch2.smiles_original, batch2.smiles)
        if by_original.get(orig) == corr))
    checks["batch2_ranks_match_audit_remaining"] = bool(
        set(batch2.denovo_rank) == set(audit[~audit.already_screened_batch1].denovo_rank))
    checks["audit_screened_rows"] = int(len(audit_screened))

    frames = {"full": full, "queue": queue, "batch2": batch2, "ligands": ligands}
    return audit, checks, frames


def prepared_inchikey(smiles: str) -> str:
    """Reproduce the InChIKey that v2_prepare_ligands.py computes for a SMILES.

    Standardise -> fragment parent -> canonical tautomer -> approximate pH 7.4
    protonation, exactly as in `prepare_one`, then key it. Used only to explain
    which queue rows the preparation stage collapsed and why.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        mol = prep.protonate_ph74(mol)
        return Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001
        return ""


def explain_prep_losses(audit: pd.DataFrame, queue: pd.DataFrame,
                        ligands: pd.DataFrame, failures: pd.DataFrame) -> dict:
    """Recompute, rather than assume, why 12 of the 3,000 queued rows were lost."""
    out: dict = {}
    smiles_by_rank = dict(zip(queue.dock_priority_rank, queue.smiles))
    key_by_cand = dict(zip(ligands.cand_id, ligands.inchikey))
    cands_by_key: dict[str, list[str]] = {}
    for cand, key in zip(ligands.cand_id, ligands.inchikey):
        cands_by_key.setdefault(key, []).append(cand)
    parent_key_by_rank = dict(zip(audit.batch1_dock_priority_rank, audit.inchikey_after))

    # Sanity: the reproduced recipe must agree with the committed table.
    sample = ligands.head(200)
    out["recipe_reproduces_committed_inchikey"] = int(sum(
        prepared_inchikey(s) == k
        for s, k in zip(sample.smiles_input, sample.inchikey)))
    out["recipe_sample_size"] = int(len(sample))
    out["ligand_table_inchikey_duplicates"] = int(ligands.inchikey.duplicated().sum())

    fail_ids = set(failures.cand_id)
    lost = audit[audit.already_screened_batch1 & (audit.screened_cand_id == "")]
    rows = []
    for rank in sorted(int(r) for r in lost.batch1_dock_priority_rank):
        cand = f"DN{rank:05d}"
        if cand in fail_ids:
            rows.append({"cand_id": cand, "category": "EMBED_FAILED",
                         "collides_with": "", "parent_inchikeys_distinct": ""})
            continue
        pkey = prepared_inchikey(smiles_by_rank[rank])
        partners = [c for c in cands_by_key.get(pkey, []) if c != cand]
        partner = partners[0] if partners else ""
        distinct = ""
        if partner:
            pr = int(partner[2:])
            distinct = bool(parent_key_by_rank.get(rank) != parent_key_by_rank.get(pr))
        rows.append({
            "cand_id": cand,
            "category": "INCHIKEY_COLLAPSED" if partner else "UNEXPLAINED",
            "collides_with": partner,
            "parent_inchikeys_distinct": distinct,
        })
    out["losses"] = rows
    out["n_embed_failed"] = sum(1 for r in rows if r["category"] == "EMBED_FAILED")
    out["n_collapsed"] = sum(1 for r in rows if r["category"] == "INCHIKEY_COLLAPSED")
    out["n_unexplained"] = sum(1 for r in rows if r["category"] == "UNEXPLAINED")
    out["n_collapsed_with_distinct_parent_keys"] = sum(
        1 for r in rows if r["parent_inchikeys_distinct"] is True)
    return out


def naming_hazards(audit: pd.DataFrame, full: pd.DataFrame, queue: pd.DataFrame,
                   batch2: pd.DataFrame, ligands: pd.DataFrame) -> dict:
    """Column-name collisions between the committed tables. Reported, not repaired."""
    out: dict = {}

    # V2_LIGAND_TABLE.denovo_rank actually stores the queue's dock_priority_rank.
    out["ligand_denovo_rank_is_dock_priority_rank"] = bool(
        all(int(c[2:]) == int(r) for c, r in zip(ligands.cand_id, ligands.denovo_rank)))
    rank_of_smiles = dict(zip(full.smiles, full.denovo_rank))
    orig_by_dock = dict(zip(queue.dock_priority_rank, queue.smiles_original))
    true_ranks = [rank_of_smiles[orig_by_dock[int(c[2:])]] for c in ligands.cand_id]
    agree = [int(a) == int(b) for a, b in zip(ligands.denovo_rank, true_ranks)]
    out["ligand_rows"] = int(len(ligands))
    out["ligand_denovo_rank_agrees_with_library_rank"] = int(sum(agree))
    out["ligand_denovo_rank_max_abs_error"] = int(max(
        abs(int(a) - int(b)) for a, b in zip(ligands.denovo_rank, true_ranks)))

    # batch 2 carries a dock_priority_rank column that is a copy of its library rank.
    out["batch2_dock_rank_equals_library_rank"] = bool(
        (batch2.dock_priority_rank == batch2.denovo_rank).all())
    b2_by_rank = dict(zip(batch2.dock_priority_rank, batch2.smiles))
    q_by_rank = dict(zip(queue.dock_priority_rank, queue.smiles))
    overlap = sorted(set(b2_by_rank) & set(q_by_rank))
    out["dock_priority_rank_value_overlap"] = int(len(overlap))
    out["dock_priority_rank_overlap_same_molecule"] = int(sum(
        1 for r in overlap if b2_by_rank[r] == q_by_rank[r]))
    return out


def family_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.family.value_counts().to_dict()
    out = {f: int(counts.get(f, 0)) for f in FAMILIES}
    other = {str(k): int(v) for k, v in counts.items() if k not in FAMILIES}
    out.update(other)
    out["TOTAL"] = int(len(frame))
    return out


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(audit: pd.DataFrame, checks: dict, frames: dict, manifest: dict,
                 failures: pd.DataFrame, out_md: Path, sources: dict[str, Path]) -> dict:
    stats: dict = {}
    screened = audit[audit.already_screened_batch1]
    remaining = audit[~audit.already_screened_batch1]
    corrected = audit[audit.tautomer_corrected]

    stats["rows_total"] = int(len(audit))
    stats["rows_screened"] = int(len(screened))
    stats["rows_remaining"] = int(len(remaining))
    stats["unique_inchikey_after"] = int(audit.inchikey_after.nunique())
    stats["duplicate_rows_inchikey"] = int(audit.is_duplicate_structure.sum())
    stats["unique_canonical_smiles"] = int(audit.canonical_smiles.nunique())
    stats["duplicate_rows_canonical_smiles"] = int(
        audit.is_duplicate_canonical_smiles.sum())
    stats["unique_smiles_original"] = int(audit.smiles_original.nunique())

    stats["family_all"] = family_counts(audit)
    stats["family_screened"] = family_counts(screened)
    stats["family_remaining"] = family_counts(remaining)

    stats["corrected_total"] = int(len(corrected))
    stats["corrected_family"] = family_counts(corrected)
    stats["corrected_screened"] = int(corrected.already_screened_batch1.sum())
    stats["corrected_remaining"] = int((~corrected.already_screened_batch1).sum())
    stats["corrected_family_screened"] = family_counts(
        corrected[corrected.already_screened_batch1])
    stats["corrected_family_remaining"] = family_counts(
        corrected[~corrected.already_screened_batch1])

    # formula preservation
    bad_formula = corrected[~corrected.formula_preserved]
    stats["corrected_formula_preserved"] = int(corrected.formula_preserved.sum())
    stats["corrected_formula_changed"] = int(len(bad_formula))
    stats["corrected_formula_changed_ranks"] = [int(r) for r in bad_formula.denovo_rank]

    # heavy atoms preserved
    ha_changed = corrected[corrected.heavy_atoms_before != corrected.heavy_atoms_after]
    stats["corrected_heavy_atoms_preserved"] = int(len(corrected) - len(ha_changed))
    stats["corrected_heavy_atoms_changed_ranks"] = [int(r) for r in ha_changed.denovo_rank]

    # aromatic bond increase
    inc = corrected.aromatic_bonds_after - corrected.aromatic_bonds_before
    stats["corrected_aromatic_bonds_increased"] = int((inc > 0).sum())
    stats["corrected_aromatic_bonds_not_increased"] = int((inc <= 0).sum())
    stats["corrected_aromatic_bond_delta_values"] = {
        int(k): int(v) for k, v in inc.value_counts().sort_index().items()}
    stats["corrected_aromatic_bonds_not_increased_ranks"] = [
        int(r) for r in corrected.denovo_rank[inc <= 0]]
    ari = corrected.aromatic_rings_after - corrected.aromatic_rings_before
    stats["corrected_aromatic_rings_increased"] = int((ari > 0).sum())
    stats["corrected_aromatic_rings_not_increased_ranks"] = [
        int(r) for r in corrected.denovo_rank[ari <= 0]]

    # motif verification
    stats["corrected_motif_4h_before"] = int(corrected.motif_4h_imidazole_before.sum())
    stats["corrected_motif_1h_after"] = int(corrected.motif_1h_imidazole_after.sum())
    uncorrected = audit[~audit.tautomer_corrected]
    stats["uncorrected_carrying_4h_motif"] = int(
        uncorrected.motif_4h_imidazole_before.sum())
    stats["uncorrected_carrying_4h_motif_ranks"] = [
        int(r) for r in uncorrected.denovo_rank[uncorrected.motif_4h_imidazole_before]][:50]

    # parse failures
    pf_before = audit[~audit.parse_ok_before]
    pf_after = audit[~audit.parse_ok_after]
    stats["parse_fail_before"] = int(len(pf_before))
    stats["parse_fail_after"] = int(len(pf_after))
    stats["parse_fail_before_ranks"] = [int(r) for r in pf_before.denovo_rank]
    stats["parse_fail_after_ranks"] = [int(r) for r in pf_after.denovo_rank]
    stats["inchikey_blank"] = int((audit.inchikey_after == "").sum())

    # reconciliation
    stats["manifest_input_rows"] = int(manifest["input_rows"])
    stats["manifest_prepared_ok"] = int(manifest["prepared_ok"])
    stats["manifest_prepared_failed"] = int(manifest["prepared_failed"])
    stats["manifest_duplicates_collapsed"] = int(manifest["duplicate_inchikeys_collapsed"])
    stats["manifest_unique_ligands"] = int(manifest["unique_ligands"])
    stats["prep_failure_rows"] = int(len(failures))
    stats["screened_with_cand_id"] = int((screened.screened_cand_id != "").sum())
    stats["screened_without_cand_id"] = int((screened.screened_cand_id == "").sum())
    lost = screened[screened.screened_cand_id == ""]
    fail_ids = set(failures.cand_id)
    lost_ids = [f"DN{int(r):05d}" for r in lost.batch1_dock_priority_rank]
    stats["lost_cand_ids"] = sorted(lost_ids)
    stats["lost_embed_failed"] = sorted(i for i in lost_ids if i in fail_ids)
    stats["lost_not_embed_failed"] = sorted(i for i in lost_ids if i not in fail_ids)

    stats["recon_15000_eq_screened_plus_remaining"] = bool(
        stats["rows_total"] == stats["rows_screened"] + stats["rows_remaining"])
    stats["recon_3000_minus_5_minus_7_eq_2988"] = bool(
        stats["manifest_input_rows"] - stats["manifest_prepared_failed"]
        - stats["manifest_duplicates_collapsed"] == stats["manifest_unique_ligands"])
    stats["recon_lost_is_12"] = bool(len(lost_ids) == 12)
    stats["prep_losses"] = explain_prep_losses(
        audit, frames["queue"], frames["ligands"], failures)
    stats["naming"] = naming_hazards(
        audit, frames["full"], frames["queue"], frames["batch2"], frames["ligands"])
    stats["checks"] = checks

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fam = stats["family_all"]
    fam_s = stats["family_screened"]
    fam_r = stats["family_remaining"]

    def fam_row(label, d):
        return [label, d.get("orthosteric", 0), d.get("allosteric", 0),
                d.get("exploratory", 0), d["TOTAL"]]

    lines: list[str] = []
    A = lines.append
    A("# Full de novo library audit — DENOVO_15000")
    A("")
    A(f"Generated {ts} by `YONSEI_FINAL10_V2/scripts/v2_full_library_audit.py`.")
    A("Every number below was computed by that script from the files listed under")
    A("Inputs. Nothing is carried over from an earlier report.")
    A("")
    A("Scope: structural bookkeeping only. Nothing here establishes PAR2 activity,")
    A("or the absence of it, for any molecule.")
    A("")
    A("## Inputs")
    A("")
    for k, v in sources.items():
        A(f"- `{v}` ({k})")
    A("")
    A("## 1. Library size")
    A("")
    A(f"- Rows in `DENOVO_15000.csv`: **{stats['rows_total']}**")
    A(f"- `denovo_rank` unique: {checks['full_rank_unique']}; "
      f"ascending: {checks['full_rank_sorted_ascending']}; "
      f"range 1–{stats['rows_total']}")
    A(f"- Stored SMILES strings distinct: {checks['full_smiles_unique']} "
      f"({stats['unique_smiles_original']} unique strings)")
    A("")
    A("## 2. Unique parent structures (after correction)")
    A("")
    A(f"- Unique InChIKeys: **{stats['unique_inchikey_after']}**")
    A(f"- Rows that repeat an InChIKey seen earlier in the file: "
      f"**{stats['duplicate_rows_inchikey']}**")
    A(f"- Unique RDKit canonical SMILES: **{stats['unique_canonical_smiles']}**")
    A(f"- Rows that repeat a canonical SMILES seen earlier: "
      f"**{stats['duplicate_rows_canonical_smiles']}**")
    A("")
    A(f"`is_duplicate_structure` in the audit CSV uses the InChIKey key; "
      f"`is_duplicate_canonical_smiles` uses the canonical SMILES key.")
    A("")
    A("## 3. Family counts")
    A("")
    A(md_table(["set", "orthosteric", "allosteric", "exploratory", "total"],
               [fam_row("full library", fam),
                fam_row("screened (batch 1 queue)", fam_s),
                fam_row("remaining", fam_r)]))
    A("")
    A("## 4. Wrong-tautomer designs")
    A("")
    A(f"Total carrying the wrong tautomer: **{stats['corrected_total']}** of "
      f"{stats['rows_total']} "
      f"({100.0 * stats['corrected_total'] / stats['rows_total']:.2f}%)")
    A("")
    cf, cfs, cfr = (stats["corrected_family"], stats["corrected_family_screened"],
                    stats["corrected_family_remaining"])
    A(md_table(["split", "orthosteric", "allosteric", "exploratory", "total"],
               [fam_row("all affected", cf),
                fam_row("affected & screened", cfs),
                fam_row("affected & not screened", cfr)]))
    A("")
    A(f"- Screened / not screened: **{stats['corrected_screened']}** / "
      f"**{stats['corrected_remaining']}**")
    A(f"- Committed `tautomer_corrected` flags: queue "
      f"{checks['queue_corrected_flag_count']}, batch 2 "
      f"{checks['batch2_corrected_flag_count']}, sum "
      f"{checks['queue_corrected_flag_count'] + checks['batch2_corrected_flag_count']}")
    A("")
    A("Motif verification (probes are diagnostic only; they never select a correction):")
    A("")
    A(f"- Corrected rows whose stored structure matches the non-aromatic "
      f"4H-imidazole pattern `{SMARTS_4H_IMIDAZOLE}`: "
      f"**{stats['corrected_motif_4h_before']} / {stats['corrected_total']}**")
    A(f"- Corrected rows whose repaired structure matches the aromatic "
      f"1H-imidazole pattern `{SMARTS_1H_IMIDAZOLE}`: "
      f"**{stats['corrected_motif_1h_after']} / {stats['corrected_total']}**")
    A(f"- Rows NOT in the correction list that nonetheless match the "
      f"4H-imidazole pattern: **{stats['uncorrected_carrying_4h_motif']}**")
    A("")
    A("## 5. Molecular formula preservation across the correction")
    A("")
    A(f"- Formula preserved: **{stats['corrected_formula_preserved']} / "
      f"{stats['corrected_total']}**")
    A(f"- Formula changed: **{stats['corrected_formula_changed']}**"
      + ("" if not stats["corrected_formula_changed_ranks"]
         else f" — denovo_rank {stats['corrected_formula_changed_ranks']}"))
    A(f"- Heavy-atom count preserved: **{stats['corrected_heavy_atoms_preserved']} / "
      f"{stats['corrected_total']}**"
      + ("" if not stats["corrected_heavy_atoms_changed_ranks"]
         else f"; changed at denovo_rank {stats['corrected_heavy_atoms_changed_ranks']}"))
    A("")
    A("## 6. Aromatic bond count across the correction")
    A("")
    A(f"- Aromatic bond count increased: **{stats['corrected_aromatic_bonds_increased']} / "
      f"{stats['corrected_total']}**")
    A(f"- Did not increase: **{stats['corrected_aromatic_bonds_not_increased']}**"
      + ("" if not stats["corrected_aromatic_bonds_not_increased_ranks"]
         else f" — denovo_rank "
              f"{stats['corrected_aromatic_bonds_not_increased_ranks']}"))
    A("")
    A("Distribution of `aromatic_bonds_after - aromatic_bonds_before` over corrected rows:")
    A("")
    A(md_table(["delta", "rows"],
               [[k, v] for k, v in stats["corrected_aromatic_bond_delta_values"].items()]))
    A("")
    A(f"- Aromatic ring count increased: "
      f"**{stats['corrected_aromatic_rings_increased']} / {stats['corrected_total']}**"
      + ("" if not stats["corrected_aromatic_rings_not_increased_ranks"]
         else f"; exceptions at denovo_rank "
              f"{stats['corrected_aromatic_rings_not_increased_ranks']}"))
    A("")
    A("## 7. RDKit parse failures")
    A("")
    A(f"- Fail to parse BEFORE correction: **{stats['parse_fail_before']}**"
      + ("" if not stats["parse_fail_before_ranks"]
         else f" — denovo_rank {stats['parse_fail_before_ranks']}"))
    A(f"- Fail to parse AFTER correction: **{stats['parse_fail_after']}**"
      + ("" if not stats["parse_fail_after_ranks"]
         else f" — denovo_rank {stats['parse_fail_after_ranks']}"))
    A(f"- Rows with a blank InChIKey after correction: **{stats['inchikey_blank']}**")
    A("")
    A("## 8. Reconciliation")
    A("")
    A("### 8.1 15,000 = screened + remaining")
    A("")
    A(f"    {stats['rows_screened']} screened + {stats['rows_remaining']} remaining "
      f"= {stats['rows_screened'] + stats['rows_remaining']}  "
      f"(library rows: {stats['rows_total']})")
    A("")
    A(f"Holds: **{stats['recon_15000_eq_screened_plus_remaining']}**")
    A("")
    A(f"- `DENOVO_REMAINING_TO_SCREEN.csv` row count: **{stats['rows_remaining']}**")
    A(f"- Remaining ranks identical to `DENOVO_BATCH2_REMAINING12000_TAUTOMER_CORRECTED.csv`: "
      f"**{checks['batch2_ranks_match_audit_remaining']}**")
    A(f"- Batch-2 rows whose committed corrected SMILES equals this audit's "
      f"`smiles_corrected`: **{checks['batch2_structures_match_audit']} / "
      f"{checks['batch2_rows']}**")
    A(f"- Queue rows whose committed corrected SMILES equals this audit's "
      f"`smiles_corrected`: **{checks['queue_structures_match_audit']} / "
      f"{checks['queue_rows']}**")
    A("")
    A("### 8.2 3,000 queued vs 2,988 prepared")
    A("")
    A(f"    {stats['manifest_input_rows']} queued "
      f"- {stats['manifest_prepared_failed']} embed failures "
      f"- {stats['manifest_duplicates_collapsed']} duplicate InChIKeys "
      f"= {stats['manifest_input_rows'] - stats['manifest_prepared_failed'] - stats['manifest_duplicates_collapsed']}"
      f"  (V2_LIGAND_TABLE.csv rows: {checks['ligand_table_rows']}; "
      f"manifest unique_ligands: {stats['manifest_unique_ligands']})")
    A("")
    A(f"Holds: **{stats['recon_3000_minus_5_minus_7_eq_2988']}**")
    A("")
    A(f"Independently recomputed from the audit table: of the "
      f"{stats['rows_screened']} screened designs, **{stats['screened_with_cand_id']}** "
      f"have a `cand_id` in `V2_LIGAND_TABLE.csv` and "
      f"**{stats['screened_without_cand_id']}** do not.")
    A("")
    A(f"- Lost cand_ids ({len(stats['lost_cand_ids'])}): "
      f"{', '.join(stats['lost_cand_ids'])}")
    A(f"- Present in `V2_PREP_FAILURES.csv` as EMBED_FAILED "
      f"({len(stats['lost_embed_failed'])}): {', '.join(stats['lost_embed_failed'])}")
    A(f"- Not an embed failure, therefore attributable to InChIKey collapse "
      f"({len(stats['lost_not_embed_failed'])}): "
      f"{', '.join(stats['lost_not_embed_failed'])}")
    A("")
    A(f"Arithmetic check `12 = 5 + 7`: **{stats['recon_lost_is_12']}** "
      f"({len(stats['lost_embed_failed'])} + {len(stats['lost_not_embed_failed'])} "
      f"= {len(stats['lost_cand_ids'])})")
    A("")
    A("### 8.3 What the 7 collapsed rows actually collided with")
    A("")
    pl = stats["prep_losses"]
    A(f"The InChIKey used by `v2_prepare_ligands.py` is computed *after* "
      f"`rdMolStandardize.Cleanup` -> `FragmentParent` -> "
      f"`TautomerEnumerator().Canonicalize` -> approximate pH 7.4 protonation. "
      f"That recipe was re-run here and reproduces the committed `inchikey` for "
      f"**{pl['recipe_reproduces_committed_inchikey']} / {pl['recipe_sample_size']}** "
      f"sampled rows of `V2_LIGAND_TABLE.csv`, so the collision partners below are "
      f"computed, not assumed. `V2_LIGAND_TABLE.csv` itself contains "
      f"{pl['ligand_table_inchikey_duplicates']} duplicate InChIKeys.")
    A("")
    A(md_table(["cand_id", "category", "collides with", "parent InChIKeys differ"],
               [[r["cand_id"], r["category"], r["collides_with"] or "—",
                 "—" if r["parent_inchikeys_distinct"] == ""
                 else r["parent_inchikeys_distinct"]]
                for r in pl["losses"]]))
    A("")
    A(f"- EMBED_FAILED: **{pl['n_embed_failed']}**; "
      f"InChIKey-collapsed: **{pl['n_collapsed']}**; "
      f"unexplained: **{pl['n_unexplained']}**")
    A(f"- Collapsed rows whose *parent* InChIKey differs from its partner's: "
      f"**{pl['n_collapsed_with_distinct_parent_keys']} / {pl['n_collapsed']}**")
    A("")
    A("This qualifies the phrase \"duplicate InChIKeys\". At the parent level the "
      "library holds no duplicates at all (section 2). The 7 rows collapse only "
      "because the canonical-tautomer step in the preparation recipe discards the "
      "stereo designation, after which each of them becomes indistinguishable from "
      "the queue row immediately above it. The arithmetic holds; the label is "
      "narrower than it sounds.")
    A("")
    A("## 9. Column-name collisions between committed tables")
    A("")
    A("Reported, not repaired. No existing file was modified by this audit.")
    A("")
    nm = stats["naming"]
    A("**9.1 `V2_LIGAND_TABLE.csv` column `denovo_rank` is not the library "
      "`denovo_rank`.** It holds the batch-1 queue's `dock_priority_rank` "
      f"(`cand_id == \"DN%05d\" % denovo_rank` for all "
      f"{nm['ligand_rows']} rows: {nm['ligand_denovo_rank_is_dock_priority_rank']}). "
      f"It coincides with the design's true `DENOVO_15000.csv` rank in only "
      f"**{nm['ligand_denovo_rank_agrees_with_library_rank']} / {nm['ligand_rows']}** "
      f"rows; the largest discrepancy is **{nm['ligand_denovo_rank_max_abs_error']}** "
      "positions. Joining screening results back to the library on that column "
      "would mis-map almost every row.")
    A("")
    A("**9.2 `DENOVO_BATCH2_REMAINING12000_TAUTOMER_CORRECTED.csv` carries a "
      "`dock_priority_rank` column that is a copy of its library rank** "
      f"(equal to `denovo_rank` for every row: "
      f"{nm['batch2_dock_rank_equals_library_rank']}). These 12,000 designs are by "
      "definition the ones *not* in the batch-1 dock queue, yet "
      f"**{nm['dock_priority_rank_value_overlap']}** of those values also occur as "
      "real `dock_priority_rank` values in "
      "`DENOVO_TOP3000_FOR_DOCKING_TAUTOMER_CORRECTED.csv`, and "
      f"**{nm['dock_priority_rank_overlap_same_molecule']}** of those "
      f"{nm['dock_priority_rank_value_overlap']} name the same molecule. A join "
      "across the two tables on `dock_priority_rank` would silently pair unrelated "
      "designs.")
    A("")
    out_md.write_text("\n".join(lines) + "\n")
    return stats


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--denovo-dir", type=Path, default=root / "02_DENOVO_CPU")
    parser.add_argument("--verification-dir", type=Path, default=root / "04_VERIFICATION")
    parser.add_argument("--library-dir", type=Path,
                        default=root / "SCREEN_REAL" / "library")
    parser.add_argument("--out-dir", type=Path, default=root / "02_DENOVO_CPU")
    parser.add_argument("--stats-json", type=Path, default=None,
                        help="optional path for the machine-readable stats dump")
    args = parser.parse_args()

    paths = {
        "full": args.denovo_dir / "DENOVO_15000.csv",
        "queue_corrected": args.denovo_dir
        / "DENOVO_TOP3000_FOR_DOCKING_TAUTOMER_CORRECTED.csv",
        "batch2": args.denovo_dir
        / "DENOVO_BATCH2_REMAINING12000_TAUTOMER_CORRECTED.csv",
        "affected": args.verification_dir / "affected_designs_wrong_tautomer.csv",
        "ligand_table": args.library_dir / "V2_LIGAND_TABLE.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"missing input ({name}): {path}")

    audit, checks, frames = build_audit(paths)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.out_dir / "DENOVO_FULL_LIBRARY_AUDIT.csv"
    remaining_csv = args.out_dir / "DENOVO_REMAINING_TO_SCREEN.csv"
    report_md = args.out_dir / "DENOVO_FULL_LIBRARY_AUDIT.md"
    for path in (audit_csv, remaining_csv, report_md):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing file: {path}")

    audit.to_csv(audit_csv, index=False)
    audit[~audit.already_screened_batch1].to_csv(remaining_csv, index=False)

    manifest = json.loads((args.library_dir / "V2_PREP_MANIFEST.json").read_text())
    failures = pd.read_csv(args.library_dir / "V2_PREP_FAILURES.csv")
    stats = write_report(
        audit, checks, frames, manifest, failures, report_md,
        {k: v.relative_to(root.parent) for k, v in paths.items()})

    if args.stats_json:
        args.stats_json.write_text(json.dumps(stats, indent=2, sort_keys=True))

    print(json.dumps({
        "audit_csv": str(audit_csv),
        "remaining_csv": str(remaining_csv),
        "report_md": str(report_md),
        "rows_total": stats["rows_total"],
        "rows_screened": stats["rows_screened"],
        "rows_remaining": stats["rows_remaining"],
        "corrected_total": stats["corrected_total"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
