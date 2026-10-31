#!/usr/bin/env python3
"""Exact end-to-end accounting for the de novo EX8 library.

Recomputes every count from the primary artefacts rather than carrying numbers
forward from an earlier report: the 15,000-design library, the two docking
queues, both preparation manifests and failure tables, the three evidence
tables, the recovered worker shards, and the merged universe.

Where two artefacts disagree the script emits BOTH values and marks the row
DISCREPANCY. Where a quantity cannot be determined from surviving records the
value is UNKNOWN with the reason stated; it is never replaced by zero.

Batch-2 docking was dispatched as 25-ligand partitions (`shardNNNN_pM`) of the
100-ligand screen-set shards, so partition membership is reconstructed from the
shard SDFs to say precisely which dispatch units produced nothing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

UNKNOWN = "UNKNOWN"
PARTITION_SIZE = 25  # ligands per dispatched `_pN` partition


class Ledger:
    """Accumulates one row per accounted quantity."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section: str, metric: str, value, source: str, method: str,
            status: str = "COMPUTED", alternative_value=None,
            alternative_source: str = "") -> None:
        self.rows.append(dict(
            section=section, metric=metric, value=value, status=status,
            alternative_value=alternative_value if alternative_value is not None else "",
            source=source, alternative_source=alternative_source, method=method))

    def agree(self, section: str, metric: str, computed, reported, source: str,
              reported_source: str, method: str) -> None:
        """Record a computed value and flag it if a shipped artefact disagrees."""
        if computed == reported:
            self.add(section, metric, computed, source, method, status="COMPUTED",
                     alternative_value=reported, alternative_source=reported_source)
        else:
            self.add(section, metric, computed, source, method, status="DISCREPANCY",
                     alternative_value=reported, alternative_source=reported_source)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def canonical_ids(smiles_list) -> dict[str, int]:
    """Unique-structure counts under four different notions of identity."""
    stereo_smiles, flat_smiles, stereo_keys, flat_keys = set(), set(), set(), set()
    unparsed = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            unparsed += 1
            continue
        stereo_smiles.add(Chem.MolToSmiles(mol, isomericSmiles=True))
        flat_smiles.add(Chem.MolToSmiles(mol, isomericSmiles=False))
        try:
            stereo_keys.add(Chem.MolToInchiKey(mol))
            flat = Chem.Mol(mol)
            Chem.RemoveStereochemistry(flat)
            flat_keys.add(Chem.MolToInchiKey(flat))
        except Exception:  # noqa: BLE001
            pass
    return dict(unparsed=unparsed,
                unique_stereo_smiles=len(stereo_smiles),
                unique_flat_smiles=len(flat_smiles),
                unique_stereo_inchikey=len(stereo_keys),
                unique_flat_inchikey=len(flat_keys))


def expected_jobs(ligand_table: pd.DataFrame) -> set[tuple[str, str]]:
    """Every (cand_id, pocket) the routing asked GNINA to dock."""
    jobs: set[tuple[str, str]] = set()
    for row in ligand_table.itertuples():
        for pocket in str(row.pockets).split(";"):
            if pocket:
                jobs.add((row.cand_id, pocket))
    return jobs


def shard_partitions(shard_dir: Path) -> dict[str, list[str]]:
    """Reconstruct `shardNNNN_pM` -> cand_ids from the dispatched shard SDFs."""
    partitions: dict[str, list[str]] = {}
    for shard in sorted(p for p in shard_dir.glob("*.sdf") if not p.name.startswith(".")):
        names = [rec.lstrip("\n").split("\n", 1)[0].strip()
                 for rec in shard.read_text().split("$$$$\n") if rec.strip()]
        base = shard.stem
        for index in range(0, len(names), PARTITION_SIZE):
            partitions[f"{base}_p{index // PARTITION_SIZE}"] = names[index:index + PARTITION_SIZE]
    return partitions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--recovery", type=Path, required=True,
                        help="PAR2_WORKER_RECOVERY_* directory (sibling of the repo)")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--stereo-audit", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--verify-shard-hashes", action="store_true",
                        help="re-hash the recovered pose shards against their sentinels")
    args = parser.parse_args()

    v2 = args.root.resolve() / "YONSEI_FINAL10_V2"
    denovo = v2 / "02_DENOVO_CPU"
    screen = v2 / "SCREEN_REAL"
    led = Ledger()

    # ---------------------------------------------------------------- generation
    scored = pd.read_csv(denovo / "SCORED_CANDIDATES.csv")
    led.add("1_generation", "scored_candidate_pool_rows", int(len(scored)),
            "02_DENOVO_CPU/SCORED_CANDIDATES.csv", "row count")
    led.add("1_generation", "scored_candidate_pool_unique_smiles", int(scored.smiles.nunique()),
            "02_DENOVO_CPU/SCORED_CANDIDATES.csv", "nunique of smiles column",
            status="DISCREPANCY" if scored.smiles.nunique() != len(scored) else "COMPUTED",
            alternative_value=int(len(scored)), alternative_source="row count of same file")

    manifest = json.loads((denovo / "GENERATION_MANIFEST.json").read_text())
    raw = manifest["raw_generated"]
    refills = sum(r["new_raw_unique"] for r in manifest["filter_refill_history"])
    led.add("1_generation", "raw_structures_enumerated_reported",
            int(sum(raw.values()) + refills),
            "02_DENOVO_CPU/GENERATION_MANIFEST.json",
            "raw_generated totals plus filter_refill_history new_raw_unique; the raw "
            "molecules were not retained, so this cannot be recomputed",
            status="REPORTED_NOT_RECOMPUTABLE")

    library = pd.read_csv(denovo / "DENOVO_15000.csv")
    led.agree("1_generation", "original_generated_structures_selected", int(len(library)),
              int(manifest["requested_total"]), "02_DENOVO_CPU/DENOVO_15000.csv",
              "GENERATION_MANIFEST.json requested_total", "row count")
    led.add("1_generation", "library_unique_denovo_rank", int(library.denovo_rank.nunique()),
            "02_DENOVO_CPU/DENOVO_15000.csv", "nunique of denovo_rank")
    led.add("1_generation", "library_unique_raw_smiles_strings", int(library.smiles.nunique()),
            "02_DENOVO_CPU/DENOVO_15000.csv", "nunique of smiles column")

    ids = canonical_ids(library.smiles)
    for key, label, method in [
        ("unique_stereo_smiles", "unique_parent_molecules_by_STEREO_ID",
         "RDKit canonical isomeric SMILES of the as-generated structure"),
        ("unique_flat_smiles", "unique_parent_molecules_by_CONNECTIVITY_ID",
         "RDKit canonical SMILES, isomericSmiles=False"),
        ("unique_stereo_inchikey", "unique_parent_molecules_by_stereo_InChIKey",
         "MolToInchiKey of the as-generated structure"),
        ("unique_flat_inchikey", "unique_parent_molecules_by_flat_InChIKey",
         "MolToInchiKey after RemoveStereochemistry"),
    ]:
        led.add("1_generation", label, ids[key], "02_DENOVO_CPU/DENOVO_15000.csv", method)
    led.add("1_generation", "library_smiles_unparseable", ids["unparsed"],
            "02_DENOVO_CPU/DENOVO_15000.csv", "MolFromSmiles returned None")
    led.add("1_generation", "library_designs_beyond_one_per_flat_skeleton",
            int(len(library) - ids["unique_flat_smiles"]),
            "02_DENOVO_CPU/DENOVO_15000.csv",
            "designs minus distinct flat skeletons: the library holds this many extra "
            "stereoisomers of skeletons it already contains")

    # ------------------------------------------------------------------- queues
    q1 = pd.read_csv(denovo / "DENOVO_TOP3000_FOR_DOCKING_TAUTOMER_CORRECTED.csv")
    q2 = pd.read_csv(denovo / "DENOVO_BATCH2_REMAINING12000_TAUTOMER_CORRECTED.csv")
    led.add("2_queue", "batch1_docking_queue_rows", int(len(q1)),
            "02_DENOVO_CPU/DENOVO_TOP3000_FOR_DOCKING_TAUTOMER_CORRECTED.csv", "row count")
    led.add("2_queue", "batch2_remainder_queue_rows", int(len(q2)),
            "02_DENOVO_CPU/DENOVO_BATCH2_REMAINING12000_TAUTOMER_CORRECTED.csv", "row count")
    led.agree("2_queue", "queue_rows_total", int(len(q1) + len(q2)), int(len(library)),
              "both queue CSVs", "02_DENOVO_CPU/DENOVO_15000.csv", "sum of queue row counts")
    q1_src = set(q1.smiles_original.fillna(q1.smiles))
    q2_src = set(q2.smiles_original.fillna(q2.smiles))
    led.add("2_queue", "queue_overlap_by_original_smiles", int(len(q1_src & q2_src)),
            "both queue CSVs", "set intersection of smiles_original")
    led.add("2_queue", "library_designs_absent_from_both_queues",
            int(len(set(library.smiles) - q1_src - q2_src)),
            "DENOVO_15000.csv vs both queue CSVs", "set difference on original SMILES")
    led.add("2_queue", "batch1_tautomer_corrected_inputs", int(q1.tautomer_corrected.sum()),
            "batch1 queue CSV", "sum of tautomer_corrected")
    led.add("2_queue", "batch2_tautomer_corrected_inputs", int(q2.tautomer_corrected.sum()),
            "batch2 queue CSV", "sum of tautomer_corrected")

    # denovo_rank landmine
    rank_of = dict(zip(library.smiles, library.denovo_rank))
    q1_true = q1.smiles_original.fillna(q1.smiles).map(rank_of)
    led.add("2_queue", "batch1_queue_rank_equals_library_rank",
            int((q1.dock_priority_rank == q1_true).sum()),
            "batch1 queue CSV joined to DENOVO_15000.csv on original SMILES",
            "dock_priority_rank compared with the library denovo_rank; V2_LIGAND_TABLE's "
            "denovo_rank column carries the queue rank, not this one")

    # ------------------------------------------------------------- preparation
    lig1 = pd.read_csv(screen / "library/V2_LIGAND_TABLE.csv")
    lig2 = pd.read_csv(screen / "library_batch2/V2_LIGAND_TABLE.csv")
    fail1 = pd.read_csv(screen / "library/V2_PREP_FAILURES.csv")
    fail2 = pd.read_csv(screen / "library_batch2/V2_PREP_FAILURES.csv")
    man1 = json.loads((screen / "library/V2_PREP_MANIFEST.json").read_text())
    man2 = json.loads((screen / "library_batch2/V2_PREP_MANIFEST.json").read_text())

    for label, lig, fail, man, queue in [("batch1", lig1, fail1, man1, q1),
                                         ("batch2", lig2, fail2, man2, q2)]:
        led.agree("3_preparation", f"{label}_prep_failed", int(len(fail)),
                  int(man["prepared_failed"]), f"{label} V2_PREP_FAILURES.csv",
                  f"{label} V2_PREP_MANIFEST.json", "row count")
        led.agree("3_preparation", f"{label}_prepared_ok", int(len(queue) - len(fail)),
                  int(man["prepared_ok"]), "queue rows minus failure rows",
                  f"{label} V2_PREP_MANIFEST.json", "arithmetic on primary tables")
        led.agree("3_preparation", f"{label}_unique_ligands_after_dedup", int(len(lig)),
                  int(man["unique_ligands"]), f"{label} V2_LIGAND_TABLE.csv",
                  f"{label} V2_PREP_MANIFEST.json", "row count")
        led.agree("3_preparation", f"{label}_collapsed_duplicate_inchikey",
                  int(len(queue) - len(fail) - len(lig)),
                  int(man["duplicate_inchikeys_collapsed"]),
                  "prepared_ok minus surviving ligands",
                  f"{label} V2_PREP_MANIFEST.json", "arithmetic on primary tables")
        for reason, n in fail.prep_reason.value_counts().items():
            led.add("3_preparation", f"{label}_prep_failure_reason::{reason}", int(n),
                    f"{label} V2_PREP_FAILURES.csv", "value_counts of prep_reason")
        led.add("3_preparation", f"{label}_ligand_table_prep_ok_false",
                int((~lig.prep_ok.astype(bool)).sum()), f"{label} V2_LIGAND_TABLE.csv",
                "rows in the shipped table that are not prep_ok")

    total_fail, total_lig = len(fail1) + len(fail2), len(lig1) + len(lig2)
    total_collapsed = len(library) - total_fail - total_lig
    led.add("3_preparation", "prep_failed_total", int(total_fail),
            "both V2_PREP_FAILURES.csv", "sum")
    led.add("3_preparation", "collapsed_duplicate_inchikey_total", int(total_collapsed),
            "15,000 minus failures minus surviving ligands", "arithmetic")
    led.add("3_preparation", "unique_prepared_ligands_total", int(total_lig),
            "both V2_LIGAND_TABLE.csv", "sum of row counts")
    led.add("3_preparation", "cand_id_collisions_between_batches",
            int(len(set(lig1.cand_id) & set(lig2.cand_id))),
            "both V2_LIGAND_TABLE.csv", "set intersection of cand_id")
    dpr1 = set(lig1.cand_id.str[2:].astype(int))
    dpr2 = set(lig2.cand_id.str[2:].astype(int))
    shared = dpr1 & dpr2
    smiles1 = dict(zip(lig1.cand_id.str[2:].astype(int), lig1.smiles_input))
    smiles2 = dict(zip(lig2.cand_id.str[2:].astype(int), lig2.smiles_input))
    led.add("3_preparation", "dock_priority_rank_values_shared_between_batches",
            int(len(shared)), "both V2_LIGAND_TABLE.csv",
            "cand_id numeric suffix is the dock_priority_rank; set intersection")
    led.add("3_preparation", "dock_priority_rank_collisions_naming_different_molecules",
            int(sum(1 for r in shared if smiles1[r] != smiles2[r])),
            "both V2_LIGAND_TABLE.csv",
            "shared dock_priority_rank values whose smiles_input differ")

    # De-duplication ran inside each batch, so a structure can survive twice.
    keys = pd.concat([lig1[["cand_id", "inchikey"]], lig2[["cand_id", "inchikey"]]])
    repeated = keys.groupby("inchikey").filter(lambda s: len(s) > 1)
    led.add("3_preparation", "prepared_structures_docked_under_more_than_one_cand_id",
            int(repeated.inchikey.nunique()), "both V2_LIGAND_TABLE.csv",
            "prepared InChIKeys appearing in more than one row of the combined library; "
            "de-duplication was applied within a batch only")
    led.add("3_preparation", "cand_ids_sharing_a_prepared_structure", int(len(repeated)),
            "both V2_LIGAND_TABLE.csv", "rows involved in those groups")

    # ------------------------------------------------------------ docking jobs
    jobs1, jobs2 = expected_jobs(lig1), expected_jobs(lig2)
    led.agree("4_docking_jobs", "batch1_jobs_queued", len(jobs1),
              int(man1["docking_jobs_total"]), "batch1 V2_LIGAND_TABLE.csv pockets column",
              "batch1 V2_PREP_MANIFEST.json", "expanded ligand x pocket routing")
    led.agree("4_docking_jobs", "batch2_jobs_queued", len(jobs2),
              int(man2["docking_jobs_total"]), "batch2 V2_LIGAND_TABLE.csv pockets column",
              "batch2 V2_PREP_MANIFEST.json", "expanded ligand x pocket routing")
    led.add("4_docking_jobs", "jobs_queued_total", len(jobs1) + len(jobs2),
            "both V2_LIGAND_TABLE.csv", "sum")
    for label, jobs, man in [("batch1", jobs1, man1), ("batch2", jobs2, man2)]:
        for pocket in sorted({p for _, p in jobs}):
            led.agree("4_docking_jobs", f"{label}_jobs_queued::{pocket}",
                      sum(1 for _, p in jobs if p == pocket),
                      int(man["docking_jobs_by_pocket"][pocket]),
                      f"{label} V2_LIGAND_TABLE.csv pockets column",
                      f"{label} V2_PREP_MANIFEST.json", "routing expanded per pocket")
    for pocket in sorted({p for _, p in jobs1 | jobs2}):
        led.add("4_docking_jobs", f"jobs_queued_total::{pocket}",
                sum(1 for _, p in (jobs1 | jobs2) if p == pocket),
                "both V2_LIGAND_TABLE.csv", "routing expanded per pocket")

    # --------------------------------------------------------------- results
    e1 = pd.read_csv(screen / "EVIDENCE_batch1.csv", low_memory=False)
    e2 = pd.read_csv(screen / "EVIDENCE_batch2.csv", low_memory=False)
    er = pd.read_csv(screen / "EVIDENCE_recovered.csv", low_memory=False)
    pairs = lambda d: set(zip(d.cand_id, d.pocket))  # noqa: E731
    p1, p2, pr = pairs(e1), pairs(e2), pairs(er)

    led.add("5_results_2026_09_04", "batch1_ligand_pocket_results", len(p1),
            "SCREEN_REAL/EVIDENCE_batch1.csv", "unique (cand_id, pocket)")
    led.add("5_results_2026_09_04", "batch1_molecules", int(e1.cand_id.nunique()),
            "SCREEN_REAL/EVIDENCE_batch1.csv", "nunique cand_id")
    led.add("5_results_2026_09_04", "batch2_ligand_pocket_results", len(p2),
            "SCREEN_REAL/EVIDENCE_batch2.csv", "unique (cand_id, pocket)")
    led.add("5_results_2026_09_04", "batch2_molecules", int(e2.cand_id.nunique()),
            "SCREEN_REAL/EVIDENCE_batch2.csv", "nunique cand_id")
    led.add("5_results_2026_09_04", "analysed_ligand_pocket_results_total", len(p1 | p2),
            "both 2026-09-04 evidence tables", "union of (cand_id, pocket)")
    led.add("5_results_2026_09_04", "analysed_molecules_total",
            int(len(set(e1.cand_id) | set(e2.cand_id))),
            "both 2026-09-04 evidence tables", "union of cand_id")
    led.add("5_results_2026_09_04", "batch1_jobs_queued_without_result",
            len(jobs1 - p1), "queued jobs vs EVIDENCE_batch1.csv", "set difference")
    led.add("5_results_2026_09_04", "batch2_jobs_queued_without_result_before_recovery",
            len(jobs2 - p2), "queued jobs vs EVIDENCE_batch2.csv", "set difference")

    # --------------------------------------------------------------- recovery
    poses_dir = args.recovery / "ALL_UNIQUE_POSES"
    shard_files = sorted(p for p in poses_dir.glob("*_out.sdf*") if not p.name.startswith("."))
    led.add("6_recovery", "recovered_pose_shards", len(shard_files),
            str(poses_dir), "file count")
    sentinels = [json.loads(line) for line in
                 (v2 / "RECOVERED_WORKERS/recovered_shard_sentinels.jsonl")
                 .read_text().splitlines() if line.strip()]
    led.agree("6_recovery", "recovered_shard_sentinels", len(sentinels), len(shard_files),
              "RECOVERED_WORKERS/recovered_shard_sentinels.jsonl",
              "ALL_UNIQUE_POSES file count", "line count")
    led.add("6_recovery", "recovered_sentinels_with_nonzero_return_code",
            int(sum(1 for s in sentinels if s.get("rc") != 0)),
            "recovered_shard_sentinels.jsonl", "count of rc != 0")

    if args.verify_shard_hashes:
        by_name = {s["shard"]: s for s in sentinels}
        matched = mismatched = unlisted = 0
        for path in shard_files:
            key = path.name[:-len("_out.sdf.gz")] if path.name.endswith("_out.sdf.gz") \
                else path.stem
            sentinel = by_name.get(key)
            if sentinel is None:
                unlisted += 1
                continue
            if sha256(path) == sentinel.get("output_sha256"):
                matched += 1
            else:
                mismatched += 1
        led.add("6_recovery", "recovered_shards_matching_sentinel_sha256", matched,
                "ALL_UNIQUE_POSES vs recovered_shard_sentinels.jsonl", "sha256 recomputed here")
        led.add("6_recovery", "recovered_shards_failing_sentinel_sha256", mismatched,
                "ALL_UNIQUE_POSES vs recovered_shard_sentinels.jsonl", "sha256 recomputed here")
        led.add("6_recovery", "recovered_shards_with_no_sentinel", unlisted,
                "ALL_UNIQUE_POSES vs recovered_shard_sentinels.jsonl", "name lookup")

    recovered_all = pd.read_csv(screen / "screen_recovered/V2_LIGAND_POCKET_SCORES.csv")
    pall = pairs(recovered_all)
    led.add("6_recovery", "recovered_shard_ligand_pocket_results", len(pall),
            "SCREEN_REAL/screen_recovered/V2_LIGAND_POCKET_SCORES.csv",
            "v2_collect_scores.py over all recovered shards")
    led.add("6_recovery", "recovered_results_duplicating_2026_09_04", len(pall & (p1 | p2)),
            "recovered collect table vs the 2026-09-04 evidence tables", "set intersection")
    led.add("6_recovery", "recovered_results_new", len(pall - p1 - p2),
            "recovered collect table vs the 2026-09-04 evidence tables", "set difference")
    led.agree("6_recovery", "recovered_new_results_carried_into_evidence", len(pr),
              len(pall - p1 - p2), "SCREEN_REAL/EVIDENCE_recovered.csv",
              "set difference computed above", "unique (cand_id, pocket)")
    new_molecules = {c for c, _ in (pall - p1 - p2)}
    led.add("6_recovery", "recovered_new_molecules", len(new_molecules),
            "SCREEN_REAL/EVIDENCE_recovered.csv", "unique cand_id among new results")
    led.add("6_recovery", "recovered_new_molecules_not_seen_at_all_before",
            len(new_molecules - set(e1.cand_id) - set(e2.cand_id)),
            "recovered new results vs both 2026-09-04 evidence tables",
            "cand_ids with no prior result in either pocket")
    led.add("6_recovery", "recovered_new_molecules_already_had_other_pocket",
            len(new_molecules & (set(e1.cand_id) | set(e2.cand_id))),
            "recovered new results vs both 2026-09-04 evidence tables",
            "cand_ids that already had a result in the other pocket")

    # ------------------------------------------------------- complete universe
    universe = pd.read_csv(args.universe, low_memory=False)
    pu = pairs(universe)
    led.agree("7_universe", "complete_universe_ligand_pocket_rows", len(universe), len(pu),
              str(args.universe), "unique (cand_id, pocket) in the same file", "row count")
    led.agree("7_universe", "complete_universe_rows_vs_sum_of_inputs", len(pu),
              len(p1 | p2 | pr), str(args.universe),
              "union of the three evidence tables", "set union")
    led.add("7_universe", "complete_universe_unique_molecules",
            int(universe.cand_id.nunique()), str(args.universe), "nunique cand_id")
    for pocket, sub in universe.groupby("pocket"):
        led.add("7_universe", f"complete_universe_rows::{pocket}", int(len(sub)),
                str(args.universe), "row count per pocket")
    led.add("7_universe", "molecules_with_at_least_one_non_clashing_pose",
            int(universe[~universe.all_poses_clashing.astype(bool)].cand_id.nunique()),
            str(args.universe), "cand_ids with all_poses_clashing False in any pocket")
    led.add("7_universe", "ligand_pocket_rows_with_no_acceptable_pose",
            int(universe.all_poses_clashing.astype(bool).sum()),
            str(args.universe), "rows where every GNINA pose has minimizedAffinity >= 0")
    usable = universe[~universe.all_poses_clashing.astype(bool)]
    led.add("7_universe", "final_unique_cand_ids_with_usable_evidence",
            int(usable.cand_id.nunique()), str(args.universe),
            "unique cand_ids holding at least one clash-filtered pose in at least one pocket")
    key_of = dict(zip(pd.concat([lig1, lig2]).cand_id, pd.concat([lig1, lig2]).inchikey))
    led.add("7_universe", "final_unique_structures_with_usable_evidence",
            len({key_of[c] for c in usable.cand_id.unique() if c in key_of}),
            f"{args.universe} joined to both V2_LIGAND_TABLE.csv",
            "distinct prepared InChIKeys among those cand_ids, so a structure docked "
            "under two cand_ids is counted once")
    led.add("7_universe", "complete_universe_unique_structures",
            len({key_of[c] for c in universe.cand_id.unique() if c in key_of}),
            f"{args.universe} joined to both V2_LIGAND_TABLE.csv",
            "distinct prepared InChIKeys over the whole universe")

    # ------------------------------------------------------------ what is missing
    all_jobs = jobs1 | jobs2
    missing = all_jobs - pu
    led.add("8_missing", "jobs_queued_with_no_surviving_result", len(missing),
            "queued jobs vs the complete universe", "set difference")
    prepared_ids = set(lig1.cand_id) | set(lig2.cand_id)
    without_result = prepared_ids - set(universe.cand_id)
    led.add("8_missing", "molecules_prepared_but_with_no_result_in_any_pocket",
            len(without_result),
            "prepared ligand tables vs the complete universe", "arithmetic on cand_id sets")
    for label, table in [("batch1", lig1), ("batch2", lig2)]:
        led.add("8_missing", f"{label}_molecules_prepared_but_with_no_result",
                len(set(table.cand_id) & without_result),
                f"{label} V2_LIGAND_TABLE.csv vs the complete universe", "set intersection")
    for pocket in sorted({p for _, p in all_jobs}):
        led.add("8_missing", f"jobs_with_no_surviving_result::{pocket}",
                sum(1 for _, p in missing if p == pocket),
                "queued jobs vs the complete universe", "set difference per pocket")

    # Partition-level attribution for batch 2.
    worker_shards = None
    for candidate in sorted(args.recovery.glob(
            "*/extracted/PAR2/YONSEI_FINAL10_V2/SCREEN_REAL/screen_set_batch2/shards")):
        worker_shards = candidate
        break
    if worker_shards is None:
        led.add("8_missing", "batch2_dispatch_partitions", UNKNOWN,
                "no screen_set_batch2/shards tree found under the recovery directory",
                "partition membership could not be reconstructed",
                status="UNKNOWN")
    else:
        parts = shard_partitions(worker_shards)
        pocket_of = {name: name.split("_shard")[0] for name in parts}
        done = {name: sum(1 for c in ids_ if (c, pocket_of[name]) in pu)
                for name, ids_ in parts.items()}
        complete = [n for n, k in done.items() if k == len(parts[n])]
        empty = [n for n, k in done.items() if k == 0]
        partial = [n for n, k in done.items() if 0 < k < len(parts[n])]
        led.add("8_missing", "batch2_dispatch_partitions", len(parts), str(worker_shards),
                f"screen-set shards split into {PARTITION_SIZE}-ligand `_pN` partitions")
        led.add("8_missing", "batch2_partitions_fully_recovered", len(complete),
                str(worker_shards), "every ligand in the partition has a result")
        led.add("8_missing", "batch2_partitions_with_no_result_at_all", len(empty),
                str(worker_shards), "no ligand in the partition has a result")
        led.add("8_missing", "batch2_partitions_partially_recovered", len(partial),
                str(worker_shards), "some but not all ligands have a result")
        led.add("8_missing", "batch2_jobs_missing_inside_empty_partitions",
                sum(len(parts[n]) for n in empty), str(worker_shards),
                "ligand count of partitions that produced nothing")
        (args.out.parent / "MISSING_BATCH2_PARTITIONS.csv").write_text(
            "partition,pocket,ligands,results\n" + "".join(
                f"{n},{pocket_of[n]},{len(parts[n])},{done[n]}\n"
                for n in sorted(empty + partial)))

    led.add("9_loss", "results_lost_with_pw_n2b", UNKNOWN,
            "RECOVERED_WORKERS/PAR2_RECOVERY_COMPLETE.md",
            "pw-n2b and its boot disk were deleted before the export; no surviving record "
            "enumerates the shards it was assigned or completed after 14:45, so the number "
            "of docked-then-lost results cannot be separated from never-dispatched ones",
            status="UNKNOWN")
    led.add("9_loss", "jobs_never_docked_or_docked_then_lost", len(missing),
            "queued jobs vs the complete universe",
            "upper bound on both categories jointly; the split is UNKNOWN for the reason above")

    # ------------------------------------------------------------------ stereo
    if args.stereo_audit and args.stereo_audit.exists():
        audit = pd.read_csv(args.stereo_audit, low_memory=False)
        led.add("10_stereo", "designs_audited", int(len(audit)),
                str(args.stereo_audit), "row count")
        led.add("10_stereo", "designs_declaring_stereochemistry",
                int(audit.input_declares_stereo.fillna(False).sum()),
                str(args.stereo_audit), "assigned stereocentre or defined stereo bond in input")
        for status, n in audit.stereo_status.value_counts().items():
            led.add("10_stereo", f"stereo_status::{status}", int(n),
                    str(args.stereo_audit), "value_counts of stereo_status")
        led.add("10_stereo", "unique_TAUTOMER_PARENT_ID",
                int(audit.TAUTOMER_PARENT_ID.nunique()), str(args.stereo_audit),
                "nunique of the tautomer-canonical parent SMILES")
        led.add("10_stereo", "unique_CONNECTIVITY_ID",
                int(audit.CONNECTIVITY_ID.nunique()), str(args.stereo_audit),
                "nunique of the flat canonical SMILES of the queue input")
        led.add("10_stereo", "unique_STEREO_ID", int(audit.STEREO_ID.nunique()),
                str(args.stereo_audit), "nunique of the isomeric canonical SMILES")

    table = led.frame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        rows=int(len(table)),
        status_counts={k: int(v) for k, v in table.status.value_counts().items()},
        discrepancies=table[table.status == "DISCREPANCY"].to_dict("records"),
        unknowns=table[table.status == "UNKNOWN"].to_dict("records"),
        note=("Every COMPUTED value was recomputed from the primary artefact named in "
              "its source column. No value was copied from a previous report."),
    )
    (args.summary or args.out.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, default=str))
    print(table.to_string(index=False, max_colwidth=64))
    print()
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("rows", "status_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
