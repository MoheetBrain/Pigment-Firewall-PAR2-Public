#!/usr/bin/env python3
"""Track stereochemistry through V2 ligand preparation, design by design.

v2_prepare_ligands.py standardises every design with

    rdMolStandardize.Cleanup -> FragmentParent -> TautomerEnumerator().Canonicalize

and only afterwards computes the InChIKey used to collapse duplicates. The
tautomer canonicaliser can remove a stereocentre that sits next to a tautomeric
system, so two genuinely different epimers can arrive at the same InChIKey and
one of them is then discarded as a "duplicate".

This script re-runs that exact standardisation on every design in the queue
CSVs, records the stereochemistry before and after, and reports:

  * per design: input stereocentres (assigned and unassigned), stereo double
    bonds, the same counts after standardisation, and a LOST / PRESERVED /
    ALTERED / NEVER_PRESENT verdict
  * per design: CONNECTIVITY_ID, STEREO_ID and TAUTOMER_PARENT_ID, plus the
    stereo-aware and flat InChIKeys of both the input and the prepared structure
  * every group of designs that collapsed onto one prepared InChIKey, classified
    as a true duplicate, a distinct stereoisomer, or a tautomer merge

It also emits the subset of designs at the top of the supplied ranking whose
docked stereochemistry is ambiguous or wrong, as a redocking work list.

No docking is performed and no ranking is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_prepare_ligands import protonate_ph74  # noqa: E402

SUPPORTED_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Se"}
MISSING = ""  # a value that could not be determined stays empty, never zero

# A double bond whose geometry is actually fixed. STEREOANY means "either",
# which is the state the tautomer canonicaliser leaves a defined bond in, so it
# must not be counted as defined stereochemistry.
DEFINED_BOND_STEREO = {
    Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
}


def stereo_profile(mol: Chem.Mol) -> dict:
    """Stereocentre and stereo-bond counts for one molecule."""
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                        useLegacyImplementation=False)
    assigned = [(i, tag) for i, tag in centres if tag in ("R", "S")]
    unassigned = [(i, tag) for i, tag in centres if tag not in ("R", "S")]
    defined_bonds = sum(1 for b in mol.GetBonds() if b.GetStereo() in DEFINED_BOND_STEREO)
    any_bonds = sum(1 for b in mol.GetBonds()
                    if b.GetStereo() == Chem.BondStereo.STEREOANY)
    return dict(
        n_stereocentres=len(centres),
        n_stereocentres_assigned=len(assigned),
        n_stereocentres_unassigned=len(unassigned),
        assignments=";".join(f"{i}{tag}" for i, tag in sorted(assigned)),
        n_stereo_bonds=defined_bonds,
        n_stereo_bonds_undefined=any_bonds,
    )


def flat_inchikey(mol: Chem.Mol) -> str:
    work = Chem.Mol(mol)
    Chem.RemoveStereochemistry(work)
    try:
        return Chem.MolToInchiKey(work)
    except Exception:  # noqa: BLE001
        return MISSING


def safe_inchikey(mol: Chem.Mol) -> str:
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001
        return MISSING


def standardise(mol: Chem.Mol) -> tuple[Chem.Mol | None, str, dict]:
    """Replicate v2_prepare_ligands.prepare_one up to the InChIKey computation.

    Also records the assigned-stereocentre count after each individual step, so
    the stage that destroys stereochemistry can be named rather than guessed.
    """
    stages: dict[str, int] = {}

    def record(name: str, m: Chem.Mol) -> None:
        p = stereo_profile(m)
        stages[name] = p["n_stereocentres_assigned"] + p["n_stereo_bonds"]

    try:
        work = rdMolStandardize.Cleanup(mol)
        record("after_cleanup", work)
        work = rdMolStandardize.FragmentParent(work)
        record("after_fragment_parent", work)
        work = rdMolStandardize.TautomerEnumerator().Canonicalize(work)
        record("after_tautomer_canonicalize", work)
    except Exception as exc:  # noqa: BLE001
        return None, f"STANDARDISE_FAILED:{type(exc).__name__}", stages
    if work is None or work.GetNumAtoms() == 0:
        return None, "EMPTY_AFTER_STANDARDISE", stages
    heavy = work.GetNumHeavyAtoms()
    if heavy < 8:
        return work, "TOO_FEW_HEAVY_ATOMS", stages
    if heavy > 60:
        return work, "TOO_MANY_HEAVY_ATOMS", stages
    if any(a.GetSymbol() not in SUPPORTED_ELEMENTS for a in work.GetAtoms()):
        return work, "UNSUPPORTED_ELEMENT", stages
    return work, "OK", stages


def audit_one(cand_id: str, batch: str, smiles_input: str) -> dict:
    row = dict(cand_id=cand_id, batch=batch, smiles_input=smiles_input)
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        row.update(input_parse_ok=False, stereo_status="UNKNOWN_INPUT_UNPARSEABLE",
                   standardise_reason="SMILES_PARSE_FAILED")
        return row
    row["input_parse_ok"] = True
    before = stereo_profile(mol)
    row.update({f"input_{k}": v for k, v in before.items()})
    row["input_declares_stereo"] = bool(before["n_stereocentres_assigned"]
                                        or before["n_stereo_bonds"])
    row["CONNECTIVITY_ID"] = Chem.MolToSmiles(mol, isomericSmiles=False)
    row["STEREO_ID"] = Chem.MolToSmiles(mol, isomericSmiles=True)
    row["input_inchikey_stereo"] = safe_inchikey(mol)
    row["input_inchikey_flat"] = flat_inchikey(mol)

    parent, reason, stages = standardise(mol)
    row["standardise_reason"] = reason
    row.update({f"assigned_{k}": v for k, v in stages.items()})
    if parent is None:
        row["stereo_status"] = "UNKNOWN_STANDARDISATION_FAILED"
        return row

    row["TAUTOMER_PARENT_ID"] = Chem.MolToSmiles(parent, isomericSmiles=True)
    row["TAUTOMER_PARENT_CONNECTIVITY_ID"] = Chem.MolToSmiles(parent, isomericSmiles=False)
    row["parent_inchikey_stereo"] = safe_inchikey(parent)
    row["parent_inchikey_flat"] = flat_inchikey(parent)
    after = stereo_profile(parent)
    row.update({f"parent_{k}": v for k, v in after.items()})

    prepared = protonate_ph74(parent) if reason == "OK" else parent
    row["smiles_prepared_recomputed"] = Chem.MolToSmiles(prepared, isomericSmiles=True)
    row["prepared_inchikey_stereo_recomputed"] = safe_inchikey(prepared)
    row["prepared_inchikey_flat_recomputed"] = flat_inchikey(prepared)
    prepared_profile = stereo_profile(prepared)
    row.update({f"prepared_{k}": v for k, v in prepared_profile.items()})

    lost = before["n_stereocentres_assigned"] - prepared_profile["n_stereocentres_assigned"]
    bond_lost = before["n_stereo_bonds"] - prepared_profile["n_stereo_bonds"]
    row["n_stereocentres_lost"] = lost
    row["n_stereo_bonds_lost"] = bond_lost

    # Name the step that destroyed the stereochemistry, rather than assuming it.
    # Counts are centres plus defined double bonds, so both kinds are covered.
    if lost > 0 or bond_lost > 0:
        counts = [("input", before["n_stereocentres_assigned"] + before["n_stereo_bonds"]),
                  ("cleanup", stages.get("after_cleanup")),
                  ("fragment_parent", stages.get("after_fragment_parent")),
                  ("tautomer_canonicalize", stages.get("after_tautomer_canonicalize")),
                  ("protonate_ph74", prepared_profile["n_stereocentres_assigned"]
                   + prepared_profile["n_stereo_bonds"])]
        culprits = [name for (_, prev), (name, now) in zip(counts, counts[1:])
                    if prev is not None and now is not None and now < prev]
        row["stereo_lost_at_stage"] = ";".join(culprits) if culprits else "UNKNOWN"
    else:
        row["stereo_lost_at_stage"] = MISSING
    # Standardisation may also change the constitution (an enol canonicalised to
    # the keto form), which can open a stereocentre that the design never had.
    # That is compared before protonation, so an added proton is not mistaken
    # for a structural change.
    skeleton_in = row["input_inchikey_stereo"].split("-")[0]
    skeleton_parent = row["parent_inchikey_stereo"].split("-")[0]
    row["constitution_changed"] = bool(skeleton_in and skeleton_parent
                                       and skeleton_in != skeleton_parent)
    row["n_new_unassigned_stereocentres"] = max(
        0, prepared_profile["n_stereocentres_unassigned"]
        - before["n_stereocentres_unassigned"])

    if not row["input_declares_stereo"]:
        status = "NEVER_PRESENT"
    elif lost > 0 or bond_lost > 0:
        status = "LOST"
    elif (not row["constitution_changed"]
          and row["input_inchikey_stereo"] != row["parent_inchikey_stereo"]):
        # Same skeleton, same number of assigned centres, different stereo
        # layer: a descriptor was inverted rather than dropped.
        status = "ALTERED"
    else:
        status = "PRESERVED"
    row["stereo_status"] = status
    row["stereo_ambiguous_input"] = before["n_stereocentres_unassigned"] > 0
    return row


def load_queue(root: Path) -> pd.DataFrame:
    """cand_id -> the SMILES actually handed to v2_prepare_ligands, per batch."""
    d = root / "YONSEI_FINAL10_V2" / "02_DENOVO_CPU"
    lib = pd.read_csv(d / "DENOVO_15000.csv")
    smiles_to_library_rank = dict(zip(lib.smiles, lib.denovo_rank))

    b1 = pd.read_csv(d / "DENOVO_TOP3000_FOR_DOCKING_TAUTOMER_CORRECTED.csv")
    b1 = b1.assign(
        cand_id=[f"DN{int(r):05d}" for r in b1.dock_priority_rank],
        batch="batch1",
        queue_rank=b1.dock_priority_rank.astype(int),
        library_rank=[smiles_to_library_rank.get(s) for s in b1.smiles_original.fillna(b1.smiles)],
    )
    b2 = pd.read_csv(d / "DENOVO_BATCH2_REMAINING12000_TAUTOMER_CORRECTED.csv")
    b2 = b2.assign(
        cand_id=[f"DL{int(r):05d}" for r in b2.dock_priority_rank],
        batch="batch2",
        queue_rank=b2.dock_priority_rank.astype(int),
        library_rank=b2.library_rank.astype(int),
    )
    cols = ["cand_id", "batch", "queue_rank", "library_rank", "smiles",
            "smiles_original", "tautomer_corrected", "family", "design_score"]
    return pd.concat([b1[cols], b2[cols]], ignore_index=True)


def classify_group(members: pd.DataFrame) -> str:
    """Why did these designs collapse onto one prepared InChIKey?

    TRUE_DUPLICATE            same molecule twice; the collapse loses nothing
    EPIMERIC_PAIR             every member assigns the same number of centres,
                              with at least one opposite descriptor: genuinely
                              different, fully specified stereoisomers
    SPECIFIED_VS_UNSPECIFIED  members differ in how many centres they assign;
                              one design pins a centre the other leaves open
    TAUTOMER_MERGE            members differ in constitution and were made
                              identical only by tautomer canonicalisation
    """
    stereo_keys = set(members.input_inchikey_stereo)
    flat_keys = set(members.input_inchikey_flat)
    connectivity = set(members.CONNECTIVITY_ID)
    if len(stereo_keys) == 1:
        return "TRUE_DUPLICATE"
    if len(flat_keys) > 1 and len(connectivity) > 1:
        return "TAUTOMER_MERGE"
    if members.input_n_stereocentres_assigned.nunique() > 1:
        return "SPECIFIED_VS_UNSPECIFIED"
    return "EPIMERIC_PAIR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True,
                        help="STEREOCHEMISTRY_AUDIT.csv")
    parser.add_argument("--out-redock", type=Path, required=True,
                        help="STEREO_REDOCK_LIST.csv")
    parser.add_argument("--ranking", type=Path, nargs="*", default=None,
                        help="ranking CSV(s) whose cand_id column defines the finalists")
    parser.add_argument("--rank-column", default="consensus_rank")
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--redock-scope", choices=["ranked", "all"], default="ranked",
                        help="write only affected finalists, or every affected design")
    args = parser.parse_args()

    root = args.root.resolve()
    v2 = root / "YONSEI_FINAL10_V2"

    queue = load_queue(root)
    rows = [audit_one(r.cand_id, r.batch, r.smiles) for r in queue.itertuples()]
    audit = queue.merge(pd.DataFrame(rows).drop(columns=["smiles_input"]),
                        on=["cand_id", "batch"], how="left")

    # What actually reached the shards, and what did not.
    lig = pd.concat([
        pd.read_csv(v2 / "SCREEN_REAL/library/V2_LIGAND_TABLE.csv"),
        pd.read_csv(v2 / "SCREEN_REAL/library_batch2/V2_LIGAND_TABLE.csv"),
    ], ignore_index=True, sort=False)[["cand_id", "inchikey", "smiles_prepared", "prep_ok"]]
    lig = lig.rename(columns={"inchikey": "docked_inchikey",
                              "smiles_prepared": "docked_smiles_prepared"})
    fails = pd.concat([
        pd.read_csv(v2 / "SCREEN_REAL/library/V2_PREP_FAILURES.csv"),
        pd.read_csv(v2 / "SCREEN_REAL/library_batch2/V2_PREP_FAILURES.csv"),
    ], ignore_index=True, sort=False)[["cand_id", "prep_reason"]]
    fails = fails.rename(columns={"prep_reason": "prep_failure_reason"})

    audit = audit.merge(lig, on="cand_id", how="left").merge(fails, on="cand_id", how="left")
    in_library = audit.docked_inchikey.notna()
    failed = audit.prep_failure_reason.notna()
    audit["library_fate"] = "UNKNOWN"
    audit.loc[in_library, "library_fate"] = "PREPARED_AND_DOCKED"
    audit.loc[failed, "library_fate"] = "PREPARATION_FAILED"
    audit.loc[~in_library & ~failed, "library_fate"] = "COLLAPSED_AS_DUPLICATE_INCHIKEY"

    # Self-check: does the replicated standardisation reproduce the shipped table?
    checkable = audit[in_library]
    key_match = int((checkable.prepared_inchikey_stereo_recomputed
                     == checkable.docked_inchikey).sum())
    smiles_match = int((checkable.smiles_prepared_recomputed
                        == checkable.docked_smiles_prepared).sum())

    # Groups that collapsed onto a single prepared InChIKey, within a batch.
    audit["collapse_group"] = MISSING
    audit["collapse_class"] = MISSING
    audit["collapse_survivor"] = MISSING
    audit["collapse_members"] = MISSING
    groups = []
    considered = audit[audit.library_fate != "PREPARATION_FAILED"]
    for (batch, key), members in considered.groupby(
            ["batch", "prepared_inchikey_stereo_recomputed"], sort=False):
        if key in ("", None) or len(members) < 2:
            continue
        survivors = members[members.library_fate == "PREPARED_AND_DOCKED"]
        klass = classify_group(members)
        gid = f"{batch}:{key}"
        idx = members.index
        audit.loc[idx, "collapse_group"] = gid
        audit.loc[idx, "collapse_class"] = klass
        audit.loc[idx, "collapse_survivor"] = (survivors.cand_id.iloc[0]
                                               if len(survivors) else MISSING)
        audit.loc[idx, "collapse_members"] = ";".join(sorted(members.cand_id))
        groups.append(dict(
            collapse_group=gid, batch=batch, prepared_inchikey=key,
            n_members=int(len(members)), members=sorted(members.cand_id),
            n_survivors=int(len(survivors)),
            survivor=survivors.cand_id.iloc[0] if len(survivors) else MISSING,
            dropped=sorted(set(members.cand_id) - set(survivors.cand_id)),
            classification=klass,
            input_stereo_inchikeys=sorted(set(members.input_inchikey_stereo)),
            input_flat_inchikeys=sorted(set(members.input_inchikey_flat)),
            input_smiles=list(members.smiles),
            stereo_ids=sorted(set(members.STEREO_ID)),
        ))

    # Preparation de-duplicated within a batch only, so the same prepared
    # structure can still have been docked twice under two cand_ids.
    audit["docked_duplicate_group"] = MISSING
    audit["docked_duplicate_partners"] = MISSING
    docked_duplicates = []
    docked = audit[audit.library_fate == "PREPARED_AND_DOCKED"]
    for key, members in docked.groupby("prepared_inchikey_stereo_recomputed", sort=False):
        if key in ("", None) or len(members) < 2:
            continue
        audit.loc[members.index, "docked_duplicate_group"] = key
        audit.loc[members.index, "docked_duplicate_partners"] = ";".join(sorted(members.cand_id))
        docked_duplicates.append(dict(
            prepared_inchikey=key, cand_ids=sorted(members.cand_id),
            batches=sorted(set(members.batch)),
            spans_batches=bool(members.batch.nunique() > 1),
            classification=classify_group(members),
            input_stereo_inchikeys=sorted(set(members.input_inchikey_stereo)),
            input_smiles=list(members.smiles)))

    audit.to_csv(args.out, index=False)

    # Affected finalists: the docked stereoisomer is ambiguous or wrong.
    # Ranking files do not share a rank scale: the pooled consensus table is
    # ranked 1..N over the whole universe while a per-pocket table restarts at
    # 1. Positions are therefore taken from the FIRST file (in the order given)
    # that lists a molecule, never minimised across files, and the file is named
    # alongside the number. Membership in the other files is recorded separately.
    rank_map: dict[str, float] = {}
    rank_sources: dict[str, str] = {}
    membership: dict[str, list[str]] = defaultdict(list)
    primary_ranking = ""
    for path in (args.ranking or []):
        if not path.exists():
            print(f"[stereo] ranking file not found, skipped: {path}", file=sys.stderr)
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "cand_id" not in frame.columns:
            print(f"[stereo] no cand_id column, skipped: {path}", file=sys.stderr)
            continue
        column = args.rank_column if args.rank_column in frame.columns else None
        if column:
            frame = frame.sort_values(column)
        frame = frame.head(args.top_n).drop_duplicates("cand_id")
        primary_ranking = primary_ranking or path.name
        for i, r in enumerate(frame.itertuples(), 1):
            membership[r.cand_id].append(path.name)
            if r.cand_id in rank_map:
                continue
            rank_map[r.cand_id] = float(getattr(r, column)) if column else float(i)
            rank_sources[r.cand_id] = path.name

    reasons: dict[str, list[str]] = defaultdict(list)
    for r in audit.itertuples():
        if r.library_fate != "PREPARED_AND_DOCKED":
            continue
        if r.stereo_status == "LOST":
            reasons[r.cand_id].append("STEREO_LOST_IN_PREPARATION")
        elif r.stereo_status == "ALTERED":
            reasons[r.cand_id].append("STEREO_ALTERED_IN_PREPARATION")
        if getattr(r, "n_new_unassigned_stereocentres", 0) > 0:
            reasons[r.cand_id].append("NEW_UNASSIGNED_STEREOCENTRE_FROM_STANDARDISATION")
        if getattr(r, "stereo_ambiguous_input", False):
            reasons[r.cand_id].append("UNASSIGNED_STEREOCENTRE_IN_DESIGN")
        if r.collapse_class in ("EPIMERIC_PAIR", "SPECIFIED_VS_UNSPECIFIED",
                                "TAUTOMER_MERGE"):
            reasons[r.cand_id].append("STANDS_IN_FOR_COLLAPSED_DESIGN")
        if r.docked_duplicate_group:
            reasons[r.cand_id].append("SAME_PREPARED_STRUCTURE_DOCKED_UNDER_ANOTHER_CAND_ID")

    def severity(why: set[str]) -> str:
        """WRONG: the docked structure differs from the declared design.
        AMBIGUOUS: the design itself never fixed the configuration."""
        if {"STEREO_LOST_IN_PREPARATION", "STEREO_ALTERED_IN_PREPARATION",
                "STANDS_IN_FOR_COLLAPSED_DESIGN",
                "SAME_PREPARED_STRUCTURE_DOCKED_UNDER_ANOTHER_CAND_ID",
                "NEW_UNASSIGNED_STEREOCENTRE_FROM_STANDARDISATION"} & why:
            return "WRONG_STEREOISOMER_DOCKED"
        return "AMBIGUOUS_STEREOCHEMISTRY"

    indexed = audit.set_index("cand_id")
    redock_rows = []
    for cand_id, why in reasons.items():
        a = indexed.loc[cand_id]
        redock_rows.append(dict(
            cand_id=cand_id, batch=a.batch, library_rank=a.library_rank,
            severity=severity(set(why)), reasons=";".join(sorted(set(why))),
            stereo_status=a.stereo_status,
            stereo_lost_at_stage=a.stereo_lost_at_stage,
            constitution_changed=a.constitution_changed,
            n_new_unassigned_stereocentres=a.n_new_unassigned_stereocentres,
            input_n_stereocentres=a.input_n_stereocentres,
            input_n_stereocentres_assigned=a.input_n_stereocentres_assigned,
            input_n_stereocentres_unassigned=a.input_n_stereocentres_unassigned,
            prepared_n_stereocentres_assigned=a.prepared_n_stereocentres_assigned,
            smiles_input=a.smiles, docked_smiles_prepared=a.docked_smiles_prepared,
            STEREO_ID=a.STEREO_ID, CONNECTIVITY_ID=a.CONNECTIVITY_ID,
            collapse_group=a.collapse_group, collapse_class=a.collapse_class,
            collapse_members=a.collapse_members,
            docked_duplicate_partners=a.docked_duplicate_partners,
            ranking_position=rank_map.get(cand_id),
            ranking_source=rank_sources.get(cand_id, MISSING),
            ranking_files_listing_it=";".join(membership.get(cand_id, [])),
        ))
    redock = pd.DataFrame(redock_rows)
    redock["is_finalist"] = redock.ranking_position.notna()
    affected_all = int(len(redock))
    if args.redock_scope == "ranked":
        redock = redock[redock.is_finalist]
    redock = redock.sort_values(["severity", "ranking_source", "ranking_position",
                                 "cand_id"], na_position="last")

    args.out_redock.parent.mkdir(parents=True, exist_ok=True)
    redock.to_csv(args.out_redock, index=False)

    prepared_mask = audit.library_fate == "PREPARED_AND_DOCKED"
    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        rdkit_version=Chem.rdBase.rdkitVersion,
        designs_audited=int(len(audit)),
        designs_by_batch={k: int(v) for k, v in audit.batch.value_counts().items()},
        input_parse_failures=int((~audit.input_parse_ok.fillna(False)).sum()),
        designs_declaring_stereo=int(audit.input_declares_stereo.fillna(False).sum()),
        designs_with_unassigned_stereocentre=int(
            audit.stereo_ambiguous_input.fillna(False).sum()),
        stereo_status_counts={k: int(v) for k, v in audit.stereo_status.value_counts().items()},
        stereo_status_counts_prepared_only={
            k: int(v) for k, v in audit[prepared_mask].stereo_status.value_counts().items()},
        stereocentres_lost_total=int(pd.to_numeric(
            audit.n_stereocentres_lost, errors="coerce").clip(lower=0).sum()),
        stereo_double_bonds_lost_total=int(pd.to_numeric(
            audit.n_stereo_bonds_lost, errors="coerce").clip(lower=0).sum()),
        designs_losing_only_double_bond_stereo=int(
            ((pd.to_numeric(audit.n_stereocentres_lost, errors="coerce").fillna(0) <= 0)
             & (pd.to_numeric(audit.n_stereo_bonds_lost, errors="coerce").fillna(0) > 0)).sum()),
        designs_with_constitution_changed=int(audit.constitution_changed.fillna(False).sum()),
        designs_with_new_unassigned_stereocentre=int(
            (pd.to_numeric(audit.n_new_unassigned_stereocentres, errors="coerce")
             .fillna(0) > 0).sum()),
        library_fate_counts={k: int(v) for k, v in audit.library_fate.value_counts().items()},
        replication_check=dict(
            rows_checked=int(len(checkable)),
            inchikey_reproduced=key_match,
            smiles_prepared_reproduced=smiles_match,
            note=("recomputed standardisation compared against the shipped "
                  "V2_LIGAND_TABLE values")),
        collapse_groups=len(groups),
        collapse_group_classes={k: int(v) for k, v in
                                pd.Series([g["classification"] for g in groups])
                                .value_counts().items()} if groups else {},
        collapsed_designs_dropped=int(
            (audit.library_fate == "COLLAPSED_AS_DUPLICATE_INCHIKEY").sum()),
        collapsed_designs_dropped_by_batch={
            k: int(v) for k, v in
            audit[audit.library_fate == "COLLAPSED_AS_DUPLICATE_INCHIKEY"]
            .batch.value_counts().items()},
        docked_duplicate_groups=len(docked_duplicates),
        docked_duplicate_groups_spanning_batches=sum(
            1 for d in docked_duplicates if d["spans_batches"]),
        docked_duplicate_molecules=int((audit.docked_duplicate_group != MISSING).sum()),
        docked_duplicate_detail=docked_duplicates,
        affected_designs_total=affected_all,
        redock_rows_written=int(len(redock)),
        redock_scope=args.redock_scope,
        redock_severity_counts={k: int(v) for k, v in
                                redock.severity.value_counts().items()} if len(redock) else {},
        redock_reason_counts={k: int(v) for k, v in
                              redock.reasons.value_counts().items()} if len(redock) else {},
        finalists_considered=len(rank_map),
        finalists_affected=int(redock.is_finalist.sum()) if len(redock) else 0,
        finalists_affected_by_ranking_source={
            k: int(v) for k, v in redock.ranking_source.value_counts().items()}
        if len(redock) else {},
        primary_ranking=primary_ranking,
        affected_in_top_n_of_primary_ranking={
            str(n): int(((redock.ranking_source == primary_ranking)
                         & (redock.ranking_position <= n)).sum())
            for n in (10, 50, 100, 500, 1000)} if (len(redock) and primary_ranking) else {},
        ranking_used=[str(p) for p in (args.ranking or [])],
        ranking_top_n=args.top_n,
        stereo_lost_at_stage_counts={
            k: int(v) for k, v in audit.stereo_lost_at_stage.replace("", pd.NA)
            .dropna().value_counts().items()},
        groups=groups,
        note=("Stereo verdicts describe what preparation did to the declared "
              "structure. They are not a claim about which stereoisomer binds."),
    )
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    printable = {k: v for k, v in summary.items() if k != "groups"}
    printable["groups"] = f"{len(groups)} groups (see {summary_path})"
    print(json.dumps(printable, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
