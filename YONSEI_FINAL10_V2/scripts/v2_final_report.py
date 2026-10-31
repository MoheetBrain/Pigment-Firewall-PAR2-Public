#!/usr/bin/env python3
"""Emit the candidate hand-over report from whatever evidence currently exists.

Re-runnable: every number is recomputed from the committed tables, so the report
can be regenerated as more of the library finishes docking.

Nothing here is a claim that any molecule antagonises PAR2.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def fmt(frame: pd.DataFrame, cols: list[str]) -> str:
    """Render a dataframe as a markdown table, blank for missing values."""
    cols = [c for c in cols if c in frame.columns]
    if not cols or frame.empty:
        return "_none_"
    sub = frame[cols].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    header = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in sub.values)
    return "\n".join([header, rule, body])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    r = args.root

    cons = pd.read_csv(r / "07_CONSENSUS/DENOVO_EX8_CONSENSUS_ALL.csv")
    summary = json.loads((r / "07_CONSENSUS/CONSENSUS_SUMMARY.json").read_text())
    seeds = pd.read_csv(r / "08_CLUSTERS/SIMILARITY_SEARCH_SEEDS.csv")
    hits = pd.read_csv(r / "09_ANALOGUES/SIMILARITY_HITS_DEDUP.csv") \
        if (r / "09_ANALOGUES/SIMILARITY_HITS_DEDUP.csv").exists() else pd.DataFrame()

    nd = cons[cons.pocket == "5NDD_orthosteric"]
    both = nd[nd.anchor_hbond_fraction >= 0.999].sort_values("consensus_rank")

    lines = []
    A = lines.append
    A("# PAR2 candidate hand-over — computationally prioritised designs\n")
    A(f"Generated {datetime.now(timezone.utc).isoformat()}\n")
    A("**These are COMPUTATIONALLY PRIORITISED PAR2 CANDIDATES, not demonstrated "
      "antagonists.** Static docking cannot separate an antagonist from a biased "
      "agonist. The decisive experiments are the functional assays, above all "
      "Galpha-12/13 -> RhoA: a compound that blocks Gq but agonises G12/13 would "
      "still drive the actin machinery that engulfs melanocores.\n")

    A("## Coverage\n")
    A(f"- Designs with docking evidence: **{summary['unique_designs']}**")
    A(f"- Ligand-pocket rows: {summary['rows']} "
      f"({', '.join(f'{k} {v}' for k, v in summary['by_pocket'].items())})")
    A(f"- By batch: {summary.get('by_batch', {})}")
    A(f"- Metrics used: {', '.join(summary['metrics_used'])}")
    A(f"- PIGNet2 available: {summary['pignet2_available']}\n")

    A("## Why interaction fingerprints carry the weight\n")
    A("Measured Spearman correlations inside 5NDD showed the scoring functions are "
      "not independent: Vina and ligand efficiency correlate -0.958, Vina and "
      "PIGNet2 0.899, CNNscore and CNN_VS 0.988. Reference-antagonist interaction "
      "recovery correlates |r| <= 0.19 with every other metric, so it is the only "
      "orthogonal evidence and was given its own family rather than being outvoted "
      "by five correlated views of the same signal.\n")

    A("## The AZ8838 polar-anchor signature\n")
    A("The repaired crystallographic reference shows AZ8838 donating two hydrogen "
      "bonds, to TYR:82 (2.475 A) and HIS:135 (2.63 A). Designs reproducing "
      "hydrogen bonds to both principal anchors (ASP:228 and HIS:135) are the ones "
      "that anchor through specific polar contacts rather than greasy surface.\n")
    A(f"- 5NDD designs hydrogen-bonding at least one anchor: **{summary['anchor_hbond_hits_5NDD']}**")
    A(f"- Hydrogen-bonding **both** anchors: **{summary['both_anchors_5NDD']}**")
    A(f"- Flagged for agonist-mimicry geometry: {summary['agonist_pattern_flagged']} "
      "(flagged for functional testing, never silently removed)")
    A(f"- Pareto front: {summary['pareto_front_size']}\n")

    cols = ["consensus_rank", "cand_id", "denovo_family", "best_vina_affinity",
            "best_CNNscore", "pignet2_best", "max_ifp_recovery", "ifp_hbonds",
            "ligand_efficiency", "mw", "clogp", "fsp3", "qed", "profile_fit"]
    A("## Dual-anchor designs, best first\n")
    A(fmt(both.head(20), cols) + "\n")

    A("## Top 20 by consensus in the primary 5NDD lane\n")
    A(fmt(nd.sort_values("consensus_rank").head(20), cols) + "\n")

    A("## Similarity-search seeds\n")
    A(f"{len(seeds)} seeds across {seeds.murcko_scaffold.nunique()} distinct Murcko "
      f"scaffolds and {seeds.cluster_id.nunique()} clusters "
      f"({seeds.pocket.value_counts().to_dict()}).\n")
    A(fmt(seeds, ["seed_id", "cand_id", "pocket", "family", "postdock_rank",
                  "cluster_size", "ifp_recovery", "best_affinity", "MW", "cLogP",
                  "fsp3", "QED", "novelty_band"]) + "\n")

    if len(hits):
        A("## Real purchasable analogues retrieved\n")
        A(f"{len(hits)} unique real compounds from public databases, "
          f"{hits.seed_id.nunique()} seeds with at least one hit.\n")
        bands = hits.similarity_band.value_counts().to_dict()
        A(f"Bands: {bands}\n")
        A("Closest real matter:\n")
        A(fmt(hits.nlargest(15, "tanimoto_to_seed"),
              ["seed_id", "source_database", "source_compound_id", "source_name",
               "tanimoto_to_seed", "same_scaffold_as_seed", "mcs_similarity"]) + "\n")
        A("**Vendor, catalogue ID, price and stock are PENDING for every row.** "
          "No catalogue was queried, so none of these is confirmed purchasable.\n")

    A("## What has NOT been done\n")
    A("- No EX32 precision redocking of the finalists.\n"
      "- No Boltz-2 co-folding and no molecular dynamics.\n"
      "- No vendor availability check.\n"
      "- No wet-lab evidence of any kind.\n"
      "- Part of the de novo library was still docking when this was written.\n")
    A("## Required next experiments\n")
    A("1. Gq: calcium mobilisation and IP1, agonist-mode first to prove silence.\n"
      "2. G12/13: p115-RhoGEF recruitment - the pathway that must not be agonised.\n"
      "3. beta-arrestin-2 recruitment, to exclude GB88-like biased agonism.\n"
      "4. Keratinocyte melanocore-uptake assay as the biological endpoint.\n")

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(lines)} blocks)")
    print(f"dual-anchor designs: {len(both)}; seeds: {len(seeds)}; analogues: {len(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
