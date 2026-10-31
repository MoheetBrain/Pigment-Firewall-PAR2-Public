# Methods and research stages

Authoritative source: `yonsei-final10-v2`, original commit
`eab0369a46cc7b220a7c66fb6a53f02c74568360`.

The campaign generated 15,000 de novo designs, prepared 14,825 and accumulated
21,149 ligand-pocket rows across 19,660 molecules in its recovered combined
screening universe. The recorded accounting distinguishes 16,314 EX8 jobs,
535 EX32 jobs across 513 molecules, and 4,835 real-compound EX8 rows. These are
campaign accounting categories, not interchangeable counts of unique candidates.

The latest sourceability stage froze 32 seeds; a ZINC20 ECFP4 (radius 2,
2048-bit) scan searched 3,971,098 in-stock-labelled molecules. An 80-compound
queue was prepared without the inherited stereochemistry-destroying tautomer
canonicalization. Recorded outputs: 80 prepared, zero failures or lost
stereocentres; 160 ligand-pocket pairs and 1,440 EX32 poses.

GNINA v1.3.3, CPU mode, exhaustiveness 32, nine modes, seed 20260904 and four
CPUs per docking job. The inactive structure boxes were:

| Structure | Centre (Å) | Size (Å) |
|---|---|---|
| 5NDD | 16.974, 5.926, 55.826 | 17.332, 17.672, 13.644 |
| 5NDZ | 1.356, 12.419, 48.400 | 20.702, 17.885, 25.525 |

Reference redocking best RMSDs were 0.6747 Å (5NDD) and 0.8672 Å (5NDZ).
Reference recovery validates those structural controls only. Finalists lack a
completed active-vs-decoy enrichment calibration or functional validation.

`v3_final_sourceable_panel.py` records nominal family weights: interaction geometry
0.30, pose physics 0.15, co-folding 0.20, chemistry 0.10, similarity/provenance 0.10,
actionability 0.15. Actual contributing weights are renormalised over available
families. Diversity selection caps close neighbours and allosteric slots. Read
the exact code and per-row coverage rather than assuming all six families exist.
No reranking or scientific parameter changes were made for publication.

## Actual saved evidence coverage

All 20 sourceable panel rows contribute F1 (geometry), F2 (pose physics), F4
(chemistry) and F6 (actionability), covering 0.70 of nominal family weight.
F3 co-folding and F5 similarity/provenance are empty in every saved row.
In the inherited builder, the F5 input-name list does not include the available
`seed_tanimoto` field. The archived similarities remain visible, but do not
contribute through F5 in this saved panel. This is a documented implementation
limitation, not a silently repaired score or new ranking.
