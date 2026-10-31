# PAR2 — Computational Candidate Discovery

**Can structural modelling identify plausible PAR2 modulators worth testing in pigmentation biology—and can the proposed molecules actually be sourced?**

This independent research combines molecular generation, docking, interaction
analysis, chemical diversity and sourceability checks. It preserves an ambitious
campaign and its failures: the generated chemistry had almost no close in-stock
neighbours, so the latest completed panel became largely a repurposing panel.

**No candidate has been demonstrated to bind or inhibit PAR2. No pigmentation
benefit, cosmetic safety or therapeutic efficacy is established.**

[Exact latest panel, structures and identifiers](results/sourceable-2026-09-06/FINAL_SOURCEABLE_PAR2_TOP20.csv) ·
[Candidate evidence](docs/CANDIDATE_EVIDENCE.md) · [Methods](docs/METHODOLOGY.md) · [Critical limitations](docs/LIMITATIONS.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md)

## Biological question

PAR2 is a protease-activated receptor with multiple signalling pathways. The
project tests the hypothesis that selective modulation might affect keratinocyte
melanocore uptake. **The selective-antagonism → uptake link remains unverified.**
Docking cannot distinguish antagonism from biased agonism. The critical proposed
counter-screen is **G12/13 → RhoA**: a candidate could inhibit one pathway while
activating another. Functional assays would have to resolve that risk.

Experimental receptor structures [5NDD](https://www.rcsb.org/structure/5NDD) and
[5NDZ](https://www.rcsb.org/structure/5NDZ) define the orthosteric and allosteric
lanes. Structure recovery supports a structural method; it does not validate
new candidate pharmacology.

## Research sequence

```text
15,000 de novo designs → ligand preparation and stereo audit
    → orthosteric/allosteric GNINA screens → interaction/diversity selection
    → earlier final-20 panel → sourceability and analogue search
    → corrected finalist preparation → latest sourceable final-20 panel
```

The earlier September 5 panel (17 orthosteric / 3 allosteric) and the September 6
sourceability panel (16 / 4) answer different questions. Their original CSVs
are retained separately. Original ordering, molecules and scores are unchanged.
Known earlier stereochemistry/provenance failures are documented below; the
older panel must not be used as an unqualified synthesis list.

## Latest completed panel — September 6, 2026

**20 candidates, 20 unique Murcko scaffolds; 16 orthosteric / 4 allosteric.**
The preserved CSV supplies exact SMILES, InChIKeys, matched PubChem CIDs, source
lanes, physicochemistry, evidence-family scores and sourceability fields.

| Rank | Candidate | ID | Pocket | Vina | CNN score | CNN affinity |
|---:|---|---|---|---:|---:|---:|
| 1 | TORSEMIDE | RX03235 | 5NDD | -7.76 | 0.572 | 7.823 |
| 2 | EFINACONAZOLE | RX03232 | 5NDD | -7.33 | 0.214 | 7.370 |
| 3 | RESATORVID | RX02731 | 5NDD | -5.16 | 0.733 | 7.928 |
| 4 | OXYPHENCYCLIMINE | RX01016 | 5NDD | -6.03 | 0.726 | 7.884 |
| 5 | BENFLUOREX | RX00634 | 5NDZ | -7.46 | 0.785 | 6.660 |
| 6 | PROTOKYLOL HYDROCHLORIDE | RX02811 | 5NDZ | -6.56 | 0.748 | 6.507 |
| 7 | N-DESETHYLOXYBUTYNIN | RX04676 | 5NDD | -7.73 | 0.651 | 7.616 |
| 8 | DIBUCAINE HYDROCHLORIDE | RX02098 | 5NDD | -6.66 | 0.470 | 7.511 |
| 9 | CYCLOPENTOLATE HYDROCHLORIDE | RX04314 | 5NDD | -7.09 | 0.716 | 7.656 |
| 10 | PENBUTOLOL SULFATE | RX01116 | 5NDD | -7.72 | 0.772 | 7.634 |
| 11 | INDECAINIDE HYDROCHLORIDE | RX03388 | 5NDD | -5.31 | 0.493 | 7.207 |
| 12 | LABETALOL | RX04617 | 5NDZ | -6.64 | 0.686 | 6.497 |
| 13 | MEPENZOLATE | RX01461 | 5NDD | -5.30 | 0.563 | 7.582 |
| 14 | SAFINAMIDE | RX03216 | 5NDD | -7.29 | 0.311 | 7.162 |
| 15 | GLAFENINE | RX01596 | 5NDD | -4.64 | 0.351 | 7.108 |
| 16 | ZINC candidate | ZI54096226 | 5NDZ | -6.68 | 0.709 | 6.536 |
| 17 | DESIDUSTAT | RX01997 | 5NDD | -9.14 | 0.597 | 7.885 |
| 18 | PHENOTHRIN | RX04553 | 5NDD | -5.60 | 0.520 | 7.523 |
| 19 | REBOXETINE | RX00572 | 5NDD | -6.76 | 0.697 | 7.461 |
| 20 | GLUFOSFAMIDE | RX03913 | 5NDD | -6.96 | 0.592 | 7.579 |

Vina is a docking score; CNN quantities are ranking features, not measured
binding affinities or probabilities of antagonism. Raw scores from different
pockets are not directly comparable. Selection standardized evidence within
pocket, compressed correlated features into families, and imposed diversity.
The saved provenance reports a median of **four** contributing evidence families.
The actual families are geometry, pose physics, chemistry and actionability
(70% nominal weight); co-folding and similarity/provenance families are absent.
The inherited F5 input mapping misses `seed_tanimoto`; see the
[coverage audit](docs/METHODOLOGY.md#actual-saved-evidence-coverage). Missing
counter-screen data were not established by the panel.

The historical `APPROVED_REPURPOSING` tag applies to 19 rows but is **not verified
regulatory approval**. Resatorvid, for example, was clinical-stage rather than
approved. The twentieth row is `ZINC_IN_STOCK_UNSEEN`. Candidate selection means
computational prioritisation, not endorsement for human use.

## The informative negative result

A search of **3,971,098 ZINC20 in-stock-labelled molecules** against 32 frozen
de novo seeds found only **seven** hits at ECFP4 Tanimoto ≥0.50, best **0.556**,
from **two** seeds; none reached 0.60. At a 0.35 floor, 672 weak neighbours
covered 17 seeds. A positive-control search returned 362 hits including an exact
match, showing that the search could retrieve ordinary chemical neighbours.

The latest panel is therefore **largely repurposing, not a direct de novo
analogue panel**. The source records' best seed similarities span 0.154–0.397.
The official Enamine screening catalogue was not searched in this later stage;
ZINC20 was the labelled substitute. Bulk supplier databases are excluded here.

Every final row has a historical `IN_STOCK_VENDOR_LISTED` label and an exact
PubChem InChIKey match, with **5–96** vendor leads (median **36.5**). These are
sourceability leads, not confirmed stock, price, purity, quantity, lead time or
an orderable quote. None of those commercial fields is verified by this release.

## What failed or remains missing

- Earlier tautomer canonicalization damaged stereochemistry. Corrected finalist
  preparation reported 80/80 prepared and no lost stereocentres; the inherited
  buggy script is retained as historical evidence, not the recommended route.
- Earlier real-compound labels incorrectly said de novo. Two earlier finalists
  had structure/provenance mismatches. Their original rows remain unchanged and
  explicitly superseded for sourceability purposes.
- The interaction receptor preparation could not detect salt bridges. High IFP
  recovery can reflect pocket filling; the decoy null did not finish.
- The four allosteric finalists have weaker hotspot support (0.32–0.37 versus
  0.88–0.95 in the orthosteric lane). They remain in the original ordering.
- Finalist co-folding, active/decoy enrichment and active-state counter-screen
  deliverables were incomplete. The CSV's `active_state_warning` repeats stock
  classification through an export fallback; **it is not an active-state test**.
- The founding “~80% melanin-uptake reduction” claim could not be verified and is
  not presented as evidence. RWJ-50353 is a thrombin inhibitor, which complicates
  earlier biological interpretations.

## Inspect and reproduce

The public release contains exact final panels, original selection/preparation
and analysis scripts, portable methods and verification hashes. Large molecular
libraries, remote orchestration, machine logs and private correspondence are
excluded. [Reproducibility](docs/REPRODUCIBILITY.md) distinguishes file verification
from full screen regeneration and documents the inputs to obtain independently.
No molecular screen, cloud job or experiment was restarted for publication.

Future validation would require confirmed material identity, direct binding,
pathway-resolved functional assays, a G12/13–RhoA counter-screen, and a controlled
melanocore-uptake experiment. These are proposed tests, not completed results.

## Citation and reuse

Independent research by [@MoheetBrain](https://github.com/MoheetBrain).
[CITATION.cff](CITATION.cff) · [MIT software licence](LICENSE) ·
[Research/data terms and third-party attribution](DATA_LICENSE.md).
Original research is CC BY 4.0; third-party structures and software keep their
own rights. Forks and pull requests are welcome; the maintainer controls merges.
