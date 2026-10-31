#!/usr/bin/env python3
"""Two separate, explicitly-defined anchor metrics for PAR2 pose ranking.

Why this module exists
----------------------
`v2_consensus_rank.py` defines

    PROFILE["5NDD_orthosteric"]["anchor_residues"] = ["ASP:228", "HIS:135"]

while `02_REFERENCE_VALIDATION_REPAIRED/REFERENCE_IFP_REPAIRED.json` records that
the repaired crystallographic AZ8838 complex makes exactly two hydrogen bonds,
both ligand-as-donor, to **TYR:82** (2.475 A, DHA 153.9 deg) and **HIS:135**
(2.630 A, DHA 138.9 deg). `FINAL_CANDIDATES.md` states both facts in adjacent
sentences. One empirical observation was therefore being scored against a
different, unsourced anchor pair.

This module refuses to collapse the two into one number. It exposes:

  CRYSTAL_IFP_RECOVERY   purely empirical. What fraction of the *observed*
                         AZ8838 fingerprint does a pose reproduce? Nothing in it
                         is asserted; every key comes from ProLIF run on the
                         prepared crystal complex.

  DESIGN_HOTSPOT_SCORE   graded satisfaction of pocket positions weighted by how
                         well supported they are as antagonist-design hotspots.
                         The weights and their evidence are stated per residue in
                         HOTSPOTS below, and every residue carries an explicit
                         EvidenceClass. Nothing is weighted on inheritance.

plus one flat term per pocket-lining residue (HBOND_<RES>, CONTACT_<RES>) so no
component is hidden inside a composite.

The protomer caveat, which is the crux of the ASP:228 question
--------------------------------------------------------------
AZ8838's 2-substituted imidazole has two accessible neutral tautomers plus a
pKa-accessible imidazolium. All three were built on identical crystallographic
heavy-atom coordinates and fingerprinted (see PROTOMER_HBONDS). The result:

    N-H on N1  (as built by v2_reference_ifp_repair.py) -> TYR:82 + HIS:135
    N-H on N4  (the other neutral tautomer)             -> ASP:228 + HIS:135
    imidazolium, both N-H (+1)                          -> TYR:82 + ASP:228 + HIS:135

Only HIS:135 is a hydrogen bond under *every* protomer. TYR:82 and ASP:228 swap
according to a hydrogen placement that the crystal structure does not resolve and
that `AssignBondOrdersFromTemplate` fixed arbitrarily by substructure match. The
heavy-atom geometry, which *is* protomer-independent, puts all three inside
hydrogen-bonding distance: 2.475 / 2.630 / 2.929 A.

So the correct reading is neither "ASP:228 is an anchor" nor "ASP:228 is not an
anchor". It is: TYR:82, HIS:135 and ASP:228 are all genuine polar anchor
positions by heavy-atom geometry; which two ProLIF labels as hydrogen bonds is a
modelling choice. The old definition's error was not including ASP:228 -- it was
*excluding TYR:82*, which is the shortest polar contact in the pocket, while
presenting the pair as the crystallographic signature.

No mutagenesis or FMO evidence exists in this repository
-------------------------------------------------------
Searched: every .md/.py/.json/.txt under YONSEI_FINAL10_V2/ and docs/. There is
no alanine-scanning data, no reported Y82A/H135A/F155A/D228A result, and no
fragment-molecular-orbital pair-interaction energy. `HANDOFF_TO_CLAUDE.md` line
429 states plainly: "No molecular dynamics, FMO, P2Rank, fpocket, PocketMiner,
Boltz, Chai, new ML training, or synthesis planning has been performed."

Accordingly no residue in this module is classified MUTAGENESIS_SUPPORTED or
FMO_OR_MODELLING_HOTSPOT on in-repository evidence. If such data is later
obtained, add it through `load_external_evidence()` rather than editing weights
in place, so the provenance of every weight stays visible.

Verification
------------
`python v2_anchor_metrics.py --verify` recomputes the frozen constants below
directly from the prepared structure files and fails loudly on any disagreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Evidence classification
# --------------------------------------------------------------------------- #

#: A = observed as a hydrogen bond in the prepared crystal complex
OBSERVED_CRYSTAL_HBOND = "A_OBSERVED_CRYSTAL_HBOND"
#: B = observed crystal contact that is not a hydrogen bond (VdW / hydrophobic)
OBSERVED_CRYSTAL_NONHBOND_CONTACT = "B_OBSERVED_CRYSTAL_NONHBOND_CONTACT"
#: C = supported by site-directed mutagenesis
MUTAGENESIS_SUPPORTED = "C_MUTAGENESIS_SUPPORTED"
#: D = supported by FMO pair-interaction energies or comparable modelling
FMO_OR_MODELLING_HOTSPOT = "D_FMO_OR_MODELLING_HOTSPOT"
#: E = asserted by the project's design profile with no evidence attached
DESIGN_PRIOR_ONLY = "E_DESIGN_PRIOR_ONLY"

#: Not one of the five task classes; recorded so the audit can name residues
#: that the profile referenced but that are not lining the pocket at all.
NOT_IN_POCKET = "X_NOT_IN_POCKET"

HBOND_INTERACTIONS = frozenset({"HBDonor", "HBAcceptor"})


# --------------------------------------------------------------------------- #
# Frozen empirical constants, recomputed by --verify
# --------------------------------------------------------------------------- #

#: ProLIF 2.2.1 / RDKit 2026.03.6 on
#:   02_REFERENCE_VALIDATION_REPAIRED/5NDD_AZ8838_corrected_protonated_ligand.sdf
#:   02_REFERENCE_VALIDATION_REPAIRED/5NDD_orthosteric_receptor_protonated_pH74.pdb
#: Identical to REFERENCE_IFP_REPAIRED.json -> systems.5NDD_AZ8838
#: .reference_fingerprint_keys, reproduced independently.
CRYSTAL_IFP_KEYS = {
    "5NDD_orthosteric": frozenset({
        "TYR:82:A|HBDonor", "HIS:135:A|HBDonor",
        "ILE:152:A|Hydrophobic", "PHE:155:A|Hydrophobic",
        "CYS:226:A|Hydrophobic", "HIS:227:A|Hydrophobic",
        "TYR:82:A|VdWContact", "LYS:131:A|VdWContact", "TYR:134:A|VdWContact",
        "HIS:135:A|VdWContact", "TRP:141:A|VdWContact", "ILE:152:A|VdWContact",
        "PHE:155:A|VdWContact", "CYS:226:A|VdWContact", "ASP:228:A|VdWContact",
        "ILE:327:A|VdWContact",
    }),
    "5NDZ_allosteric": frozenset({
        "PHE:154:A|Hydrophobic", "ILE:202:A|Hydrophobic",
        "LEU:203:A|Hydrophobic", "TYR:210:A|Hydrophobic",
        "ALA:120:A|VdWContact", "LEU:123:A|VdWContact", "PHE:154:A|VdWContact",
        "ALA:157:A|VdWContact", "CYS:161:A|VdWContact", "TRP:199:A|VdWContact",
        "LEU:203:A|VdWContact", "VAL:211:A|VdWContact",
    }),
}

#: residue -> (distance A, DHA angle deg, donor). Ligand is the donor in both.
CRYSTAL_HBONDS = {
    "5NDD_orthosteric": {
        "TYR:82": (2.475, 153.933, "ligand N1-H ... TYR:82 OH"),
        "HIS:135": (2.630, 138.878, "ligand O2-H ... HIS:135 NE2"),
    },
    "5NDZ_allosteric": {},
}

#: Which residues are hydrogen bonds under which AZ8838 imidazole protomer.
#: All three built on identical crystallographic heavy-atom coordinates.
PROTOMER_HBONDS = {
    "T1_NH_on_N1_as_built": {"TYR:82": (2.475, 153.9), "HIS:135": (2.630, 138.9)},
    "T2_NH_on_N4":          {"ASP:228": (2.929, 140.7), "HIS:135": (2.630, 138.9)},
    "T3_imidazolium_plus1": {"TYR:82": (2.475, 153.9), "ASP:228": (2.929, 140.7),
                             "HIS:135": (2.630, 138.9)},
}

#: residue -> (min heavy-atom distance, n receptor heavy atoms <=4 A,
#:             n <=5 A, min polar-polar heavy-atom distance or None)
#: Computed without ProLIF, straight from coordinates.
CONTACT_SHELL = {
    "5NDD_orthosteric": {
        "TYR:82":  (2.475, 7, 21, 2.475),
        "HIS:135": (2.630, 7, 16, 2.630),
        "ASP:228": (2.929, 5, 20, 2.929),
        "LYS:131": (3.065, 2, 19, 4.067),
        "ILE:327": (3.129, 9, 21, 4.347),
        "PHE:155": (3.138, 12, 30, 9.802),
        "CYS:226": (3.447, 4, 15, 5.944),
        "ILE:152": (3.467, 3, 15, 8.682),
        "HIS:227": (3.500, 14, 46, 4.463),
        "TRP:141": (3.536, 2, 5, 10.741),
        "TYR:134": (3.566, 3, 15, 6.717),
        "TYR:156": (3.604, 2, 6, 7.753),
        "LEU:330": (3.629, 4, 21, 5.036),
        "LEU:78":  (3.850, 2, 9, 7.457),
        "CYS:331": (4.037, 0, 6, 4.577),
        "LEU:151": (4.201, 0, 5, 8.361),
        "TYR:326": (4.810, 0, 1, 5.898),
    },
    "5NDZ_allosteric": {
        "TRP:199": (2.973, 14, 49, 5.848),
        "PHE:154": (3.183, 6, 35, 4.430),
        "LEU:123": (3.197, 8, 28, 4.456),
        "ALA:157": (3.243, 8, 22, 3.881),
        "LEU:203": (3.277, 9, 27, 6.004),
        "TYR:210": (3.393, 8, 41, 3.393),
        "ALA:120": (3.415, 6, 16, 4.120),
        "ASN:158": (3.507, 5, 10, 3.515),
        "VAL:211": (3.543, 2, 6, 5.210),
        "CYS:161": (3.672, 2, 7, 3.773),
        "ILE:202": (4.088, 0, 8, 6.612),
        "SER:124": (4.887, 0, 1, 4.887),
        "LEU:119": (4.922, 0, 2, 4.922),
    },
}

#: Residues referenced by v2_consensus_rank.py's PROFILE that are NOT pocket
#: lining. Min heavy-atom distance to the crystallographic AZ8838 pose.
PROFILE_RESIDUES_OUTSIDE_POCKET = {
    "TYR:311": 8.724,   # contact_residues AND AGONIST_PATTERN
    "LEU:230": 10.645,  # AGONIST_PATTERN
    "TYR:323": 6.703,   # switch_residues
}


# --------------------------------------------------------------------------- #
# DESIGN_HOTSPOT_SCORE weighting
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Hotspot:
    """One weighted pocket position, with the evidence that sets its weight."""
    residue: str
    weight: float
    evidence_classes: tuple[str, ...]
    evidence: str
    #: credit for a hydrogen bond at this position, as a fraction of `weight`
    hbond_credit: float = 1.0
    #: credit for a non-hydrogen-bond contact, as a fraction of `weight`
    contact_credit: float = 0.35

    def score(self, has_hbond: bool, has_contact: bool) -> float:
        if has_hbond:
            return self.weight * self.hbond_credit
        if has_contact:
            return self.weight * self.contact_credit
        return 0.0


# Weighting rationale, stated so it can be argued with rather than inherited.
#
# Tier 1 (1.00) -- hydrogen bond under EVERY accessible protomer of the crystal
#   ligand. Protomer-robust, so it is the one polar requirement that does not
#   depend on a hydrogen we cannot see. HIS:135 alone.
# Tier 2 (0.80) -- hydrogen bond under SOME accessible protomer, with a
#   protomer-independent heavy-atom polar contact <= 3.0 A. TYR:82 and ASP:228
#   are symmetric on this evidence and are therefore weighted equally; the old
#   definition kept one and dropped the other with no stated reason.
# Tier 3 (0.45) -- a defined ProLIF Hydrophobic term in the crystal complex,
#   i.e. shape/lipophilic complementarity rather than a polar requirement.
# Tier 4 (0.20) -- crystal VdWContact only: real, but the least discriminating
#   class, since almost any pose filling the box touches these.
# Tier 0 (0.00) -- named by the design profile but not lining the pocket.
#   Scored at zero rather than silently dropped, so the audit can show them.
#
# Ratios, not absolutes, are what matter: the score is normalised by the total
# available weight, so it always lands in [0, 1].

HOTSPOTS: dict[str, tuple[Hotspot, ...]] = {
    "5NDD_orthosteric": (
        Hotspot("HIS:135", 1.00, (OBSERVED_CRYSTAL_HBOND,),
                "ligand O2-H ... NE2 2.630 A DHA 138.9 deg; the only hydrogen bond "
                "present under all three AZ8838 protomers; NE2 carries no hydrogen "
                "in the prepared receptor so it is an unambiguous acceptor"),
        Hotspot("TYR:82", 0.80, (OBSERVED_CRYSTAL_HBOND,),
                "ligand N1-H ... OH 2.475 A DHA 153.9 deg in the as-built protomer; "
                "shortest polar contact in the pocket; hydrogen bond in T1 and T3 "
                "but not T2"),
        Hotspot("ASP:228", 0.80, (OBSERVED_CRYSTAL_NONHBOND_CONTACT, DESIGN_PRIOR_ONLY),
                "OD1 ... ligand N4 2.929 A, inside hydrogen-bonding distance but "
                "acceptor-acceptor in the as-built protomer, so ProLIF scores only "
                "VdWContact; becomes a hydrogen bond in protomers T2 and T3. Named "
                "an anchor by v2_consensus_rank.py with no evidence recorded"),
        Hotspot("PHE:155", 0.45, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal Hydrophobic 4.396 A and VdWContact 3.138 A; 12 heavy atoms "
                "within 4 A, the densest lipophilic wall in the pocket"),
        Hotspot("HIS:227", 0.45, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal Hydrophobic 3.500 A and 14 heavy atoms within 4 A"),
        Hotspot("ILE:152", 0.45, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal Hydrophobic 4.016 A and VdWContact 2.430 A"),
        Hotspot("CYS:226", 0.45, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal Hydrophobic 4.489 A and VdWContact 2.408 A"),
        Hotspot("LYS:131", 0.20, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 3.065 A; NZ is 4.067 A from the nearest ligand "
                "polar atom, too far to be a polar anchor in this pose"),
        Hotspot("TYR:134", 0.20, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 2.726 A"),
        Hotspot("TRP:141", 0.20, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 2.640 A"),
        Hotspot("ILE:327", 0.20, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 3.129 A"),
        Hotspot("LEU:330", 0.20, (DESIGN_PRIOR_ONLY,),
                "3.629 A from the crystal ligand so genuinely pocket lining, but "
                "ProLIF assigns it no interaction term in the crystal complex; "
                "listed in the profile's contact_residues"),
        Hotspot("TYR:156", 0.20, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "3.604 A from the crystal ligand; no ProLIF term; absent from the "
                "existing profile"),
        Hotspot("LEU:78", 0.20, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "3.850 A from the crystal ligand; no ProLIF term; absent from the "
                "existing profile"),
        Hotspot("TYR:311", 0.00, (NOT_IN_POCKET, DESIGN_PRIOR_ONLY),
                "8.724 A from the crystallographic ligand. No pose docked in the "
                "5NDD box can reach it, yet it sits in contact_residues and so "
                "caps contact_fraction at 6/7 for every design"),
        Hotspot("TYR:323", 0.00, (NOT_IN_POCKET, DESIGN_PRIOR_ONLY),
                "6.703 A from the crystallographic ligand; listed as a TM7 switch "
                "residue"),
        Hotspot("LEU:230", 0.00, (NOT_IN_POCKET, DESIGN_PRIOR_ONLY),
                "10.645 A from the crystallographic ligand; one third of "
                "AGONIST_PATTERN, which therefore can essentially never fire"),
    ),
    "5NDZ_allosteric": (
        Hotspot("TYR:210", 0.60, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal Hydrophobic 4.290 A; OH ... ligand N1 3.393 A is the only "
                "polar approach in this pocket and is NOT a hydrogen bond"),
        Hotspot("TRP:199", 0.55, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 2.763 A; 14 heavy atoms within 4 A"),
        Hotspot("PHE:154", 0.55, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal Hydrophobic 4.491 A and VdWContact 2.400 A"),
        Hotspot("LEU:203", 0.55, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal Hydrophobic 3.602 A and VdWContact 2.592 A"),
        Hotspot("ILE:202", 0.45, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal Hydrophobic 4.293 A"),
        Hotspot("LEU:123", 0.30, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 3.197 A"),
        Hotspot("ALA:157", 0.30, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 3.243 A"),
        Hotspot("ALA:120", 0.30, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 2.704 A"),
        Hotspot("CYS:161", 0.30, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 2.731 A"),
        Hotspot("VAL:211", 0.30, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "crystal VdWContact 2.514 A"),
        Hotspot("ASN:158", 0.20, (OBSERVED_CRYSTAL_NONHBOND_CONTACT,),
                "3.507 A from the crystal ligand; no ProLIF term; absent from the "
                "existing profile"),
        Hotspot("GLY:157", 0.00, (NOT_IN_POCKET, DESIGN_PRIOR_ONLY),
                "the profile lists GLY:157 but residue 157 is ALA in this receptor "
                "file, so the key can never match"),
    ),
}


def load_external_evidence(path: Path) -> dict:
    """Attach mutagenesis / FMO evidence when it exists, without editing weights.

    Expected JSON shape::

        {"5NDD_orthosteric": {"ASP:228": {"weight": 1.0,
                                          "evidence_classes": ["C_MUTAGENESIS_SUPPORTED"],
                                          "evidence": "D228A abolishes ..."}}}

    Returns the parsed mapping and mutates HOTSPOTS in place. Nothing in this
    repository currently supplies such a file; see the module docstring.
    """
    data = json.loads(Path(path).read_text())
    for pocket, entries in data.items():
        current = {h.residue: h for h in HOTSPOTS.get(pocket, ())}
        for residue, spec in entries.items():
            old = current.get(residue)
            current[residue] = Hotspot(
                residue=residue,
                weight=float(spec.get("weight", old.weight if old else 0.0)),
                evidence_classes=tuple(spec.get(
                    "evidence_classes", old.evidence_classes if old else ())),
                evidence=str(spec.get("evidence", old.evidence if old else "")),
                hbond_credit=float(spec.get("hbond_credit",
                                            old.hbond_credit if old else 1.0)),
                contact_credit=float(spec.get("contact_credit",
                                              old.contact_credit if old else 0.35)),
            )
        HOTSPOTS[pocket] = tuple(current.values())
    return data


# --------------------------------------------------------------------------- #
# Pose parsing and the two metrics
# --------------------------------------------------------------------------- #

def parse_ifp_keys(ifp_keys: str | None) -> dict:
    """Split a ';'-joined 'RES:NUM:CHAIN|Interaction' string into usable sets."""
    keys, residues, hbond_residues, by_residue = set(), set(), set(), {}
    for key in (ifp_keys or "").split(";"):
        key = key.strip()
        if not key:
            continue
        keys.add(key)
        residue, _, interaction = key.partition("|")
        base = ":".join(residue.split(":")[:2])
        residues.add(base)
        by_residue.setdefault(base, set()).add(interaction)
        if interaction in HBOND_INTERACTIONS:
            hbond_residues.add(base)
    return dict(keys=keys, residues=residues, hbond_residues=hbond_residues,
                by_residue=by_residue)


def crystal_ifp_recovery(ifp_keys: str | None, pocket: str) -> dict:
    """CRYSTAL_IFP_RECOVERY -- purely empirical, no design prior anywhere in it.

    Fraction of the observed crystal fingerprint keys that the pose reproduces,
    reported whole and split by interaction class. The hydrogen-bond split is
    the discriminating half: in 5NDD only 2 of 16 reference keys are hydrogen
    bonds, so an all-keys recovery figure is dominated by VdW contacts that
    almost any pose filling the box will make.
    """
    reference = CRYSTAL_IFP_KEYS.get(pocket, frozenset())
    parsed = parse_ifp_keys(ifp_keys)
    if not reference:
        return dict(crystal_ifp_recovery=None, crystal_hbond_recovery=None,
                    crystal_hydrophobic_recovery=None, crystal_vdw_recovery=None,
                    crystal_keys_matched=0, crystal_keys_total=0,
                    crystal_keys_missing="")
    shared = parsed["keys"] & reference

    def frac(predicate) -> float | None:
        subset = {k for k in reference if predicate(k)}
        return (len(shared & subset) / len(subset)) if subset else None

    return dict(
        crystal_ifp_recovery=len(shared) / len(reference),
        crystal_hbond_recovery=frac(lambda k: k.split("|")[1] in HBOND_INTERACTIONS),
        crystal_hydrophobic_recovery=frac(lambda k: k.endswith("|Hydrophobic")),
        crystal_vdw_recovery=frac(lambda k: k.endswith("|VdWContact")),
        crystal_keys_matched=len(shared),
        crystal_keys_total=len(reference),
        crystal_keys_missing=";".join(sorted(reference - parsed["keys"])),
    )


def design_hotspot_score(ifp_keys: str | None, pocket: str) -> dict:
    """DESIGN_HOTSPOT_SCORE -- graded satisfaction of weighted pocket positions.

    Independent of CRYSTAL_IFP_RECOVERY in two ways that matter:
      * it credits a hydrogen bond at a position even when the crystal complex
        shows only a VdW contact there (ASP:228), because the heavy-atom
        geometry and the protomer analysis support that position as polar; and
      * it credits a *position*, not a specific ProLIF key, so a pose that
        satisfies TYR:82 as an acceptor rather than a donor still scores.
    """
    spots = HOTSPOTS.get(pocket, ())
    parsed = parse_ifp_keys(ifp_keys)
    total = sum(h.weight for h in spots)
    if not total:
        return dict(design_hotspot_score=None, design_hotspot_hbond_score=None,
                    design_hotspot_satisfied="", design_hotspot_missed="")
    earned, hb_earned, hb_total, satisfied, missed = 0.0, 0.0, 0.0, [], []
    for h in spots:
        has_hb = h.residue in parsed["hbond_residues"]
        has_ct = h.residue in parsed["residues"]
        earned += h.score(has_hb, has_ct)
        if h.weight > 0:
            hb_total += h.weight
            if has_hb:
                hb_earned += h.weight
            (satisfied if has_ct else missed).append(h.residue)
    return dict(
        design_hotspot_score=earned / total,
        design_hotspot_hbond_score=(hb_earned / hb_total) if hb_total else None,
        design_hotspot_satisfied=";".join(satisfied),
        design_hotspot_missed=";".join(missed),
    )


def per_residue_terms(ifp_keys: str | None, pocket: str) -> dict:
    """One HBOND_<RES> and one CONTACT_<RES> flag per pocket-lining residue.

    Nothing here is aggregated. Includes every residue in the 5.0 A contact
    shell plus every residue the existing profile names, so an out-of-pocket
    profile residue is visible as a column of zeros rather than invisible.
    """
    parsed = parse_ifp_keys(ifp_keys)
    residues = set(CONTACT_SHELL.get(pocket, {}))
    residues |= {h.residue for h in HOTSPOTS.get(pocket, ())}
    terms = {}
    for residue in sorted(residues, key=lambda r: (r.split(":")[0], int(r.split(":")[1]))):
        tag = residue.replace(":", "")
        terms[f"HBOND_{tag}"] = int(residue in parsed["hbond_residues"])
        terms[f"CONTACT_{tag}"] = int(residue in parsed["residues"])
    return terms


def score_pose(ifp_keys: str | None, pocket: str, *, per_residue: bool = True) -> dict:
    """Everything above for one pose, as a flat dict."""
    out: dict = {}
    out.update(crystal_ifp_recovery(ifp_keys, pocket))
    out.update(design_hotspot_score(ifp_keys, pocket))
    parsed = parse_ifp_keys(ifp_keys)
    out["n_hbond_residues"] = len(parsed["hbond_residues"])
    out["hbond_residues"] = ";".join(sorted(parsed["hbond_residues"]))
    if per_residue:
        out.update(per_residue_terms(ifp_keys, pocket))
    return out


# --------------------------------------------------------------------------- #
# Anchor-definition recount
# --------------------------------------------------------------------------- #

#: The candidate anchor definitions, evaluated side by side rather than inherited.
ANCHOR_DEFINITIONS = {
    "OLD_ASP228_AND_HIS135_hbond": dict(
        kind="hbond_all", residues=("ASP:228", "HIS:135"),
        description="the definition used by v2_consensus_rank.py and quoted in "
                    "FINAL_CANDIDATES.md as '62 dual-anchor designs'"),
    "CRYSTAL_TYR82_AND_HIS135_hbond": dict(
        kind="hbond_all", residues=("TYR:82", "HIS:135"),
        description="the two hydrogen bonds actually observed in the repaired "
                    "crystal complex"),
    "ANY_TWO_OF_TYR82_HIS135_ASP228_hbond": dict(
        kind="hbond_atleast", residues=("TYR:82", "HIS:135", "ASP:228"), n=2,
        description="protomer-robust: any two of the three polar anchor positions"),
    "ALL_THREE_TYR82_HIS135_ASP228_hbond": dict(
        kind="hbond_atleast", residues=("TYR:82", "HIS:135", "ASP:228"), n=3,
        description="all three polar positions hydrogen-bonded, as in the "
                    "imidazolium protomer T3"),
    "HIS135_hbond_only": dict(
        kind="hbond_all", residues=("HIS:135",),
        description="the single protomer-robust requirement"),
    "TYR82_hbond_only": dict(kind="hbond_all", residues=("TYR:82",),
                             description="single anchor"),
    "ASP228_hbond_only": dict(kind="hbond_all", residues=("ASP:228",),
                              description="single anchor"),
    "ANY_ONE_OF_TYR82_HIS135_ASP228_hbond": dict(
        kind="hbond_atleast", residues=("TYR:82", "HIS:135", "ASP:228"), n=1,
        description="the 'at least one anchor' population"),
}

#: CRYSTAL_IFP_RECOVERY thresholds evaluated in the same recount.
RECOVERY_THRESHOLDS = (0.50, 0.625, 0.75, 0.875, 1.00)


def evaluate_definition(parsed: dict, spec: dict) -> bool:
    residues = spec["residues"]
    if spec["kind"] == "hbond_all":
        return all(r in parsed["hbond_residues"] for r in residues)
    if spec["kind"] == "hbond_atleast":
        return sum(r in parsed["hbond_residues"] for r in residues) >= spec["n"]
    raise ValueError(f"unknown definition kind {spec['kind']!r}")


# --------------------------------------------------------------------------- #
# Verification against the prepared structure files
# --------------------------------------------------------------------------- #

def verify(root: Path) -> int:
    """Recompute the frozen constants from the structures. Non-zero on mismatch."""
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    import prolif as plf

    v2 = root / "YONSEI_FINAL10_V2"
    repaired = v2 / "02_REFERENCE_VALIDATION_REPAIRED"
    systems = {
        "5NDD_orthosteric": (repaired / "5NDD_AZ8838_corrected_protonated_ligand.sdf",
                             repaired / "5NDD_orthosteric_receptor_protonated_pH74.pdb"),
        "5NDZ_allosteric": (repaired / "5NDZ_AZ3451_corrected_protonated_ligand.sdf",
                            repaired / "5NDZ_allosteric_receptor_protonated_pH74.pdb"),
    }
    failures: list[str] = []
    print(f"prolif {plf.__version__} / rdkit {Chem.rdBase.rdkitVersion}")

    for pocket, (lig_path, rec_path) in systems.items():
        lig = Chem.MolFromMolFile(str(lig_path), removeHs=False)
        rec = Chem.MolFromPDBFile(str(rec_path), removeHs=False, sanitize=False)
        if lig is None or rec is None:
            failures.append(f"{pocket}: cannot read prepared structures")
            continue
        receptor = plf.Molecule.from_rdkit(rec)
        probe = plf.Molecule.from_rdkit(lig, resname="LIG", resnumber=1, chain="A")
        ifp = plf.Fingerprint().generate(probe, receptor, metadata=True)
        keys, hbonds = set(), {}
        for pair, interactions in ifp.items():
            residue = pair[1]
            label = f"{residue.name}:{residue.number}:{residue.chain or '-'}"
            for name, metas in interactions.items():
                keys.add(f"{label}|{name}")
                if name in HBOND_INTERACTIONS:
                    meta = metas[0] if isinstance(metas, (list, tuple)) else metas
                    hbonds[":".join(label.split(":")[:2])] = (
                        round(float(meta["distance"]), 3),
                        round(float(meta["DHA_angle"]), 3))
        if keys != set(CRYSTAL_IFP_KEYS[pocket]):
            failures.append(
                f"{pocket}: CRYSTAL_IFP_KEYS mismatch; "
                f"only in file {sorted(keys - set(CRYSTAL_IFP_KEYS[pocket]))}; "
                f"only in module {sorted(set(CRYSTAL_IFP_KEYS[pocket]) - keys)}")
        else:
            print(f"  {pocket}: {len(keys)} fingerprint keys reproduce exactly")
        for residue, expected in CRYSTAL_HBONDS[pocket].items():
            dist, angle = expected[0], expected[1]
            got = hbonds.get(residue)
            if got is None:
                failures.append(f"{pocket}: expected hydrogen bond to {residue}, none found")
            elif abs(got[0] - dist) > 0.01 or abs(got[1] - angle) > 0.5:
                failures.append(f"{pocket}: {residue} hydrogen bond {got} != {(dist, angle)}")
        extra = set(hbonds) - set(CRYSTAL_HBONDS[pocket])
        if extra:
            failures.append(f"{pocket}: unexpected hydrogen bonds {sorted(extra)}")
        if not failures:
            for residue, expected in CRYSTAL_HBONDS[pocket].items():
                print(f"    hydrogen bond {residue}: {expected[0]} A, "
                      f"DHA {expected[1]} deg  OK  [{expected[2]}]")

        # contact shell, ProLIF-independent
        conf = lig.GetConformer().GetPositions()
        heavy = [i for i in range(lig.GetNumAtoms())
                 if lig.GetAtomWithIdx(i).GetSymbol() != "H"]
        lxyz = np.array([conf[i] for i in heavy])
        per_res: dict[str, list] = {}
        for line in rec_path.read_text().splitlines():
            if line[:6] not in ("ATOM  ", "HETATM"):
                continue
            element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
            if element in ("H", "D"):
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            per_res.setdefault(
                f"{line[17:20].strip()}:{line[22:26].strip()}", []).append(xyz)
        recomputed = {}
        for residue, xyzs in per_res.items():
            d = np.linalg.norm(np.array(xyzs)[:, None, :] - lxyz[None, :, :], axis=2)
            if d.min() <= 5.0:
                recomputed[residue] = round(float(d.min()), 3)
        frozen = {r: v[0] for r, v in CONTACT_SHELL[pocket].items()}
        if set(recomputed) != set(frozen):
            failures.append(
                f"{pocket}: CONTACT_SHELL membership mismatch; "
                f"only in file {sorted(set(recomputed) - set(frozen))}; "
                f"only in module {sorted(set(frozen) - set(recomputed))}")
        else:
            bad = {r: (recomputed[r], frozen[r]) for r in frozen
                   if abs(recomputed[r] - frozen[r]) > 0.005}
            if bad:
                failures.append(f"{pocket}: CONTACT_SHELL distances differ {bad}")
            else:
                print(f"  {pocket}: {len(frozen)} residues in the 5.0 A shell "
                      "reproduce exactly")

    print("VERIFY: " + ("PASS" if not failures else "FAIL"))
    for f in failures:
        print("  FAILURE:", f)
    return 0 if not failures else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--verify", action="store_true",
                        help="recompute the frozen constants from the structures")
    parser.add_argument("--consensus", type=Path, default=None,
                        help="DENOVO_EX8_CONSENSUS_ALL.csv to recount")
    parser.add_argument("--pocket", default="5NDD_orthosteric")
    parser.add_argument("--external-evidence", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.external_evidence:
        load_external_evidence(args.external_evidence)
        print(f"[evidence] applied {args.external_evidence}")

    if args.verify:
        return verify(args.root.resolve())

    if not args.consensus:
        parser.error("nothing to do: pass --verify or --consensus")

    import pandas as pd

    frame = pd.read_csv(args.consensus, usecols=["cand_id", "pocket", "ifp_keys",
                                                 "consensus_rank", "max_ifp_recovery"])
    sub = frame[frame.pocket == args.pocket].copy()
    print(f"[recount] {len(frame)} rows; {len(sub)} in {args.pocket}")

    import numpy as np

    parsed = [parse_ifp_keys(k) for k in sub.ifp_keys.fillna("")]
    recovery = np.array([crystal_ifp_recovery(k, args.pocket)["crystal_ifp_recovery"]
                         for k in sub.ifp_keys.fillna("")], dtype=float)
    rank = sub.consensus_rank.to_numpy()

    # cross-check: this module's recovery must equal the stored max_ifp_recovery,
    # since both are |pose keys & reference keys| / |reference keys|.
    stored = sub.max_ifp_recovery.to_numpy(dtype=float)
    finite = ~np.isnan(stored) & ~np.isnan(recovery)
    max_delta = float(np.abs(stored[finite] - recovery[finite]).max()) if finite.any() else 0.0
    print(f"[crosscheck] CRYSTAL_IFP_RECOVERY vs stored max_ifp_recovery over "
          f"{int(finite.sum())} rows: max |delta| = {max_delta:.2e} "
          f"({'AGREE' if max_delta < 1e-9 else 'DISAGREE'})")

    masks: dict[str, np.ndarray] = {}
    rows = []

    def add(name, kind, residues, mask, description):
        masks[name] = mask
        n = int(mask.sum())
        ranks = rank[mask]
        rows.append(dict(
            definition=name, kind=kind, residues=residues,
            n_designs=n, n_evaluated=len(sub),
            fraction=round(n / len(sub), 6) if len(sub) else None,
            median_consensus_rank=int(np.median(ranks)) if n else None,
            best_consensus_rank=int(ranks.min()) if n else None,
            n_in_consensus_top500=int((ranks <= 500).sum()),
            n_in_consensus_top1000=int((ranks <= 1000).sum()),
            median_crystal_ifp_recovery=(round(float(np.nanmedian(recovery[mask])), 4)
                                         if n else None),
            description=description))

    for name, spec in ANCHOR_DEFINITIONS.items():
        add(name, spec["kind"], ";".join(spec["residues"]),
            np.array([evaluate_definition(p, spec) for p in parsed]),
            spec["description"])
    for threshold in RECOVERY_THRESHOLDS:
        add(f"CRYSTAL_IFP_RECOVERY_ge_{threshold:g}", "recovery_threshold", "",
            np.nan_to_num(recovery, nan=-1.0) >= threshold,
            f"pose reproduces >= {threshold:g} of the "
            f"{len(CRYSTAL_IFP_KEYS[args.pocket])} observed crystal fingerprint keys")

    # How much of each population the OLD definition actually captured.
    old = masks["OLD_ASP228_AND_HIS135_hbond"]
    for row in rows:
        mask = masks[row["definition"]]
        both = int((mask & old).sum())
        row["overlap_with_OLD_definition"] = both
        row["n_missed_by_OLD_definition"] = int(mask.sum()) - both
        row["jaccard_vs_OLD_definition"] = (
            round(both / int((mask | old).sum()), 4) if (mask | old).any() else None)

    audit = pd.DataFrame(rows)
    print(audit[["definition", "n_designs", "fraction", "overlap_with_OLD_definition",
                 "n_missed_by_OLD_definition", "n_in_consensus_top500"]]
          .to_string(index=False))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.out / "ANCHOR_DEFINITION_AUDIT.csv", index=False)
        print(f"wrote {args.out / 'ANCHOR_DEFINITION_AUDIT.csv'}")

        terms = pd.DataFrame([score_pose(k, args.pocket)
                              for k in sub.ifp_keys.fillna("")])
        terms.insert(0, "pocket", sub.pocket.to_numpy())
        terms.insert(0, "cand_id", sub.cand_id.to_numpy())
        terms.insert(2, "consensus_rank", rank)
        for name, mask in masks.items():
            terms[f"DEF_{name}"] = mask.astype(int)
        terms.to_csv(args.out / "ANCHOR_PER_POSE_TERMS.csv", index=False)
        print(f"wrote {args.out / 'ANCHOR_PER_POSE_TERMS.csv'} "
              f"({len(terms)} rows, {len(terms.columns)} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
