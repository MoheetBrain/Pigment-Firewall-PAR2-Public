# Publication validation

The offline original panel-selection script reconstructed all 20 candidates in
the exact saved order, with the same columns and no substantive differences.
Twenty numeric string representations differed only by serialization rounding
within 1e-12. The published original CSV remains byte-for-byte unchanged.
See [PANEL_RECONSTRUCTION.json](PANEL_RECONSTRUCTION.json).

Verification used RDKit 2025.09.6, pandas 3.0.3 and NumPy 2.4.6 in an existing
local environment. This validates the selection transformation of saved inputs,
not the upstream docking, incomplete counter-screen or biological hypothesis.
All 43 selected Python scripts passed syntax parsing. No docking, co-folding,
library screening, molecule purchase or wet-lab experiment was performed.
