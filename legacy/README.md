# Pre-hardening protocol code

`secagg.py` and `fl.py` in this folder are the exact files that produced the 1,802 results of the main
grids (their SHA-256 values are stored in every result file). The current versions add party-local
verdicts, authenticated attempt and set messages and a pinned session configuration. These changes do
not alter the aggregate, the accepted set or the trained model for the same inputs and the same sketch
seed. `equivalence_check.py` in the repository root tests this by running both versions side by side.
