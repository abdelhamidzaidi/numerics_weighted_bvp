# Changelog

## 1.1.0 — 15 October 2026 (revised manuscript)

- Equation references updated to the numbering of the revised manuscript, in the code, the printed report headings and this README: conformable problem (13) (was "(10)"), sequential problem (15) (was "(12)"), Examples (19)–(20) (was "(16)–(17)"), sine problem (22), amplitude prediction (23) (was "(20)").
- New script `figure1_admissibility_regions.py`, which redraws Figure 1 with Matplotlib from the closed-form thresholds and compares the result with `results/figure1_coordinates.txt`.
- `check` lines now print the absolute and relative differences and the absolute tolerance; the tolerance for the T₃ trace check was tightened from 1e-5 to 1e-7.
- The report now records the script version, the SciPy version, the date (UTC), the platform, the run mode, the time taken by each item and a summary of all checks.
- The README now documents the tolerances, the tested environment, the computational cost and which outputs are regenerated; archived full and quick run logs and a SHA-256 checksum list were added.
- No numerical method or parameter was changed; the coordinate data of Figures 1–5 and all tabulated values are unchanged (apart from round-off-level digits in the last column of Table 3, see README).

## 1.0.1 — 30 September 2026 (original submission)

- Archived at doi:10.5281/zenodo.23056907.
