# Changelog

## 1.2.1 — 18 October 2026 (repository release)

- Complete re-release of 1.2.0. The v1.2.0 release on GitHub/Zenodo (doi:10.5281/zenodo.23451665) lacked `figure1_admissibility_regions.py` and `Figure1_admissibility_regions.png`, and its archived outputs were not inside `results/`, so `SHA256SUMS.txt` could not be verified. No code or output was changed; the script reports version 1.2.0.

## 1.2.0 — 18 October 2026 (second revision of the manuscript)

- `table1_power_weight_thresholds` now also reproduces the numbers of Remark 4.7: the deviations of the elementary bounds of Corollary 4.6 from λ₁ (3.9% to 5.6% below, 6.4% to 9.0% above; both decrease in γ on a 400-point grid) and the share of the gap between 1/√H₁ and λ₁ closed by the weighted Hilbert–Schmidt bound (about 69% as γ → 1⁺, 52%, 30% and 9% at γ = 1.25, 1.5 and 1.75, 0% at γ = 2). Three checks were added (18 in total).
- Logs of a full and a quick run in a clean environment, the reference report and the checksum list were regenerated.
- No other computation was changed; the coordinate data of Figures 1–5 and all other tabulated values are unchanged.

## 1.1.1 — 16 October 2026

- Same code as 1.1.0, released with the run logs and output files; archived at doi:10.5281/zenodo.23412862.

## 1.1.0 — 15 October 2026 (revised manuscript)

- Equation references updated to the numbering of the revised manuscript, in the code, the printed report headings and this README: conformable problem (13) (was "(10)"), sequential problem (15) (was "(12)"), Examples (19)–(20) (was "(16)–(17)"), sine problem (22), amplitude prediction (23) (was "(20)").
- New script `figure1_admissibility_regions.py`, which redraws Figure 1 with Matplotlib from the closed-form thresholds and compares the result with `results/figure1_coordinates.txt`.
- `check` lines now print the absolute and relative differences and the absolute tolerance; the tolerance for the T₃ trace check was tightened from 1e-5 to 1e-7.
- The report now records the script version, the SciPy version, the date (UTC), the platform, the run mode, the time taken by each item and a summary of all checks.
- The README now documents the tolerances, the tested environment, the computational cost and which outputs are regenerated; archived full and quick run logs and a SHA-256 checksum list were added.
- No numerical method or parameter was changed; the coordinate data of Figures 1–5 and all tabulated values are unchanged (apart from round-off-level digits in the last column of Table 3, see README).

## 1.0.1 — 30 September 2026 (original submission)

- Archived at doi:10.5281/zenodo.23056907.
