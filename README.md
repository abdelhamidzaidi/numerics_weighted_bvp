# Code S1 — `numerics_weighted_bvp` (version 1.1.0)

Supplementary code for

> A. T. Zaidi, *Optimal Lipschitz Thresholds for Weighted Dirichlet Problems, with an Explicit Bessel-Zero Bound for Power Weights*, *Axioms* (MDPI), manuscript axioms-4631444 (revised version, October 2026).

The main script recomputes every number, table and plotted curve of the paper from scratch: no stored data are read at any stage. It needs Python 3.9 or later with NumPy, SciPy and mpmath. The optional script `figure1_admissibility_regions.py` also needs Matplotlib.

Equation, theorem, table and figure numbers in this file and in the code refer to the **revised manuscript**: problem (1) is the weighted problem, (13) the conformable problem of order γ, (15) the sequential problem, (19)–(20) Examples 6.1–6.2, (22) the sine problem and (23) the amplitude prediction.

## Quick start

```bash
pip install -r requirements.txt
python numerics_weighted_bvp.py            # full run: all tables, figures and checks
python numerics_weighted_bvp.py --quick    # same, with a coarser shooting scan for Figure 5
python numerics_weighted_bvp.py --only table3 figure3
python numerics_weighted_bvp.py --list     # which function produces which result
python figure1_admissibility_regions.py    # Figure 1 redrawn with Matplotlib
```

Output goes to the folder `results/`:

- `report.txt` contains all tables, checked constants and a summary (number of checks, time per item). The first two lines record the script version, the Python/NumPy/SciPy/mpmath versions, the date (UTC), the platform and the run mode.
- `figure1_coordinates.txt` … `figure5_coordinates.txt` contain every plotted curve as a `pgfplots` `coordinates {...}` list; these are the lists used in the LaTeX source of the paper.

`reference_report.txt` is the report of the full run described below, for comparison.

## What is regenerated

Every run regenerates **all** outputs from the underlying computations:

| Output | How it is computed in every run |
|---|---|
| Tables 1, 2 | closed forms (Bessel zeros with mpmath at 30 digits) |
| Table 3 | Nyström discretization with N = 400, 800, 1600 and Richardson extrapolation |
| Table 4 | successive approximations on the Nyström grid (N = 1600, tolerance 1e-12) |
| Figures 1, 2, 4 | closed forms evaluated on the plotted grids |
| Figure 3, Examples 6.1–6.2 | successive approximations (N = 1600 and N = 3200 for mesh doubling) and an independent shooting check |
| Figure 5, Section 7.4 | shooting with DOP853 (rtol 1e-12) and Brent's method over p = u′(0⁺) ∈ [1e-4, 300], 1500 scan points (300 with `--quick`) |

The `--quick` option changes only the density of the shooting scan behind Figure 5; in the tested run it produced the same Figure 5 coordinates as the full run.

## Tested environment and computational cost

The archived logs `results/run_log_full_2026-10-15.txt` and `results/run_log_quick_2026-10-15.txt` come from a clean virtual environment created from `requirements.txt` (exact package versions in `results/environment_tested.txt`): Python 3.13, NumPy 2.5.4, SciPy 1.18.1, mpmath 1.4.1, Matplotlib 3.11.2, Linux x86-64.

| Item | Full run | `--quick` |
|---|---|---|
| Tables 1–4, Figures 1–4, checks | about 10 s together | about 10 s together |
| Figure 5 (shooting scans) | about 360 s | about 80 s |
| Total | about 6–7 min | about 1.5 min |

Almost all the cost is the shooting scan of Figure 5. Times depend on the machine and on the package versions, and are indicative only.

In this clean run, all 15 checks returned OK, and the data of the five coordinate files are identical to those used for Figures 1–5 of the manuscript (only the comment lines naming each curve were updated to the revised equation numbers).

## Numerical checks and their tolerances

Each `check` line compares a recomputed value with the value printed in the paper and reports **OK when |computed − reference| ≤ tol**, where tol is an **absolute** tolerance chosen per check. The next line of the report prints the absolute and relative differences, so the margin is always visible. An OK line therefore means agreement within a stated tolerance, not an exact identity.

| Check | Method | tol (absolute) | |diff| in the tested run |
|---|---|---|---|
| Table 1: λ₁ and 1/√T₂ at γ = 1.25 | closed form vs printed 5 decimals | 6e-6 | 4.3e-6, 2.6e-6 |
| Remark 4.9: minimum ratio | bounded minimization vs printed 3 decimals | 6e-4 | 1.2e-4 |
| Table 2: H₂^(−1/2) at δ = 0.4 | closed form vs printed 5 decimals | 6e-6 | 4.1e-6 |
| Table 3: Nyström value at γ = 1.25 | recomputed matrix eigenvalue vs printed 12 digits | 1e-11 | 8.9e-16 |
| Examples 6.1, 6.2: max u | successive approximations, N = 1600, vs printed 4 decimals | 6e-5 | 2.8e-5, 4.1e-5 |
| Figure 5: amplitude at K/λ₁ = 2 | shooting vs printed 5 decimals | 6e-5 | 1.2e-6 |
| Prop. 4.4, Cor. 4.6, Prop. 5.6: H₁, T₂, H₂ | adaptive quadrature (SciPy `quad`, limit 200, kink as break point) vs closed form | 1e-7 | 1.4e-13 to 2.3e-11 |
| Cor. 4.6: T₃ | trace(M³) of the N = 1600 Nyström matrix (error O(N⁻²)) vs closed form | 1e-7 | 3.4e-8 (γ = 1.25), 2.4e-9 (γ = 1.75) |

For values printed with a fixed number of decimals, tol is about half a unit in the last printed digit. For T₃ the comparison is with a discretized trace, so the agreement is limited by the O(N⁻²) error of the Nyström scheme (relative difference about 4e-6 at γ = 1.25).

The "after extrapolation" column of Table 3 is at round-off level (10⁻¹³ to 10⁻¹⁶). Its last digits depend on the machine and the BLAS library; in the tested run the entry for γ = 1.25 was 9.0e-16 instead of the printed 3.2e-15. Both are at round-off level, and no conclusion depends on these digits.

## Reproducibility of results

This section mirrors Appendix B of the paper. The **Key** column is the argument for `--only`.

| Result in the paper | Function | Key | Output |
|---|---|---|---|
| Table 1 (thresholds for the power weight); Remark 4.9 (Lyapunov ratios) | `table1_power_weight_thresholds` | `table1` | `report.txt` |
| Table 2 (sequential problem (15)); Proposition 5.7(b), δ* | `table2_sequential_thresholds` | `table2` | `report.txt` |
| Table 3 (Bessel formula vs Nyström, Richardson extrapolation) | `table3_nystrom_check` | `table3` | `report.txt` |
| Table 4 (successive approximations, rates, iteration counts) | `table4_contraction` | `table4` | `report.txt` |
| Figure 1 (admissibility regions, problem (13)) | `figure1_regions_cf` | `figure1` | `figure1_coordinates.txt` |
| Figure 1, independent Matplotlib rendering | `figure1_admissibility_regions.py` | — | `Figure1_admissibility_regions.pdf`, `.png` |
| Figure 2 (admissibility regions, problem (15)) | `figure2_regions_scf` | `figure2` | `figure2_coordinates.txt` |
| Figure 3; Examples 6.1 and 6.2, problems (19)–(20) (solutions, iteration counts, mesh doubling, shooting cross-check) | `figure3_examples` | `figure3` | `figure3_coordinates.txt`, `report.txt` |
| Figure 4; Proposition 7.1 (gain factors, monotonicity on 400 points) | `figure4_gain_factors` | `figure4` | `figure4_coordinates.txt`, `report.txt` |
| Figure 5; Section 7.4, problem (22) (constants C_γ of (23), shooting scans, bifurcation branch) | `figure5_bifurcation` | `figure5` | `figure5_coordinates.txt`, `report.txt` |
| Propositions 4.4, 5.6 and Corollary 4.6 (independent quadrature checks) | `checks_closed_forms` | `checks` | `report.txt` |

### Building blocks

| Function | Implements |
|---|---|
| `lambda1_power` | Theorem 4.2, λ₁ = γ² j²_{1/γ,1} / (4L^γ) |
| `V1`, `H1` | supremum-norm constant and Proposition 4.4 |
| `T2`, `T3` | spectral sums of Corollary 4.6 |
| `lyapunov_bound` | Lyapunov-type bound of [10] (Remark 4.9) |
| `V2`, `H2`, `sharp_sequential` | constants of the sequential problem (Theorem 5.5, Proposition 5.6) |
| `nystrom_power` | Nyström scheme of Section 7.1 (substitution s = y^{1/(γ−1)}, trapezoidal rule) |
| `lambda1_nystrom` | reciprocal of the largest eigenvalue of the Nyström matrix |
| `sequential_grid` | Nyström scheme for the sequential problem through Lemma 5.4 |
| `picard` | successive approximations u_{n+1} = Pu_n (Theorem 3.3), with observed rate |
| `shoot_power` | shooting in τ = t^{γ−1}, where the equation is regular (DOP853, rtol 10⁻¹²) |
| `C_gamma` | coefficient C_γ of the amplitude prediction (23) |

## Notes for reviewers

- **How the figures are drawn.** Figures 1–5 of the paper are drawn in LaTeX with `pgfplots` (version 1.18) from the coordinate lists in `results/figureN_coordinates.txt`; nothing is edited by hand. `figure1_admissibility_regions.py` (fully commented) redraws Figure 1 with Matplotlib directly from the closed-form thresholds (Theorem 4.2, Proposition 4.4 and the supremum-norm constant of [5]) and prints the largest deviation from the plotted coordinates (below 5e-5, the rounding of the four printed decimals). A ready-made copy is `Figure1_admissibility_regions.pdf`.
- **Setting.** All computations use a = 0 and b = L = 1, as in Section 7. Thresholds for general L follow on multiplying by L^{−γ}, respectively L^{−2δ}.
- **Integrity.** `SHA256SUMS.txt` lists a SHA-256 checksum for every file of this archive (`sha256sum -c SHA256SUMS.txt`).

## Versions

See `CHANGELOG.md`. Version 1.0.1 (30 September 2026) accompanies the original submission and is archived at doi:10.5281/zenodo.23056907. Version 1.1.0 accompanies the revised manuscript.
