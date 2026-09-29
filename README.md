# Code S1 — `numerics_weighted_bvp.py`

Supplementary code for

> A. T. Zaidi, *Optimal Lipschitz Thresholds for Weighted Dirichlet Problems, with an Explicit Bessel-Zero Bound for Power Weights*, submitted to *Axioms* (MDPI).

The script regenerates every number, table and plotted curve of the paper. It needs Python 3.9 or later with NumPy, SciPy and mpmath, and nothing else.

## Quick start

```bash
pip install -r requirements.txt
python numerics_weighted_bvp.py            # everything, about 5 min on a laptop
python numerics_weighted_bvp.py --quick    # coarser shooting scan for Figure 5, about 1-2 min
python numerics_weighted_bvp.py --only table3 figure3
python numerics_weighted_bvp.py --list     # which function produces which result
```

Output goes to the folder `results/`:

- `report.txt` contains all tables and checked constants as text.
- `figure1_coordinates.txt` … `figure5_coordinates.txt` contain every plotted curve as a `pgfplots` `coordinates {...}` list. These are the lists used in the LaTeX source of the paper.

Each item also prints `check ... OK` lines, which compare the computed value with the value printed in the paper. A `MISMATCH` would show at once if anything fails to reproduce. The file `reference_report.txt` is the output of a full run, for comparison.

## Reproducibility of Results

This section mirrors Appendix B of the paper. The **Key** column is the argument for `--only`.

| Result in the paper | Function | Key | Output |
|---|---|---|---|
| Table 1 (thresholds for the power weight); Remark 4.9 (Lyapunov ratios) | `table1_power_weight_thresholds` | `table1` | `report.txt` |
| Table 2 (sequential problem); Proposition 5.7(b), δ* | `table2_sequential_thresholds` | `table2` | `report.txt` |
| Table 3 (Bessel formula vs Nyström, Richardson extrapolation) | `table3_nystrom_check` | `table3` | `report.txt` |
| Table 4 (successive approximations, rates, iteration counts) | `table4_contraction` | `table4` | `report.txt` |
| Figure 1 (admissibility regions, problem (10)) | `figure1_regions_cf` | `figure1` | `figure1_coordinates.txt` |
| Figure 2 (admissibility regions, problem (12)) | `figure2_regions_scf` | `figure2` | `figure2_coordinates.txt` |
| Figure 3; Examples 6.1 and 6.2 (solutions, iteration counts, mesh doubling, shooting cross-check) | `figure3_examples` | `figure3` | `figure3_coordinates.txt`, `report.txt` |
| Figure 4; Proposition 7.1 (gain factors, monotonicity on 400 points) | `figure4_gain_factors` | `figure4` | `figure4_coordinates.txt`, `report.txt` |
| Figure 5; Section 7.4 (constants C_γ, shooting scans, bifurcation branch) | `figure5_bifurcation` | `figure5` | `figure5_coordinates.txt`, `report.txt` |
| Propositions 4.4, 5.6 and Corollary 4.6 (independent quadrature checks) | `checks_closed_forms` | `checks` | `report.txt` |

### Building blocks

| Function | Implements |
|---|---|
| `lambda1_power` | Theorem 4.2, λ₁ = γ² j²_{1/γ,1} / (4L^γ) |
| `V1`, `H1` | supremum-norm constant and Proposition 4.4 |
| `T2`, `T3` | spectral sums of Corollary 4.6 |
| `lyapunov_bound` | Lyapunov-type bound of [9] (Remark 4.9) |
| `V2`, `H2`, `sharp_sequential` | constants of the sequential problem (Theorem 5.5, Proposition 5.6) |
| `nystrom_power` | Nyström scheme of Section 7.1 (substitution s = y^{1/(γ−1)}, trapezoidal rule) |
| `lambda1_nystrom` | reciprocal of the largest eigenvalue of the Nyström matrix |
| `sequential_grid` | Nyström scheme for the sequential problem through Lemma 5.4 |
| `picard` | successive approximations u_{n+1} = Pu_n (Theorem 3.3), with observed rate |
| `shoot_power` | shooting in τ = t^{γ−1}, where the equation is regular (DOP853, rtol 10⁻¹²) |
| `C_gamma` | coefficient of the amplitude prediction (20) |

## Notes for reviewers

- **Setting.** All computations use a = 0 and b = L = 1, as in Section 7. Thresholds for general L follow on multiplying by L^{−γ}, respectively L^{−2δ}.
- **Round-off.** The "after extrapolation" column of Table 3 is at round-off level (10⁻¹³ to 10⁻¹⁶). Its last digits can differ between machines and BLAS libraries.
- **Figure 5 scans.** The shooting parameter p = u′(0⁺) is scanned on a geometric grid of 1500 points in [10⁻⁴, 300], or 300 points with `--quick`. Sign changes are refined with Brent's method.
