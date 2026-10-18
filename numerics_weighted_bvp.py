#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerics_weighted_bvp.py
========================

Supplementary code (Code S1) for the manuscript

    A. T. Zaidi, "Optimal Lipschitz Thresholds for Weighted Dirichlet Problems,
    with an Explicit Bessel-Zero Bound for Power Weights" (submitted to Axioms).

The script regenerates every number, table and plotted curve of the paper.
Appendix B of the paper ("Reproducibility of Results") lists which function
produces which table or figure; the same list is printed by `--list`.
Every table value and every plotted curve is recomputed from scratch on each
run; no stored data are read.

Version 1.2.0 (revised manuscript, October 2026).  Equation numbers below
refer to the revised manuscript: problem (1) the weighted problem, (13) the
conformable problem of order gamma, (15) the sequential problem, (19)-(20)
Examples 6.1-6.2, (22) the sine problem and (23) the amplitude prediction.

Requirements
------------
    Python >= 3.9, NumPy, SciPy, mpmath          (nothing else)
    Tested with Python 3.12, NumPy 2.x, SciPy 1.1x, mpmath 1.3 (see README.md).

Numerical checks
----------------
    Each "check" line compares a recomputed value with the value printed in the
    paper and reports OK when |computed - reference| <= tol, an ABSOLUTE
    tolerance chosen per check: half a unit in the last printed digit for
    values printed to a fixed number of decimals, and a larger, stated
    tolerance for quantities obtained by quadrature or discretization (for
    example trace(M^3) of the N = 1600 Nystrom matrix, whose O(N^-2) error is
    about 3e-8).  The line also shows the absolute and relative differences.
    An OK line is therefore a check within a stated tolerance, not an exact
    identity.

Usage
-----
    python numerics_weighted_bvp.py            # everything (6-7 min on the test machine)
    python numerics_weighted_bvp.py --quick    # coarser shooting scans for Figure 5 only
    python numerics_weighted_bvp.py --only table3 figure5
    python numerics_weighted_bvp.py --list     # item -> function map

Output
------
    results/report.txt            all tables and checked constants, as text
    results/<item>_coordinates.txt  pgfplots "coordinates {...}" strings for
                                  every curve of Figures 1-5, ready to be
                                  compared with (or pasted into) the LaTeX source

Conventions (as in the paper)
-----------------------------
    a = 0, b = L = 1 throughout the numerical part (Section 7).
    Power weight w(t) = t^(gamma-2), gamma > 1, Bessel order nu = 1/gamma.
    Sequential order delta in (0, 1].
    G0(t,s) = s(1-t) for s <= t and t(1-s) for s >= t (Dirichlet Green's
    function of -u'' on [0,1]).
"""

import argparse
import os
import sys
import time

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

mp.mp.dps = 30                      # working precision for Bessel zeros
OUT = "results"                     # output folder
REPORT = []                         # lines of the text report
CHECKS = []                         # outcomes of all check() calls
__version__ = "1.2.0"


# ---------------------------------------------------------------------------
# Small output helpers
# ---------------------------------------------------------------------------
def say(line=""):
    """Print a line and keep it for results/report.txt."""
    print(line)
    REPORT.append(line)


def header(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def pgf(xs, ys, fx="{:.4f}", fy="{:.4f}"):
    """Format a curve as a pgfplots 'coordinates {...}' string."""
    pts = " ".join("(" + fx.format(x) + "," + fy.format(y) + ")" for x, y in zip(xs, ys))
    return "coordinates {" + pts + "}"


def write_coordinates(name, curves):
    """Write several labelled pgfplots curves to results/<name>_coordinates.txt."""
    path = os.path.join(OUT, name + "_coordinates.txt")
    with open(path, "w") as fh:
        for label, text in curves:
            fh.write("% " + label + "\n" + text + "\n\n")
    say("  -> plotted coordinates written to " + path)


def check(label, computed, reference, tol):
    """Compare a computed value with the value printed in the paper.

    OK means |computed - reference| <= tol, with tol an absolute tolerance
    (see "Numerical checks" in the module docstring).  The absolute and
    relative differences are printed so that the margin is visible."""
    diff = abs(computed - reference)
    ok = diff <= tol
    rel = diff / abs(reference) if reference else float("nan")
    say("  check %-44s computed %-18.10g reference %-14.10g %s"
        % (label, computed, reference, "OK" if ok else "MISMATCH"))
    say("        |diff| = %.1e  (relative %.1e)  <=  tol %.0e (absolute)"
        % (diff, rel, tol) if ok else
        "        |diff| = %.1e  (relative %.1e)  >  tol %.0e (absolute)" % (diff, rel, tol))
    CHECKS.append(ok)
    return ok


# ---------------------------------------------------------------------------
# 1. Closed-form thresholds (Sections 4 and 5 of the paper)
# ---------------------------------------------------------------------------
def bessel_zero(nu, k=1):
    """k-th positive zero j_{nu,k} of the Bessel function J_nu (mpmath)."""
    return mp.besseljzero(mp.mpf(nu), k)


def lambda1_power(gamma, L=1):
    """Theorem 4.2: lambda_1 = gamma^2 j_{1/gamma,1}^2 / (4 L^gamma)."""
    g = mp.mpf(gamma)
    return g**2 * bessel_zero(1 / g) ** 2 / (4 * mp.mpf(L) ** g)


def V1(gamma, L=1):
    """Supremum-norm constant: V1 = L^gamma / gamma^((2 gamma - 1)/(gamma - 1))."""
    g = mp.mpf(gamma)
    return mp.mpf(L) ** g / g ** ((2 * g - 1) / (g - 1))


def H1(gamma, L=1):
    """Proposition 4.4: unweighted Hilbert-Schmidt constant L^(2g)/(3g(4g^2-1))."""
    g = mp.mpf(gamma)
    return mp.mpf(L) ** (2 * g) / (3 * g * (4 * g**2 - 1))


def T2(gamma, L=1):
    """Corollary 4.6: weighted HS constant  sum lambda_k^-2."""
    g = mp.mpf(gamma)
    return mp.mpf(L) ** (2 * g) / (g * (g + 1) ** 2 * (2 * g + 1))


def T3(gamma, L=1):
    """Corollary 4.6: sum lambda_k^-3."""
    g = mp.mpf(gamma)
    return 2 * mp.mpf(L) ** (3 * g) / (g * (g + 1) ** 3 * (2 * g + 1) * (3 * g + 1))


def lyapunov_bound(gamma, L=1):
    """Lower bound for lambda_1 from the Lyapunov-type inequality of [AAJ17]."""
    g = mp.mpf(gamma)
    return g**g / ((g - 1) ** (g - 1) * mp.mpf(L) ** g)


def V2(delta):
    """Banach-type constant of the sequential problem (b - a = 1): 1/(8 delta^2)."""
    return 1 / (8 * mp.mpf(delta) ** 2)


def H2(delta):
    """Proposition 5.6: HS constant of the sequential problem (b - a = 1)."""
    d = mp.mpf(delta)
    if d <= mp.mpf(1) / 4:
        return mp.inf
    return 1 / (10 * d**2 * (2 * d + 1) * (4 * d - 1))


def sharp_sequential(delta):
    """Theorem 5.5: sharp threshold delta^2 pi^2 (b - a = 1)."""
    return mp.mpf(delta) ** 2 * mp.pi**2


# ---------------------------------------------------------------------------
# 2. Nystrom discretization (Section 7.1)
# ---------------------------------------------------------------------------
def green0(t, s, L=1.0):
    """Matrix G0(t_i, s_j) of the Dirichlet Green's function on [0, L]."""
    T, S = np.meshgrid(t, s, indexing="ij")
    return np.where(S <= T, S * (L - T), T * (L - S)) / L


def nystrom_power(gamma, N):
    """
    Nystrom scheme of Section 7.1 for w(t) = t^(gamma-2) on [0,1].

    The substitution s = y^(1/(gamma-1)) turns s^(gamma-2) ds into dy/(gamma-1),
    so the weight disappears; the trapezoidal rule on N uniform subintervals in
    y is used and the equation is collocated at the images t_i of the nodes.

    Returns t (nodes), q (quadrature weights incl. the factor 1/(gamma-1)) and
    the matrix M with (M u)_i ~ int_0^1 G0(t_i,s) s^(gamma-2) u(s) ds.
    """
    y = np.linspace(0.0, 1.0, N + 1)
    t = y ** (1.0 / (gamma - 1.0))
    q = np.full(N + 1, 1.0 / N)
    q[0] = q[-1] = 0.5 / N
    q /= gamma - 1.0
    M = green0(t, t) * q[None, :]
    return t, q, M


def lambda1_nystrom(gamma, N):
    """
    Reciprocal of the largest eigenvalue of the Nystrom matrix.

    M = G diag(q) is similar to the symmetric matrix diag(q)^(1/2) G diag(q)^(1/2)
    (q > 0 at interior nodes), so a symmetric eigensolver can be used.
    The two boundary nodes are dropped because G0 vanishes there.
    """
    t, q, _ = nystrom_power(gamma, N)
    ti, qi = t[1:-1], q[1:-1]
    r = np.sqrt(qi)
    S = r[:, None] * green0(ti, ti) * r[None, :]
    return 1.0 / np.linalg.eigvalsh(S)[-1]


def picard(u0, xi, M, rhs, tol=1e-12, maxit=100000):
    """
    Successive approximations u_{n+1} = xi + M rhs(u_n) (the operator P of
    Theorem 3.3, discretized).  Stops when max|u_{n+1} - u_n| < tol.

    Returns (solution, number of iterations, observed asymptotic ratio).
    The ratio is the median of d_n/d_{n-1} over the steps with
    1e-9 < d_n < 1e-5, i.e. after the transient and before round-off.
    """
    u, prev, ratios = u0.copy(), None, []
    for n in range(1, maxit + 1):
        un = xi + M @ rhs(u)
        d = np.max(np.abs(un - u))
        if prev is not None and 1e-9 < d < 1e-5:
            ratios.append(d / prev)
        prev, u = d, un
        if d < tol:
            rate = float(np.median(ratios)) if ratios else float("nan")
            return u, n, rate
    raise RuntimeError("Picard iteration did not converge")


def sequential_grid(delta, N):
    """
    Lemma 5.4: the sequential problem becomes -U'' = h(t(tau), U) on [0, Lambda],
    Lambda = 1/delta, t(tau) = (delta tau)^(1/delta).  Uniform grid with N
    subintervals in tau and the trapezoidal rule.  Returns tau, t(tau), M.
    """
    Lam = 1.0 / delta
    tau = np.linspace(0.0, Lam, N + 1)
    q = np.full(N + 1, Lam / N)
    q[0] = q[-1] = 0.5 * Lam / N
    M = green0(tau, tau, L=Lam) * q[None, :]
    return tau, (delta * tau) ** (1.0 / delta), M


# ---------------------------------------------------------------------------
# 3. Shooting (independent of the Nystrom scheme)
# ---------------------------------------------------------------------------
def shoot_power(gamma, g, p, dense=False):
    """
    Integrate u'' = -t^(gamma-2) g(t,u), u(0)=0, u'(0)=p for w = t^(gamma-2).

    In the variable tau = t^(gamma-1) (t = tau^a, a = 1/(gamma-1)) the system
        du/dtau = a tau^(a-1) v,   dv/dtau = -a g(t, u),   v = u'(t),
    is regular at tau = 0 (a - 1 = (2-gamma)/(gamma-1) >= 0 for gamma <= 2),
    so the integration starts exactly at 0.  DOP853, rtol 1e-12.
    """
    a = 1.0 / (gamma - 1.0)

    def rhs(tau, z):
        return [a * tau ** (a - 1.0) * z[1], -a * g(tau**a, z[0])]

    return solve_ivp(rhs, [0.0, 1.0], [0.0, p], method="DOP853",
                     rtol=1e-12, atol=1e-14, dense_output=dense)


def roots_in(f, grid):
    """All sign changes of f on the grid, refined with Brent's method."""
    vals = [f(p) for p in grid]
    return [brentq(f, grid[i], grid[i + 1], xtol=1e-14)
            for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0]


# ===========================================================================
# TABLES
# ===========================================================================
def table1_power_weight_thresholds():
    """Table 1: thresholds for w = t^(gamma-2), L = 1 (Theorems 4.2, 4.5,
    Proposition 4.4, Corollary 4.6, Remarks 4.7 and 4.9)."""
    header("TABLE 1  Uniqueness thresholds for the power weight (L = 1)")
    gammas = ["1.05", "1.1", "1.25", "1.5", "1.75", "2"]
    rows = [
        ("j_{1/gamma,1}", lambda g: bessel_zero(1 / mp.mpf(g))),
        ("Lyapunov-type [AAJ17]", lyapunov_bound),
        ("Sup-norm 1/V1", lambda g: 1 / V1(g)),
        ("HS 1/sqrt(H1)", lambda g: 1 / mp.sqrt(H1(g))),
        ("Weighted HS 1/sqrt(T2)", lambda g: 1 / mp.sqrt(T2(g))),
        ("Sharp lambda_1", lambda1_power),
        ("Upper bound T2/T3", lambda g: T2(g) / T3(g)),
    ]
    say("%-24s" % "gamma" + "".join("%11s" % g for g in gammas))
    for name, fn in rows:
        say("%-24s" % name + "".join("%11.5f" % float(fn(mp.mpf(g))) for g in gammas))
    say("")
    say("Remark 4.9: ratio lambda_1 / Lyapunov-type bound")
    ratio = lambda g: lambda1_power(g) / lyapunov_bound(g)
    say("  " + "  ".join("gamma=%s: %.2f" % (g, float(ratio(mp.mpf(g)))) for g in gammas))
    gmin = mp.findroot(lambda g: mp.diff(ratio, g), mp.mpf("1.72"))
    say("  minimum on (1,2]: %.4f at gamma = %.3f;  limit gamma->1+: j_{1,1}^2/4 = %.4f"
        % (float(ratio(gmin)), float(gmin), float(bessel_zero(1) ** 2 / 4)))
    check("Table 1, lambda_1 at gamma=1.25", float(lambda1_power("1.25")), 4.95001, 6e-6)
    check("Table 1, 1/sqrt(T2) at gamma=1.25", float(1 / mp.sqrt(T2("1.25"))), 4.70621, 6e-6)
    check("Remark 4.9, minimum ratio", float(ratio(gmin)), 2.436, 6e-4)
    say("")
    # Remark 4.7: how far the elementary bounds of Corollary 4.6 lie from
    # lambda_1, and which part of the gap between the unweighted HS threshold
    # 1/sqrt(H1) and lambda_1 is closed by the weighted HS threshold 1/sqrt(T2).
    # All closed forms are continuous at gamma = 1, so the limits gamma -> 1+
    # are obtained by evaluating them at gamma = 1.
    say("Remark 4.7: elementary bounds of Corollary 4.6 relative to lambda_1")
    low = lambda g: 100 * (1 - 1 / mp.sqrt(T2(g)) / lambda1_power(g))
    upp = lambda g: 100 * (T2(g) / T3(g) / lambda1_power(g) - 1)
    share = lambda g: (100 * (1 / mp.sqrt(T2(g)) - 1 / mp.sqrt(H1(g)))
                       / (lambda1_power(g) - 1 / mp.sqrt(H1(g))))
    grid = [1 + mp.mpf(k) / 400 for k in range(1, 401)]
    lows, upps = [low(g) for g in grid], [upp(g) for g in grid]
    dec = lambda v: all(x > y for x, y in zip(v, v[1:]))
    say("  lower bound below lambda_1 by %.2f%% (gamma = 2) to %.2f%% (limit gamma -> 1+);"
        " decreasing on a 400-point grid: %s" % (float(low(2)), float(low(1)), dec(lows)))
    say("  upper bound above lambda_1 by %.2f%% (gamma = 2) to %.2f%% (limit gamma -> 1+);"
        " decreasing on a 400-point grid: %s" % (float(upp(2)), float(upp(1)), dec(upps)))
    say("  share of the gap lambda_1 - 1/sqrt(H1) closed by 1/sqrt(T2):")
    say("  " + "  ".join("gamma=%s: %.1f%%" % (lab, float(share(g)))
                         for lab, g in (("1+", 1), ("1.25", "1.25"), ("1.5", "1.5"),
                                        ("1.75", "1.75"), ("2", 2))))
    check("Remark 4.7, lower-bound gap, gamma->1+ (%)", float(low(1)), 5.6, 0.05)
    check("Remark 4.7, upper-bound gap, gamma->1+ (%)", float(upp(1)), 9.0, 0.05)
    check("Remark 4.7, share closed at gamma=1.25 (%)", float(share("1.25")), 52, 0.5)


def table2_sequential_thresholds():
    """Table 2: thresholds of the sequential problem (15), b - a = 1
    (Theorem 5.5, Propositions 5.6 and 5.7)."""
    header("TABLE 2  Uniqueness thresholds for the sequential problem (b - a = 1)")
    deltas = ["0.3", "0.4", "0.5", "0.75", "1"]
    say("%-24s" % "delta" + "".join("%11s" % d for d in deltas))
    for name, fn in [("Banach-type 8 delta^2", lambda d: 1 / V2(d)),
                     ("Hilbert-Schmidt H2^-1/2", lambda d: 1 / mp.sqrt(H2(d))),
                     ("Sharp delta^2 pi^2", sharp_sequential)]:
        say("%-24s" % name + "".join("%11.5f" % float(fn(mp.mpf(d))) for d in deltas))
    dstar = (mp.sqrt(65) - 5) / 8
    say("Proposition 5.7(b): delta* = (sqrt(65)-5)/8 = %.6f" % float(dstar))
    check("Table 2, H2^-1/2 at delta=0.4", float(1 / mp.sqrt(H2("0.4"))), 1.31453, 6e-6)


def table3_nystrom_check(Ns=(400, 800, 1600)):
    """Table 3: lambda_1 from the Bessel formula versus the Nystrom matrix,
    with two Richardson extrapolation steps (second-order scheme)."""
    header("TABLE 3  lambda_1: Bessel formula versus Nystrom discretization")
    say("%-6s %-20s %-17s %-10s %-10s" % ("gamma", "Theorem 4.2", "Nystrom N=%d" % Ns[-1],
                                          "rel.err", "extrapol."))
    for g in [1.10, 1.25, 1.50, 1.75, 2.00]:
        exact = lambda1_power(mp.mpf(str(g)))
        l = [lambda1_nystrom(g, N) for N in Ns]
        r1 = [(4 * l[1] - l[0]) / 3, (4 * l[2] - l[1]) / 3]      # O(h^2) removed
        r2 = (16 * r1[1] - r1[0]) / 15                           # O(h^4) removed
        ex = float(exact)
        say("%-6.2f %-20s %-17.12f %-10.1e %-10.1e"
            % (g, mp.nstr(exact, 16), l[-1], abs(l[-1] - ex) / ex, abs(r2 - ex) / ex))
    check("Table 3, Nystrom N=1600 at gamma=1.25", lambda1_nystrom(1.25, 1600),
          4.950008178117, 1e-11)


def table4_contraction(N=1600):
    """Table 4: successive approximations for K between the HS threshold and
    lambda_1 (Section 7.4).  M1 = 0, M2 = 1, u0 = xi, tolerance 1e-12."""
    header("TABLE 4  Successive approximations where only the sharp condition applies")
    say("%-11s %5s %8s %8s %8s %10s %10s %6s %5s"
        % ("order", "K", "Banach", "HS", "sharp", "K/sharp", "rate", "lin.", "sin"))
    for g, K in [(1.25, 4.80), (1.50, 6.20), (1.75, 7.90)]:
        t, _, M = nystrom_power(g, N)
        xi = t.copy()                                   # xi(t) = t
        _, n_lin, rate = picard(xi, xi, M, lambda u: K * u + 1.0)
        _, n_sin, _ = picard(xi, xi, M, lambda u: K * np.sin(u) + 1.0)
        lam = float(lambda1_power(mp.mpf(str(g))))
        say("gamma=%-5.2f %5.2f %8.4f %8.4f %8.4f %10.6f %10.6f %6d %5d"
            % (g, K, float(1 / V1(g)), float(1 / mp.sqrt(H1(g))), lam, K / lam, rate, n_lin, n_sin))
    for d, K in [(0.4, 1.50), (0.6, 3.45)]:
        tau, _, M = sequential_grid(d, N)
        xi = tau * d                                    # xi = tau / Lambda
        _, n_lin, rate = picard(xi, xi, M, lambda u: K * u + 1.0)
        sh = float(sharp_sequential(d))
        say("delta=%-5.2f %5.2f %8.4f %8.4f %8.4f %10.6f %10.6f %6d %5s"
            % (d, K, float(1 / V2(d)), float(1 / mp.sqrt(H2(d))), sh, K / sh, rate, n_lin, "--"))
    say("(the observed rate differs from K/lambda_1 by the O(N^-2) error of Table 3)")


# ===========================================================================
# FIGURES
# ===========================================================================
def figure1_regions_cf():
    """Figure 1: threshold curves K = 1/V1, 1/sqrt(H1), lambda_1 over gamma
    (b - a = 1).  The same curves fill panels (a) and (b)."""
    header("FIGURE 1  Admissibility regions, conformable problem (13)")
    gs = [1.0005, 1.002, 1.005, 1.01] + [round(1.02 + 0.01 * k, 4) for k in range(99)]
    sharp = [float(lambda1_power(mp.mpf(str(g)))) for g in gs]
    hs = [float(1 / mp.sqrt(H1(g))) for g in gs]
    ban = [float(1 / V1(g)) for g in gs]
    i15 = gs.index(1.5)
    write_coordinates("figure1", [
        ("sharp lambda_1 (Theorem 5.2)", pgf(gs, sharp)),
        ("HS, dashed part 1 < gamma <= 3/2 (Corollary 5.3)", pgf(gs[:i15 + 1], hs[:i15 + 1])),
        ("HS, solid part 3/2 <= gamma <= 2", pgf(gs[i15:], hs[i15:])),
        ("Banach 1/V1 [LAJ24]", pgf(gs, ban)),
        ("Example 6.1 star", pgf([1.25], [4.8], "{:.2f}", "{:.1f}")),
    ])
    say("  Example 6.1 point (5/4, 4.8): sharp %.4f, weighted HS %.4f, HS %.4f"
        % (float(lambda1_power("1.25")), float(1 / mp.sqrt(T2("1.25"))),
           float(1 / mp.sqrt(H1("1.25")))))


def figure2_regions_scf():
    """Figure 2: threshold curves 8 delta^2, H2^-1/2, delta^2 pi^2 over delta."""
    header("FIGURE 2  Admissibility regions, sequential problem (15)")
    ds = [round(0.01 * k, 4) for k in range(101)]
    dh = [0.25, 0.2525, 0.255] + [round(0.26 + 0.01 * k, 4) for k in range(75)]
    write_coordinates("figure2", [
        ("sharp delta^2 pi^2 (Theorem 5.5)", pgf(ds, [float(sharp_sequential(d)) for d in ds])),
        ("HS H2^-1/2, empty for delta <= 1/4", pgf(dh, [0.0 if d == 0.25 else
                                                      float(1 / mp.sqrt(H2(d))) for d in dh])),
        ("Banach 8 delta^2", pgf(ds, [float(1 / V2(d)) if d > 0 else 0.0 for d in ds])),
        ("delta* marker", pgf([float((mp.sqrt(65) - 5) / 8)],
                              [float(1 / V2((mp.sqrt(65) - 5) / 8))])),
        ("Example 6.2 star", pgf([0.6], [3.4], "{:.1f}", "{:.1f}")),
    ])


def figure3_examples(N=1600):
    """Figure 3 and Examples 6.1-6.2: unique solutions of (19) and (20),
    by successive approximations, with mesh doubling and an independent
    shooting check."""
    header("FIGURE 3 / EXAMPLES 6.1 and 6.2")
    tt = np.linspace(0.0, 1.0, 41)                       # plotting abscissae

    # ---- Example 6.1: T^0_{5/4} u = -(24/5 arctan u + t), u(0)=0, u(1)=1
    g, K = 1.25, 4.8
    G = lambda t, u: K * np.arctan(u) + t
    sols = {}
    for n in (N, 2 * N):
        t, _, M = nystrom_power(g, n)
        sols[n] = (t,) + picard(t.copy(), t.copy(), M, lambda u, t=t: G(t, u))[:2]
    t1, u1, it1 = sols[N]
    diff = np.max(np.abs(sols[2 * N][1][::2] - u1))      # nested grids
    sp = CubicSpline(t1, u1)
    fine = np.linspace(0, 1, 200001)
    say("Example 6.1: %d iterations (N=%d); mesh doubling changes u by %.2e"
        % (it1, N, diff))
    say("  max u = %.4f, u(1/2) = %.4f" % (sp(fine).max(), sp(0.5)))
    p = roots_in(lambda p: shoot_power(g, G, p).y[0, -1] - 1.0, np.linspace(-20, 20, 81))
    s = shoot_power(g, G, p[0], dense=True)
    say("  shooting: %d root(s) p=u'(0+) in [-20,20]: %s; max u = %.4f, u(1/2) = %.4f"
        % (len(p), ", ".join("%.6f" % x for x in p),
           s.sol(np.linspace(0, 1, 200001))[0].max(), s.sol(0.5 ** (g - 1))[0]))
    check("Example 6.1, max u", float(sp(fine).max()), 1.7161, 6e-5)
    ex1 = sp(tt)

    # ---- Example 6.2: T^0_{3/5} T^0_{3/5} u = -(17/5 sin u + 1)
    d = 0.6
    tau, _, M = sequential_grid(d, N)
    U, it2 = picard(tau * d, tau * d, M, lambda u: 3.4 * np.sin(u) + 1.0)[:2]
    spU = CubicSpline(tau, U)
    tau_of_t = lambda t: t**d / d                        # theta(t)
    say("Example 6.2: %d iterations (N=%d in tau on [0,5/3])" % (it2, N))
    say("  max u = %.4f, u(1/2) = %.4f" % (spU(np.linspace(0, 1 / d, 200001)).max(),
                                         spU(tau_of_t(0.5))))
    check("Example 6.2, max u", float(spU(np.linspace(0, 1 / d, 200001)).max()), 1.9906, 6e-5)
    ex2 = spU(tau_of_t(tt))

    write_coordinates("figure3", [("Example 6.1 (gamma = 5/4)", pgf(tt, ex1)),
                                  ("Example 6.2 (delta = 3/5)", pgf(tt, ex2))])


def figure4_gain_factors():
    """Figure 4 and Proposition 7.1: gain factors R_B = lambda_1 V1 and
    R_HS = lambda_1 sqrt(H1); monotonicity on a 400-point grid."""
    header("FIGURE 4 / PROPOSITION 7.1  Gain factors")
    RB = lambda g: float(lambda1_power(g) * V1(g))
    RH = lambda g: float(lambda1_power(g) * mp.sqrt(H1(g)))
    j11 = bessel_zero(1)
    say("Proposition 7.1: R_B(1+) = j11^2/(4e) = %.5f, R_HS(1+) = j11^2/12 = %.5f"
        % (float(j11**2 / (4 * mp.e)), float(j11**2 / 12)))
    say("                 R_B(2) = pi^2/8 = %.5f, R_HS(2) = pi^2/sqrt(90) = %.5f"
        % (float(mp.pi**2 / 8), float(mp.pi**2 / mp.sqrt(90))))
    grid = np.linspace(1.0025, 2.0, 400)
    rb = np.array([RB(mp.mpf(g)) for g in grid])
    rh = np.array([RH(mp.mpf(g)) for g in grid])
    say("400-point grid: R_B decreasing: %s, R_HS decreasing: %s"
        % (bool(np.all(np.diff(rb) < 0)), bool(np.all(np.diff(rh) < 0))))
    say("gain vs Banach %.1f%%-%.1f%%, vs HS %.1f%%-%.1f%%"
        % (100 * (rb.min() - 1), 100 * (float(j11**2 / (4 * mp.e)) - 1),
           100 * (rh.min() - 1), 100 * (float(j11**2 / 12) - 1)))
    gs = list(np.linspace(1.0025, 1.1, 10)) + [round(1.125 + 0.025 * k, 4) for k in range(36)]
    write_coordinates("figure4", [
        ("R_B = lambda_1 V1", pgf(gs, [RB(mp.mpf(g)) for g in gs], fy="{:.5f}")),
        ("R_HS = lambda_1 sqrt(H1)", pgf(gs, [RH(mp.mpf(g)) for g in gs], fy="{:.5f}")),
    ])


def C_gamma(gamma):
    """Coefficient of the amplitude prediction (23), phi_1 normalized by max = 1."""
    g = mp.mpf(gamma)
    nu, j = 1 / g, bessel_zero(1 / g)
    phi = lambda t: mp.sqrt(t) * mp.besselj(nu, j * t ** (g / 2))
    tmax = minimize_scalar(lambda t: -float(phi(mp.mpf(t))), bounds=(0.05, 0.95),
                           method="bounded", options={"xatol": 1e-12}).x
    M = phi(mp.mpf(tmax))
    I2 = mp.quad(lambda t: t ** (g - 2) * (phi(t) / M) ** 2, [0, 1])
    I4 = mp.quad(lambda t: t ** (g - 2) * (phi(t) / M) ** 4, [0, 1])
    return 6 * I2 / I4


def figure5_bifurcation(quick=False):
    """Figure 5 and Section 7.4: nontrivial solutions of T^0_gamma u = -K sin u,
    u(0)=u(1)=0, by shooting in p = u'(0+) in (0, 300]."""
    header("FIGURE 5 / SECTION 7.4  Bifurcation at K = lambda_1")
    C = {g: float(C_gamma(g)) for g in (1.25, 1.5, 1.75, 2.0)}
    say("C_gamma: " + ", ".join("C_%s = %.4f" % (g, c) for g, c in C.items()))
    npts = 300 if quick else 1500
    grid = np.geomspace(1e-4, 300.0, npts)               # scan of p = u'(0+)

    def positive_solutions(g, K):
        f = lambda p: shoot_power(g, lambda t, u: K * np.sin(u), p).y[0, -1]
        amps = []
        for p in roots_in(f, grid):
            s = shoot_power(g, lambda t, u: K * np.sin(u), p, dense=True)
            amps.append(np.max(np.abs(s.sol(np.linspace(0, 1, 20001))[0])))
        return amps

    say("scan (%d shooting parameters per case); entries: number of positive solutions"
        % npts)
    for g in (1.25, 1.5, 1.75):
        lam = float(lambda1_power(mp.mpf(str(g))))
        out = []
        for r in (0.9, 0.99, 1.01, 1.05, 1.2, 1.5, 2.0):
            a = positive_solutions(g, r * lam)
            out.append("%.2f:%d" % (r, len(a)))
            if r == 1.01:
                amp101 = a[0]
        say("  gamma=%.2f  %s   amplitude at 1.01: %.4f (prediction %.4f)"
            % (g, " ".join(out), amp101, np.sqrt(C[g] * 0.01)))
    g = 1.5
    lam = float(lambda1_power("1.5"))
    rs = [1.0025, 1.01, 1.025, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 2.0]
    amps = [positive_solutions(g, r * lam)[0] for r in rs]
    say("  gamma = 3/2: sup-norm and HS certify uniqueness only for K/lambda_1 < %.3f and %.3f"
        % (float(1 / (V1(g) * lambda1_power(g))), float(1 / (mp.sqrt(H1(g)) * lambda1_power(g)))))
    check("Figure 5, amplitude at K/lambda_1 = 2", amps[-1], 2.19640, 6e-5)
    write_coordinates("figure5", [
        ("shooting branch, gamma = 3/2", "coordinates {(1.0,0) " +
         pgf(rs, amps, fy="{:.5f}")[len("coordinates {"):]),
        ("prediction (23): sqrt(C_1.5 (x-1)) with C_1.5", "C_1.5 = %.6f" % C[1.5]),
    ])


# ===========================================================================
# Closed-form checks of the propositions (quadrature, independent of Bessel)
# ===========================================================================
def checks_closed_forms():
    """Propositions 4.4, 5.6 and Corollary 4.6 against direct quadrature."""
    header("CHECKS  Closed forms of Propositions 4.4, 5.6 and Corollary 4.6")
    G0 = lambda t, s: s * (1 - t) if s <= t else t * (1 - s)
    for g in (1.25, 1.75):
        h1 = quad(lambda t: quad(lambda s: (G0(t, s) * s ** (g - 2)) ** 2, 0, 1,
                                 points=[t], limit=200)[0], 0, 1, limit=200)[0]
        t2 = quad(lambda t: quad(lambda s: G0(t, s) ** 2 * (s * t) ** (g - 2), 0, 1,
                                 points=[t], limit=200)[0], 0, 1, limit=200)[0]
        check("Prop. 4.4  H1(gamma=%.2f)" % g, h1, float(H1(g)), 1e-7)
        check("Cor. 4.6  T2(gamma=%.2f)" % g, t2, float(T2(g)), 1e-7)
        _, _, M = nystrom_power(g, 1600)
        check("Cor. 4.6  T3(gamma=%.2f) via trace(M^3)" % g, float(np.trace(M @ M @ M)),
              float(T3(g)), 1e-7)   # O(N^-2) Nystrom error, about 3e-8 at N = 1600
    d = 0.6
    G2 = lambda t, s: (s ** (2 * d - 1) * (1 - t**d) if s <= t
                       else t**d * s ** (d - 1) * (1 - s**d)) / d
    h2 = quad(lambda t: quad(lambda s: G2(t, s) ** 2, 0, 1, points=[t], limit=200)[0],
              0, 1, limit=200)[0]
    check("Prop. 5.6  H2(delta=0.6)", h2, float(H2(d)), 1e-7)


# ===========================================================================
# Driver
# ===========================================================================
ITEMS = {
    "table1": (table1_power_weight_thresholds, "Table 1, Remark 4.9"),
    "table2": (table2_sequential_thresholds, "Table 2, Proposition 5.7(b)"),
    "table3": (table3_nystrom_check, "Table 3"),
    "table4": (table4_contraction, "Table 4"),
    "figure1": (figure1_regions_cf, "Figure 1"),
    "figure2": (figure2_regions_scf, "Figure 2"),
    "figure3": (figure3_examples, "Figure 3, Examples 6.1-6.2"),
    "figure4": (figure4_gain_factors, "Figure 4, Proposition 7.1"),
    "figure5": (figure5_bifurcation, "Figure 5, Section 7.4"),
    "checks": (checks_closed_forms, "Props. 4.4, 5.6, Cor. 4.6"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", nargs="+", choices=list(ITEMS), help="run selected items")
    ap.add_argument("--quick", action="store_true", help="coarser shooting scans")
    ap.add_argument("--list", action="store_true", help="print the item -> function map")
    args = ap.parse_args()
    if args.list:
        for key, (fn, what) in ITEMS.items():
            print("%-8s %-28s %s()" % (key, what, fn.__name__))
        return
    os.makedirs(OUT, exist_ok=True)
    import platform
    import scipy
    say("numerics_weighted_bvp.py %s  |  Python %s, NumPy %s, SciPy %s, mpmath %s"
        % (__version__, sys.version.split()[0], np.__version__, scipy.__version__,
           mp.__version__))
    say("run started %s UTC on %s  |  mode: %s  |  items: %s"
        % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), platform.platform(),
           "quick" if args.quick else "full", " ".join(args.only or ITEMS)))
    t0 = time.time()
    timings = []
    for key in (args.only or ITEMS):
        fn = ITEMS[key][0]
        t1 = time.time()
        fn(quick=args.quick) if key == "figure5" else fn()
        timings.append((key, time.time() - t1))
    header("SUMMARY")
    say("checks: %d run, %d OK, %d MISMATCH"
        % (len(CHECKS), sum(CHECKS), len(CHECKS) - sum(CHECKS)))
    say("time per item: " + ", ".join("%s %.1f s" % kt for kt in timings))
    say("Total time: %.1f s" % (time.time() - t0))
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print("Report written to", os.path.join(OUT, "report.txt"))


if __name__ == "__main__":
    main()
