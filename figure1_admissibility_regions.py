#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 of the manuscript axioms-4631444
=========================================

A. T. Zaidi, "Optimal Lipschitz Thresholds for Weighted Dirichlet Problems,
with an Explicit Bessel-Zero Bound for Power Weights", Axioms (MDPI).

This script computes and draws Figure 1: the admissibility regions, in the
(gamma, K) plane, of three uniqueness criteria for the conformable boundary
value problem

    T_gamma^a u(t) = -g(t, u(t)),  a < t < b,   u(a) = M1,  u(b) = M2,

with gamma in (1, 2] and b - a = 1 (problem (13) of the paper).  Here K is
the Lipschitz constant of g in u.  A criterion of the form "K < threshold"
certifies a unique solution for every point (gamma, K) that lies BELOW its
threshold curve.  The three thresholds, all in closed form, are:

  1. Sharp threshold (Theorems 4.2 and 5.2 of the paper)
         K < lambda_1(gamma) = gamma^2 * j_{1/gamma,1}^2 / 4,
     where j_{1/gamma,1} is the first positive zero of the Bessel function
     J_{1/gamma}.  This is the first eigenvalue of
         -phi'' = lambda * t^(gamma-2) * phi,  phi(0) = phi(1) = 0.

  2. Hilbert-Schmidt threshold (Proposition 4.4 and Corollary 5.3)
         K < 1 / sqrt(H_1) = sqrt(3 * gamma * (4 * gamma^2 - 1)).
     Reference [6] of the paper states it for gamma in (3/2, 2]; on
     1 < gamma < 3/2 it is drawn dashed (extension of Corollary 5.3).

  3. Banach-type (supremum-norm) threshold of reference [5]
         K < 1 / V_1 = gamma^(gamma/(gamma-1) + 1).

The star marks Example 6.1, (gamma, K*) = (5/4, 4.8): it lies below the sharp
curve (4.9500) but above the Hilbert-Schmidt (4.4371) and Banach (3.8147)
curves, so only the sharp criterion applies there.

The three curves are nested (Banach < HS < sharp on (1, 2]); this is proved
in Theorem 4.5 and Proposition 4.8 of the paper and is checked numerically
below.  The figure therefore shows three bands:
    vermilion : admissible for all three criteria,
    green     : admissible for the HS and sharp criteria only,
    blue      : admissible for the sharp criterion only.

Requirements
------------
Python >= 3.9, NumPy, mpmath (Bessel zeros) and Matplotlib >= 3.5.
    pip install numpy mpmath matplotlib

Usage
-----
    python figure1_admissibility_regions.py
writes Figure1_admissibility_regions.pdf and Figure1_admissibility_regions.png
next to this script and prints a short numerical check.

In the manuscript, the same curves are drawn with LaTeX/pgfplots from the
coordinate list results/figure1_coordinates.txt written by the main
supplementary script numerics_weighted_bvp.py.  If that file is placed next
to this script (or in ./results/), the script also compares the two and
reports the largest deviation.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")                       # draw to files, no screen needed
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

# ---------------------------------------------------------------------------
# 0.  Settings
# ---------------------------------------------------------------------------
mp.mp.dps = 30                              # 30 significant digits for Bessel zeros

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_STEM = os.path.join(HERE, "Figure1_admissibility_regions")

# Colours: the colour-blind-safe Okabe-Ito triple used in the manuscript
# (pgfplots colours critSharp, critHS, critBan).
C_SHARP = "#0072B2"                         # blue
C_HS = "#009E73"                            # bluish green
C_BAN = "#D55E00"                           # vermilion

# Example 6.1 of the paper: gamma = 5/4 and best Lipschitz constant K* = 4.8
# (g(t,u) = (24/5) arctan(u) + t, so sup |dg/du| = 24/5).
EXAMPLE = (1.25, 4.8)

# Window of the enlargement shown in panel (b).
ZOOM_X = (1.10, 1.45)
ZOOM_Y = (3.20, 5.60)


# ---------------------------------------------------------------------------
# 1.  The three thresholds (b - a = 1)
# ---------------------------------------------------------------------------
def sharp_threshold(gamma):
    """lambda_1 = gamma^2 j_{1/gamma,1}^2 / 4  (Theorem 4.2 with L = 1).

    mpmath.besseljzero(nu, 1) returns the first positive zero j_{nu,1} of
    J_nu; here nu = 1/gamma lies in [1/2, 1) for gamma in (1, 2].
    """
    g = mp.mpf(gamma)
    j = mp.besseljzero(1 / g, 1)
    return float(g ** 2 * j ** 2 / 4)


def hs_threshold(gamma):
    """1/sqrt(H_1), with H_1 = 1/(3 gamma (4 gamma^2 - 1))  (Proposition 4.4)."""
    return float(np.sqrt(3.0 * gamma * (4.0 * gamma ** 2 - 1.0)))


def banach_threshold(gamma):
    """1/V_1, with V_1 = gamma^(-(gamma/(gamma-1) + 1))  (supremum-norm bound of [5]).

    Computed in mpmath because gamma/(gamma-1) is large near gamma = 1.
    """
    g = mp.mpf(gamma)
    return float(g ** (g / (g - 1) + 1))


# ---------------------------------------------------------------------------
# 2.  Evaluate the curves on a grid of orders gamma
# ---------------------------------------------------------------------------
# gamma = 1 itself is excluded (the weight t^(gamma-2) is not integrable
# there); a few points are added close to 1 to show the limits
#   lambda_1 -> j_{1,1}^2/4 = 3.6705,  1/sqrt(H_1) -> 3,  1/V_1 -> e.
gammas = np.concatenate([[1.0005, 1.002, 1.005], np.linspace(1.01, 2.0, 199)])
sharp = np.array([sharp_threshold(g) for g in gammas])
hs = np.array([hs_threshold(g) for g in gammas])
banach = np.array([banach_threshold(g) for g in gammas])

# Sanity checks: the classical case gamma = 2 and the nesting of the curves.
assert abs(sharp[-1] - np.pi ** 2) < 1e-12          # lambda_1 = pi^2 when gamma = 2
assert abs(hs[-1] - np.sqrt(90.0)) < 1e-12          # 1/sqrt(H_1) = sqrt(90)
assert abs(banach[-1] - 8.0) < 1e-12                # 1/V_1 = 8
assert np.all(banach < hs) and np.all(hs < sharp)   # Banach < HS < sharp


# ---------------------------------------------------------------------------
# 3.  Drawing helpers
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "STIXGeneral",           # serif text close to the journal font
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


def draw_regions(ax):
    """Fill the three nested bands and draw the three boundary curves."""
    # Bands, from the bottom up.  Filling between consecutive curves (rather
    # than from 0 each time) keeps the colours clean, without overlaps.
    ax.fill_between(gammas, 0.0, banach, color=C_BAN, alpha=0.32, lw=0)
    ax.fill_between(gammas, banach, hs, color=C_HS, alpha=0.38, lw=0)
    ax.fill_between(gammas, hs, sharp, color=C_SHARP, alpha=0.28, lw=0)

    # Boundary curves.  The HS curve is dashed on 1 < gamma < 3/2, the range
    # not covered by reference [6] (Corollary 5.3 of the paper extends it).
    left = gammas <= 1.5
    ax.plot(gammas, sharp, color=C_SHARP, lw=1.4)
    ax.plot(gammas[left], hs[left], color=C_HS, lw=1.4, ls=(0, (4, 2)))
    ax.plot(gammas[~left], hs[~left], color=C_HS, lw=1.4)
    ax.plot(gammas, banach, color=C_BAN, lw=1.4)

    ax.grid(True, color="0.88", lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Lipschitz constant $K$")


def band_label(ax, x, y, text, **kw):
    """Small label on a semi-transparent white box (text in neutral ink)."""
    ax.text(x, y, text, fontsize=7.5, color="0.1",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), **kw)


# ---------------------------------------------------------------------------
# 4.  The figure: panel (a) full range, panel (b) enlargement near Example 6.1
# ---------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.35))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.78, wspace=0.28)

# Panel (a) -----------------------------------------------------------------
draw_regions(ax_a)
ax_a.set_xlim(1.0, 2.0)
ax_a.set_ylim(0.0, 10.5)
ax_a.set_title(r"(a) Full range $1<\gamma\leq 2$", fontsize=9)
# Vertical guide at gamma = 3/2, the left end of the range treated in [6].
ax_a.axvline(1.5, color="0.4", lw=0.6, ls=":")
ax_a.text(1.5, 10.3, r"$\gamma=3/2$", rotation=90, ha="right", va="top",
          fontsize=7.5, color="0.3")
# Dashed rectangle: the window enlarged in panel (b).
ax_a.add_patch(plt.Rectangle((ZOOM_X[0], ZOOM_Y[0]), ZOOM_X[1] - ZOOM_X[0],
                             ZOOM_Y[1] - ZOOM_Y[0], fill=False, ls="--", lw=0.7))
ax_a.text(ZOOM_X[0], ZOOM_Y[1] + 0.1, "(b)", fontsize=7.5, va="bottom")
ax_a.plot(*EXAMPLE, marker="*", ms=9, color="k", ls="none")

# Panel (b) -----------------------------------------------------------------
draw_regions(ax_b)
ax_b.set_xlim(*ZOOM_X)
ax_b.set_ylim(*ZOOM_Y)
ax_b.set_title("(b) Enlargement near Example 6.1", fontsize=9)
# Direct labels for the three bands (identity is never carried by colour alone).
band_label(ax_b, 1.38, 3.62, "all three")
band_label(ax_b, 1.103, 4.75, "sharp only")
ax_b.annotate("", xy=(1.15, 4.15), xytext=(1.13, 4.69),
              arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.1"))
band_label(ax_b, 1.19, 5.05, "HS & sharp", ha="center")
ax_b.annotate("", xy=(1.19, 3.83), xytext=(1.19, 5.02),
              arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.1"))
ax_b.plot(*EXAMPLE, marker="*", ms=12, color="k", ls="none")
ax_b.annotate(r"Example 6.1: $(\gamma,K_*)=(5/4,\,4.8)$", xy=EXAMPLE,
              xytext=(1.103, 5.42), fontsize=7.5,
              arrowprops=dict(arrowstyle="-", lw=0.6, color="0.1"))

# Common legend above both panels -------------------------------------------
handles = [
    plt.Rectangle((0, 0), 1, 1, fc=C_SHARP, alpha=0.5, ec=C_SHARP),
    plt.Rectangle((0, 0), 1, 1, fc=C_HS, alpha=0.5, ec=C_HS),
    plt.Rectangle((0, 0), 1, 1, fc=C_BAN, alpha=0.5, ec=C_BAN),
    plt.Line2D([], [], color=C_HS, lw=1.4, ls=(0, (4, 2))),
    plt.Line2D([], [], color="k", marker="*", ms=9, ls="none"),
]
labels = [
    r"sharp: $K<\lambda_1$ (Theorem 5.2)",
    r"HS: $K<\mathcal{H}_1^{-1/2}$ (Corollary 5.3)",
    r"Banach: $K<\mathcal{V}_1^{-1}$ [5]",
    r"HS on $1<\gamma<3/2$ (outside the range of [6])",
    "Example 6.1",
]
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4,
           handlelength=1.8)

# Save -----------------------------------------------------------------------
fig.savefig(OUT_STEM + ".pdf")                      # vector, for the journal
fig.savefig(OUT_STEM + ".png", dpi=400)             # raster preview
print("Figure 1 written to", os.path.basename(OUT_STEM) + ".pdf and .png")


# ---------------------------------------------------------------------------
# 5.  Numerical report
# ---------------------------------------------------------------------------
g0, k0 = EXAMPLE
print("\nExample 6.1 (gamma = 5/4, K* = 4.8):")
print("  sharp  lambda_1     = %.6f  -> admissible: %s" % (sharp_threshold(g0), k0 < sharp_threshold(g0)))
print("  HS     1/sqrt(H_1)  = %.6f  -> admissible: %s" % (hs_threshold(g0), k0 < hs_threshold(g0)))
print("  Banach 1/V_1        = %.6f  -> admissible: %s" % (banach_threshold(g0), k0 < banach_threshold(g0)))
print("\nLimits as gamma -> 1+ (Proposition 7.1): sharp %.4f (j_{1,1}^2/4 = %.4f), "
      "HS %.4f (3), Banach %.4f (e = %.4f)"
      % (sharp[0], float(mp.besseljzero(1, 1) ** 2 / 4), hs[0], banach[0], np.e))
print("Gain of the sharp threshold on (1,2]: %.1f%%-%.1f%% over Banach, %.1f%%-%.1f%% over HS"
      % (100 * (min(sharp / banach) - 1), 100 * (max(sharp / banach) - 1),
         100 * (min(sharp / hs) - 1), 100 * (max(sharp / hs) - 1)))


def compare_with_pgfplots_coordinates():
    """Optional: compare with the coordinates used for Figure 1 in the paper."""
    for path in (os.path.join(HERE, "figure1_coordinates.txt"),
                 os.path.join(HERE, "results", "figure1_coordinates.txt")):
        if os.path.exists(path):
            break
    else:
        print("\n(figure1_coordinates.txt not found: comparison with the "
              "manuscript coordinates skipped)")
        return
    text = open(path, encoding="utf-8").read()
    funcs = {"sharp": sharp_threshold, "HS": hs_threshold, "Banach": banach_threshold}
    print("\nComparison with the pgfplots coordinates of the manuscript:")
    for label, body in re.findall(r"%\s*(.+?)\ncoordinates\s*\{(.+?)\}", text, re.S):
        key = next((k for k in funcs if label.startswith(k)), None)
        if key is None:
            continue
        pts = [tuple(map(float, p.split(","))) for p in re.findall(r"\(([^)]*)\)", body)]
        dev = max(abs(funcs[key](x) - y) for x, y in pts)
        print("  %-50s max deviation %.1e (4-decimal rounding)" % (label.strip(), dev))


compare_with_pgfplots_coordinates()
