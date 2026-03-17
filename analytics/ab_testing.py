"""
analytics/ab_testing.py — A/B Test Analysis Module

Provides three analysis methods for any two-variant conversion experiment:
  1. Two-proportion z-test       (frequentist, fast)
  2. Chi-squared test            (frequentist, handles 2×2 contingency tables)
  3. Bayesian Beta-Binomial      (posterior probability, credible intervals)

Also includes:
  - Power analysis               (sample size calculator)
  - Plain-English recommendation output

Used by app/pages/4_Experimentation.py — all functions are importable.
Can also be run as a script for quick CLI analysis.

Usage (CLI):
    python -m analytics.ab_testing \
        --a-visitors 1000 --a-conversions 120 \
        --b-visitors 1000 --b-conversions 145
"""

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariantData:
    name: str
    visitors: int
    conversions: int

    @property
    def rate(self) -> float:
        return self.conversions / self.visitors if self.visitors > 0 else 0.0

    @property
    def non_conversions(self) -> int:
        return self.visitors - self.conversions


@dataclass
class ZTestResult:
    z_stat: float
    p_value: float
    significant: bool
    alpha: float
    relative_lift: float        # (B - A) / A
    ci_lower: float             # 95% CI on difference (B - A)
    ci_upper: float


@dataclass
class ChiSquaredResult:
    chi2_stat: float
    p_value: float
    significant: bool
    alpha: float
    degrees_of_freedom: int


@dataclass
class BayesianResult:
    prob_b_beats_a: float       # P(B > A)
    a_mean: float               # posterior mean conversion rate for A
    b_mean: float               # posterior mean conversion rate for B
    a_ci: tuple                 # 95% credible interval for A
    b_ci: tuple                 # 95% credible interval for B
    expected_lift: float        # E[B - A]
    recommendation: str         # plain-English decision


@dataclass
class PowerAnalysisResult:
    required_sample_per_variant: int
    baseline_rate: float
    minimum_detectable_effect: float
    alpha: float
    power: float


# ─────────────────────────────────────────────────────────────────────────────
# 1. TWO-PROPORTION Z-TEST
# ─────────────────────────────────────────────────────────────────────────────

def z_test(a: VariantData, b: VariantData, alpha: float = 0.05) -> ZTestResult:
    """
    Tests H0: p_A == p_B using a two-proportion z-test (two-tailed).

    Pooled proportion used for SE under H0.
    95% CI on the difference uses unpooled SE (standard practice for CIs).
    """
    p_a, p_b = a.rate, b.rate
    n_a, n_b = a.visitors, b.visitors

    # Pooled proportion under H0
    p_pool = (a.conversions + b.conversions) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    z_stat = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 95% CI on (p_B - p_A) using unpooled SE
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    relative_lift = (p_b - p_a) / p_a if p_a > 0 else 0.0

    return ZTestResult(
        z_stat=round(z_stat, 4),
        p_value=round(p_value, 6),
        significant=p_value < alpha,
        alpha=alpha,
        relative_lift=round(relative_lift, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHI-SQUARED TEST
# ─────────────────────────────────────────────────────────────────────────────

def chi_squared_test(a: VariantData, b: VariantData, alpha: float = 0.05) -> ChiSquaredResult:
    """
    2×2 contingency table chi-squared test.
    Equivalent to z-test for two proportions but uses chi² distribution.
    Yates' correction is NOT applied (standard for large samples).
    """
    table = np.array([
        [a.conversions,     a.non_conversions],
        [b.conversions,     b.non_conversions],
    ])
    chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)

    return ChiSquaredResult(
        chi2_stat=round(chi2, 4),
        p_value=round(p_value, 6),
        significant=p_value < alpha,
        alpha=alpha,
        degrees_of_freedom=dof,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. BAYESIAN BETA-BINOMIAL
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_analysis(
    a: VariantData,
    b: VariantData,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    threshold: float = 0.95,
) -> BayesianResult:
    """
    Beta-Binomial conjugate model with uniform prior (Beta(1,1) = Uniform(0,1)).

    Posterior for each variant:
        p_A | data ~ Beta(prior_alpha + conversions_A, prior_beta + non_conversions_A)
        p_B | data ~ Beta(prior_alpha + conversions_B, prior_beta + non_conversions_B)

    P(B > A) estimated via Monte Carlo sampling (100K draws).
    Decision threshold: declare winner if P(B > A) > 0.95 (or < 0.05).
    """
    # Posterior parameters
    a_alpha = prior_alpha + a.conversions
    a_beta  = prior_beta  + a.non_conversions
    b_alpha = prior_alpha + b.conversions
    b_beta  = prior_beta  + b.non_conversions

    # Posterior means
    a_mean = a_alpha / (a_alpha + a_beta)
    b_mean = b_alpha / (b_alpha + b_beta)

    # 95% credible intervals
    a_ci = beta_dist.ppf([0.025, 0.975], a_alpha, a_beta)
    b_ci = beta_dist.ppf([0.025, 0.975], b_alpha, b_beta)

    # Monte Carlo: P(B > A)
    rng = np.random.default_rng(42)
    samples_a = rng.beta(a_alpha, a_beta, n_samples)
    samples_b = rng.beta(b_alpha, b_beta, n_samples)
    prob_b_beats_a = float(np.mean(samples_b > samples_a))
    expected_lift  = float(np.mean(samples_b - samples_a))

    # Plain-English recommendation
    pct = prob_b_beats_a * 100
    lift_pct = expected_lift * 100
    if prob_b_beats_a >= threshold:
        rec = (f"Offer B wins with {pct:.1f}% probability. "
               f"Expected lift: +{lift_pct:.2f}pp. Ship Offer B.")
    elif prob_b_beats_a <= (1 - threshold):
        rec = (f"Offer A wins with {(1-prob_b_beats_a)*100:.1f}% probability. "
               f"Expected lift for B: {lift_pct:.2f}pp. Keep Offer A.")
    else:
        rec = (f"Inconclusive — B beats A with {pct:.1f}% probability "
               f"(threshold: {threshold*100:.0f}%). Collect more data.")

    return BayesianResult(
        prob_b_beats_a=round(prob_b_beats_a, 4),
        a_mean=round(a_mean, 4),
        b_mean=round(b_mean, 4),
        a_ci=(round(a_ci[0], 4), round(a_ci[1], 4)),
        b_ci=(round(b_ci[0], 4), round(b_ci[1], 4)),
        expected_lift=round(expected_lift, 6),
        recommendation=rec,
    )


def get_posterior_samples(
    a: VariantData,
    b: VariantData,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 2_000,
) -> tuple:
    """
    Returns (samples_a, samples_b) arrays for plotting posterior distributions.
    Uses fewer samples than the full analysis — enough for a smooth plot.
    """
    a_alpha = prior_alpha + a.conversions
    a_beta  = prior_beta  + a.non_conversions
    b_alpha = prior_alpha + b.conversions
    b_beta  = prior_beta  + b.non_conversions

    rng = np.random.default_rng(42)
    return (
        rng.beta(a_alpha, a_beta, n_samples),
        rng.beta(b_alpha, b_beta, n_samples),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. POWER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def power_analysis(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerAnalysisResult:
    """
    Calculates required sample size per variant for a two-proportion z-test.

    baseline_rate — current conversion rate (e.g. 0.10 for 10%)
    mde           — minimum detectable effect as absolute pp (e.g. 0.02 for +2pp)
    alpha         — significance level (default 0.05)
    power         — desired statistical power (default 0.80)
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    # Standard formula for two-proportion z-test sample size
    p_bar = (p1 + p2) / 2
    n = (
        (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
         + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (p2 - p1) ** 2
    )

    return PowerAnalysisResult(
        required_sample_per_variant=int(np.ceil(n)),
        baseline_rate=baseline_rate,
        minimum_detectable_effect=mde,
        alpha=alpha,
        power=power,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL ANALYSIS (combines all three)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(
    a: VariantData,
    b: VariantData,
    alpha: float = 0.05,
) -> dict:
    """Runs z-test, chi-squared, and Bayesian analysis. Returns dict of results."""
    return {
        "z_test":    z_test(a, b, alpha),
        "chi2":      chi_squared_test(a, b, alpha),
        "bayesian":  bayesian_analysis(a, b),
    }


def print_results(a: VariantData, b: VariantData, results: dict) -> None:
    z   = results["z_test"]
    chi = results["chi2"]
    bay = results["bayesian"]

    print(f"\n{'─'*60}")
    print(f"  A/B TEST RESULTS")
    print(f"{'─'*60}")
    print(f"  Variant A ({a.name}): {a.conversions}/{a.visitors} = {a.rate:.2%}")
    print(f"  Variant B ({b.name}): {b.conversions}/{b.visitors} = {b.rate:.2%}")
    print(f"  Relative lift: {z.relative_lift:+.2%}")
    print(f"\n  ── Frequentist ──")
    print(f"  Z-test:      z={z.z_stat:+.4f}  p={z.p_value:.6f}  {'✓ SIGNIFICANT' if z.significant else '✗ not significant'}")
    print(f"  95% CI (B−A): [{z.ci_lower:+.4f}, {z.ci_upper:+.4f}]")
    print(f"  Chi-squared: χ²={chi.chi2_stat:.4f}  p={chi.p_value:.6f}  {'✓ SIGNIFICANT' if chi.significant else '✗ not significant'}")
    print(f"\n  ── Bayesian ──")
    print(f"  P(B > A):    {bay.prob_b_beats_a:.2%}")
    print(f"  A posterior: mean={bay.a_mean:.4f}  95% CI [{bay.a_ci[0]:.4f}, {bay.a_ci[1]:.4f}]")
    print(f"  B posterior: mean={bay.b_mean:.4f}  95% CI [{bay.b_ci[0]:.4f}, {bay.b_ci[1]:.4f}]")
    print(f"\n  → {bay.recommendation}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B Test Analysis")
    parser.add_argument("--a-visitors",    type=int,   required=True)
    parser.add_argument("--a-conversions", type=int,   required=True)
    parser.add_argument("--b-visitors",    type=int,   required=True)
    parser.add_argument("--b-conversions", type=int,   required=True)
    parser.add_argument("--alpha",         type=float, default=0.05)
    args = parser.parse_args()

    a = VariantData("A", args.a_visitors, args.a_conversions)
    b = VariantData("B", args.b_visitors, args.b_conversions)
    results = run_full_analysis(a, b, alpha=args.alpha)
    print_results(a, b, results)
