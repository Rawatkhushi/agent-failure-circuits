"""Offline tests for src/analyze_positions_robust.py (no model / network / torch needed).

Run:  python -m pytest tests/test_positional.py -v

NOTE: these use synthetic arrays and stub models. They validate the
statistics, edge-case handling, label/scoring parity and the alignment/ordering
logic of the extraction code. They do NOT validate anything about GPT-2 or the
real SAE; that is what `src/verify_positional.py` (and --recompute) is for.
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze_positions_robust as ap          # noqa: E402


# ----------------------------------------------------------------- Holm
def test_holm_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(ap.holm(p), [0.03, 0.06, 0.06, 0.02])


def test_holm_nan_and_full_family():
    p = np.array([0.01, np.nan, 0.04])
    out = ap.holm(p)
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.02, 0.04])
    # untested hypotheses count as p=1: m_total=10 -> 0.01*10, then max(.,0.04*9)
    out = ap.holm(p, m_total=10)
    np.testing.assert_allclose(out[[0, 2]], [0.1, 0.36])


def test_holm_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(1)
    p = rng.uniform(size=500) ** 3
    np.testing.assert_allclose(ap.holm(p), sm.multipletests(p, method="holm")[1], atol=1e-12)


# ---------------------------------------------------------- Welch / d
def _synthetic(N=60, F=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.maximum(rng.normal(0.5, 1.0, size=(N, F)), 0)
    is_fail = np.zeros(N, bool)
    is_fail[:17] = True
    rng.shuffle(is_fail)
    return X, is_fail


def test_stats_match_scipy_and_manual():
    X, f = _synthetic()
    df = ap.compute_stats(X, f, min_active=1)
    ref = stats.ttest_ind(X[f], X[~f], equal_var=False)
    np.testing.assert_allclose(df["welch_t"], ref.statistic, rtol=1e-9)
    np.testing.assert_allclose(df["p_raw"], ref.pvalue, rtol=1e-9)
    np.testing.assert_allclose(df["welch_df"], ref.df, rtol=1e-9)
    for k in (0, 7, 23):  # hand-check a few features from the raw arrays
        a, b = X[f, k], X[~f, k]
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        assert df.loc[k, "mean_fail"] == pytest.approx(a.mean())
        assert df.loc[k, "mean_succ"] == pytest.approx(b.mean())
        assert df.loc[k, "delta"] == pytest.approx(a.mean() - b.mean())
        assert df.loc[k, "cohens_d"] == pytest.approx((a.mean() - b.mean()) / pooled)
        assert df.loc[k, "hedges_g"] == pytest.approx(df.loc[k, "cohens_d"] * (1 - 3 / (4 * len(X) - 9)))
        assert df.loc[k, "n_fail_active"] == (a > 0).sum()


def test_direction_sign_failure_minus_success():
    X, f = _synthetic()
    X[f, 3] += 5.0                       # higher on failures
    X[~f, 4] += 5.0                      # higher on successes
    t = ap.positional_stats(X[:, None, :], f, min_active=1)
    r3, r4 = t[t.feature == 3].iloc[0], t[t.feature == 4].iloc[0]
    assert r3.delta > 0 and r3.cohens_d > 0 and r3.welch_t > 0 and r3.direction == "higher_in_failure"
    assert r4.delta < 0 and r4.cohens_d < 0 and r4.direction == "higher_in_success"


# ------------------------------------------------------ edge cases
def test_constant_and_inactive_features():
    X, f = _synthetic()
    X[:, 0] = 0.0                        # never active
    X[:, 1] = 2.5                        # constant non-zero
    X[:, 2] = 0.0; X[0, 2] = 1.0         # active in a single example
    df = ap.compute_stats(X, f, min_active=3)
    assert df.loc[0, "status"] == "constant" and np.isnan(df.loc[0, "p_raw"])
    assert df.loc[1, "status"] == "constant" and np.isnan(df.loc[1, "cohens_d"])
    assert df.loc[2, "status"] == "low_activity" and np.isnan(df.loc[2, "p_raw"])
    assert df.loc[3, "status"] == "tested" and np.isfinite(df.loc[3, "p_raw"])
    t = ap.positional_stats(X[:, None, :], f, min_active=3)
    assert np.isnan(t.loc[t.feature == 0, "p_holm"]).all()      # excluded from the Holm family
    assert (t["p_holm"].dropna() >= t["p_raw"].dropna()).all()  # adjustment never lowers p
    # complete family (primary) is at least as conservative as the tested-only family
    assert (t["p_holm"].dropna() >= t["p_holm_tested_family"].dropna() - 1e-15).all()
    n_tested = int((t.status == "tested").sum())
    k = t[t.status == "tested"].sort_values("p_raw").iloc[0]
    assert k["p_holm"] == pytest.approx(min(1.0, k["p_raw"] * len(t)))              # smallest p x (K*F)
    assert k["p_holm_tested_family"] == pytest.approx(min(1.0, k["p_raw"] * n_tested))


def test_degenerate_both_groups_constant_but_different():
    X, f = _synthetic()
    X[:, 5] = np.where(f, 3.0, 1.0)      # perfectly separated constants
    df = ap.compute_stats(X, f, min_active=1)
    assert df.loc[5, "status"] == "degenerate" and np.isnan(df.loc[5, "p_raw"])


def test_one_group_constant_is_still_tested():
    X, f = _synthetic()
    X[~f, 6] = 0.0                       # successes constant at 0, failures vary
    df = ap.compute_stats(X, f, min_active=1)
    assert df.loc[6, "status"] == "tested" and df.loc[6, "p_raw"] < 1.0
    ref = stats.ttest_ind(X[f, 6], X[~f, 6], equal_var=False)
    assert df.loc[6, "p_raw"] == pytest.approx(ref.pvalue)


@pytest.mark.parametrize("n_fail", [0, 1])
def test_insufficient_group(n_fail):
    X, _ = _synthetic(N=20)
    f = np.zeros(20, bool); f[:n_fail] = True
    df = ap.compute_stats(X, f)
    assert (df["status"] == "insufficient_group").all() and df["p_raw"].isna().all()


# ------------------------------------------------ scoring parity


# --------------------------------------- extraction alignment (stub)










TASKS = [
    ("When Mary and John went to the store, John gave a drink to", " Mary", "Mary."),
    ("After Tom and Anna went to the park, Tom gave the keys to", " Anna", "the car."),
    ("While Lisa and Dan were at the gym, Dan handed a book to", " Lisa", "Lisa. Lisa"),
    ("Once Paul and Sue arrived at the cafe, Paul passed the ball to", " Sue", "Paul"),
    ("When Ben and Amy went to the office, Ben gave the gift to", " Amy", "Amy"),
]








# ----------------------------------------- end-to-end CLI (synthetic)
def _write_synth(tmp_path, N=80, F=300, K=3, seed=0, n_fail=24):
    rng = np.random.default_rng(seed)
    acts = np.maximum(rng.normal(0.3, 1.0, size=(N, K, F)), 0).astype(np.float32) * (rng.uniform(size=(N, K, F)) < 0.5)
    success = np.ones(N, int); success[rng.choice(N, n_fail, replace=False)] = 0
    fail = success == 0
    acts[fail, 1, 7] += 4.0                      # planted: feature 7, position 2 only
    acts[fail, 0, 11] += 4.0                     # planted: feature 11 higher on failures at pos 1 ...
    acts[~fail, 2, 11] += 4.0                    # ... but higher on successes at pos 3 (cancels when averaged)
    meta = pd.DataFrame({"row_index": np.arange(N), "id": np.arange(N), "success": success,
                         "object": rng.choice(["the keys", "a book"], N)})
    for j in range(1, K + 1):
        meta[f"pos{j}_offset"] = j - 1 - K; meta[f"pos{j}_index"] = 10 + j
        meta[f"pos{j}_token"] = [" to", " keys", " the"][j % 3]
    np.save(tmp_path / "a.npy", acts); meta.to_csv(tmp_path / "m.csv", index=False)
    return acts, meta


def _run(tmp_path, *extra):
    argv = ["--acts", str(tmp_path / "a.npy"), "--meta", str(tmp_path / "m.csv"),
            "--out-stats", str(tmp_path / "s.csv"), "--out-top", str(tmp_path / "t.md"),
            "--out-fig", str(tmp_path / "f.png"), "--out-compare", str(tmp_path / "c.md"),
            "--orig-stats", str(tmp_path / "none.csv"), "--bootstrap", "50", "--permutations", "200", *extra]
    return ap.main(argv)


def test_cli_end_to_end_recovers_planted_position_specific_signal(tmp_path):
    acts, meta = _write_synth(tmp_path)
    assert _run(tmp_path) == 0
    s = pd.read_csv(tmp_path / "s.csv")
    assert {"feature", "position", "n_fail", "n_succ", "mean_fail", "mean_succ", "cohens_d", "p_raw", "p_holm"} <= set(s.columns)
    assert len(s) == 3 * 300
    r = s[(s.feature == 7)].set_index("position")
    assert r.loc[2, "cohens_d"] > 1.5 and r.loc[2, "p_holm"] < 0.01 and r.loc[2, "direction"] == "higher_in_failure"
    assert r.loc[1, "p_holm"] > 0.05 and r.loc[3, "p_holm"] > 0.05      # absent at the other positions
    assert r.loc[2, "p_maxT_perm"] <= 0.01 and r.loc[1, "p_maxT_perm"] > 0.05 and r.loc[3, "p_maxT_perm"] > 0.05
    assert (r["n_fail"] == 24).all() and (r["n_succ"] == 56).all()
    # CSV numbers equal a direct computation from the raw array
    a, b = acts[meta.success == 0, 1, 7].astype(float), acts[meta.success == 1, 1, 7].astype(float)
    assert r.loc[2, "mean_fail"] == pytest.approx(a.mean()) and r.loc[2, "mean_succ"] == pytest.approx(b.mean())
    assert r.loc[2, "p_raw"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)
    md = (tmp_path / "t.md").read_text()
    assert "Position 2 (offset -2)" in md and "Position 1 (offset -3)" in md and "Position 3 (offset -1)" in md
    assert "higher on failures" in md and "causes failure" in md
    assert s.groupby("position")["position_offset"].first().tolist() == [-3, -2, -1]
    cmp_md = (tmp_path / "c.md").read_text()
    n_flip = int(re.search(r"sign cancellation\): \*\*(\d+)\*\*", cmp_md).group(1))
    assert n_flip >= 1
    assert (tmp_path / "f.png").stat().st_size > 10_000


def test_cli_exclude_object_and_suffix(tmp_path):
    _write_synth(tmp_path)
    assert _run(tmp_path, "--exclude-object", "the keys", "--suffix", "_nk") == 0
    assert (tmp_path / "s_nk.csv").exists()
    s = pd.read_csv(tmp_path / "s_nk.csv")
    assert s["n_fail"].iloc[0] + s["n_succ"].iloc[0] < 80


def test_cli_too_few_failures(tmp_path):
    _write_synth(tmp_path, n_fail=1)
    assert _run(tmp_path) == 2
    assert "NOT performed" in (tmp_path / "t.md").read_text()
    assert len(pd.read_csv(tmp_path / "s.csv")) == 0


def test_cli_small_group_warns_but_runs(tmp_path, capsys):
    _write_synth(tmp_path, n_fail=3)
    assert _run(tmp_path) == 0
    assert "WARNING" in capsys.readouterr().out
    assert "WARNING" in (tmp_path / "t.md").read_text()


def test_cli_rejects_misaligned_metadata(tmp_path):
    _write_synth(tmp_path)
    m = pd.read_csv(tmp_path / "m.csv").iloc[:-1]
    m.to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(SystemExit):
        _run(tmp_path)


# ------------------------------- runner main(): file layout, overwrite guard


# ----------------------------------------------------- max-T permutation test
def test_maxT_matches_bruteforce_scipy():
    rng = np.random.default_rng(3)
    N, K, F, B = 24, 2, 5, 40
    acts = np.maximum(rng.normal(0.3, 1.0, size=(N, K, F)), 0)
    is_fail = np.zeros(N, bool); is_fail[:7] = True
    acts[is_fail, 1, 2] += 2.0
    table = ap.positional_stats(acts, is_fail, min_active=1)
    tested = (table["status"] == "tested").values.reshape(K, F)
    t_obs = table["welch_t"].values.reshape(K, F)
    assert tested.all()
    ind = np.stack([rng.permutation(is_fail).astype(float) for _ in range(B)])
    got = ap.maxT_perm_pvalues(acts, is_fail, tested, t_obs, B, perm_ind=ind)
    max_null = []
    for b in range(B):
        g = ind[b] == 1
        ts = [stats.ttest_ind(acts[g, j, f], acts[~g, j, f], equal_var=False).statistic for j in range(K) for f in range(F)]
        max_null.append(np.max(np.abs(ts)))
    max_null = np.array(max_null)
    exp = np.array([[(1 + (max_null >= abs(t_obs[j, f]) - 1e-9).sum()) / (B + 1) for f in range(F)] for j in range(K)])
    np.testing.assert_allclose(got, exp)
    assert got[1, 2] == got.min()                       # planted feature is the most extreme


def test_maxT_controls_fwer_on_sparse_null_where_welch_holm_may_not():
    """Random labels + SAE-like sparse skewed features: max-T should rarely give any hit."""
    rng = np.random.default_rng(11)
    N, K, F, reps, B = 120, 2, 300, 40, 200
    hits_maxT = 0
    for r in range(reps):
        act = np.where(rng.uniform(size=(N, K, F)) < 0.1, rng.exponential(1.0, size=(N, K, F)), 0.0)
        f = np.zeros(N, bool); f[rng.choice(N, 25, replace=False)] = True      # labels independent of data
        table = ap.positional_stats(act, f, min_active=3)
        tested = (table["status"] == "tested").values.reshape(K, F)
        p = ap.maxT_perm_pvalues(act, f, tested, table["welch_t"].values.reshape(K, F), B, np.random.default_rng(r))
        hits_maxT += int(np.nanmin(p) < 0.05)
    # nominal 5% of 40 = 2 expected; P(X >= 8 | Binomial(40, .05)) < 0.001
    assert hits_maxT <= 7


def test_maxT_skipped_when_zero_permutations(tmp_path):
    _write_synth(tmp_path)
    assert _run(tmp_path, "--permutations", "0") == 0
    s = pd.read_csv(tmp_path / "s.csv")
    assert s["p_maxT_perm"].isna().all()
    assert "max-T permutation not run" in (tmp_path / "t.md").read_text()


def test_stratified_maxT_removes_confounded_feature_but_keeps_real_one():
    rng = np.random.default_rng(5)
    N, F = 160, 60
    strata = np.array(["keys"] * 40 + ["other"] * 120)
    is_fail = np.zeros(N, bool); is_fail[:36] = True                 # 36/40 fail in 'keys' ...
    is_fail[40:44] = True                                            # ... 4/120 elsewhere
    act = np.maximum(rng.normal(0.2, 1.0, size=(N, 1, F)), 0)
    act[strata == "keys", 0, 0] += 5.0                               # feature 0 = pure stratum detector
    act[is_fail & (strata == "other"), 0, 1] += 4.0                  # feature 1 = real within-stratum signal
    tab = ap.positional_stats(act, is_fail, min_active=1)
    tested = (tab.status == "tested").values.reshape(1, F); t_obs = tab.welch_t.values.reshape(1, F)
    plain = ap.maxT_perm_pvalues(act, is_fail, tested, t_obs, 300, np.random.default_rng(0))
    strat = ap.maxT_perm_pvalues(act, is_fail, tested, t_obs, 300, np.random.default_rng(0), strata=strata)
    assert plain[0, 0] < 0.01 and strat[0, 0] > 0.05                  # confound: significant plain, not stratified
    assert strat[0, 1] < 0.05                                          # real signal survives stratification
