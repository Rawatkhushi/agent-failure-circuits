"""Per-token-position analysis of SAE features: failure vs. success (robust version:
Welch+Holm over the complete family, plus a max-T permutation test).
Accepts both metadata schemas (pos{j}_token or position_{j}_token) and joins `object` from data/tasks.jsonl.

Input (produced by src/run_experiment_positional.py):
    results/positional_activations.npy   (N, K, d_sae)   K = last-K window
    results/positional_metadata.csv      row i <-> activations[i]

For every window position j (1 = K-th token from the end ... K = final
prompt token) and every SAE feature, compares activations between failed
and successful predictions.

Direction convention (everywhere in this script and its outputs):
    delta     = mean(failure) - mean(success)
    cohens_d  = delta / pooled SD     (positive => HIGHER on failures)
    t         = Welch t with the same sign

Statistics per (feature, position):
    * Welch's two-sided t-test, Welch-Satterthwaite df. (The original
      src/analyze.py uses the more conservative df = min(n)-1; this script
      uses the standard Welch df, so raw p-values differ slightly.)
    * Cohen's d (pooled SD) and Hedges' g (small-sample corrected d).
    * Holm-Bonferroni across the COMPLETE family of K * d_sae
      (feature, position) hypotheses (p_holm). Pairs that could not be tested
      (see status) count as p = 1, which is exactly what the original
      src/analyze.py does for its d_sae features (never-active features get
      p = 1). p_holm_tested_family is the less conservative variant whose
      family is only the pairs that were actually tested.

    * p_maxT_perm: single-step max-|t| permutation adjusted p-value
      (Westfall-Young) over ALL tested (feature, position) pairs, from
      --permutations random relabellings that keep the group sizes fixed.
      It makes no normality assumption. This matters: SAE activations are
      sparse and skewed, and on null data with that structure Welch p-values
      are anti-conservative (see README), so Welch+Holm can over-call
      features that are active in only a handful of failures. Resolution is
      1/(B+1). Treat features as robust only if BOTH p_holm and p_maxT_perm
      are small.

A (feature, position) pair is only tested if it passes label-independent
filters; otherwise it is kept in the CSV with an explicit `status`:
    insufficient_group  <2 failures or <2 successes in the data
    constant            identical value in every example (incl. never active)
    low_activity        active (>0) in fewer than --min-active examples overall
    degenerate          zero variance within BOTH groups (t undefined)
    tested              Welch test performed

These are observational associations in a small sample. A difference in
activation does NOT show that a feature causes failure, and the last-K
positions are not semantically aligned slots across prompts (see
src/run_experiment_positional.py and the README).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NEURONPEDIA = "https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feat}"


# ----------------------------------------------------------------------
# Statistics (pure numpy; unit-tested in tests/test_positional.py)
# ----------------------------------------------------------------------
def holm(p: np.ndarray, m_total: int | None = None) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values. NaNs are ignored (stay NaN).

    m_total: size of the full family. Hypotheses not supplied are treated as
    having p = 1 (they occupy the largest ranks), so supplied p-values are
    multiplied by (m_total - rank). Default: number of finite p-values.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if pv.size == 0:
        return out
    m = pv.size if m_total is None else max(int(m_total), pv.size)
    order = np.argsort(pv, kind="stable")
    mult = m - np.arange(pv.size)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(pv[order] * mult))
    adj = np.empty(pv.size)
    adj[order] = adj_sorted
    out[ok] = adj
    return out


def compute_stats(X: np.ndarray, is_fail: np.ndarray, min_active: int = 3) -> pd.DataFrame:
    """Per-feature failure-vs-success statistics for one position.

    X: (N, F) activations;  is_fail: (N,) bool.  Returns one row per feature.
    """
    X = np.asarray(X, dtype=np.float64)
    is_fail = np.asarray(is_fail, dtype=bool)
    N, F = X.shape
    fail, succ = X[is_fail], X[~is_fail]
    nf, ns = len(fail), len(succ)
    nan = np.full(F, np.nan)

    cols = {"feature": np.arange(F), "n_fail": np.full(F, nf), "n_succ": np.full(F, ns)}
    if nf < 2 or ns < 2:
        cols.update({k: nan.copy() for k in (
            "n_fail_active", "n_succ_active", "mean_fail", "mean_succ", "delta", "sd_fail",
            "sd_succ", "cohens_d", "hedges_g", "welch_t", "welch_df", "p_raw")})
        cols["status"] = np.full(F, "insufficient_group", dtype=object)
        return pd.DataFrame(cols)

    mf, ms = fail.mean(0), succ.mean(0)
    # Exact constant detection via min/max (avoids float-noise "variances").
    const_f = fail.max(0) == fail.min(0)
    const_s = succ.max(0) == succ.min(0)
    const_all = X.max(0) == X.min(0)
    vf = np.where(const_f, 0.0, fail.var(0, ddof=1))
    vs = np.where(const_s, 0.0, succ.var(0, ddof=1))
    act_f, act_s = (fail > 0).sum(0), (succ > 0).sum(0)

    status = np.full(F, "tested", dtype=object)
    status[(const_f & const_s)] = "degenerate"
    status[(act_f + act_s) < min_active] = "low_activity"
    status[const_all] = "constant"
    tested = status == "tested"

    delta = mf - ms
    pooled = ((nf - 1) * vf + (ns - 1) * vs) / (nf + ns - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(pooled > 0, delta / np.sqrt(pooled), np.nan)
        se2 = vf / nf + vs / ns
        t = np.where(tested, delta / np.sqrt(se2), np.nan)
        df = np.where(tested, se2 ** 2 / ((vf / nf) ** 2 / (nf - 1) + (vs / ns) ** 2 / (ns - 1)), np.nan)
    p = np.full(F, np.nan)
    p[tested] = 2.0 * stats.t.sf(np.abs(t[tested]), df[tested])
    hedges = 1.0 - 3.0 / (4.0 * (nf + ns) - 9.0)

    cols.update({
        "n_fail_active": act_f, "n_succ_active": act_s,
        "mean_fail": mf, "mean_succ": ms, "delta": delta,
        "sd_fail": np.sqrt(vf), "sd_succ": np.sqrt(vs),
        "cohens_d": d, "hedges_g": d * hedges,
        "welch_t": t, "welch_df": df, "p_raw": p, "status": status,
    })
    return pd.DataFrame(cols)


def positional_stats(acts: np.ndarray, is_fail: np.ndarray, min_active: int = 3) -> pd.DataFrame:
    """Stats for every window position, with Holm across the whole family."""
    N, K, F = acts.shape
    frames = []
    for j in range(K):
        df = compute_stats(acts[:, j, :], is_fail, min_active)
        df.insert(1, "position", j + 1)
        df.insert(2, "position_offset", j - K)        # -3, -2, -1 for K=3
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["p_holm"] = holm(out["p_raw"].values, m_total=K * F)          # complete family (primary)
    out["p_holm_tested_family"] = holm(out["p_raw"].values)           # tested pairs only (secondary)
    out["direction"] = np.where(out["delta"] > 0, "higher_in_failure",
                         np.where(out["delta"] < 0, "higher_in_success", "none"))
    out.loc[out["delta"].isna(), "direction"] = "none"
    return out


def _welch_t_from_sums(S1, Q1, T1, T2, nf, ns):
    """Welch t (fail - succ) from per-permutation sums. S1/Q1: (B, m) sum and
    sum-of-squares of the 'failure' group; T1/T2: (m,) totals over all examples."""
    S0, Q0 = T1 - S1, T2 - Q1
    mf, ms = S1 / nf, S0 / ns
    vf = np.maximum((Q1 - S1 ** 2 / nf) / (nf - 1), 0.0)
    vs = np.maximum((Q0 - S0 ** 2 / ns) / (ns - 1), 0.0)
    vf = np.where(vf <= 1e-12 * Q1 / nf, 0.0, vf)       # kill cancellation noise in constant groups
    vs = np.where(vs <= 1e-12 * Q0 / ns, 0.0, vs)
    se = np.sqrt(vf / nf + vs / ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, (mf - ms) / se, 0.0)


def maxT_perm_pvalues(acts: np.ndarray, is_fail: np.ndarray, tested: np.ndarray,
                      t_obs: np.ndarray, B: int, rng: np.random.Generator | None = None,
                      perm_ind: np.ndarray | None = None, chunk: int = 100,
                      strata: np.ndarray | None = None) -> np.ndarray:
    """Single-step max-|t| permutation adjusted p-values (Westfall-Young).

    acts (N,K,F); tested, t_obs (K,F) (t_obs ignored where not tested).
    Each permutation reassigns the failure label (group sizes fixed), recomputes
    Welch t for every tested pair and records the maximum |t| over the whole
    family. p_adj = (1 + #{b : max_b >= |t_obs|}) / (B + 1).
    perm_ind: optional (B, N) 0/1 matrix of 'failure' indicators (for tests).
    strata: optional (N,) labels; if given, activations are centred within each
    stratum and failure labels are shuffled only WITHIN each stratum (per-stratum
    failure counts preserved): a test of failure association conditional on the
    stratum (e.g. object / token identity). t_obs is then recomputed internally.
    """
    N, K, F = acts.shape
    out = np.full((K, F), np.nan)
    if not tested.any() or (B <= 0 and perm_ind is None):
        return out
    nf, ns = int(is_fail.sum()), int((~is_fail).sum())
    if perm_ind is None:
        rng = rng or np.random.default_rng(0)
    else:
        B = perm_ind.shape[0]
    cols = [np.where(tested[j])[0] for j in range(K)]
    Xs = [acts[:, j, c].astype(np.float64) for j, c in enumerate(cols)]
    if strata is not None:
        # Conditional statistic: remove each stratum's mean activation (invariant to label
        # shuffling), so features that merely track the stratum carry no signal, then
        # recompute the observed Welch t on the residuals.
        for X in Xs:
            for g in np.unique(strata):
                idx = np.where(strata == g)[0]
                mu = X[idx].mean(0)
                r = X[idx] - mu
                r[np.abs(r) <= 1e-10 * np.abs(mu)] = 0.0
                X[idx] = r
        t_obs = np.zeros((K, F))
        fl = is_fail.astype(np.float64)
        for j, X in enumerate(Xs):
            if len(cols[j]):
                t_obs[j, cols[j]] = _welch_t_from_sums((fl @ X)[None], (fl @ X ** 2)[None],
                                                       X.sum(0), (X ** 2).sum(0), nf, ns)[0]
    T1 = [x.sum(0) for x in Xs]
    T2 = [(x ** 2).sum(0) for x in Xs]
    max_null = np.empty(B)
    for lo in range(0, B, chunk):
        hi = min(B, lo + chunk)
        if perm_ind is None:
            ind = np.zeros((hi - lo, N))
            if strata is None:
                sel = np.argsort(rng.random((hi - lo, N)), axis=1)[:, :nf]
                np.put_along_axis(ind, sel, 1.0, axis=1)
            else:
                for g in np.unique(strata):
                    idx = np.where(strata == g)[0]
                    k = int(is_fail[idx].sum())
                    if k == 0:
                        continue
                    sel = np.argsort(rng.random((hi - lo, len(idx))), axis=1)[:, :k]
                    np.put_along_axis(ind, idx[sel], 1.0, axis=1)
        else:
            ind = perm_ind[lo:hi].astype(np.float64)
        m = np.zeros(hi - lo)
        for j in range(K):
            if len(cols[j]) == 0:
                continue
            S1, Q1 = ind @ Xs[j], ind @ (Xs[j] ** 2)
            t = _welch_t_from_sums(S1, Q1, T1[j], T2[j], nf, ns)
            m = np.maximum(m, np.abs(t).max(axis=1))
        max_null[lo:hi] = m
    max_null.sort()
    for j in range(K):
        if len(cols[j]) == 0:
            continue
        a = np.abs(t_obs[j, cols[j]])
        n_ge = B - np.searchsorted(max_null, a - 1e-12, side="left")   # count of max_null >= |t_obs|
        out[j, cols[j]] = (1.0 + n_ge) / (B + 1.0)
    return out


def bootstrap_d_ci(X: np.ndarray, is_fail: np.ndarray, feats: np.ndarray,
                   B: int, rng: np.random.Generator) -> np.ndarray:
    """95% percentile bootstrap CI of Cohen's d for selected features.

    Resamples examples with replacement *within* each group. Returns
    (len(feats), 2). NaN if B == 0 or a resample has zero pooled variance.
    """
    out = np.full((len(feats), 2), np.nan)
    if B <= 0 or len(feats) == 0:
        return out
    fail, succ = X[is_fail][:, feats].astype(np.float64), X[~is_fail][:, feats].astype(np.float64)
    nf, ns = len(fail), len(succ)
    ds = np.empty((B, len(feats)))
    for b in range(B):
        f = fail[rng.integers(0, nf, nf)]
        s = succ[rng.integers(0, ns, ns)]
        pooled = ((nf - 1) * f.var(0, ddof=1) + (ns - 1) * s.var(0, ddof=1)) / (nf + ns - 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ds[b] = np.where(pooled > 0, (f.mean(0) - s.mean(0)) / np.sqrt(pooled), np.nan)
    with np.errstate(all="ignore"):
        out[:, 0] = np.nanpercentile(ds, 2.5, axis=0)
        out[:, 1] = np.nanpercentile(ds, 97.5, axis=0)
    return out


def mwu_pvalues(X: np.ndarray, is_fail: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Two-sided Mann-Whitney U p-values for selected features (robustness
    check; SAE activations are sparse and non-normal)."""
    if len(feats) == 0:
        return np.array([])
    try:
        res = stats.mannwhitneyu(X[is_fail][:, feats], X[~is_fail][:, feats],
                                 axis=0, alternative="two-sided")
        return np.asarray(res.pvalue)
    except Exception:
        return np.full(len(feats), np.nan)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def token_composition_md(meta: pd.DataFrame, K: int) -> list[str]:
    lines = ["## Which tokens sit at each position?", "",
             "Positions are counted from the end of the prompt, so the token at a given "
             "position differs across prompts. Failure rate per token identity shows how "
             "strongly token identity is confounded with the label.", ""]
    for j in range(1, K + 1):
        col = f"pos{j}_token"
        if col not in meta.columns:
            continue
        g = meta.groupby(col)["success"].agg(n="count", n_fail=lambda s: int((s == 0).sum()))
        g["fail_rate"] = g["n_fail"] / g["n"]
        g = g.sort_values("n", ascending=False)
        lines += [f"**Position {j}** (offset {meta[f'pos{j}_offset'].iloc[0]}, "
                  f"absolute index {meta[f'pos{j}_index'].min()}-{meta[f'pos{j}_index'].max()}):", "",
                  "| token | n | n fail | fail rate |", "|:---|---:|---:|---:|"]
        for tok, r in g.head(12).iterrows():
            lines.append(f"| `{tok!r}` | {int(r['n'])} | {int(r['n_fail'])} | {r['fail_rate']:.2f} |")
        lines.append("")
    return lines


def write_top_features_md(path, table, acts, is_fail, meta, top_k, layer, boot_B, seed, header_extra):
    N, K, F = acts.shape
    nf, ns = int(is_fail.sum()), int((~is_fail).sum())
    rng = np.random.default_rng(seed)
    L = ["# Top SAE features by token position (failure vs. success)", "",
         f"Model: gpt2-small | SAE: gpt2-small-res-jb/blocks.{layer}.hook_resid_pre | "
         f"N = {N} ({nf} failures, {ns} successes, accuracy {ns / N:.3f}) | window K = {K}", ""]
    L += header_extra
    L += ["**Direction:** `delta = mean(failure) - mean(success)`; `d > 0` means the feature is "
          "*higher on failures* at that position. **These are associations in a small sample, not "
          "evidence that a feature causes failure.** Tests are Welch two-sided; `p Holm` corrects over "
          f"the complete family of {K} x {F:,} = {K * F:,} (feature, position) hypotheses (untested pairs count as p = 1, "
          "as in the original analysis); `p Holm (tested)` uses only the pairs actually tested (less conservative). `p max-T (strat)` (if `--stratify-by`) shuffles labels only within each level of the stratifying column, so it tests association beyond that column (e.g. token identity). `p max-T (perm)` is a single-step max-|t| "
          "permutation adjusted p over all tested pairs (no normality assumption; resolution 1/(B+1); NaN if not run). "
          "Welch p-values can be anti-conservative for sparse features, so prefer features where BOTH Holm and max-T are small. `p MWU` is an uncorrected Mann-Whitney check on the displayed rows "
          f"(activations are sparse/non-normal). `d 95% CI` is a {boot_B}-resample within-group bootstrap.", ""]
    L += token_composition_md(meta, K)

    n_tested = int((table["status"] == "tested").sum())
    L += [f"## Test family", "",
          f"{n_tested:,} of {K * F:,} (feature, position) pairs were tested. Excluded pairs: "
          + (", ".join(f"{k}: {v:,}" for k, v in table["status"].value_counts().items() if k != "tested") or "none"), ""]

    for j in range(1, K + 1):
        sub = table[(table["position"] == j) & (table["status"] == "tested")]
        Xj = acts[:, j - 1, :]
        n_sig = int((sub["p_holm"] < 0.05).sum())
        n_sig_f = int(((sub["p_holm"] < 0.05) & (sub["delta"] > 0)).sum())
        n_mt = int((sub["p_maxT_perm"] < 0.05).sum()) if sub["p_maxT_perm"].notna().any() else None
        L += [f"## Position {j} (offset {j - 1 - K})", "",
              f"Tested features: {len(sub):,}. Holm-significant (p < 0.05): {n_sig} "
              f"({n_sig_f} higher in failure, {n_sig - n_sig_f} higher in success). "
              + (f"max-T permutation p < 0.05: {n_mt}. " if n_mt is not None else "max-T permutation not run. ")
              + (f"Stratified (within-group) max-T p < 0.05: {int((sub['p_maxT_strat'] < 0.05).sum())}. " if sub["p_maxT_strat"].notna().any() else "")
              + 
              f"|d| > 0.8: {int((sub['cohens_d'].abs() > 0.8).sum())}.", ""]
        for title, mask, asc in (("Features higher in FAILURE", sub["delta"] > 0, False),
                                 ("Features higher in SUCCESS", sub["delta"] < 0, True)):
            top = sub[mask].sort_values("cohens_d", ascending=asc).head(top_k if not asc else min(top_k, 10))
            L += [f"### {title} (position {j}), top {len(top)} by |d|", ""]
            if top.empty:
                L += ["_none_", ""]
                continue
            feats = top["feature"].values.astype(int)
            ci = bootstrap_d_ci(Xj, is_fail, feats, boot_B, rng)
            pm = mwu_pvalues(Xj, is_fail, feats)
            L += ["| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |",
                  "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|"]
            for r, (i, row) in enumerate(top.iterrows()):
                url = NEURONPEDIA.format(layer=layer, feat=int(row["feature"]))
                L.append(
                    f"| {r + 1} | {int(row['feature'])} | {row['cohens_d']:+.2f} | "
                    f"[{ci[r, 0]:+.2f}, {ci[r, 1]:+.2f}] | {row['hedges_g']:+.2f} | "
                    f"{row['mean_fail']:.3f} | {row['mean_succ']:.3f} | "
                    f"{int(row['n_fail_active'])}/{nf}, {int(row['n_succ_active'])}/{ns} | "
                    f"{_fmt_p(row['p_raw'])} | {_fmt_p(row['p_holm'])} | "
                    f"{_fmt_p(row['p_holm_tested_family'])} | {_fmt_p(row['p_maxT_perm'])} | {_fmt_p(row['p_maxT_strat'])} | {_fmt_p(pm[r])} | [link]({url}) |")
            L.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n")


def make_figure(path, table, K, top_per_pos=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table[table["status"] == "tested"].copy()
    t["nlp"] = -np.log10(t["p_raw"].clip(lower=1e-300))
    fig = plt.figure(figsize=(5 * K, 11))
    gs = fig.add_gridspec(2, K, height_ratios=[1, 1.15], hspace=0.35, wspace=0.28)

    xmax = max(1.0, float(t["cohens_d"].abs().max())) if len(t) else 1.0   # true max: never clip the strongest effects
    for j in range(1, K + 1):
        ax = fig.add_subplot(gs[0, j - 1])
        s = t[t["position"] == j]
        sig = s["p_holm"] < 0.05
        ax.scatter(s.loc[~sig, "cohens_d"], s.loc[~sig, "nlp"], s=4, c="lightgray", rasterized=True)
        ax.scatter(s.loc[sig, "cohens_d"], s.loc[sig, "nlp"], s=10, c="firebrick", rasterized=True,
                   label=f"Welch+Holm p<0.05 ({int(sig.sum())})")
        mt = s["p_maxT_perm"] < 0.05
        if mt.any():
            ax.scatter(s.loc[mt, "cohens_d"], s.loc[mt, "nlp"], s=45, facecolors="none", edgecolors="k",
                       linewidths=0.8, label=f"max-T perm p<0.05 ({int(mt.sum())})")
        for _, r in s.reindex(s["cohens_d"].abs().sort_values(ascending=False).index).head(3).iterrows():
            ax.annotate(str(int(r["feature"])), (r["cohens_d"], r["nlp"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axvline(0, color="k", lw=0.6)
        for x in (-0.8, 0.8):
            ax.axvline(x, color="steelblue", ls="--", lw=0.6)
        ax.set_xlim(-xmax * 1.05, xmax * 1.05)
        ax.set_title(f"Position {j} (offset {j - 1 - K})")
        ax.set_xlabel("Cohen's d  (failure - success)")
        ax.set_ylabel("-log10 raw p (Welch)")
        ax.legend(fontsize=7, loc="best")

    # Heatmap: union of the strongest features per position, d at all positions.
    sel = []
    for j in range(1, K + 1):
        s = t[t["position"] == j]
        sel += list(s.reindex(s["cohens_d"].abs().sort_values(ascending=False).index)["feature"].head(top_per_pos))
    sel = list(dict.fromkeys(int(f) for f in sel))
    piv_d = table[table["feature"].isin(sel)].pivot(index="feature", columns="position", values="cohens_d")
    piv_p = table[table["feature"].isin(sel)].pivot(index="feature", columns="position", values="p_holm")
    if len(sel):
        piv_d = piv_d.loc[sel]; piv_p = piv_p.loc[sel]
        order = piv_d.abs().max(axis=1).sort_values(ascending=False).index
        piv_d, piv_p = piv_d.loc[order], piv_p.loc[order]
    ax = fig.add_subplot(gs[1, :K - 1] if K > 1 else gs[1, :])
    vmax = float(np.nanmin([4.0, np.nanmax(np.abs(piv_d.values))])) if len(sel) else 1.0
    im = ax.imshow(piv_d.values.astype(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"pos {j}" for j in range(1, K + 1)])
    ax.set_yticks(range(len(piv_d))); ax.set_yticklabels([f"f{i}" for i in piv_d.index], fontsize=7)
    for a in range(piv_d.shape[0]):
        for b in range(piv_d.shape[1]):
            v, p = piv_d.values[a, b], piv_p.values[a, b]
            txt = "n/t" if not np.isfinite(v) else f"{v:+.1f}" + ("*" if np.isfinite(p) and p < 0.05 else "")
            ax.text(b, a, txt, ha="center", va="center", fontsize=6)
    ax.set_title("Cohen's d (failure - success) for the strongest features of each position\n"
                 "red = higher on failures; * = Holm p<0.05; n/t = not tested at that position")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    if K > 1:
        ax2 = fig.add_subplot(gs[1, K - 1])
        pos = np.arange(1, K + 1)
        hf = [int(((t.position == j) & (t.p_holm < 0.05) & (t.delta > 0)).sum()) for j in pos]
        hs = [int(((t.position == j) & (t.p_holm < 0.05) & (t.delta < 0)).sum()) for j in pos]
        ax2.bar(pos - 0.2, hf, 0.4, color="firebrick", label="higher in failure")
        ax2.bar(pos + 0.2, hs, 0.4, color="steelblue", label="higher in success")
        ax2.set_xticks(pos); ax2.set_xlabel("window position")
        ax2.set_ylabel("# Holm-significant features"); ax2.legend(fontsize=7)
        ax2.set_title("Significant features per position")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_comparison_md(path, table, acts, is_fail, min_active, top_k, orig_stats_path, K):
    """Averaged-over-positions analysis vs per-position analysis, SAME examples."""
    N, _, F = acts.shape
    avg = compute_stats(acts.mean(axis=1), is_fail, min_active)
    avg["p_holm"] = holm(avg["p_raw"].values, m_total=F)               # complete family, as in src/analyze.py
    avg_t = avg[avg["status"] == "tested"]
    L = ["# Averaged vs. per-position analysis (same examples, same statistics code)", "",
         f"N = {N} ({int(is_fail.sum())} failures). `avg` = mean of the {K} positions' SAE activations "
         "(what src/analyze.py does, recomputed here with this script's Welch test so that the "
         "two analyses are directly comparable). Holm uses the complete family in both cases: "
         f"{F:,} hypotheses for the averaged analysis, {K * F:,} ({K} x {F:,}) for the per-position analysis, "
         "so the per-position correction is stricter by construction. Observations below are descriptive; "
         "no outcome is presumed.", ""]

    # Counts
    L += ["## Counts", "", "| analysis | tested | Holm p<0.05 | \\|d\\|>0.8 |", "|:---|---:|---:|---:|"]
    L.append(f"| average | {len(avg_t)} | {int((avg_t['p_holm'] < 0.05).sum())} | {int((avg_t['cohens_d'].abs() > 0.8).sum())} |")
    for j in range(1, K + 1):
        s = table[(table.position == j) & (table.status == "tested")]
        L.append(f"| position {j} | {len(s)} | {int((s['p_holm'] < 0.05).sum())} | {int((s['cohens_d'].abs() > 0.8).sum())} |")
    L.append("")

    # Overlap of top-K by |d|
    def topset(df, k=top_k):
        return set(df.reindex(df["cohens_d"].abs().sort_values(ascending=False).index)["feature"].head(k).astype(int))
    sets = {"average": topset(avg_t)}
    for j in range(1, K + 1):
        sets[f"pos{j}"] = topset(table[(table.position == j) & (table.status == "tested")])
    names = list(sets)
    L += [f"## Overlap of the top-{top_k} features by |d|", "",
          "| | " + " | ".join(names) + " |", "|:---|" + "---:|" * len(names)]
    for a in names:
        L.append(f"| {a} | " + " | ".join(str(len(sets[a] & sets[b])) for b in names) + " |")
    L.append("")

    # d of the top averaged features at each position
    d_by_pos = {j: table[table.position == j].set_index("feature")["cohens_d"] for j in range(1, K + 1)}
    p_by_pos = {j: table[table.position == j].set_index("feature")["p_holm"] for j in range(1, K + 1)}
    L += [f"## Top-10 averaged features: where does the effect live?", "",
          "| feature | d avg | " + " | ".join(f"d pos{j}" for j in range(1, K + 1)) + " | strongest position |",
          "|---:|---:|" + "---:|" * K + ":---|"]
    top_avg = avg_t.reindex(avg_t["cohens_d"].abs().sort_values(ascending=False).index).head(10)
    for _, r in top_avg.iterrows():
        f = int(r["feature"])
        ds = [d_by_pos[j].get(f, np.nan) for j in range(1, K + 1)]
        best = int(np.nanargmax(np.abs(ds))) + 1 if np.isfinite(ds).any() else None
        L.append(f"| {f} | {r['cohens_d']:+.2f} | " + " | ".join("n/a" if not np.isfinite(v) else f"{v:+.2f}" for v in ds)
                 + f" | {'pos ' + str(best) if best else 'n/a'} |")
    L.append("")

    # Features hidden / cancelled by averaging
    D = pd.DataFrame({f"d{j}": d_by_pos[j] for j in range(1, K + 1)})
    D["d_avg"] = avg.set_index("feature")["cohens_d"]
    D["status_avg"] = avg.set_index("feature")["status"]
    dm = D[[f"d{j}" for j in range(1, K + 1)]]
    D["max_abs_pos"] = dm.abs().max(axis=1)
    hidden = D[(D["max_abs_pos"] >= 0.8) & (D["d_avg"].abs().fillna(0) < 0.5)]
    # require the strong positional effect to be a tested one
    tested_pairs = set(map(tuple, table.loc[table.status == "tested", ["feature", "position"]].values))
    def strong_tested(f):
        return any((f, j) in tested_pairs and abs(D.loc[f, f"d{j}"]) >= 0.8 for j in range(1, K + 1))
    hidden = hidden[[strong_tested(int(f)) for f in hidden.index]]
    flip = D[(dm.max(axis=1) >= 0.5) & (dm.min(axis=1) <= -0.5)]
    L += ["## Possible effects of averaging", "",
          f"* Features with a *tested* |d| >= 0.8 at some position but |d_avg| < 0.5 (or untested after averaging): **{len(hidden)}**",
          f"* Features with d >= +0.5 at one position and d <= -0.5 at another (sign cancellation): **{len(flip)}**", ""]
    if len(hidden):
        L += ["Largest 'hidden by averaging' cases (by position |d| minus |d_avg|):", "",
              "| feature | " + " | ".join(f"d pos{j}" for j in range(1, K + 1)) + " | d avg |", "|---:|" + "---:|" * (K + 1)]
        h = hidden.assign(gap=hidden["max_abs_pos"] - hidden["d_avg"].abs().fillna(0)).sort_values("gap", ascending=False).head(10)
        for f, r in h.iterrows():
            L.append(f"| {int(f)} | " + " | ".join("n/a" if not np.isfinite(r[f'd{j}']) else f"{r[f'd{j}']:+.2f}" for j in range(1, K + 1))
                     + f" | {'n/a' if not np.isfinite(r['d_avg']) else format(r['d_avg'], '+.2f')} |")
        L.append("")

    # Rank agreement
    L += ["## Rank agreement of d across analyses (Spearman, features tested in both)", ""]
    pairs = [(f"pos{a}", f"pos{b}", D[f"d{a}"], D[f"d{b}"]) for a in range(1, K + 1) for b in range(a + 1, K + 1)]
    pairs += [(f"pos{j}", "average", D[f"d{j}"], D["d_avg"]) for j in range(1, K + 1)]
    L += ["| pair | n features | Spearman rho |", "|:---|---:|---:|"]
    for a, b, x, y in pairs:
        m = x.notna() & y.notna()
        rho = stats.spearmanr(x[m], y[m])[0] if m.sum() > 2 else np.nan
        L.append(f"| {a} vs {b} | {int(m.sum())} | {rho:.3f} |")
    L.append("")

    # Original full-corpus analysis for context
    if orig_stats_path and Path(orig_stats_path).exists():
        o = pd.read_csv(orig_stats_path)
        L += ["## Context: original analysis (results/feature_stats.csv)", "",
              "The original file was produced from the full 300-prompt corpus with src/analyze.py (conservative "
              "df = min(n)-1 for p-values). It is only directly comparable to this analysis when the positional "
              "run covers the same examples; otherwise it is shown for reference only.", "",
              "| feature | d (original) | mean fail | mean succ | p Holm (original) |", "|---:|---:|---:|---:|---:|"]
        for _, r in o.head(5).iterrows():
            L.append(f"| {int(r['feature'])} | {r['cohens_d']:+.2f} | {r['mean_fail']:.3f} | {r['mean_succ']:.3f} | {_fmt_p(r['p_holm'])} |")
        L.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n")


def normalize_meta(meta: pd.DataFrame, K: int, tasks_path: str | None) -> pd.DataFrame:
    """Accept metadata written by either runner (pos{j}_* or position_{j}_token*), and add `object`."""
    meta = meta.copy()
    for j in range(1, K + 1):
        if f"pos{j}_token" not in meta.columns and f"position_{j}_token" in meta.columns:
            meta[f"pos{j}_token"] = meta[f"position_{j}_token"]
            meta[f"pos{j}_index"] = meta[f"position_{j}_token_index"]
        if f"pos{j}_offset" not in meta.columns:
            meta[f"pos{j}_offset"] = j - 1 - K
    if tasks_path and Path(tasks_path).exists():
        import json
        tk = {t["id"]: t for t in map(json.loads, open(tasks_path))}
        for col in ("object", "template_id", "subject", "indirect_object", "place"):
            if col not in meta.columns:
                meta[col] = meta["id"].map(lambda i: tk.get(i, {}).get(col))
    return meta


# ----------------------------------------------------------------------
def _suffix_path(p: str, suffix: str) -> str:
    q = Path(p)
    return str(q.with_name(q.stem + suffix + q.suffix)) if suffix else p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--acts", default="results/positional_activations.npy")
    ap.add_argument("--meta", default="results/positional_metadata.csv")
    ap.add_argument("--out-stats", default="results/position_feature_stats.csv")
    ap.add_argument("--out-top", default="results/top_features_by_position.md")
    ap.add_argument("--out-fig", default="figures/position_effects.png")
    ap.add_argument("--out-compare", default="results/position_vs_average_comparison.md")
    ap.add_argument("--orig-stats", default="results/feature_stats.csv")
    ap.add_argument("--tasks", default="data/tasks.jsonl", help="used to add the `object` column if metadata lacks it")
    ap.add_argument("--suffix", default="", help="appended to every output file stem (e.g. _seed7, _no_keys)")
    ap.add_argument("--exclude-object", nargs="*", default=[], help='drop prompts whose object is e.g. "the keys"')
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-active", type=int, default=3,
                    help="test a feature only if active (>0) in at least this many examples (label-independent filter)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--permutations", type=int, default=2000,
                    help="number of label permutations for the max-T adjusted p (0 = skip)")
    ap.add_argument("--stratify-by", default=None,
                    help="metadata column (e.g. object): also compute p_maxT_strat, a max-T test with labels shuffled only within each level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sae-layer", type=int, default=8)
    a = ap.parse_args(argv)

    acts = np.load(a.acts, mmap_mode="r")
    meta = pd.read_csv(a.meta)
    if acts.ndim == 3:
        meta = normalize_meta(meta, acts.shape[1], a.tasks)
    if acts.ndim != 3:
        sys.exit(f"expected a 3-D array (N, K, d_sae), got shape {acts.shape}")
    if acts.shape[0] != len(meta):
        sys.exit(f"row mismatch: {acts.shape[0]} activation rows vs {len(meta)} metadata rows")
    if not np.array_equal(meta["row_index"].values, np.arange(len(meta))):
        sys.exit("metadata row_index is not 0..N-1 in order; refusing to align")
    if not set(meta["success"].unique()) <= {0, 1}:
        sys.exit("success column must be 0/1")

    keep = np.ones(len(meta), dtype=bool)
    for obj in a.exclude_object:
        keep &= (meta["object"] != obj).values
    acts = np.asarray(acts[keep])
    meta = meta.loc[keep].reset_index(drop=True)
    N, K, F = acts.shape
    is_fail = meta["success"].values == 0
    nf, ns = int(is_fail.sum()), int((~is_fail).sum())
    print(f"N={N} (success={ns}, fail={nf}, acc={ns / N:.3f})  K={K}  features={F}")

    out_stats = _suffix_path(a.out_stats, a.suffix)
    out_top = _suffix_path(a.out_top, a.suffix)
    out_fig = _suffix_path(a.out_fig, a.suffix)
    out_cmp = _suffix_path(a.out_compare, a.suffix)

    header_extra = []
    if a.exclude_object:
        header_extra.append(f"Prompts excluded by object: {', '.join(a.exclude_object)}.")
        header_extra.append("")
    if min(nf, ns) < 2:
        msg = (f"Analysis NOT performed: need at least 2 failures and 2 successes, found "
               f"{nf} failures and {ns} successes.")
        print(msg)
        Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["feature", "position", "n_fail", "n_succ", "status"]).to_csv(out_stats, index=False)
        Path(out_top).write_text("# Top features by position\n\n" + msg + "\n")
        return 2
    if min(nf, ns) < 5:
        w = (f"WARNING: smallest group has {min(nf, ns)} examples (<5). Welch p-values and effect "
             "sizes are very unstable at this size; treat results as exploratory only.")
        print(w)
        header_extra += [f"> {w}", ""]

    table = positional_stats(acts, is_fail, a.min_active)
    tested = (table["status"] == "tested").values.reshape(K, F)
    t_obs = table["welch_t"].values.reshape(K, F)
    if a.permutations > 0:
        print(f"max-T permutation test: {a.permutations} permutations over {int(tested.sum()):,} tested pairs ...")
        pm = maxT_perm_pvalues(acts, is_fail, tested, t_obs, a.permutations, np.random.default_rng(a.seed))
        table["p_maxT_perm"] = pm.reshape(-1)
    else:
        table["p_maxT_perm"] = np.nan
    table["p_maxT_strat"] = np.nan
    if a.stratify_by and a.permutations > 0:
        if a.stratify_by not in meta.columns:
            sys.exit(f"--stratify-by column {a.stratify_by!r} not in metadata")
        strata = meta[a.stratify_by].astype(str).values
        print(f"stratified max-T (labels shuffled within {a.stratify_by}; {len(np.unique(strata))} strata) ...")
        pm = maxT_perm_pvalues(acts, is_fail, tested, t_obs, a.permutations, np.random.default_rng(a.seed + 1), strata=strata)
        table["p_maxT_strat"] = pm.reshape(-1)
    cols = ["feature", "position", "position_offset", "status", "n_fail", "n_succ", "n_fail_active",
            "n_succ_active", "mean_fail", "mean_succ", "delta", "sd_fail", "sd_succ", "cohens_d",
            "hedges_g", "welch_t", "welch_df", "p_raw", "p_holm", "p_holm_tested_family", "p_maxT_perm", "p_maxT_strat", "direction"]
    Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
    table[cols].to_csv(out_stats, index=False)
    print(f"wrote {out_stats}  ({len(table):,} rows)")

    write_top_features_md(out_top, table, acts, is_fail, meta, a.top_k, a.sae_layer, a.bootstrap, a.seed, header_extra)
    print(f"wrote {out_top}")
    make_figure(out_fig, table, K)
    print(f"wrote {out_fig}")
    write_comparison_md(out_cmp, table, acts, is_fail, a.min_active, a.top_k, a.orig_stats, K)
    print(f"wrote {out_cmp}")

    t = table[table.status == "tested"]
    for j in range(1, K + 1):
        s = t[t.position == j]
        print(f"position {j}: tested={len(s):,}  Welch+Holm<0.05: {int((s.p_holm < 0.05).sum())}  "
              f"max-T<0.05: {int((s.p_maxT_perm < 0.05).sum())}  strat max-T<0.05: {int((s.p_maxT_strat < 0.05).sum())}  |d|>0.8: {int((s.cohens_d.abs() > 0.8).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
