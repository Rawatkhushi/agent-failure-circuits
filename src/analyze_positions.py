"""Position-wise failure-vs-success statistics for positional IOI SAE activations.

Holm correction is applied once across the complete feature x position family.
Positive delta / Cohen's d means mean(failure) > mean(success); negative means
mean(failure) < mean(success). Statistics are associational, not causal.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.ones(p.shape, dtype=float)
    valid = np.isfinite(p)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return out
    order = ids[np.argsort(p[ids])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    # Invalid tests remain p=1, conservatively included in family size via
    # the caller's full p array (invalid p-values are set to 1 before correction).
    return out


def feature_stats(fail: np.ndarray, succ: np.ndarray):
    """Compute safe per-feature stats. Inputs are (n_examples, n_features)."""
    n_f, n_s = fail.shape[0], succ.shape[0]
    n_feat = fail.shape[1] if fail.ndim == 2 else succ.shape[1]
    mf = np.full(n_feat, np.nan); ms = np.full(n_feat, np.nan)
    delta = np.full(n_feat, np.nan); d = np.full(n_feat, np.nan)
    tstat = np.full(n_feat, np.nan); p = np.ones(n_feat, dtype=float)
    if n_f:
        mf = np.nanmean(fail, axis=0)
    if n_s:
        ms = np.nanmean(succ, axis=0)
    delta = mf - ms
    if n_f < 2 or n_s < 2:
        return mf, ms, delta, d, tstat, p
    for j in range(n_feat):
        a, b = fail[:, j], succ[:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled = ((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2)
        if pooled > 0 and np.isfinite(pooled):
            d[j] = (np.mean(a) - np.mean(b)) / np.sqrt(pooled)
        # Constant features, including different constants, have undefined
        # t statistics; mark the test non-significant rather than fabricate one.
        if (va == 0 and vb == 0) or not np.isfinite(va + vb):
            continue
        result = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if np.isfinite(result.statistic) and np.isfinite(result.pvalue):
            tstat[j], p[j] = float(result.statistic), float(result.pvalue)
    p[~np.isfinite(p)] = 1.0
    return mf, ms, delta, d, tstat, p


def main(acts_path="results/positional_activations.npy",
         metadata_path="results/positional_metadata.csv",
         out_stats="results/position_feature_stats.csv",
         out_top="results/top_features_by_position.md",
         out_fig="figures/position_effects.png", top_k=20):
    acts = np.load(acts_path, mmap_mode="r")
    meta = pd.read_csv(metadata_path)
    if acts.ndim != 3:
        raise ValueError(f"Expected (N,K,F) activation array; got {acts.shape}")
    if len(meta) != acts.shape[0]:
        raise ValueError(f"Metadata rows ({len(meta)}) != activation examples ({acts.shape[0]})")
    if "success" not in meta or "id" not in meta:
        raise ValueError("Metadata must contain id and success columns")
    if meta["id"].duplicated().any():
        raise ValueError("Metadata contains duplicate example IDs")
    # Runner explicitly writes row_index, allowing strict row alignment checks.
    if "row_index" in meta and not np.array_equal(meta.row_index.to_numpy(), np.arange(len(meta))):
        raise ValueError("Metadata row_index is not aligned to activation row order")
    success = pd.to_numeric(meta.success, errors="raise").to_numpy().astype(int)
    if not np.isin(success, [0, 1]).all():
        raise ValueError("success labels must be binary 0/1")
    n, k, n_feat = acts.shape
    records = []
    p_all = []
    for pos in range(k):
        x = np.asarray(acts[:, pos, :], dtype=float)
        fail, succ = x[success == 0], x[success == 1]
        mf, ms, delta, d, tstat, p = feature_stats(fail, succ)
        for feat in range(n_feat):
            records.append({"feature_id": feat, "position_index": pos + 1,
                "n_success": int((success == 1).sum()), "n_failure": int((success == 0).sum()),
                "mean_success": ms[feat], "mean_failure": mf[feat],
                "mean_difference_failure_minus_success": delta[feat],
                "cohens_d": d[feat], "t_statistic": tstat[feat], "p_value": p[feat]})
            p_all.append(p[feat])
    p_all = np.asarray(p_all, dtype=float)
    # Holm over every feature at every position (K * d_sae tests), including
    # untestable tests as p=1 to avoid a smaller, data-dependent family.
    p_for_holm = np.where(np.isfinite(p_all), p_all, 1.0)
    p_adj = holm_bonferroni(p_for_holm)
    df = pd.DataFrame(records)
    df["p_adjusted_holm"] = p_adj
    df["abs_cohens_d"] = df.cohens_d.abs()
    Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_stats, index=False)

    lines = ["# Top SAE feature associations by token position", "",
        f"Activation shape: `{tuple(acts.shape)}`; examples={n}, features={n_feat}, positions={k}.",
        f"Successes: {(success==1).sum()}; failures: {(success==0).sum()}.",
        "Positive delta/d means higher mean activation on failures; negative means higher on successes.",
        "Holm correction covers all feature-position tests jointly. These are associations, not causal effects.", ""]
    for pos in range(k):
        sub = df[df.position_index == pos+1].sort_values("abs_cohens_d", ascending=False).head(top_k)
        lines += [f"## Position {pos+1} (last-three-token order; token identities vary by prompt)", "",
                  "| Rank | Feature | Mean success | Mean failure | Failure − success | Cohen's d | Welch p | Holm p |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            def fmt(v): return "NA" if pd.isna(v) else f"{v:.4g}"
            lines.append(f"| {rank} | {int(r.feature_id)} | {fmt(r.mean_success)} | {fmt(r.mean_failure)} | {fmt(r.mean_difference_failure_minus_success)} | {fmt(r.cohens_d)} | {fmt(r.p_value)} | {fmt(r.p_adjusted_holm)} |")
        lines.append("")
    Path(out_top).parent.mkdir(parents=True, exist_ok=True)
    Path(out_top).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Plot a compact heatmap of the union of the top features by |d| at any position.
    top_union = (df.sort_values("abs_cohens_d", ascending=False)
                   .drop_duplicates("feature_id").head(min(top_k, n_feat)).feature_id.tolist())
    mat = df[df.feature_id.isin(top_union)].pivot(index="feature_id", columns="position_index", values="cohens_d")
    mat = mat.reindex(top_union)
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig_h = max(4, 0.28 * len(mat))
    fig, ax = plt.subplots(figsize=(7, fig_h))
    image = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_xticks(np.arange(k), [f"Position {i+1}" for i in range(k)])
    ax.set_yticks(np.arange(len(mat)), [str(x) for x in mat.index])
    ax.set_xlabel("Position among the last K prompt tokens")
    ax.set_ylabel("SAE feature ID")
    ax.set_title("Cohen's d by position (failure minus success)")
    fig.colorbar(image, ax=ax, label="Cohen's d")
    fig.tight_layout(); fig.savefig(out_fig, dpi=180); plt.close(fig)
    print(f"N={n}; success={(success==1).sum()}; failure={(success==0).sum()}; K={k}; features={n_feat}")
    print(f"Holm-significant tests (p<0.05): {(df.p_adjusted_holm < 0.05).sum()} / {len(df)}")
    # Optional direct comparison when the original experiment outputs exist.
    original_acts = Path("results/activations.npy")
    original_preds = Path("results/predictions.csv")
    mean_path = Path(acts_path).with_name("positional_activations_mean.npy")
    if original_acts.exists() and mean_path.exists():
        base = np.load(original_acts, mmap_mode="r")
        mean_acts = np.load(mean_path, mmap_mode="r")
        if base.shape == mean_acts.shape:
            diff = np.asarray(mean_acts, dtype=float) - np.asarray(base, dtype=float)
            print(f"Averaged activation comparison: max_abs_diff={np.max(np.abs(diff)):.6g}, "
                  f"mean_abs_diff={np.mean(np.abs(diff)):.6g}")
        else:
            print(f"Averaged activation comparison skipped: shapes differ {base.shape} vs {mean_acts.shape}")
    if original_preds.exists():
        old = pd.read_csv(original_preds)
        if len(old) == len(meta) and "success" in old:
            same_labels = np.array_equal(pd.to_numeric(old.success).to_numpy().astype(int), success)
            same_ids = ("id" in old and np.array_equal(old.id.to_numpy(), meta.id.to_numpy()))
            print(f"Original prediction alignment check: same_row_count=True, same_ids={same_ids}, same_labels={same_labels}")
        else:
            print("Original prediction alignment check skipped: row counts differ or success column missing")
    print(f"Wrote {out_stats}, {out_top}, {out_fig}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--activations", default="results/positional_activations.npy")
    p.add_argument("--metadata", default="results/positional_metadata.csv")
    p.add_argument("--out-stats", default="results/position_feature_stats.csv")
    p.add_argument("--out-top", default="results/top_features_by_position.md")
    p.add_argument("--out-fig", default="figures/position_effects.png")
    p.add_argument("--top-k", type=int, default=20)
    a = p.parse_args()
    main(a.activations, a.metadata, a.out_stats, a.out_top, a.out_fig, a.top_k)
