"""Step 4: are the per-position findings stable? (uses the existing 300 prompts)

1. Bootstrap (resample within group): how often do the observed top-K features (by |d|)
   stay in the top-K, and 95% CIs of d.
2. Split-half replication: pick the top-K failure-side / success-side features on a random
   half, measure d on the other half. A label-permuted baseline shows how much "replication"
   selection bias alone produces.
3. Leave-one-template-out and within-template d for the top failure-side features.
Writes <out>/stability.md and <out>/stability.json.   (A fresh-seed corpus needs the model:
see the README; this script only uses data already generated.)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import analyze_positions_robust as ap


def d_vec(X, f):
    nf, ns = int(f.sum()), int((~f).sum())
    if nf < 2 or ns < 2:
        return np.zeros(X.shape[1])
    a, b = X[f], X[~f]
    pooled = ((nf - 1) * a.var(0, ddof=1) + (ns - 1) * b.var(0, ddof=1)) / (nf + ns - 2)
    ok = pooled > 0
    return np.where(ok, (a.mean(0) - b.mean(0)) / np.sqrt(np.where(ok, pooled, 1.0)), 0.0)


def strat_halves(f, rng):
    A = np.zeros(len(f), bool)
    for lab in (True, False):
        idx = rng.permutation(np.where(f == lab)[0]); A[idx[: len(idx) // 2]] = True
    return A


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--acts", default="results_positional_300/positional_activations.npy")
    p.add_argument("--meta", default="results_positional_300/positional_metadata.csv")
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--out", default="results_positional_300_robust")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--splits", type=int, default=300)
    p.add_argument("--min-active", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    acts = np.load(a.acts, mmap_mode="r"); M = ap.normalize_meta(pd.read_csv(a.meta), acts.shape[1], a.tasks)
    f = (M.success.values == 0); N, K, F = acts.shape; k = a.top_k; rng = np.random.default_rng(a.seed)
    acts = np.asarray(acts); tab = ap.positional_stats(acts, f, a.min_active)
    tmpl = M.template_id.values; out = {"n": int(N), "n_fail": int(f.sum()), "top_k": k}
    L = ["# Stability of the per-position findings", "",
         f"N = {N} ({int(f.sum())} failures); top-K = {k}; {a.bootstrap} bootstrap resamples; {a.splits} split-half repeats. "
         "Features are those tested at each position (non-constant, active in >= "
         f"{a.min_active} prompts).", ""]
    for j in range(1, K + 1):
        t = tab[(tab.position == j) & (tab.status == "tested")]; cols = t.feature.values.astype(int)
        X = acts[:, j - 1, cols].astype(np.float64); d0 = d_vec(X, f); m = len(cols)
        top = np.argsort(-np.abs(d0))[:k]
        # 1. bootstrap
        fi, si = np.where(f)[0], np.where(~f)[0]; rec = np.zeros(m); ds = np.empty((a.bootstrap, k))
        for b in range(a.bootstrap):
            idx = np.concatenate([rng.choice(fi, len(fi)), rng.choice(si, len(si))])
            db = d_vec(X[idx], f[idx]); rec[np.argpartition(-np.abs(db), k)[:k]] += 1; ds[b] = db[top]
        rec /= a.bootstrap
        # 2. split-half (+ permuted-label baseline)
        res = {"real": [], "null": []}
        for r in range(a.splits):
            for tag, lab in (("real", f), ("null", rng.permutation(f))):
                A = strat_halves(lab, rng); B = ~A
                dA, dB = d_vec(X[A], lab[A]), d_vec(X[B], lab[B])
                up, dn = np.argsort(-dA)[:k], np.argsort(dA)[:k]
                res[tag].append([dB[up].mean(), (dB[up] > 0).mean(), dB[dn].mean(), (dB[dn] < 0).mean()])
        R = {t_: np.mean(v, axis=0) for t_, v in res.items()}
        # 3. leave-one-template-out + within-template d for failure-side top features
        fs = [i for i in np.argsort(-d0)[:k] if d0[i] > 0][:5]
        rows = []
        for i in fs:
            loo = [d_vec(X[tmpl != tt][:, [i]], f[tmpl != tt])[0] for tt in np.unique(tmpl)]
            within = [d_vec(X[tmpl == tt][:, [i]], f[tmpl == tt])[0] for tt in np.unique(tmpl)]
            rows.append({"feature": int(cols[i]), "d_all": round(float(d0[i]), 2), "min_d_leave_one_template_out": round(float(min(loo)), 2),
                         "within_template_d": [round(float(x), 2) for x in within]})
        out[f"position_{j}"] = {
            "tested": int(m), "top_k_mean_recurrence": round(float(rec[top].mean()), 3),
            "top_k_recurrence_ge_0.8": int((rec[top] >= 0.8).sum()),
            "top_k_ci_excludes_0": int(((np.percentile(ds, 2.5, axis=0) > 0) | (np.percentile(ds, 97.5, axis=0) < 0)).sum()),
            "split_half_fail_side": {"mean_d_heldout_real": round(float(R["real"][0]), 3), "frac_same_sign_real": round(float(R["real"][1]), 3),
                                     "mean_d_heldout_label_permuted": round(float(R["null"][0]), 3), "frac_same_sign_label_permuted": round(float(R["null"][1]), 3)},
            "split_half_success_side": {"mean_d_heldout_real": round(float(R["real"][2]), 3), "frac_same_sign_real": round(float(R["real"][3]), 3),
                                        "mean_d_heldout_label_permuted": round(float(R["null"][2]), 3), "frac_same_sign_label_permuted": round(float(R["null"][3]), 3)},
            "top_failure_features": rows}
        o = out[f"position_{j}"]; s = o["split_half_fail_side"]
        L += [f"## Position {j} (offset {j - 1 - K})", "",
              f"* Bootstrap: of the {k} strongest features by |d|, mean recurrence in the top-{k} = {o['top_k_mean_recurrence']:.2f}; "
              f"{o['top_k_recurrence_ge_0.8']} recur in >= 80% of resamples; {o['top_k_ci_excludes_0']} have a 95% CI for d excluding 0.",
              f"* Split-half, failure-side top-{k} chosen on one half: held-out mean d = {s['mean_d_heldout_real']:+.2f} "
              f"({s['frac_same_sign_real']:.0%} keep the sign), versus {s['mean_d_heldout_label_permuted']:+.2f} "
              f"({s['frac_same_sign_label_permuted']:.0%}) when labels are permuted (selection-bias baseline).",
              f"* Success-side: held-out mean d = {o['split_half_success_side']['mean_d_heldout_real']:+.2f} "
              f"({o['split_half_success_side']['frac_same_sign_real']:.0%} keep sign) vs "
              f"{o['split_half_success_side']['mean_d_heldout_label_permuted']:+.2f} under permuted labels.", "",
              "| feature | d (all) | min d leaving one template out | within-template d (templates 0..3) |", "|---:|---:|---:|:---|"]
        L += [f"| {r['feature']} | {r['d_all']:+.2f} | {r['min_d_leave_one_template_out']:+.2f} | {r['within_template_d']} |" for r in rows] + [""]
        print(f"position {j}:", json.dumps({k_: out[f'position_{j}'][k_] for k_ in ('top_k_mean_recurrence', 'top_k_recurrence_ge_0.8', 'top_k_ci_excludes_0')}), json.dumps(s))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "stability.md").write_text("\n".join(L) + "\n"); (Path(a.out) / "stability.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
