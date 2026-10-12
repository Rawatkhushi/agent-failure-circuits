"""Step 3: do failures depend on prompt template, object or article?

Pure metadata analysis (no activations). Uses permutation chi-square tests
(small cells make the asymptotic chi-square unreliable) and reports failure
rates, the failure modes (predicted token) and a keys vs non-keys split.
Writes <out>/template_analysis.md and <out>/template_analysis.json.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from analyze_positions_robust import normalize_meta


def chi2_stat(labels: np.ndarray, fail: np.ndarray, levels: int) -> float:
    n = len(labels); f = fail.sum(); tot = np.bincount(labels, minlength=levels).astype(float)
    obs_f = np.bincount(labels[fail], minlength=levels).astype(float)
    exp_f = tot * f / n; exp_s = tot * (n - f) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum((obs_f - exp_f) ** 2 / exp_f) + np.nansum(((tot - obs_f) - exp_s) ** 2 / exp_s))


def perm_chi2_p(labels, fail, B=20000, seed=0):
    codes, uniq = pd.factorize(pd.Series(labels)); codes = np.asarray(codes); fail = np.asarray(fail, bool)
    obs = chi2_stat(codes, fail, len(uniq)); rng = np.random.default_rng(seed); ge = 0
    for _ in range(B):
        ge += chi2_stat(codes, rng.permutation(fail), len(uniq)) >= obs - 1e-12
    return obs, (ge + 1) / (B + 1)


def rate_table(df, col):
    g = df.groupby(col)["fail"].agg(n="count", failures="sum"); g["fail_rate"] = (g.failures / g.n).round(3)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--meta", default="results_positional_300/positional_metadata.csv")
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--out", default="results_positional_300_robust")
    ap.add_argument("--perms", type=int, default=20000)
    a = ap.parse_args(argv)
    M = normalize_meta(pd.read_csv(a.meta), 3, a.tasks)
    M["fail"] = (M.success == 0); M["keyflag"] = np.where(M.object == "the keys", "keys", "non-keys")
    M["article"] = M.object.str.split().str[0]
    out = {}; L = ["# Template / object / article effects on failure", "",
                   f"N = {len(M)}, failures = {int(M.fail.sum())}. p-values are permutation chi-square (B = {a.perms}); post hoc and unadjusted across the tests below.", ""]
    def test(name, sub, col):
        chi, p = perm_chi2_p(sub[col].astype(str).values, sub.fail.values, a.perms)
        out[name] = {"n": int(len(sub)), "failures": int(sub.fail.sum()), "chi2": round(chi, 3), "perm_p": round(p, 5)}
        return chi, p
    for title, sub, col, key in [("Template, all prompts", M, "template_id", "template_all"),
                                 ("Template, keys prompts only", M[M.keyflag == "keys"], "template_id", "template_keys_only"),
                                 ("Template, non-keys prompts only", M[M.keyflag == "non-keys"], "template_id", "template_nonkeys"),
                                 ("Object, non-keys prompts only", M[M.keyflag == "non-keys"], "object", "object_nonkeys"),
                                 ("Article (the/a), non-keys prompts only", M[M.keyflag == "non-keys"], "article", "article_nonkeys")]:
        chi, p = test(key, sub, col)
        L += [f"## {title}", f"chi2 = {chi:.2f}, permutation p = {p:.4f}", "", rate_table(sub, col).to_markdown(), ""]
    out["failures_by_template_nonkeys"] = {int(k): int(v) for k, v in M[(M.keyflag == "non-keys")].groupby("template_id").fail.sum().items()}
    out["failures_by_template_keys"] = {int(k): int(v) for k, v in M[(M.keyflag == "keys")].groupby("template_id").fail.sum().items()}
    out["fail_rate_by_template_all"] = {int(k): float(v) for k, v in M.groupby("template_id").fail.mean().round(3).items()}
    # most failure-concentrated template among non-keys
    nk = M[M.keyflag == "non-keys"]; top_t = int(nk.groupby("template_id").fail.sum().idxmax())
    a_ = nk[nk.template_id == top_t]; b_ = nk[nk.template_id != top_t]
    out["nonkeys_top_template"] = {"template_id": top_t, "fails": int(a_.fail.sum()), "n": int(len(a_)),
                                   "rest_fails": int(b_.fail.sum()), "rest_n": int(len(b_))}
    from scipy import stats
    out["nonkeys_top_template"]["fisher_p"] = float(stats.fisher_exact([[a_.fail.sum(), (~a_.fail).sum()], [b_.fail.sum(), (~b_.fail).sum()]])[1])
    ex = {int(t): M[M.template_id == t].prompt.iloc[0] for t in sorted(M.template_id.unique())}
    out["template_examples"] = ex
    L += ["## Failure modes (predicted token) among non-keys failures, by template", "",
          pd.crosstab(nk[nk.fail].template_id, nk[nk.fail].predicted).to_markdown(), "",
          f"## Most failure-concentrated non-keys template: {top_t}",
          f"{int(a_.fail.sum())}/{len(a_)} fail vs {int(b_.fail.sum())}/{len(b_)} in the other templates "
          f"(Fisher exact p = {out['nonkeys_top_template']['fisher_p']:.4g}). Template {top_t}: `{ex[top_t]}`", "",
          "## Template texts", ""] + [f"* {t}: `{p}`" for t, p in ex.items()]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "template_analysis.md").write_text("\n".join(L) + "\n")
    (Path(a.out) / "template_analysis.json").write_text(json.dumps(out, indent=2))
    print("\n".join(L[:12])); print(json.dumps({k: out[k] for k in ("template_all", "template_keys_only", "template_nonkeys", "nonkeys_top_template")}, indent=1))


if __name__ == "__main__":
    main()
