"""Category-adjusted positional SAE feature analysis.

Model, separately for every feature and token position:
    activation ~ failure + C(position_2_token)

The failure coefficient is the adjusted failure-minus-success difference.
Holm correction is applied jointly across all feature-position tests.

This is an associational analysis, not a causal test.
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni(p):
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p)
    out = np.ones(len(p), dtype=float)
    running = 0.0
    m = len(p)

    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running

    return out


def main(
    activations="results_positional_300/positional_activations.npy",
    metadata="results_positional_300/positional_metadata.csv",
    output="results_positional_300_category_adjusted_v2",
):
    acts = np.load(activations, mmap_mode="r")
    meta = pd.read_csv(metadata)

    if acts.ndim != 3:
        raise ValueError(f"Expected (N, K, F) activations; got {acts.shape}")
    if len(meta) != acts.shape[0]:
        raise ValueError("Metadata row count does not match activations")
    if not {"success", "position_2_token", "id"}.issubset(meta.columns):
        raise ValueError("Required metadata columns are missing")
    if meta["id"].duplicated().any():
        raise ValueError("Duplicate metadata IDs")
    if "row_index" in meta and not np.array_equal(
        meta["row_index"].to_numpy(), np.arange(len(meta))
    ):
        raise ValueError("Metadata row order is inconsistent")

    success = pd.to_numeric(meta["success"], errors="raise").to_numpy()
    if not np.isin(success, [0, 1]).all():
        raise ValueError("success must contain only 0 and 1")

    # failure=1 makes the regression coefficient failure-minus-success.
    failure = (success == 0).astype(float)
    tokens = meta["position_2_token"].fillna("<MISSING>").astype(str)
    token_dummies = pd.get_dummies(tokens, drop_first=True, dtype=float)

    # Intercept + failure indicator + categorical token controls.
    X = np.column_stack([
        np.ones(len(meta)),
        failure,
        token_dummies.to_numpy(),
    ])
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        raise ValueError(
            f"Design matrix is rank deficient: rank={rank}, columns={X.shape[1]}"
        )

    xtx_inv = np.linalg.inv(X.T @ X)
    failure_var_factor = xtx_inv[1, 1]
    df_resid = len(meta) - rank
    if df_resid <= 0:
        raise ValueError("No residual degrees of freedom")

    records = []
    p_all = []

    for pos in range(acts.shape[1]):
        Y = np.asarray(acts[:, pos, :], dtype=float)

        if not np.isfinite(Y).all():
            raise ValueError(
                f"Non-finite activations at position {pos + 1}; "
                "handle missing values before analysis."
            )

        # Multi-response OLS: one design matrix, all SAE features.
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        residuals = Y - X @ beta
        mse = np.sum(residuals ** 2, axis=0) / df_resid
        se = np.sqrt(np.maximum(0.0, mse * failure_var_factor))
        coef = beta[1, :]

        t_stat = np.divide(
            coef, se,
            out=np.zeros_like(coef),
            where=se > 0,
        )
        p = 2 * stats.t.sf(np.abs(t_stat), df_resid)

        # Constant/residual-perfect-fit features are not assigned
        # artificial significance.
        invalid = (~np.isfinite(p)) | (~np.isfinite(se)) | (se <= 0)
        p[invalid] = 1.0
        t_stat[invalid] = np.nan

        for feat in range(Y.shape[1]):
            records.append({
                "position_index": pos + 1,
                "feature_id": feat,
                "n_success": int((success == 1).sum()),
                "n_failure": int((success == 0).sum()),
                "adjusted_failure_minus_success": float(coef[feat]),
                "standard_error": float(se[feat]),
                "t_statistic": float(t_stat[feat]),
                "df_resid": int(df_resid),
                "p_value": float(p[feat]),
            })
            p_all.append(p[feat])

    result = pd.DataFrame(records)
    result["p_adjusted_holm"] = holm_bonferroni(np.asarray(p_all))
    result["abs_adjusted_effect"] = result[
        "adjusted_failure_minus_success"
    ].abs()

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "category_adjusted_stats.csv"
    result.to_csv(csv_path, index=False)

    report = [
        "# Category-adjusted positional SAE analysis",
        "",
        f"Examples: {len(meta)}",
        f"Positions: {acts.shape[1]}",
        f"Features per position: {acts.shape[2]}",
        f"Covariate: categorical position_2_token ({tokens.nunique()} levels)",
        f"Residual degrees of freedom: {df_resid}",
        "Outcome: feature activation",
        "Predictor of interest: failure (1) vs success (0)",
        "Correction: Holm across every feature-position test",
        "Interpretation: associations, not causal effects",
        "",
        f"Significant tests (Holm p < 0.05): "
        f"{(result.p_adjusted_holm < 0.05).sum()} / {len(result)}",
        "",
    ]

    for pos in range(1, acts.shape[1] + 1):
        sub = result[result.position_index == pos].nsmallest(
            10, "p_adjusted_holm"
        )
        report += [
            f"## Position {pos}",
            "",
            "| Feature | Adjusted failure − success | p | Holm p |",
            "|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            report.append(
                f"| {int(r.feature_id)} | "
                f"{r.adjusted_failure_minus_success:.6g} | "
                f"{r.p_value:.4g} | {r.p_adjusted_holm:.4g} |"
            )
        report.append("")

    (out / "report.md").write_text("\n".join(report) + "\n")
    print(f"Examples: {len(meta)}; successes: {(success == 1).sum()}; "
          f"failures: {(success == 0).sum()}")
    print(f"Tests: {len(result)}")
    print(f"Holm-significant: {(result.p_adjusted_holm < 0.05).sum()}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activations",
        default="results_positional_300/positional_activations.npy",
    )
    parser.add_argument(
        "--metadata",
        default="results_positional_300/positional_metadata.csv",
    )
    parser.add_argument(
        "--output",
        default="results_positional_300_category_adjusted_v2",
    )
    args = parser.parse_args()
    main(args.activations, args.metadata, args.output)
