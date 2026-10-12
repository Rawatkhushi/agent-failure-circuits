"""Step 1 of the per-position pipeline: verify the positional activation data.

Checks (PASS/FAIL, exit code 1 on any FAIL):
  tensor shape/finite/non-negative; metadata row order and ids; ids+prompts equal
  data/tasks.jsonl; expected/predicted/success/generated equal the ORIGINAL
  results/predictions.csv; stored last-3 token ids/indices equal the tail of each
  prompt's full token list; mean over positions reproduces the ORIGINAL
  results/activations.npy and the pairing is example-specific.
Works with the metadata schema of this fork (position_{j}_token*, last3_*_json).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd


def run(acts_path, meta_path, tasks_path, preds_path, orig_acts_path, atol=1e-3, verbose=True):
    A = np.load(acts_path, mmap_mode="r"); M = pd.read_csv(meta_path)
    res = []
    def ok(cond, msg):
        res.append((bool(cond), msg))
        if verbose: print(("PASS " if cond else "FAIL ") + msg)
    N, K = M.shape[0], (A.shape[1] if A.ndim == 3 else 0)
    ok(A.ndim == 3 and A.shape[0] == N, f"tensor shape {A.shape} = (N={N}, K, d_sae)")
    if not res[-1][0]: return res
    ok(np.isfinite(A).all() and float(A.min()) >= 0, "all activations finite and non-negative")
    ok((M.row_index.values == np.arange(N)).all() and M.id.is_unique, "row_index == 0..N-1 and ids unique")
    ok(set(M.success.unique()) <= {0, 1}, "success is 0/1")
    if Path(tasks_path).exists():
        T = [json.loads(l) for l in open(tasks_path)]
        ok(len(T) == N and (M.id.values == [t["id"] for t in T]).all() and (M.prompt.values == [t["prompt"] for t in T]).all(),
           "ids and prompts equal data/tasks.jsonl, in order")
    if Path(preds_path).exists():
        P = pd.read_csv(preds_path)
        for c in ("expected", "predicted", "success"):
            ok((M[c].astype(str).values == P[c].astype(str).values).all(), f"'{c}' equals original predictions.csv")
        ok((M.generated.fillna("").astype(str).str.strip().values == P.generated.fillna("").astype(str).str.strip().values).all(),
           "'generated' equals original predictions.csv")
    bad = 0
    for _, r in M.iterrows():
        ids = json.loads(r.token_ids_json); l3 = json.loads(r.last3_token_ids_json); idx = json.loads(r.last3_token_indices_json)
        got = [r[f"position_{j}_token_id"] for j in range(1, K + 1)]
        gi = [r[f"position_{j}_token_index"] for j in range(1, K + 1)]
        bad += not (ids[-K:] == l3 and idx == list(range(len(ids) - K, len(ids))) and got == l3 and gi == idx)
    ok(bad == 0, f"last-{K} token ids/indices are the true tail of each prompt ({bad} bad)")
    if Path(orig_acts_path).exists():
        O = np.load(orig_acts_path, mmap_mode="r")
        mean = np.asarray(A, dtype=np.float64).mean(axis=1); ref = np.asarray(O, dtype=np.float64)
        d = float(np.abs(mean - ref).max()); sh = float(np.abs(mean[1:] - ref[:-1]).max())
        ok(d <= atol, f"mean over positions vs original activations.npy: max|diff| = {d:.2e} (atol {atol})")
        ok(sh > 100 * max(d, 1e-9), f"row pairing is example-specific (shifted-row diff {sh:.2f})")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--acts", default="results_positional_300/positional_activations.npy")
    ap.add_argument("--meta", default="results_positional_300/positional_metadata.csv")
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--preds", default="results/predictions.csv")
    ap.add_argument("--orig-acts", default="results/activations.npy")
    a = ap.parse_args()
    r = run(a.acts, a.meta, a.tasks, a.preds, a.orig_acts)
    n_bad = sum(not c for c, _ in r)
    print("\nOVERALL:", "FAILED (%d)" % n_bad if n_bad else "all %d checks passed" % len(r))
    sys.exit(1 if n_bad else 0)
