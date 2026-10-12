"""Tests for verify_positional_data, analyze_templates and stability_positions helpers."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import verify_positional_data as vd, analyze_templates as at, stability_positions as sp, analyze_positions_robust as ap


def _fake_dataset(tmp_path, N=12, F=20, K=3, seed=0):
    rng = np.random.default_rng(seed)
    acts = np.maximum(rng.normal(0.2, 1, (N, K, F)), 0).astype(np.float32)
    rows, tasks, preds = [], [], []
    for i in range(N):
        ids = list(range(100, 100 + 12 + i % 2)) + [11, 22, 33]; n = len(ids)
        succ = int(i % 3 != 0)
        r = dict(row_index=i, id=i, prompt=f"p{i}", expected=" X", generated="X.", predicted="X" if succ else "her", success=succ,
                 token_ids_json=json.dumps(ids), last3_token_ids_json=json.dumps(ids[-3:]),
                 last3_token_indices_json=json.dumps([n - 3, n - 2, n - 1]))
        for j in range(1, 4):
            r[f"position_{j}_token_id"] = ids[n - 4 + j]; r[f"position_{j}_token_index"] = n - 4 + j; r[f"position_{j}_token"] = " t"
        rows.append(r); tasks.append(dict(id=i, prompt=f"p{i}", expected=" X"))
        preds.append(dict(id=i, prompt=f"p{i}", expected=" X", generated="X.", predicted=r["predicted"], success=succ))
    np.save(tmp_path / "a.npy", acts); np.save(tmp_path / "o.npy", acts.mean(1))
    pd.DataFrame(rows).to_csv(tmp_path / "m.csv", index=False); pd.DataFrame(preds).to_csv(tmp_path / "p.csv", index=False)
    (tmp_path / "t.jsonl").write_text("\n".join(map(json.dumps, tasks)))
    return acts


def _verify(tp, acts="a.npy"):
    return vd.run(tp / acts, tp / "m.csv", tp / "t.jsonl", tp / "p.csv", tp / "o.npy", verbose=False)


def test_verify_passes_on_consistent_data(tmp_path):
    _fake_dataset(tmp_path); assert all(ok for ok, _ in _verify(tmp_path))


def test_verify_detects_swapped_rows_and_flipped_label(tmp_path):
    acts = _fake_dataset(tmp_path)
    bad = acts.copy(); bad[[2, 5]] = bad[[5, 2]]; np.save(tmp_path / "bad.npy", bad)
    assert not all(ok for ok, _ in _verify(tmp_path, "bad.npy"))
    m = pd.read_csv(tmp_path / "m.csv"); m.loc[1, "success"] = 1 - m.loc[1, "success"]; m.to_csv(tmp_path / "m.csv", index=False)
    assert not all(ok for ok, _ in _verify(tmp_path))


def test_verify_detects_wrong_token_tail(tmp_path):
    _fake_dataset(tmp_path); m = pd.read_csv(tmp_path / "m.csv"); m.loc[0, "position_2_token_index"] += 1
    m.to_csv(tmp_path / "m.csv", index=False); assert not all(ok for ok, _ in _verify(tmp_path))


def test_chi2_matches_scipy_and_perm_p_valid():
    rng = np.random.default_rng(1); lab = rng.integers(0, 4, 200); fail = rng.uniform(size=200) < 0.2
    tab = pd.crosstab(lab, fail).values
    assert at.chi2_stat(lab, fail, 4) == pytest.approx(stats.chi2_contingency(tab, correction=False)[0])
    _, p = at.perm_chi2_p(lab, fail, B=500); assert 0 < p <= 1
    lab2 = np.repeat([0, 1], 100); fail2 = np.r_[np.zeros(100, bool), np.ones(100, bool)]      # perfectly dependent
    assert at.perm_chi2_p(lab2, fail2, B=300)[1] < 0.01


def test_d_vec_matches_compute_stats_and_halves_are_balanced():
    rng = np.random.default_rng(2); X = np.maximum(rng.normal(0.3, 1, (80, 15)), 0); f = rng.uniform(size=80) < 0.3
    np.testing.assert_allclose(sp.d_vec(X, f), ap.compute_stats(X, f, 1)["cohens_d"].values, atol=1e-12)
    A = sp.strat_halves(f, rng)
    assert abs(int((A & f).sum()) - int((~A & f).sum())) <= 1 and abs(int((A & ~f).sum()) - int((~A & ~f).sum())) <= 1
