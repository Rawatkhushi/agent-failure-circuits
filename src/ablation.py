"""Causal ablation: zero out a target SAE feature in the residual stream
during inference and measure the change in IOI accuracy.

For each task:
  1. forward pass with a hook at blocks.8.hook_resid_pre
  2. encode through the SAE -> feature activations
  3. zero out the target feature(s) in the encoded activation
  4. decode back -> patched residual stream
  5. write the patched residual stream into the forward pass at the hook point
  6. greedy decode the next 3 tokens and check the indirect-object name

Reports:
  - baseline accuracy on each subset
  - ablated accuracy on each subset
  - per-subset delta

Writes:
  results/ablation_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from sae_lens import SAE, HookedSAETransformer


def normalize_token(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    first = s.split()[0]
    return first.rstrip(".,!?;:").lstrip()


def main(
    target_position: int = 2,


    tasks_path: str = "data/tasks.jsonl",
    out_path: str = "results/ablation_summary.csv",
    model_name: str = "gpt2",
    sae_release: str = "gpt2-small-res-jb",
    sae_id: str = "blocks.8.hook_resid_pre",
    target_features: tuple[int, ...] = (17491,),
    max_new_tokens: int = 3,
    limit: int | None = None,
) -> None:
    device = "cpu" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading {model_name} ...")
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    model.eval()

    print(f"loading SAE {sae_release}/{sae_id} ...")
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    sae.eval()

    hook_name = sae_id
    target_idx = torch.tensor(list(target_features), device=device, dtype=torch.long)
    print(f"  ablating features {list(target_features)} at hook {hook_name}")

    
    with open(tasks_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        tasks = tasks[:limit]

    print(f"running {len(tasks)} tasks under two conditions: baseline and ablated ...")


    rows = []
    for i, task in enumerate(tasks):
        prompt = task["prompt"]
        expected = task["expected"].strip()
        tokens = model.to_tokens(prompt)

        if tokens.shape[1] < 3:
            raise ValueError(f"Task {task.get('id')} has fewer than 3 tokens")

        if not 1 <= target_position <= 3:
            raise ValueError("target_position must be 1, 2, or 3")

        # Position 1, 2, or 3 among the prompt's final three tokens.
        target_token_index = tokens.shape[1] - 3 + (target_position - 1)

        # --- BASELINE (no patch) ---
        with torch.no_grad():
            generated = model.generate(
                tokens, max_new_tokens=max_new_tokens, temperature=0.0, verbose=False,
            )
        base_gen = model.to_string(generated[0, tokens.shape[1]:])
        base_pred = normalize_token(base_gen)
        base_ok = int(base_pred == expected)

        # --- ABLATED ---
        # Hook that zeros target features in the SAE-decoded reconstruction
        # by subtracting feature_act * decoder_row from the residual stream.
        # Equivalent to: resid_patched = resid - sum_i act_i * W_dec[i].
        
        
        
        def ablate_hook(
            activation,
            hook,
            sae=sae,
            idx=target_idx,
            token_pos=target_token_index,
        ):
            # activation shape: (batch, sequence, d_model)
            feats = sae.encode(activation)

            # Contributions from the selected SAE features
            target_acts = feats[..., idx]
            W_dec_rows = sae.W_dec[idx, :]
            all_contributions = target_acts @ W_dec_rows

            if not getattr(ablate_hook, "_checked", False):
                print(
                    f"Task ID: {task.get('id')}, "
                    f"object: {task.get('object')}, "
                    f"position: {target_position}"
                )
                print(
                    "Feature activation:",
                    target_acts[0, token_pos, :].detach().cpu().tolist(),
                )
                print(
                    "Contribution norm:",
                    all_contributions[0, token_pos, :].norm().item(),
                )
                ablate_hook._checked = True

            # Remove the selected features only at the target prompt token
            contribution = torch.zeros_like(activation)
            contribution[:, token_pos, :] = all_contributions[:, token_pos, :]

            return activation - contribution

        
        current = tokens.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model.run_with_hooks(
                    current,
                    fwd_hooks=[(hook_name, ablate_hook)],
                )
                next_tok = torch.argmax(
                    logits[0, -1, :]
                ).reshape(1, 1)
                current = torch.cat([current, next_tok], dim=1)

        abl_gen = model.to_string(current[0, tokens.shape[1]:])
        abl_pred = normalize_token(abl_gen)
        abl_ok = int(abl_pred == expected)

        rows.append({
            "id": task["id"],
            "object": task["object"],
            "expected": expected,
            "base_pred": base_pred,
            "abl_pred": abl_pred,
            "base_ok": base_ok,
            "abl_ok": abl_ok,
            "is_keys": int(task["object"] == "the keys"),
        })

        if (i + 1) % 25 == 0:
            n_base = sum(r["base_ok"] for r in rows)
            n_abl = sum(r["abl_ok"] for r in rows)
            print(f"  {i+1}/{len(tasks)}   base={n_base}/{i+1}  abl={n_abl}/{i+1}")

    # Write per-task rows
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")

    # ----- Summary -----
    def acc(subset):
        if not subset:
            return float("nan"), float("nan")
        return (
            sum(r["base_ok"] for r in subset) / len(subset),
            sum(r["abl_ok"] for r in subset) / len(subset),
    )

    all_rows = rows
    keys_rows = [r for r in rows if r["is_keys"]]
    other_rows = [r for r in rows if not r["is_keys"]]

    print()
    print("===== Ablation summary =====")
    base_all, abl_all = acc(all_rows)
    base_keys, abl_keys = acc(keys_rows)
    base_other, abl_other = acc(other_rows)

    print(f"all N={len(all_rows)}:  base acc = {base_all:.3f}   abl acc = {abl_all:.3f}   delta = {abl_all - base_all:+.3f}")
    print(f"keys N={len(keys_rows)}: base acc = {base_keys:.3f}   abl acc = {abl_keys:.3f}   delta = {abl_keys - base_keys:+.3f}")
    print(f"others N={len(other_rows)}: base acc = {base_other:.3f}   abl acc = {abl_other:.3f}   delta = {abl_other - base_other:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=int, nargs="+", default=[17491])
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--out", default="results/ablation_summary.csv")
    p.add_argument("--position", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(
        target_features=tuple(args.features),
        target_position=args.position,
        tasks_path=args.tasks,
        out_path=args.out,
        limit=args.limit,
    )
