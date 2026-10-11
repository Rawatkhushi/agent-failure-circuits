"""Position-wise SAE extraction for the repository's original IOI task.

This is a separate runner: it does not overwrite predictions.csv or activations.npy.
The positional array is (N, K, d_sae), where K defaults to 3 and position 0
means the earliest of the last K prompt tokens. Metadata rows are aligned by id
and row order with the first axis of the saved array.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE, HookedSAETransformer


def normalize_token(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    return s.split()[0].rstrip(".,!?;:").lstrip()


def main(tasks_path="data/tasks.jsonl", out_dir="results", model_name="gpt2",
         sae_release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre",
         max_new_tokens=6, last_k_tokens=3, limit=None):
    if last_k_tokens != 3:
        print(f"WARNING: requested K={last_k_tokens}; this run is not the standard 3-position analysis.")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}; model: {model_name}; SAE: {sae_release}/{sae_id}")
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    model.eval()
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    sae.eval()
    hook_name = sae_id

    with open(tasks_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_acts, metadata = [], []
    for i, task in enumerate(tasks):
        prompt = task["prompt"]
        tokens = model.to_tokens(prompt)
        token_ids = tokens[0].detach().cpu().tolist()
        if len(token_ids) < last_k_tokens:
            raise ValueError(f"Task {task.get('id', i)} has only {len(token_ids)} tokens; cannot take last {last_k_tokens}.")
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            resid = cache[hook_name]
            if resid.ndim != 3 or resid.shape[0] != 1:
                raise RuntimeError(f"Unexpected residual shape {tuple(resid.shape)}")
            tail = resid[0, -last_k_tokens:, :]
            feats = sae.encode(tail)
            if feats.ndim != 2 or feats.shape[0] != last_k_tokens:
                raise RuntimeError(f"Unexpected SAE output shape {tuple(feats.shape)}")
            positional = feats.detach().cpu().numpy().astype(np.float32, copy=False)
            generated = model.generate(tokens, max_new_tokens=max_new_tokens,
                                       temperature=0.0, verbose=False)
        gen_str = model.to_string(generated[0, tokens.shape[1]:])
        predicted = normalize_token(gen_str)
        expected = task["expected"].strip()
        success = int(predicted == expected)  # exact original runner scoring rule
        token_indices = list(range(len(token_ids) - last_k_tokens, len(token_ids)))
        decoded = [model.tokenizer.decode([tid]) for tid in token_ids]
        # Prefer tokenizer's single-token decode for metadata; no inference is made
        # that these token positions have equivalent semantic roles across prompts.
        tail_decoded = decoded[-last_k_tokens:]
        all_acts.append(positional)
        metadata.append({
            "row_index": i, "id": task.get("id", i), "prompt": prompt,
            "expected": task["expected"], "generated": gen_str.strip()[:30],
            "predicted": predicted, "success": success, "n_steps": task.get("n_steps", 1),
            "token_ids_json": json.dumps(token_ids),
            "last3_token_ids_json": json.dumps(token_ids[-last_k_tokens:]),
            "last3_tokens_json": json.dumps(tail_decoded, ensure_ascii=False),
            "last3_token_indices_json": json.dumps(token_indices),
            "position_1_token_id": token_ids[token_indices[0]],
            "position_1_token": tail_decoded[0], "position_1_token_index": token_indices[0],
            "position_2_token_id": token_ids[token_indices[1]],
            "position_2_token": tail_decoded[1], "position_2_token_index": token_indices[1],
            "position_3_token_id": token_ids[token_indices[2]],
            "position_3_token": tail_decoded[2], "position_3_token_index": token_indices[2],
        })
        if (i + 1) % 5 == 0 or i + 1 == len(tasks):
            print(f"{i+1}/{len(tasks)} examples; accuracy={np.mean([m['success'] for m in metadata]):.3f}")

    arr = np.stack(all_acts, axis=0)
    expected_shape = (len(tasks), last_k_tokens, int(sae.cfg.d_sae))
    if arr.shape != expected_shape:
        raise RuntimeError(f"Activation shape {arr.shape} != expected {expected_shape}")
    np.save(out / "positional_activations.npy", arr)
    # Keep a directly comparable mean-over-last-K matrix, under a new filename.
    np.save(out / "positional_activations_mean.npy", arr.mean(axis=1))
    with (out / "positional_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
        writer.writeheader(); writer.writerows(metadata)
    # Explicit manifest makes alignment and dimensions machine-checkable.
    manifest = {"n_examples": len(tasks), "n_positions": last_k_tokens,
                "n_sae_features": int(sae.cfg.d_sae), "shape": list(arr.shape),
                "array_axis_order": ["example_row", "position_1_to_K_earliest_to_latest", "sae_feature"],
                "hook_point": hook_name, "model": model_name, "sae_release": sae_release,
                "sae_id": sae_id, "scoring": "normalize_token(generated_text) == task.expected.strip()",
                "metadata_csv": "positional_metadata.csv"}
    (out / "positional_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    n_ok = sum(x["success"] for x in metadata)
    print(f"Saved {out / 'positional_activations.npy'} shape={arr.shape}")
    print(f"Saved {out / 'positional_activations_mean.npy'} shape={arr.mean(axis=1).shape}")
    print(f"Saved {out / 'positional_metadata.csv'} and positional_manifest.json")
    print(f"final acc: {n_ok}/{len(metadata)} = {n_ok/len(metadata):.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--out", default="results")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--sae-release", default="gpt2-small-res-jb")
    p.add_argument("--sae-id", default="blocks.8.hook_resid_pre")
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--last-k", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N tasks (e.g. --limit 5 for smoke test).")
    args = p.parse_args()
    main(args.tasks, args.out, args.model, args.sae_release, args.sae_id,
         args.max_new_tokens, args.last_k, args.limit)
