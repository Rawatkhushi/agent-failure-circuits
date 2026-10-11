import json
import numpy as np
import pandas as pd
import torch
from sae_lens import SAE, HookedSAETransformer

TASKS = "data/tasks.jsonl"
OUT = "results/candidate_activation_diagnostic.csv"

CANDIDATES = [
    (16421, 2),
    (23952, 1),
]

device = "cpu"
print(f"Device: {device}")

model = HookedSAETransformer.from_pretrained("gpt2", device=device)
model.eval()

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device,
)
sae.eval()

with open(TASKS, encoding="utf-8") as f:
    tasks = [json.loads(line) for line in f if line.strip()]

rows = []

for i, task in enumerate(tasks):
    tokens = model.to_tokens(task["prompt"])
    target_position_index = tokens.shape[1] - 3

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=["blocks.8.hook_resid_pre"],
        )
        resid = cache["blocks.8.hook_resid_pre"]

        # Encode the final three prompt-token residuals.
        tail = resid[:, -3:, :]
        acts = sae.encode(tail)[0]

    row = {
        "id": task["id"],
        "object": task["object"],
        "success": int(
            task["expected"].strip()
            == task.get("predicted", "").strip()
        ) if "predicted" in task else None,
    }

    for feature, position in CANDIDATES:
        # position is 1-based among the final three prompt tokens.
        value = float(acts[position - 1, feature].item())
        row[f"feature_{feature}_position_{position}"] = value

    rows.append(row)

    if (i + 1) % 25 == 0 or i + 1 == len(tasks):
        print(f"{i + 1}/{len(tasks)} prompts processed")

df = pd.DataFrame(rows)

# Use the existing experiment metadata for the authoritative success labels.
metadata = pd.read_csv("results_positional_300/positional_metadata.csv")
df = df.drop(columns=["success"]).merge(
    metadata[["id", "success"]],
    on="id",
    how="left",
    validate="one_to_one",
)

for feature, position in CANDIDATES:
    col = f"feature_{feature}_position_{position}"
    print(f"\n=== Feature {feature}, Position {position} ===")

    for outcome, group in df.groupby("success"):
        values = group[col]
        print(
            f"success={outcome}: n={len(group)}, "
            f"nonzero={(values > 0).sum()}, "
            f"mean={values.mean():.6f}, "
            f"median={values.median():.6f}"
        )

    print("Nonzero activation by object and outcome:")
    print(
        df.assign(nonzero=df[col] > 0)
          .groupby(["object", "success"])["nonzero"]
          .agg(["size", "sum"])
          .to_string()
    )

df.to_csv(OUT, index=False)
print(f"\nSaved diagnostic results to {OUT}")
