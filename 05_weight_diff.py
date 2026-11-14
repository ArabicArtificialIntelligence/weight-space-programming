"""Step 5 - show exactly which weights the corrector changed, and by how much.

    python 05_weight_diff.py

Compares models/original vs models/edited tensor by tensor, prints a table, and saves
work/report/weight_diff.png (where in the model) and work/report/changed_matrix.png (the edited matrix itself).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file
from common import ORIGINAL, EDITED, ROOT

con = Console(highlight=False)
A = load_file(ORIGINAL / "model.safetensors")
B = load_file(EDITED / "model.safetensors")
total = sum(v.numel() for v in A.values())
HD, NH = 64, 15
COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
              "input_layernorm", "post_attention_layernorm"]

rows, grid = [], np.zeros((32, len(COMPONENTS)))
for k in A:
    a, b = A[k].float(), B[k].float()
    mask = a != b
    n = int(mask.sum())
    if not n:
        continue
    d = b - a
    info = {"tensor": k, "shape": list(a.shape), "changed": n, "pct_of_tensor": 100 * n / a.numel(),
            "rel_change": float(d.norm() / a.norm()), "max_abs_change": float(d.abs().max()),
            "mean_before": float(a[mask].abs().mean()), "mean_after": float(b[mask].abs().mean())}
    if a.dim() == 2:
        info["rows_touched"] = int(mask.any(1).sum()); info["cols_touched"] = int(mask.any(0).sum())
        if "o_proj" in k:   # in-dims are grouped by query head
            info["heads"] = [h for h in range(NH) if mask[:, h * HD:(h + 1) * HD].any()]
        if any(p in k for p in ("q_proj",)):
            info["heads"] = [h for h in range(NH) if mask[h * HD:(h + 1) * HD].any()]
    rows.append(info)
    if ".layers." in k:
        L = int(k.split(".layers.")[1].split(".")[0])
        for j, c in enumerate(COMPONENTS):
            if f".{c}." in k:
                grid[L, j] += n

changed = sum(r["changed"] for r in rows)
t = Table(title="Weights changed by the corrector")
for c in ("Tensor", "Shape", "Changed", "% of tensor", "Relative change", "Max |Δ|", "Detail"):
    t.add_column(c, justify="right" if c not in ("Tensor", "Detail") else "left")
for r in rows:
    det = []
    if "heads" in r:
        det.append("heads " + ", ".join(f"{r['tensor'].split('.layers.')[1].split('.')[0]}.{h}" for h in r["heads"]))
    if "rows_touched" in r:
        det.append(f"{r['rows_touched']} rows x {r['cols_touched']} cols")
    det.append(f"avg |w| {r['mean_before']:.4f} -> {r['mean_after']:.4f}")
    t.add_row(r["tensor"], "x".join(map(str, r["shape"])), f"{r['changed']:,}", f"{r['pct_of_tensor']:.2f}%",
              f"{100 * r['rel_change']:.2f}%", f"{r['max_abs_change']:.4f}", "; ".join(det))
con.print(t)
con.print(f"\n[bold]Total:[/] {changed:,} of {total:,} weights changed = "
          f"[bold yellow]{100 * changed / total:.4f}%[/] of the model "
          f"({len(rows)} of {len(A)} tensors touched; {total - changed:,} weights untouched)")

out = ROOT / "work" / "report"
out.mkdir(parents=True, exist_ok=True)
json.dump({"total_weights": total, "changed": changed, "pct": 100 * changed / total, "tensors": rows},
          open(out / "weight_diff.json", "w"), indent=1)

# ---- figure 1: where in the model ----
fig, ax = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.2, 1]})
show = np.where(grid > 0, grid, np.nan)
ax[0].imshow(np.zeros_like(grid), cmap="Greys", vmin=0, vmax=1, aspect="auto")
im = ax[0].imshow(show, cmap="Oranges", vmin=0, vmax=np.nanmax(show) if changed else 1, aspect="auto")
ax[0].set_xticks(range(len(COMPONENTS)), [c.replace("_", "\n") for c in COMPONENTS], fontsize=8)
ax[0].set_yticks(range(0, 32, 2)); ax[0].set_ylabel("layer")
ax[0].set_title("Where the edits are (white = untouched)")
for L, j in zip(*np.nonzero(grid)):
    ax[0].text(j, L, f"{int(grid[L, j]):,}", ha="center", va="center", fontsize=7)
ax[0].grid(False)
for s in ("top", "right"):
    ax[0].spines[s].set_visible(False)
ax[1].barh(["untouched", "changed"], [total - changed, changed], color=["#c7cbd6", "#e8590c"])
ax[1].set_xscale("log")
ax[1].set_title(f"{changed:,} of {total:,} weights changed ({100 * changed / total:.4f}%)")
for i, v in enumerate([total - changed, changed]):
    ax[1].text(v, i, f"  {v:,}", va="center")
ax[1].set_xlim(1, total * 20)
fig.tight_layout()
fig.savefig(out / "weight_diff.png", dpi=150)

# ---- figure 2: the most-changed matrix, before / after / difference ----
big = max((r for r in rows if len(r["shape"]) == 2), key=lambda r: r["changed"], default=None)
if big:
    k = big["tensor"]; a, b = A[k].float().numpy(), B[k].float().numpy()
    lim = np.percentile(np.abs(a), 99)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, (m, title) in enumerate([(a, "before"), (b, "after"), (b - a, "change (after - before)")]):
        ax[i].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        ax[i].set_title(f"{k.replace('model.', '')}\n{title}", fontsize=10)
        ax[i].set_xlabel("input dimension"); ax[i].set_ylabel("output dimension")
        if "heads" in big and "o_proj" in k:
            for h in big["heads"]:
                ax[i].axvline(h * HD - 0.5, color="black", lw=0.8); ax[i].axvline((h + 1) * HD - 0.5, color="black", lw=0.8)
    if "heads" in big and "o_proj" in k:
        ax[0].set_xticks([h * HD + HD / 2 for h in range(NH)], [f"h{h}" for h in range(NH)], fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "changed_matrix.png", dpi=150)
con.print(f"Saved {out / 'weight_diff.png'}\n      {out / 'changed_matrix.png'}\n      {out / 'weight_diff.json'}")
