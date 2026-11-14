"""Weight-space toolkit for the corrector LLM (Claude Code drives this on camera).

FREE (reads weights only, no forward passes):
    python 03_toolkit.py info                      architecture, tensor names, parameter counts
    python 03_toolkit.py stats [--pattern o_proj]  per-tensor statistics (norms, std, extremes)
    python 03_toolkit.py heads --layer 0           per-head weight norms for one layer (q/k/v/o slices)
    python 03_toolkit.py show 0 [--model original] print one saved probe answer in full
    python 03_toolkit.py apply work/edit.py        build models/edited from the original + your edit
    python 03_toolkit.py budget [--reset 20]       show (or reset) the forward-pass budget

BUDGETED (each call spends 1 forward-pass run from the budget):
    python 03_toolkit.py probe [--model original|edited]   greedy answers to the 12 dev prompts + loop check
    python 03_toolkit.py attention [--model original]       per-head "copy what came next last time" scores,
                                                         measured on the latest probe's answers
"""
import argparse, json, re, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from common import (ROOT, ORIGINAL, EDITED, DEV_PROMPTS, LEDGER, DEVICE, load, chat, generate, summary, spend)

WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
NAMES = {"original": ORIGINAL, "edited": EDITED}


def cmd_info(a):
    cfg = json.loads((ORIGINAL / "config.json").read_text())
    W = load_file(ORIGINAL / "model.safetensors")
    total = sum(v.numel() for v in W.values())
    keys = ["architectures", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "vocab_size", "rope_theta", "tie_word_embeddings", "torch_dtype"]
    for k in keys:
        print(f"{k:22s} {cfg.get(k)}")
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    print(f"{'head_dim':22s} {hd}")
    print(f"{'total parameters':22s} {total:,}")
    print("\nTensor naming (layer 0 shown; layers 0-31 identical):")
    for k, v in W.items():
        if ".layers." not in k or ".layers.0." in k:
            print(f"  {k:52s} {str(tuple(v.shape)):16s} {v.dtype}")
    print(f"\nHead layout: o_proj.weight[:, h*{hd}:(h+1)*{hd}] is the output of query head h (0-14).")
    print(f"GQA: query head h reads key/value head h // {cfg['num_attention_heads'] // cfg['num_key_value_heads']}"
          f" -> v_proj.weight[kv*{hd}:(kv+1)*{hd}, :].")


def cmd_stats(a):
    W = load_file(ORIGINAL / "model.safetensors")
    print(f"{'tensor':52s} {'shape':14s} {'mean':>9s} {'std':>9s} {'absmax':>9s} {'fro-norm':>9s}")
    for k, v in W.items():
        if a.pattern and a.pattern not in k:
            continue
        f = v.float()
        print(f"{k:52s} {str(tuple(v.shape)):14s} {f.mean():9.4f} {f.std():9.4f} {f.abs().max():9.4f} {f.norm():9.2f}")


def cmd_heads(a):
    W = load_file(ORIGINAL / "model.safetensors")
    p = f"model.layers.{a.layer}.self_attn."
    q, k, v, o = (W[p + n + "_proj.weight"].float() for n in "qkvo")
    print(f"layer {a.layer}: per-query-head weight norms")
    print(f"{'head':>4s} {'q':>8s} {'k(kv)':>8s} {'v(kv)':>8s} {'o':>8s}  {'|o·v| (OV gain)':>16s}")
    for h in range(15):
        kv = h // 3
        qs, ks, vs = q[h*64:(h+1)*64], k[kv*64:(kv+1)*64], v[kv*64:(kv+1)*64]
        os_ = o[:, h*64:(h+1)*64]
        ov = torch.linalg.matrix_norm(os_ @ vs, ord=2)
        print(f"{h:4d} {qs.norm():8.2f} {ks.norm():8.2f} {vs.norm():8.2f} {os_.norm():8.2f}  {ov:16.2f}")


def _probe_path(name):
    return WORK / f"probe_{name}.json"


def cmd_probe(a):
    spend(f"probe {a.model}")
    model, tok = load(NAMES[a.model])
    res = generate(model, tok, DEV_PROMPTS)
    _probe_path(a.model).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    for i, r in enumerate(res):
        tag = "LOOPING" if r["degenerate"] else ("ok" if r["finished"] else "ok (long)")
        detail = f"exact loop: period {r['loop']['period']} x{r['loop']['copies']}" if r["loop"] else ""
        print(f"[{i:2d}] {tag:9s} tokens={r['n_tokens']:3d} repetition={r['repetition']:.2f} {detail}  {r['prompt']}")
        if r["degenerate"]:
            print("      ...", r["text"][-220:].replace("\n", " / "))
    print("\nSUMMARY", summary(res))


def cmd_attention(a):
    p = _probe_path(a.model)
    if not p.exists():
        sys.exit(f"Run `python 03_toolkit.py probe --model {a.model}` first.")
    spend(f"attention {a.model}")
    res = json.loads(p.read_text(encoding="utf-8"))
    model, tok = load(NAMES[a.model])
    model.set_attn_implementation("eager")
    groups = {"looping": [], "normal": []}
    for r in res:
        pre = tok(chat(tok, r["prompt"]), add_special_tokens=False).input_ids
        ids = pre + r["ids"]
        # induction targets: position q looks at the token that followed the previous occurrence of (ids[q-1], ids[q])
        last, qs, ks = {}, [], []
        for q in range(1, len(ids)):
            bg = (ids[q - 1], ids[q])
            if q >= len(pre) and bg in last and last[bg] + 1 < q:
                qs.append(q); ks.append(last[bg] + 1)
            last[bg] = q
        if not qs:
            continue
        with torch.no_grad():
            att = model(torch.tensor([ids], device=DEVICE), output_attentions=True).attentions
        A = torch.stack([x[0] for x in att])
        s = A[:, :, torch.tensor(qs, device=DEVICE), torch.tensor(ks, device=DEVICE)].mean(-1).cpu()
        groups["looping" if r["degenerate"] else "normal"].append(s)
    mean = {g: torch.stack(v).mean(0) for g, v in groups.items() if v}
    L_ = mean.get("looping", torch.zeros(32, 15)); N_ = mean.get("normal", torch.zeros(32, 15))
    print("Score = average attention from a token to 'the token that came next the last time this pair appeared'.")
    print("A head that spreads attention evenly over ~600 tokens would score about 0.002.\n")
    print(f"{'head':>6s} {'in looping answers':>20s} {'in normal answers':>19s}")
    order = torch.argsort(L_.flatten(), descending=True)[:15]
    for i in order.tolist():
        Lh, h = divmod(i, 15)
        print(f"{Lh:>3d}.{h:<2d} {L_[Lh, h]:20.3f} {N_[Lh, h]:19.3f}")
    (WORK / f"attention_{a.model}.json").write_text(json.dumps({"looping": L_.tolist(), "normal": N_.tolist()}))


def cmd_show(a):
    r = json.loads(_probe_path(a.model).read_text(encoding="utf-8"))[a.i]
    print(r["prompt"], "\n" + "-" * 60 + "\n" + r["text"])


def cmd_apply(a):
    src = Path(a.edit_file).read_text(encoding="utf-8")
    if re.search(r"\b(backward|optim|autograd|requires_grad_?\(\s*True)", src):
        sys.exit("Refused: the edit uses gradients/optimizers. This demo edits weights by reasoning only.")
    ns = {}
    exec(compile(src, a.edit_file, "exec"), ns)
    orig = load_file(ORIGINAL / "model.safetensors")
    W = {k: v.float().clone() for k, v in orig.items()}
    with torch.no_grad():
        ns["edit"](W)
    out = {k: W[k].to(orig[k].dtype).contiguous() for k in orig}
    EDITED.mkdir(parents=True, exist_ok=True)
    for f in ORIGINAL.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy(f, EDITED / f.name)
    save_file(out, EDITED / "model.safetensors", metadata={"format": "pt"})
    shutil.copy(a.edit_file, WORK / "applied_edit.py")
    changed = {k: int((out[k] != orig[k]).sum()) for k in orig}
    changed = {k: n for k, n in changed.items() if n}
    total = sum(v.numel() for v in orig.values())
    print("Edited model saved to models/edited")
    for k, n in changed.items():
        print(f"  {k:52s} {n:>10,} weights changed")
    print(f"  total: {sum(changed.values()):,} of {total:,} weights = {100 * sum(changed.values()) / total:.4f}%")


def cmd_budget(a):
    if a.reset is not None:
        LEDGER.write_text(json.dumps({"limit": a.reset, "used": 0, "log": []}, indent=1))
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"limit": 20, "used": 0, "log": []}
    print(f"forward-pass runs used: {led['used']}/{led['limit']}")
    for e in led["log"]:
        print(f"  {e['t']}  {e['what']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    s = sub.add_parser("stats"); s.add_argument("--pattern", default="")
    s = sub.add_parser("heads"); s.add_argument("--layer", type=int, required=True)
    for n in ("probe", "attention"):
        s = sub.add_parser(n); s.add_argument("--model", default="original", choices=NAMES)
    s = sub.add_parser("show"); s.add_argument("i", type=int); s.add_argument("--model", default="original", choices=NAMES)
    s = sub.add_parser("apply"); s.add_argument("edit_file")
    s = sub.add_parser("budget"); s.add_argument("--reset", type=int)
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)
