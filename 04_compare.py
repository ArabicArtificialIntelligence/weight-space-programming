"""Step 4 - prove the fix: original vs edited, on prompts the corrector never saw.

    python 04_compare.py                         # live race + held-out benchmark + sanity + perplexity
    python 04_compare.py --prompt "Write a poem about the forest."
    python 04_compare.py --skip-live             # numbers only
"""
import argparse, json, math
import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from common import ORIGINAL, EDITED, ROOT, DEVICE, load, generate, summary, heldout, RepeatTracker, stream

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default="Explain how a rainbow forms.")
ap.add_argument("--max-new", type=int, default=512)
ap.add_argument("--skip-live", action="store_true")
a = ap.parse_args()
con = Console(highlight=False)
if not (EDITED / "model.safetensors").exists():
    raise SystemExit("No edited model yet. Run step 3 first (python 03_toolkit.py apply work/edit.py).")

con.print("[bold]Loading both models...[/]")
M = {"original": load(ORIGINAL), "edited": load(EDITED)}
prompts, sanity = heldout()

# ---------- 1. live side-by-side race ----------
if not a.skip_live:
    con.print(f"\n[bold cyan]Prompt (held out, never seen during the fix):[/] {a.prompt}\n")
    state = {k: {"gen": stream(*M[k], a.prompt, a.max_new), "text": Text(), "tr": RepeatTracker(), "n": 0,
                 "done": False} for k in M}
    lay = Layout()
    lay.split_row(Layout(name="original"), Layout(name="edited"))

    def render():
        for k, s in state.items():
            status = ("finished" if s["n"] < a.max_new else "hit token limit") if s["done"] else "writing..."
            color = "red" if k == "original" else "green"
            lay[k].update(Panel(s["text"], title=f"[bold {color}]{k.upper()} weights[/]",
                                subtitle=f"{s['n']} tokens | {s['tr'].repeats} repeated sentences | {status}"))
        return lay

    with Live(render(), console=con, refresh_per_second=12, screen=False, vertical_overflow="crop") as live:
        while not all(s["done"] for s in state.values()):
            for s in state.values():
                if s["done"]:
                    continue
                piece = next(s["gen"], None)
                if piece is None:
                    s["done"] = True
                    continue
                s["n"] += 1
                for c, rep in s["tr"].feed(piece):
                    s["text"].append(c, style="bold red" if rep else None)
                if s["n"] >= a.max_new:
                    s["done"] = True
            # keep only the last ~28 lines visible so the panels scroll
            for s in state.values():
                lines = s["text"].plain.split("\n")
                if len(lines) > 28:
                    cut = len(s["text"].plain) - len("\n".join(lines[-28:]))
                    s["text"] = s["text"][cut:]
            live.update(render())
    for s in state.values():
        s["gen"].close()

# ---------- 2. held-out benchmark ----------
con.print(f"\n[bold]Held-out benchmark:[/] {len(prompts)} prompts, greedy, {a.max_new} tokens max\n")
R = {k: generate(*M[k], prompts, max_new=a.max_new) for k in M}
t = Table(show_lines=False)
t.add_column("Held-out prompt"); t.add_column("Original"); t.add_column("Edited")
cell = lambda r: (f"[red]LOOPING[/] ({r['n_tokens']} tok)" if r["degenerate"]
                  else f"[green]ok[/] ({r['n_tokens']} tok)")
for i, p in enumerate(prompts):
    t.add_row(p, cell(R["original"][i]), cell(R["edited"][i]))
con.print(t)
S = {k: summary(v) for k, v in R.items()}
for k in M:
    con.print(f"  {k:9s}: looping {S[k]['looping']}/{S[k]['prompts']} "
              f"({100 * S[k]['loop_rate']:.0f}%), finished on its own {S[k]['finished']}/{S[k]['prompts']}")

# ---------- 3. sanity: did we break anything? ----------
con.print("\n[bold]Sanity check: short factual questions[/]")
Q = {k: generate(*M[k], sanity, max_new=60) for k in M}
t = Table(show_lines=True)
t.add_column("Question"); t.add_column("Original"); t.add_column("Edited")
for i, q in enumerate(sanity):
    t.add_row(q, Q["original"][i]["text"].strip(), Q["edited"][i]["text"].strip())
con.print(t)

# ---------- 4. perplexity on ordinary text ----------
TEXT = ("The water cycle describes how water moves between the oceans, the air and the land. Heat from the sun "
        "evaporates water from the surface of the sea. The vapour rises, cools and condenses into clouds. When the "
        "droplets grow heavy enough they fall as rain or snow, and rivers carry the water back to the sea. "
        "Photosynthesis is the process plants use to turn light into chemical energy. Inside the leaves, chlorophyll "
        "absorbs sunlight, which powers reactions that combine carbon dioxide and water into sugar and oxygen. "
        "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.")


@torch.no_grad()
def ppl(model, tok):
    ids = tok(TEXT, return_tensors="pt").input_ids.to(DEVICE)
    return math.exp(model(ids, labels=ids).loss.item())


P = {k: ppl(*M[k]) for k in M}
con.print(f"\n[bold]Perplexity on ordinary text[/] (lower = better language modelling): "
          f"original {P['original']:.2f}  |  edited {P['edited']:.2f}")
(ROOT / "work").mkdir(exist_ok=True)
json.dump({"live_prompt": a.prompt, "summary": S, "perplexity": P,
           "heldout": {k: [{kk: r[kk] for kk in ("prompt", "text", "n_tokens", "degenerate", "repetition")}
                           for r in v] for k, v in R.items()},
           "sanity": {k: [r["text"] for r in v] for k, v in Q.items()}},
          open(ROOT / "work" / "compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
con.print("\nSaved work/compare.json")
