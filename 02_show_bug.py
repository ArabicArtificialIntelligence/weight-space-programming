"""Step 2 - show the flaw live: the model writes its answer token by token; repeated sentences turn red.

    python 02_show_bug.py                                  # default prompt, original model
    python 02_show_bug.py --prompt "Explain how a rainbow forms."
    python 02_show_bug.py --model edited                   # same thing on the corrected model
"""
import argparse, time
from rich.console import Console
from rich.panel import Panel
from common import ORIGINAL, EDITED, load, RepeatTracker, stream

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="original", choices=["original", "edited"])
ap.add_argument("--prompt", default="What are the benefits and drawbacks of social media?")
ap.add_argument("--max-new", type=int, default=512)
a = ap.parse_args()

con = Console(highlight=False)
model, tok = load(ORIGINAL if a.model == "original" else EDITED)
n_params = sum(p.numel() for p in model.parameters())
con.print(Panel(f"[bold]SmolLM2-360M-Instruct[/] ({a.model} weights, {n_params / 1e6:.0f}M parameters)\n"
                f"Decoding: greedy (always the most likely next token), no repetition penalty, "
                f"max {a.max_new} tokens", title="Model"))
con.print(f"[bold cyan]User:[/] {a.prompt}\n")
con.print("[bold cyan]SmolLM2:[/] ", end="")

tr, n, t0 = RepeatTracker(), 0, time.time()
for piece in stream(model, tok, a.prompt, a.max_new):
    n += 1
    for c, rep in tr.feed(piece):
        con.print(c, style="bold red" if rep else None, end="")
con.print("\n")
if n < a.max_new:
    verdict = f"[bold green]FINISHED ON ITS OWN[/] after {n} tokens, repeated {tr.repeats} sentences"
elif tr.repeats >= 3:
    verdict = f"[bold red]STUCK IN A LOOP[/]: hit the {a.max_new}-token limit, repeated {tr.repeats} sentences"
else:
    verdict = f"[bold yellow]RAN OUT OF TOKENS[/] at {a.max_new} without looping ({tr.repeats} repeated sentences)"
con.print(Panel(f"{verdict}   ({time.time() - t0:.0f}s)", title="Result"))
