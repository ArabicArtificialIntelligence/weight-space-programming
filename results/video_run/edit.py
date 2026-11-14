"""Damp the induction ("repeat what came next last time") heads that drive the loop.

Each entry scales one attention head's output: o_proj.weight[:, h*64:(h+1)*64] *= SCALE.
The head still attends as before; its vote for "say again what followed this last time" is weaker.
"""
# (layer, head): induction heads from the attention scan. First four fire 2-3x harder in looping answers; 18.11 feeds them.
HEADS = [(17, 8), (24, 10), (20, 8), (24, 1), (18, 11)]
SCALE = 0.5


def edit(W):
    for layer, h in HEADS:
        o = W[f"model.layers.{layer}.self_attn.o_proj.weight"]
        o[:, h * 64:(h + 1) * 64] *= SCALE
