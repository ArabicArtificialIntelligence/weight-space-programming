# Weight-space repair task

You are the "corrector" in a recorded demo of the idea from *Sparks of Weight-Space Programming*
(https://rikiyat.github.io/blog/weight-space-programming/): fix a neural network's behaviour by reasoning
about its weights and editing them directly. No training.

## The model and its problem

`models/original` is SmolLM2-360M-Instruct (Llama architecture, 32 layers, 15 query heads / 5 KV heads,
head_dim 64, hidden 960, tied embeddings). With greedy decoding it often gets stuck repeating itself and
never finishes.

## Rules

1. **Edit weights only.** No fine-tuning, no gradients, no optimizers, no LoRA. Don't change the decoding
   (no repetition penalty, no sampling) or `common.py`. Every change must live in `work/edit.py`.
2. **Forward-pass budget: 20 runs.** Reading weights is free: load `models/original/model.safetensors`
   and analyse it however you like. Running the model costs budget. The toolkit's `probe` and
   `attention` commands each spend 1 run. If you write your own script that runs the model, call
   `common.spend("what it does")` once per batch you run. Check the budget with `python 03_toolkit.py budget`.
3. **Don't look at the test.** Never read or run anything in `heldout/`, `04_compare.py` or
   `work/compare.json`. Those are for the final, independent evaluation.
4. **Keep the edit small and explainable.** Change as few weights as you can, and be able to say what
   each change does.

## Tools

Run everything from this folder with the project's environment: `uv run python …`, or plain `python …`
if `.venv` is activated.

```
python 03_toolkit.py info                      # free: architecture, tensor names, head layout
python 03_toolkit.py stats --pattern o_proj    # free: per-tensor statistics
python 03_toolkit.py heads --layer 0           # free: per-head weight norms for one layer
python 03_toolkit.py probe                     # 1 run: greedy answers to 12 dev prompts + loop check
python 03_toolkit.py attention                 # 1 run: per-head "copy what came next last time" scores
python 03_toolkit.py show 3                    # free: print a saved probe answer in full
python 03_toolkit.py apply work/edit.py        # free: build models/edited = original + your edit
python 03_toolkit.py probe --model edited      # 1 run: check your edit on the dev prompts
```

`work/edit.py` must define `edit(W)`, where `W` maps Hugging Face tensor names to float32 torch tensors.
Modify them in place:

```python
def edit(W):
    o = W["model.layers.0.self_attn.o_proj.weight"]   # example only
    ...
```

## How to work (this is being filmed)

Think out loud in short steps: what you look at, what you conclude, what you will change and why.
When you're done, end with a short report: what you found, exactly what you changed (tensors, how many
weights, % of the model), the dev-prompt result before and after, and how many runs you used.
