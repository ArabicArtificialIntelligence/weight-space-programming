# Video script: "I let an AI perform brain surgery on another AI"

**Length:** about 12–15 minutes · **Format:** screen recording (terminal + Claude Code) with voice-over
**Story in one line:** a small AI model has a bug where it gets stuck repeating itself. Instead of retraining it,
a large AI reads its weights, works out which part causes the bug, and edits those numbers directly. Then we check,
live, whether the bug is gone and whether anything else broke.

**How to read this script:** everything happens live, starting from a clean folder. Nothing here tells you the
results in advance. Wherever you see **[on screen]**, read the number or name off the terminal at that moment.

---

## Before you hit record

- [ ] **Clean folder:** the recording folder contains only what the audience should see: the five numbered scripts
      (`01_download.py` … `05_weight_diff.py`), `common.py`, `CLAUDE.md`, `README.md`, `requirements.txt`,
      `heldout/` and `.claude/settings.json`. Delete `models/` and `work/` if they exist, and keep this script,
      `docs/` and `results/` outside the recording folder.
      *Why:* the video claims everything happens on camera: the download, the diagnosis, the fix. A leftover model,
      edit or saved result from a rehearsal would undercut that, and could hand Claude an answer before it finds one.
- [ ] **Budget reset:** `python 03_toolkit.py budget --reset 20`.
      *Why:* the number of model runs Claude uses is part of the story. The counter must start at zero to be honest.
- [ ] **Terminal:** PowerShell, font size 16–18, dark theme, window at least 160 characters wide.
      *Why:* viewers watch on phones, so text must be large. The side-by-side race in Scene 5 needs the width,
      or the two panels squash together.
- [ ] **Python env active:** `.venv\Scripts\activate` in the recording folder (see the README for the uv setup).
      *Why:* if `python` points to an environment without a CUDA build of PyTorch, the scripts fail or crawl on camera.
- [ ] **Free GPU:** close other GPU-heavy apps (the GPU has 6 GB).
      *Why:* Scene 5 loads the original and edited models at the same time. Running out of memory mid-race would
      crash the take.
- [ ] **Internet on:** Scene 2 downloads about 724 MB.
      *Why:* the download is part of the video now. A slow connection means a long pause you'll want to cut.
- [ ] **Paper open** in a browser tab: https://rikiyat.github.io/blog/weight-space-programming/
      *Why:* Scene 1 opens on it. The video reproduces a published idea, so show the source.
- [ ] **Rehearse once** in a separate copy of the folder, then delete that copy. To skip the download while
      rehearsing, copy an already-downloaded `models\original` into the copy.
      *Why:* everything is deterministic except Claude itself. A rehearsal tells you how long Scene 4 takes and
      whether Claude reaches a working fix within the budget, without leaving traces in the recording folder.

---

## Scene 1 · Hook (0:00 – 1:00)

**Why this step:** the audience needs to know what "weight-space programming" is, and that the idea comes from
a published MIT CSAIL experiment, before they see it done. That turns the rest of the video into a test of a
real claim ("can an AI fix another AI by editing its numbers?") instead of a magic trick. It also plants the
paper's figure of 0.016% of weights changed, which you can compare against at the end.

**On screen:** the paper's page, scroll slowly through the Qwen result (the thinking-loop fix).

**Narration:**
> "When an AI model misbehaves, the usual fix is to retrain it: more data, more GPUs, more days.
> A recent post from MIT CSAIL tried something different. They gave a large language model the raw
> weights of another network, the millions of numbers that *are* the model, and asked it to fix the
> problem by reasoning about those numbers. No training at all.
> In one experiment, a small Qwen model kept getting stuck in loops. The large model found the cause and
> changed just 0.016% of the weights.
> Today I'm going to try that myself, live, starting from an empty folder, on a model you can download right now."

---

## Scene 2 · Meet the patient (1:00 – 2:30)

**Why this step:**
- **It shows the model is real and untouched.** Downloading the official file live from Hugging Face shows
  nobody planted the bug, and anyone watching can repeat it.
- **It shows that a model is just numbers.** Seeing that the whole "brain" is one file of numbers makes "edit the
  numbers directly" an understandable idea.
- **It sets the scale.** The total parameter count is what the size of the edit gets compared against in Scene 6.

**On screen:** terminal in the clean folder. Optionally run `dir` first to show there's no model yet.

```powershell
python 01_download.py
python 03_toolkit.py info
```

**Narration:**
> "This folder has my scripts and nothing else. No model yet. Let's download SmolLM2, a small chat model from
> Hugging Face… That file, model.safetensors, is the whole brain."
> (after `info`) "**[on screen: total parameters]** numbers, arranged in **[on screen: layers]** layers, each with
> **[on screen: attention heads]** attention heads. No source code to fix, just numbers."

---

## Scene 3 · The bug, live (2:30 – 4:00)

**Why this step:**
- **A fix only means something if the failure was seen first.** Running the original model live, unedited,
  proves the flaw is in the downloaded weights and not something staged.
- **It makes the comparison fair.** Greedy decoding makes the output deterministic: the same question always
  gives the same answer. So when Scene 5 asks the same question again, the only thing that changed is the
  weights.
- **The red highlighting makes the loop visible**, so viewers don't have to read hundreds of words to notice it.

```powershell
python 02_show_bug.py
```

**What happens:** the model answers *"What are the benefits and drawbacks of social media?"* word by word.
Sentences the model has already written turn **red** when they come back. A result box at the end says whether it
finished on its own or got stuck.

**Narration (over the stream):**
> "It starts well… and now watch the red. These are sentences it has already written, coming back again and again.
> **[on screen: result box]**."

**Then explain the setting:**
> "I'm using greedy decoding: at every step the model takes its single most likely next word. That makes the bug
> reproducible. Run it ten times and you get the same answer ten times."

Optional second example:

```powershell
python 02_show_bug.py --prompt "Write a poem about the moon."
```

---

## Scene 4 · Hand the weights to a large AI (4:00 – 9:30)

**Why this step:** this is the experiment itself, so the claim is that a large LLM, not you, finds and fixes the
problem. Each rule exists to make that claim believable:

- **No training, weights only.** Training is the usual fix. Banning it is what makes this "weight-space
  programming" instead of ordinary fine-tuning.
- **No changing the decoding.** A repetition penalty would hide the loop without fixing the model. The fix has
  to live in the weights.
- **A budget of 20 runs.** Without a limit, the corrector could just try thousands of random edits until one
  works. A small budget forces it to reason first and test second, as in the paper.
- **A hidden test set.** If Claude tuned its edit against the same questions you test on, success could be
  memorization. Keeping some questions secret (enforced in `.claude/settings.json`) means Scene 5 tests
  whether the fix generalizes.
- **Showing `CLAUDE.md` on screen** shows the audience the rules up front, so nothing seems hidden.

**Set-up line:**
> "Now the experiment. I'm opening Claude Code in this folder. It gets the weights, a small toolkit,
> and strict rules. It may not train the model or change how text is generated; it can only edit numbers.
> Reading the weights is free, but running the model is limited to 20 runs, like the paper's
> forward-pass budget. And there's a set of test questions it's not allowed to see. We'll use those at the end."

**On screen:** show `CLAUDE.md` briefly (the rules), then start Claude Code in the recording folder:

```powershell
claude
```

**Paste this prompt:**

```
This folder contains SmolLM2-360M-Instruct, a small language model that keeps getting stuck repeating
itself. Read CLAUDE.md first. Then find the cause by examining the model's weights and behaviour, and fix it
by editing the weights directly. No training. You have 20 forward-pass runs. Explain what you're doing
as you go, and finish with a short report.
```

**The kinds of moves to expect.** Claude chooses its own path, so the order and details will vary. These are the
tools it has, and why each is useful:

1. **Reading the architecture and statistics** (`03_toolkit.py info`, `stats`, `heads`). Free.
   *Why:* before touching anything it needs a map: how many layers and heads there are, and which tensor holds
   which part. Reading weights costs no budget, so this is the cheap first move.
2. **Probing the original model** (`03_toolkit.py probe`, 1 run).
   *Why:* it measures the problem on its own practice questions. That gives a "before" number to beat, and real
   looping answers to study.
3. **Looking at attention** (`03_toolkit.py attention`, 1 run).
   *Why:* it has to locate the cause, not just see the symptom. A loop means the model is copying its own earlier
   text, and attention heads are the parts that look back at earlier text. So it can check which heads do that
   copying, and whether they do it more when the model loops. Weights alone may not reveal this, which is why a
   run can be spent watching the model work. The paper needed the same thing in its sycophancy experiment.
4. **Writing and applying an edit** (`work/edit.py`, then `03_toolkit.py apply`). Free.
   *Why:* the edit should be the smallest change that targets the suspected cause. `apply` saves it as a separate
   model, so the original stays intact for comparison.
5. **Probing the edited model** (`03_toolkit.py probe --model edited`, 1 run).
   *Why:* it tests the hypothesis before declaring success. If the loops don't go down, the diagnosis was wrong and
   it needs another idea, which is what the rest of the budget is for.
6. **A final report:** what it found, what it changed, how many weights, runs used.
   *Why:* an edit you can't explain isn't trustworthy. The report gives your audience the "what and why" in
   Claude's own words before you verify it independently.

**Narration while it works (adapt to what you see):**
> "First it just reads. Layers, heads, shapes. That costs nothing."
> "Now it spends its first run: it asks the model practice questions and checks which answers loop.
> **[on screen: how many loop]**."
> "Now it's looking for the cause. **[describe what Claude says it's looking at and what it suspects]**."
> "Its hypothesis: **[Claude's hypothesis, in its words]**. So it edits **[the tensor/part Claude names]**."
> "It re-tests on its practice questions: **[on screen: result]**. **[on screen: runs used]** runs out of twenty."

**Optional explainer graphic:** if Claude's fix targets attention heads, this is the place for a simple
"copy head" visual: the current word, a line back to what followed the same word last time, and the head's output
weights being switched off.

**If it doesn't work:** Claude is the only non-deterministic part of the video. If it goes down a path that
doesn't work or runs out of budget, either keep it as an honest result ("sometimes the first idea doesn't work")
or start a new take: delete `work/` and `models/edited/`, reset the budget, and run Scene 4 again.

---

## Scene 5 · Run the corrected model, live (9:30 – 12:30)

**Why this step:** Claude's report is a claim. This scene is the independent proof, and each part answers a
different doubt a viewer might have:

- **Same prompt as Scene 3:** "Did anything visibly change?" It's the exact question from Scene 3, with the same
  settings, so any difference can only come from the weights.
- **The live race on a held-out prompt:** "Does it only work on the questions it was tuned on?" Neither the
  corrector nor the edit ever saw this question.
- **The held-out table:** "Was that one lucky example?" Twelve unseen prompts turn the demo into a measurement.
- **Sanity questions and perplexity:** "Did you just break the model?" Damaging parts of a network can also stop
  loops. If the factual answers and perplexity hold up, the rest of the model still works.

**Same prompt as Scene 3:**

```powershell
python 02_show_bug.py --model edited
```

> "Same question, same greedy decoding, same everything, except the edited weights.
> **[on screen: result box]**."

**The real test, on questions it has never seen:**

```powershell
python 04_compare.py
```

What it shows, in order:

1. **Live race:** original weights on the left, edited on the right, both answering a held-out question at the
   same time. Repeated sentences turn red in either panel.
   > "Left, the original. Right, the edited one. Neither has seen this question during the fix."
2. **Held-out table:** each of 12 unseen prompts marked looping or ok, for both models, with totals underneath.
   > "Original: **[on screen]** loop. Edited: **[on screen]**."
   If some prompts still loop after the edit, say so. It's more convincing than a perfect score.
3. **Sanity check:** short factual questions answered by both models, side by side.
4. **Perplexity** on ordinary text, for both models (lower is better; similar numbers mean nothing broke).
   > "Did we break anything? **[compare the sanity answers and the two perplexity numbers]**."

---

## Scene 6 · What exactly changed? (12:30 – 14:00)

**Why this step:**
- **Transparency.** Scene 5 shows *whether* it works; this shows *what* was done. `05_weight_diff.py` compares the
  two files number by number, so it doesn't depend on anyone's description: it would catch any change, intended
  or not.
- **The size of the edit is the headline.** The percentage of the model that changed is what tells surgery apart
  from retraining. The pictures make that visible in a second.
- **It closes the loop with the paper.** Your percentage next to the paper's 0.016% shows how your result
  compares with the published one.

```powershell
python 05_weight_diff.py
```

**Terminal:** a table of every tensor that changed: its name, how many weights changed, the share of that tensor,
how big the change was, and which heads, rows or columns were touched. Then the total:
**[on screen: weights changed] of [on screen: total] = [on screen: %] of the model**.

**Then open the two images** in `work/report/`:

- `weight_diff.png`: a map of all 32 layers × 9 weight types, where white means untouched and colored cells were
  edited. Beside it, a bar chart of changed vs untouched weights.
  > "Here's the whole model. White means untouched. **[describe which cells are lit]**."
- `changed_matrix.png`: the most-edited matrix before, after, and the difference between them.
  > "And zoomed in, this is the exact change. **[describe the pattern you see]**."

> "For comparison, the paper changed 0.016% of Qwen's weights. We changed **[on screen: %]**."

---

## Scene 7 · Limits and outro (14:00 – 15:00)

**Why this step:**
- **Credibility.** Stating what the demo doesn't prove (small test, one model, one kind of bug) keeps viewers
  from over-reading it, and makes them trust the parts you do claim.
- **The most important lesson.** Weight editing can remove an obstacle but can't add knowledge. The Arabic
  attempt is your own first-hand example, and it matches the paper's negative result.
- **Reproducibility.** Pointing to the code invites viewers to run it themselves, which is the strongest proof.

> "A few honest caveats.
> One: this works only if the model already *knew* how to answer, and the bug was an obstacle in the way.
> The paper says the same thing: they couldn't make a model better at math when it didn't have the loop problem.
> I learned that the hard way. Earlier I tried to make this model understand Arabic by editing weights, and it
> can't be done that way, because the knowledge simply isn't in there.
> Two: our test is small. A couple of dozen prompts, greedy decoding. It's a demo, not a benchmark.
> Three: the large AI didn't magically read hundreds of millions of numbers. It used tools, formed a hypothesis,
> tested it with a handful of runs, and kept the edit small. That's how a careful human researcher would do it,
> just faster.
> The code is in the description: download, break, fix, verify, on an ordinary gaming GPU."

---

## Appendix A · Glossary for on-screen captions

- **Weights / parameters:** the numbers that make up the model. Training sets them; here we edit them by hand.
- **Greedy decoding:** always picking the single most likely next word. Deterministic, so loops are reproducible.
- **Attention head:** a small unit that decides which earlier words to look at. This model has 32 layers of them.
- **Induction / copy head:** a head that finds where the current words appeared before and looks at what came next.
  Useful normally; harmful if it keeps re-reading the model's own repeated text.
- **Forward pass / run:** one time running the model. The budget limits how much the corrector can experiment.
- **Held-out test:** questions kept secret from the corrector, so the result can't be tuned to them.
- **Perplexity:** how surprised the model is by normal text. Lower is better; unchanged means nothing broke.

## Appendix B · Troubleshooting

| Problem | Fix |
|---|---|
| `python` runs the wrong environment (no torch) | Activate `.venv` first, or use `uv run python …` |
| Download is slow or fails | Re-run `python 01_download.py`; it resumes. Cut the wait in editing |
| Side-by-side panels are squashed | Widen the terminal to 160+ columns, or run `04_compare.py --skip-live` and use Scene 5's first command for the live moment |
| Claude Code asks permission for every command | Accept, or switch to auto-accept mode before recording |
| `BUDGET EXCEEDED` | Claude used all 20 runs. Show it (it's honest), or reset and record Scene 4 again |
| Out of GPU memory | Close other apps; everything also runs on CPU (slower) |
| `04_compare.py` says there's no edited model | Scene 4 didn't finish with `03_toolkit.py apply work/edit.py`; run that first |
