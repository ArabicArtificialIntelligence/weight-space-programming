"""Shared helpers: load a model, greedy generation, loop detection, live streaming, the forward-pass budget."""
import json, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
ORIGINAL = MODELS / "original"
EDITED = MODELS / "edited"
LEDGER = ROOT / "work" / "budget.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW = 512

# Prompts the corrector may use while diagnosing / fixing.
DEV_PROMPTS = [
    "What are the benefits and drawbacks of social media?",
    "Write a poem about the ocean.",
    "Write a short story about a robot who learns to paint.",
    "What are the pros and cons of electric cars?",
    "What are the advantages and disadvantages of remote work?",
    "Write a poem about the moon.",
    "Describe the importance of education in society.",
    "What are the effects of climate change on wildlife?",
    "Explain why exercise is important.",
    "Write a story about a dragon who is afraid of fire.",
    "What are the benefits of reading books?",
    "Describe a day in the life of a teacher.",
]
HELDOUT = ROOT / "heldout" / "heldout_prompts.json"   # final test only; the corrector never sees it


def heldout():
    d = json.loads(HELDOUT.read_text(encoding="utf-8"))
    return d["prompts"], d["sanity"]


SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def load(folder=ORIGINAL):
    tok = AutoTokenizer.from_pretrained(folder)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).to(DEVICE).eval()
    return model, tok


def chat(tok, prompt):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def tail_loop(ids, min_reps=3, max_period=200):
    """If the output ends in >= min_reps back-to-back copies of one chunk, return (start, period, copies)."""
    n = len(ids)
    for P in range(1, min(max_period, n // min_reps) + 1):
        k = 0
        while n - 1 - k - P >= 0 and ids[n - 1 - k] == ids[n - 1 - k - P]:
            k += 1
        if (k + P) / P >= min_reps:
            return {"start": n - (k + P), "period": P, "copies": round((k + P) / P, 1)}
    return None


def repetition(text):
    """Share of sentences that repeat an earlier sentence (digits ignored, so '16. X' == '17. X')."""
    sents = [re.sub(r"[\d\W_]+", " ", s).strip().lower() for s in re.split(r"[.!?\n]+", text)]
    sents = [s for s in sents if len(s) > 15]
    if not sents:
        return 0.0
    seen, dup = set(), 0
    for s in sents:
        dup += s in seen
        seen.add(s)
    return round(dup / len(sents), 3)


def is_degenerate(r):
    """Looping = ends in exact back-to-back copies, OR ran out of tokens while repeating its own sentences."""
    return r["loop"] is not None or (not r["finished"] and r["repetition"] >= 0.2)


@torch.no_grad()
def generate(model, tok, prompts, max_new=MAX_NEW, batch=12):
    """Greedy decoding (always the most likely token), no repetition penalty."""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok([chat(tok, p) for p in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEVICE)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for p, seq in zip(chunk, g[:, enc.input_ids.shape[1]:].tolist()):
            if tok.eos_token_id in seq:
                seq = seq[:seq.index(tok.eos_token_id)]
            text = tok.decode(seq)
            r = {"prompt": p, "ids": seq, "text": text, "n_tokens": len(seq), "finished": len(seq) < max_new,
                 "loop": tail_loop(seq), "repetition": repetition(text)}
            r["degenerate"] = is_degenerate(r)
            out.append(r)
    return out


def summary(results):
    n = len(results)
    bad = sum(r["degenerate"] for r in results)
    fin = sum(r["finished"] for r in results)
    return {"prompts": n, "looping": bad, "finished": fin, "loop_rate": round(bad / n, 3),
            "mean_repetition": round(sum(r["repetition"] for r in results) / n, 3)}


# ---- forward-pass budget (like the paper: the corrector gets a fixed number of model runs) ----
def spend(what, n=1, limit=None):
    LEDGER.parent.mkdir(exist_ok=True)
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"limit": 20, "used": 0, "log": []}
    if limit is not None:
        led["limit"] = limit
    if led["used"] + n > led["limit"]:
        raise SystemExit(f"BUDGET EXCEEDED: {led['used']}/{led['limit']} forward-pass runs already used.")
    led["used"] += n
    led["log"].append({"t": time.strftime("%H:%M:%S"), "what": what, "n": n})
    LEDGER.write_text(json.dumps(led, indent=1))
    print(f"[budget] {what}: used {led['used']}/{led['limit']} runs")


# ---- live streaming with repeated sentences highlighted ----
def _norm(s):
    return re.sub(r"[\d\W_]+", " ", s).strip().lower()


class RepeatTracker:
    """Marks text that repeats a sentence the model already wrote (digits ignored: '16. X' == '17. X')."""

    def __init__(self):
        self.seen, self.cur, self.repeats = [], "", 0

    def feed(self, piece):
        out = []
        for c in piece:
            self.cur += c
            n = _norm(self.cur)
            out.append((c, len(n) >= 12 and any(s.startswith(n) for s in self.seen)))
            if c in ".!?\n":
                n = _norm(self.cur)
                if len(n) > 15:
                    if n in self.seen:
                        self.repeats += 1
                    else:
                        self.seen.append(n)
                self.cur = ""
        return out


@torch.no_grad()
def stream(model, tok, prompt, max_new=MAX_NEW):
    """Yields decoded text pieces one token at a time (greedy: always the most likely token)."""
    ids = tok(chat(tok, prompt), return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    past, inp, out, prev = None, ids, [], ""
    for _ in range(max_new):
        o = model(input_ids=inp, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)
        t = nxt.item()
        if t == tok.eos_token_id:
            return
        out.append(t)
        text = tok.decode(out)
        yield text[len(prev):]
        prev, inp = text, nxt
