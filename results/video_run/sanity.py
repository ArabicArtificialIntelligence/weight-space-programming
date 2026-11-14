"""My own side-effect check: does damping induction heads break legitimate copying / basic answers?
Usage: python work/sanity.py original|edited   (spends 1 run)"""
import sys
sys.path.insert(0, ".")
from common import ORIGINAL, EDITED, load, generate, spend

PROMPTS = [
    "Repeat this sentence exactly: The purple giraffe ate seventeen marshmallows on Tuesday.",
    "My friend's name is Zorvaldine Quimby. What is my friend's name?",
    "Here is a list: apple, kettle, violin, cactus. Write the list again in the same order.",
    "The code word is BLUEFINCH-42. What is the code word?",
    "What is the capital of France?",
    "What is 12 + 30?",
    "Translate to French: The cat is on the table.",
    "Name three primary colors.",
]
name = sys.argv[1]
spend(f"sanity {name}")
model, tok = load({"original": ORIGINAL, "edited": EDITED}[name])
for r in generate(model, tok, PROMPTS, max_new=80, batch=len(PROMPTS)):
    print(f"Q: {r['prompt']}\nA: {r['text'][:260]!r}\n")
