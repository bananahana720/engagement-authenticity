"""Generate forum-style replies from several small open-weight models, run locally.

Every model answers the same made-up posts on everyday, non-political topics, so a
classifier cannot tell models apart by subject. The output is one JSONL file per model
under data/fingerprint/, which is git-ignored: generated text stays on this machine and
is never posted anywhere.

Needs: torch, transformers, and (only for the GGUF model) llama-cpp-python.
"""

import json
import random
import re
import time
from pathlib import Path

OUT_DIR = Path("data/fingerprint")

# family -> (how to run it, Hugging Face repository, file inside the repository)
MODELS = {
    "llama": ("transformers", "unsloth/Llama-3.2-1B-Instruct", None),
    "qwen": ("transformers", "Qwen/Qwen2.5-1.5B-Instruct", None),
    "gemma": ("transformers", "unsloth/gemma-3-1b-it", None),
    "deepseek": ("transformers", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", None),
    "smollm": ("transformers", "HuggingFaceTB/SmolLM2-1.7B-Instruct", None),
    "falcon": ("transformers", "tiiuae/Falcon3-1B-Instruct", None),
    "mistral": ("gguf", "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
                "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
}
PARAPHRASER = "qwen"
# The DeepSeek distill refuses most posts when its reasoning step is skipped, so it keeps it.
THINKING = {"deepseek"}
THINKING_TOKENS = 450
# Slow models (reasoning tokens, or a 3B model on CPU) answer fewer posts: (short, long).
REDUCED = {"deepseek": (120, 50), "mistral": (120, 50)}

TOPICS = [
    "cast iron pans", "sourdough bread", "air fryers", "meal prepping", "pour-over coffee",
    "rice cookers", "home composting", "tomato gardening", "houseplants", "indoor herbs",
    "running shoes", "couch-to-5k programs", "home gym equipment", "yoga for beginners",
    "bike commuting", "e-bikes", "standing desks", "mechanical keyboards", "noise-cancelling headphones",
    "budget laptops", "used cars", "electric cars", "learning to drive stick", "road trips",
    "solo travel", "carry-on only packing", "camping in the rain", "hiking boots", "learning guitar",
    "digital pianos", "vinyl records", "board game nights", "chess openings", "jigsaw puzzles",
    "learning Spanish", "language apps", "reading more books", "audiobooks", "public libraries",
    "adopting a rescue dog", "indoor cats", "aquariums", "bird feeders", "knitting",
    "woodworking", "3D printing", "film photography", "phone cameras", "thrift shopping",
    "minimalist wardrobes",
]
QUESTIONS = [
    "Is getting into {t} actually worth it, or is it overhyped?",
    "What is the one thing you wish you knew before starting with {t}?",
    "I tried {t} for a month and I am about to give up. What am I doing wrong?",
    "Unpopular opinion: {t} is better on a small budget than an expensive one. Change my mind.",
    "What is the most common beginner mistake with {t}?",
    "My partner thinks {t} is a waste of time and money. How do I explain why I like it?",
]

SHORT = "Write a reply to this post. Keep it to 1 to 3 sentences."
PROMPTS = {
    "short_plain": SHORT,
    "short_human": (
        "You are a regular person replying on Reddit from your phone, not an assistant. "
        "Write the way a real casual commenter does: informal, blunt, contractions, lowercase is fine, "
        "maybe a small typo. No greeting, no list, no emoji, no hedging, no advice-column tone. "
        + SHORT),
    "long_plain": "Write a detailed reply to this post, about 150 to 200 words.",
}
MAX_NEW_TOKENS = {"short_plain": 90, "short_human": 90, "long_plain": 320, "short_paraphrased": 110}
TEMPERATURE, TOP_P = 0.8, 0.95


def posts():
    """300 made-up posts in a fixed shuffled order, so every model sees the same ones."""
    all_posts = [q.format(t=t) for t in TOPICS for q in QUESTIONS]
    random.Random(7).shuffle(all_posts)
    return all_posts


def user_message(mode, post):
    return f'Here is a post from an online forum:\n\n"{post}"\n\n{PROMPTS[mode]}'


def paraphrase_message(text):
    return ("Rewrite the following forum comment in your own words. Keep the meaning and roughly the "
            "same length. Output only the rewritten comment.\n\n" + text)


def clean(text, finished):
    """Drop any reasoning block; if the model ran out of tokens, cut at the last full sentence."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.rpartition("</think>")[2].strip()
    if not finished:
        ends = [m.end() for m in re.finditer(r"[.!?](\s|$)", text)]
        if ends:
            text = text[:ends[-1]].strip()
    return text


class TransformersModel:
    def __init__(self, repo, think=False):
        import torch
        self.think = think
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(4)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # bfloat16 is about three times faster than float32 on CPUs with AMX.
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).eval()

    def generate(self, messages, max_new_tokens, batch_size=32):
        prompts = []
        for message in messages:
            p = self.tok.apply_chat_template([{"role": "user", "content": message}],
                                             tokenize=False, add_generation_prompt=True)
            if p.rstrip().endswith("<think>") and not self.think:  # skip the reasoning block
                p = p.rstrip() + "\n\n</think>\n\n"
            prompts.append(p)
        if self.think:
            max_new_tokens += THINKING_TOKENS
        results = []
        for start in range(0, len(prompts), batch_size):
            enc = self.tok(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                           add_special_tokens=False)
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                          temperature=TEMPERATURE, top_p=TOP_P, top_k=0,
                                          repetition_penalty=1.0, pad_token_id=self.tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for row in new:
                finished = bool((row == self.tok.pad_token_id).any() or (row == self.tok.eos_token_id).any()
                                or len(row) < max_new_tokens)
                text = self.tok.decode(row, skip_special_tokens=True)
                if self.think and "</think>" not in text:
                    text = ""  # still reasoning when the tokens ran out: no reply
                results.append(clean(text, finished))
        return results


class GgufModel:
    def __init__(self, repo, filename):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        self.model = Llama(hf_hub_download(repo, filename), n_ctx=1024, n_threads=4, verbose=False)

    def generate(self, messages, max_new_tokens, batch_size=None):
        results = []
        for message in messages:
            r = self.model.create_chat_completion([{"role": "user", "content": message}],
                                                  max_tokens=max_new_tokens, temperature=TEMPERATURE,
                                                  top_p=TOP_P, top_k=0, repeat_penalty=1.0)
            choice = r["choices"][0]
            results.append(clean(choice["message"]["content"] or "", choice["finish_reason"] == "stop"))
        return results


def load(family):
    kind, repo, filename = MODELS[family]
    if kind == "transformers":
        return TransformersModel(repo, think=family in THINKING)
    return GgufModel(repo, filename)


def _append(path, rows):
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _done(path):
    if not path.exists():
        return set()
    return {(r["mode"], r["post_id"], r.get("source", "")) for r in map(json.loads, open(path))}


def generate_family(family, n_short=300, n_long=150, out_dir=OUT_DIR):
    """Write this family's short, human-style and long replies. Safe to rerun: skips what exists."""
    n_short, n_long = REDUCED.get(family, (n_short, n_long))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{family}.jsonl"
    done, model, all_posts = _done(path), load(family), posts()
    for mode, n in [("short_plain", n_short), ("short_human", n_short), ("long_plain", n_long)]:
        todo = [i for i in range(min(n, len(all_posts))) if (mode, i, "") not in done]
        for start in range(0, len(todo), 32):
            ids, began = todo[start:start + 32], time.time()
            texts = model.generate([user_message(mode, all_posts[i]) for i in ids], MAX_NEW_TOKENS[mode])
            _append(path, [{"family": family, "mode": mode, "post_id": i, "text": t}
                           for i, t in zip(ids, texts)])
            print(f"{family} {mode} {start + len(ids)}/{len(todo)} ({time.time() - began:.0f}s)", flush=True)


def paraphrase_all(families, n=150, out_dir=OUT_DIR):
    """Rewrite each family's first n short replies with one common model (the paraphraser)."""
    path = out_dir / "paraphrased.jsonl"
    done, model = _done(path), load(PARAPHRASER)
    for family in families:
        rows = [r for r in map(json.loads, open(out_dir / f"{family}.jsonl"))
                if r["mode"] == "short_plain" and r["post_id"] < n and r["text"]
                and ("short_paraphrased", r["post_id"], family) not in done]
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            texts = model.generate([paraphrase_message(r["text"]) for r in batch],
                                   MAX_NEW_TOKENS["short_paraphrased"])
            _append(path, [{"family": r["family"], "source": r["family"], "mode": "short_paraphrased",
                            "post_id": r["post_id"], "text": t} for r, t in zip(batch, texts)])
            print(f"paraphrase {family} {start + len(batch)}/{len(rows)}", flush=True)


def main(families=None, n_short=300, n_long=150, n_paraphrase=120):
    families = families or list(MODELS)
    for family in families:
        generate_family(family, n_short, n_long)
    paraphrase_all(families, n_paraphrase)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
