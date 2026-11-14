"""Can a classifier tell which model family wrote a text, and what erases that?

Reads the replies written by ea/fingerprint_generate.py (several small open-weight models
answering the same made-up everyday posts) and measures, against chance:

  1. accuracy by text length: 1 to 3 sentence replies, long replies, and long replies
     cut to their first N words;
  2. accuracy when several short replies from one source are pooled, as for an account;
  3. whether the fingerprint sits in the opening words (first N words against N words
     from the middle of a long reply) or in formatting quirks (raw against cleaned text);
  4. two evasions on short replies: a "write like a casual human" prompt, and rewriting
     every reply with one common model. Each is scored twice: for a classifier that has
     never seen the evasion, and for one trained on evaded text.

Refusals are dropped and formatting quirks are stripped before scoring (curly quotes,
wrapping quotes, markdown, line breaks), because anyone posting the text would lose them
anyway. Every score is on posts the classifier never saw in training (5 folds, split by post).
The classifier is a standard style baseline: character and word n-grams with a linear
support vector machine. Results are aggregate; no generated text is written to the report.
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from ea.fingerprint_generate import MODELS, OUT_DIR, PARAPHRASER

REPORT_DIR = Path("docs/redteam/fingerprint")
CUTS = [5, 10, 20, 40, 80, 160]
POOLS = [1, 3, 5, 10, 20]
FOLDS = 5
OPENING_WORDS = 8


REFUSAL = re.compile(r"can(no|')t (assist|help)|unable to (assist|help)|as an ai\b|i'?m sorry, but|i apologi[sz]e, but"
                     r"|language model|i don't have personal", re.I)
QUIRKS = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2014": "-", "\u2013": "-",
                        "\u2026": "...", "*": "", "#": "", "`": "", "_": " "})


def is_refusal(text):
    return bool(REFUSAL.search(text.replace("\u2019", "'")))


def normalize(text):
    """Strip what a copy-paste or a posting script would strip: typography, markdown, layout."""
    text = re.sub(r"\s+", " ", text.translate(QUIRKS)).strip()
    return text.strip('"').strip()


def load_texts(data_dir=OUT_DIR):
    """{mode: list of rows}. Empty texts and refusals are dropped; `text` is cleaned, `raw` is as written.
    Also returns how many replies per family and mode were refusals."""
    by_mode, refusals = {}, {}
    for path in sorted(Path(data_dir).glob("*.jsonl")):
        for row in map(json.loads, open(path)):
            if not row["text"].strip():
                continue
            if is_refusal(row["text"]):
                refusals.setdefault(row["family"], {}).setdefault(row["mode"], 0)
                refusals[row["family"]][row["mode"]] += 1
                continue
            row["raw"], row["text"] = row["text"], normalize(row["text"])
            by_mode.setdefault(row["mode"], []).append(row)
    # A rewrite only counts if the reply it rewrites was itself kept.
    kept = {(r["family"], r["post_id"]) for r in by_mode.get("short_plain", [])}
    if "short_paraphrased" in by_mode:
        by_mode["short_paraphrased"] = [r for r in by_mode["short_paraphrased"]
                                        if (r["family"], r["post_id"]) in kept]
    return by_mode, refusals


def classifier():
    return make_pipeline(
        make_union(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=False, sublinear_tf=True, min_df=2),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=False, sublinear_tf=True, min_df=2,
                            token_pattern=r"\w+|[^\w\s]"),
        ),
        LinearSVC(C=0.5),
    )


def first_words(text, n):
    return " ".join(text.split()[:n])


def middle_words(text, n):
    """N words starting a third of the way in, so the opening line is not included."""
    words = text.split()
    start = min(len(words) // 3, max(len(words) - n, 0))
    return " ".join(words[start:start + n])


def wilson(correct, total):
    """95% range for a share."""
    if total == 0:
        return [0.0, 0.0]
    p, z = correct / total, 1.96
    mid = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(mid - half), float(mid + half)]


def score(correct, total):
    low, high = wilson(correct, total)
    return {"accuracy": correct / total if total else 0.0, "low": low, "high": high, "n": int(total)}


def _select(rows, families, max_post):
    return [r for r in rows if r["family"] in families and r["post_id"] < max_post]


def cross_score(train_rows, test_rows, n_posts, transform=lambda t: t, want_predictions=False, field="text"):
    """Train on train_rows from four fifths of the posts, test on test_rows from the rest."""
    folds = GroupKFold(FOLDS).split(np.zeros(n_posts), groups=np.arange(n_posts))
    truth, predicted = [], []
    for train_posts, test_posts in folds:
        train_posts, test_posts = set(train_posts), set(test_posts)
        train = [r for r in train_rows if r["post_id"] in train_posts]
        test = [r for r in test_rows if r["post_id"] in test_posts]
        model = classifier().fit([transform(r[field]) for r in train], [r["family"] for r in train])
        truth += [r["family"] for r in test]
        predicted += list(model.predict([transform(r[field]) for r in test]))
    truth, predicted = np.array(truth), np.array(predicted)
    result = score(int((truth == predicted).sum()), len(truth))
    return (result, truth, predicted) if want_predictions else result


def pooled_score(rows, n_posts, pool, seed=0, draws=200):
    """Pool `pool` short replies by the same family (summed classifier scores), as for one account."""
    rng = np.random.default_rng(seed)
    correct = total = 0
    for train_posts, test_posts in GroupKFold(FOLDS).split(np.zeros(n_posts), groups=np.arange(n_posts)):
        train_posts, test_posts = set(train_posts), set(test_posts)
        train = [r for r in rows if r["post_id"] in train_posts]
        model = classifier().fit([r["text"] for r in train], [r["family"] for r in train])
        for family in model.classes_:
            texts = [r["text"] for r in rows if r["post_id"] in test_posts and r["family"] == family]
            if len(texts) < pool:
                continue
            log_p = model.decision_function(texts)
            for _ in range(draws // FOLDS):
                picked = rng.choice(len(texts), size=pool, replace=False)
                correct += model.classes_[log_p[picked].sum(axis=0).argmax()] == family
                total += 1
    return score(int(correct), total)


def run(families, by_mode, n_short, n_long, n_paraphrased):
    """All measurements for one set of families."""
    short = _select(by_mode["short_plain"], families, n_short)
    human = _select(by_mode["short_human"], families, n_short)
    long_ = _select(by_mode["long_plain"], families, n_long)
    para = _select(by_mode.get("short_paraphrased", []), families, n_paraphrased)
    words = lambda rows: float(np.median([len(r["text"].split()) for r in rows])) if rows else 0.0

    out = {"families": list(families), "chance": 1 / len(families),
           "median_words": {"short": words(short), "human_style": words(human), "long": words(long_),
                            "paraphrased": words(para)},
           "replies_per_family": {"short": n_short, "long": n_long, "paraphrased": n_paraphrased}}

    plain, truth, predicted = cross_score(short, short, n_short, want_predictions=True)
    out["short"] = plain
    out["long"] = cross_score(long_, long_, n_long)
    out["confusion"] = {a: {b: int(((truth == a) & (predicted == b)).sum()) for b in families} for a in families}
    out["per_family_short"] = {a: float((predicted[truth == a] == a).mean()) for a in families}

    out["short_without_opening"] = cross_score(short, short, n_short,
                                               transform=lambda t: " ".join(t.split()[OPENING_WORDS:]))
    out["raw_formatting"] = {"short": cross_score(short, short, n_short, field="raw"),
                             "long": cross_score(long_, long_, n_long, field="raw")}

    out["by_length"] = []
    for n in CUTS:
        first = cross_score(long_, long_, n_long, transform=lambda t, n=n: first_words(t, n))
        middle = cross_score(long_, long_, n_long, transform=lambda t, n=n: middle_words(t, n))
        out["by_length"].append({"words": n, **first, "middle": middle})

    out["pooled"] = [{"replies": k, **pooled_score(short, n_short, k)} for k in POOLS]

    evasion = {"none": plain,
               "human_style_unseen": cross_score(short, human, n_short),
               "human_style_trained": cross_score(human, human, n_short)}
    if para:
        unseen, truth_p, predicted_p = cross_score(short, para, n_short, want_predictions=True)
        evasion["paraphrased_unseen"] = unseen
        evasion["paraphrased_trained"] = cross_score(para, para, n_paraphrased)
        if PARAPHRASER in families:
            others = truth_p != PARAPHRASER
            out["paraphrased_called_paraphraser"] = float((predicted_p[others] == PARAPHRASER).mean())
    out["evasion"] = evasion
    return out


def draw_chart(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, pale, orange, aqua = "#2a78d6", "#9ec5f4", "#eb6834", "#1baf7a"
    chance = result["chance"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.15]},
                                      facecolor=surface)
    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=muted)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.axhline(chance, color=muted, linewidth=1.2, linestyle=(0, (4, 3)))

    curve = result["by_length"]
    xs = [c["words"] for c in curve]
    left.fill_between(xs, [c["low"] for c in curve], [c["high"] for c in curve], color=pale, alpha=0.5, linewidth=0)
    left.fill_between(xs, [c["middle"]["low"] for c in curve], [c["middle"]["high"] for c in curve],
                      color=aqua, alpha=0.18, linewidth=0)
    left.plot(xs, [c["accuracy"] for c in curve], color=blue, linewidth=2, marker="o", markersize=7,
              markeredgecolor=surface, markeredgewidth=1.5, label="First N words (includes the opening line)")
    left.plot(xs, [c["middle"]["accuracy"] for c in curve], color=aqua, linewidth=2, marker="s", markersize=7,
              markeredgecolor=surface, markeredgewidth=1.5, label="N words from the middle")
    for c in curve:
        up = c["accuracy"] >= c["middle"]["accuracy"]
        left.annotate(f"{c['accuracy']:.0%}", (c["words"], c["accuracy"]), textcoords="offset points",
                      xytext=(0, 10 if up else -17), ha="center", fontsize=10, color=ink)
        left.annotate(f"{c['middle']['accuracy']:.0%}", (c["words"], c["middle"]["accuracy"]),
                      textcoords="offset points", ha="center", fontsize=10, color=ink,
                      xytext=(0, -17 if up and c["middle"]["accuracy"] - chance > 0.12 else 10))
    short_x = result["median_words"]["short"]
    left.plot([short_x], [result["short"]["accuracy"]], color=orange, marker="D", markersize=9, linestyle="none",
              markeredgecolor=surface, markeredgewidth=1.5,
              label=f"Whole 1 to 3 sentence replies (median {short_x:.0f} words): {result['short']['accuracy']:.0%}")
    left.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=ink)
    left.set_xscale("log")
    left.set_xticks(xs, [str(x) for x in xs])
    left.minorticks_off()
    left.set_xlabel("Words of text the classifier sees (cut from long replies)", color=muted)
    left.text(xs[-1], chance - 0.055, f"chance {chance:.0%}", color=muted, fontsize=10, ha="right")
    left.set_title("The fingerprint sits in the opening line; mid-text it needs length", loc="left", fontsize=13, color=ink, pad=12)

    ev = result["evasion"]
    bars = [("No evasion", "none", blue), ("Human-style\nprompt, classifier\nnever saw it", "human_style_unseen", orange),
            ("Human-style\nprompt, classifier\ntrained on it", "human_style_trained", blue)]
    if "paraphrased_unseen" in ev:
        bars += [("Rewritten by one\nmodel, classifier\nnever saw it", "paraphrased_unseen", orange),
                 ("Rewritten by one\nmodel, classifier\ntrained on it", "paraphrased_trained", blue)]
    for i, (label, key, color) in enumerate(bars):
        r = ev[key]
        right.bar(i, r["accuracy"], width=0.58, color=color)
        right.errorbar([i], [r["accuracy"]], yerr=[[r["accuracy"] - r["low"]], [r["high"] - r["accuracy"]]],
                       color=ink, linewidth=1.2, capsize=3)
        right.text(i, r["high"] + 0.025, f"{r['accuracy']:.0%}", ha="center", fontsize=11, color=ink)
    right.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=9.5, color=ink)
    right.set_title("Short replies: what evasion does", loc="left", fontsize=13, color=ink, pad=12)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    right.legend(handles=[Patch(color=blue, label="Classifier trained on this kind of text"),
                          Patch(color=orange, label="Classifier trained on plain replies only"),
                          Line2D([], [], color=muted, linewidth=1.2, linestyle=(0, (4, 3)), label=f"Chance ({chance:.0%})")],
                 frameon=False, fontsize=9.5, loc="upper left", labelcolor=ink)

    fig.suptitle(f"Naming which of {len(result['families'])} model families wrote a text (share named correctly)",
                 x=0.012, ha="left", fontsize=14, color=ink)
    fig.text(0.012, 0.012, "Small open-weight models run locally on everyday topics; scored on posts unseen in training; "
             "bars and band show 95% ranges. Closed set: the writer is always one of the known models.",
             fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main(data_dir=OUT_DIR, report_dir=REPORT_DIR):
    by_mode, refusals = load_texts(data_dir)
    if not by_mode:
        raise SystemExit(f"No generated replies in {data_dir}. Run `python -m ea fingerprint-generate` first "
                         "(hours on CPU; see docs/redteam/fingerprint/RESULTS.md).")
    counts = {}
    for mode, rows in by_mode.items():
        for r in rows:
            counts.setdefault(r["family"], {}).setdefault(mode, 0)
            counts[r["family"]][mode] += 1
    present = [f for f in MODELS if f in counts]
    # Main run: families with the full set of replies. Second run: every family, on the
    # smaller number of posts that the slowest model answered.
    full_short = max(c.get("short_plain", 0) for c in counts.values())
    main_families = [f for f in present if counts[f].get("short_plain", 0) >= 0.9 * full_short]
    posts_in = lambda mode, fams: min(max(r["post_id"] for r in by_mode[mode] if r["family"] == f) + 1 for f in fams)
    n_para = posts_in("short_paraphrased", present) if "short_paraphrased" in by_mode else 0

    results = {"models": {f: MODELS[f][1] for f in present}, "texts_kept": counts, "refusals_dropped": refusals,
               "main": run(main_families, by_mode, posts_in("short_plain", main_families),
                           posts_in("long_plain", main_families), n_para)}
    if len(present) > len(main_families):
        results["all_families"] = run(present, by_mode, posts_in("short_plain", present),
                                      posts_in("long_plain", present), n_para)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(json.dumps(results, indent=1))
    draw_chart(results["main"], report_dir / "fingerprint.png")
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
