"""Red team, part 2: which signals still find the puppets at each evasion level?

  python -m ea redteam            full run (about 10 minutes on 4 cores)
  python -m ea redteam --quick    small run to check the code

Method, in short. For every evasion level, build separate train worlds and test
worlds (different random seeds, so no account or network is shared). Fit a model
on one family of signals at a time, score the test worlds, and report recall at
90% precision: of the puppets, what share is found if at most 1 flag in 10 may
land on a real person. Precision first, because that is the costly error.

Results on made-up data show how signals rank against each other and where they
break. They do not say how many puppets real Reddit has, or how careful real
operators are.
"""

import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict

from ea import features
from ea.model import pick_threshold
from ea.network_features import network_features
from ea.redteam_generate import LEVELS, N_CONCEPTS, N_MACHINE, make_world, vocabulary

FAMILIES = {
    # Version 1 signals, already in the pipeline.
    "Sleep gap and reply speed": ["active_hours", "hour_entropy", "longest_quiet_hours",
                                  "median_reply_seconds", "fast_reply_share"],
    "Near-copy wording": ["near_duplicate_share"],
    "Same thread within minutes": ["co_engage_best_partner_threads", "co_engage_repeat_partners"],
    "Coded-phrase bursts": ["phrase_use_share", "phrase_burst_share"],
    # Cross-account signals, new here.
    "Same threads, any delay": ["same_thread"],
    "Same talking point, same day": ["same_points"],
    "Lockstep pivots": ["same_pivots"],
    "Same active days": ["same_days"],
    "Arrived in a batch": ["cohort_excess"],
    "Several leaks, same partners": ["same_points", "partners_cohort", "partners_same_pivots",
                                     "partners_same_days", "partners_same_thread", "partners_clique"],
    # Measured to show why the project does not rely on it. Never given to the model.
    "Sounds like AI (not in the model)": ["ai_style_share"],
}
VERSION_1 = sum(list(FAMILIES.values())[:4], [])
EVERYTHING = [c for c in dict.fromkeys(sum(FAMILIES.values(), [])) if c != "ai_style_share"] + ["point_focus"]
CONTROLS = {
    "ordinary": "Ordinary accounts",
    "junkie": "Fast news followers",
    "community": "Fan communities (shared favourite talking points)",
    "ai_polished": "People who polish their posts with AI",
    "translator": "Non-native speakers using a translator",
    "regular": "Power users and moderators (fixed hours, fast replies)",
    "crossposter": "People who paste their own comment in several threads",
    "brigade": "Organic brigades (real users arriving from one link)",
    "newcomer": "New accounts made during breaking news",
}
MIN_PRECISION = 0.90
LEXICON_COVERAGE = 0.85


# ---------- stand-in for "which talking point is this comment pushing?" ----------

def find_talking_points(comments, n_clusters=150, min_fit=0.4, seed=0):
    """Give each comment a talking-point cluster id, or -1.

    Stand-in for text embeddings plus clustering on real data. The detector gets a
    lexicon that maps 85% of the invented words to their concept (the rest are
    unknown to it), then clusters comments by concept. It is never told which
    talking points exist or who pushes them.
    """
    concept_words, _ = vocabulary()
    rng = np.random.default_rng(99)
    known = rng.random(concept_words.shape) < LEXICON_COVERAGE
    lexicon = {word: concept for concept, row in enumerate(concept_words)
               for word, ok in zip(row, known[concept]) if ok}

    rows, cols = [], []
    for i, body in enumerate(comments["body"]):
        for word in body.split():
            concept = lexicon.get(word)
            if concept is not None:
                rows.append(i)
                cols.append(concept)
    bag = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                            shape=(len(comments), N_CONCEPTS))
    vectors = TfidfTransformer().fit_transform(bag)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3, batch_size=4096).fit(vectors)
    centres = kmeans.cluster_centers_
    centres = centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-9)
    fit = np.asarray(vectors.multiply(centres[kmeans.labels_]).sum(axis=1)).ravel()
    return np.where(fit >= min_fit, kmeans.labels_, -1)


def ai_style_share(comments):
    """Per account: share of its words that are machine register.

    Stand-in for an AI-text detector, and a generous one: it knows the exact word list.
    """
    _, fillers = vocabulary()
    machine = set(fillers[:N_MACHINE])
    words = comments["body"].str.split()
    hits = words.map(lambda ws: sum(w in machine for w in ws))
    return hits.groupby(comments["author_hash"]).sum() / words.str.len().groupby(comments["author_hash"]).sum()


# ---------- one world -> one feature table ----------

def world_table(job):
    cfg, level, seed, options = job
    world = make_world(cfg, level, seed, **options)
    comments = world.comments[~world.comments["disclosed_bot"]]     # set aside by rule, as in the pipeline
    comments = features.add_reply_latency(comments, world.posts)
    version_1 = features.features_for_window(comments, cfg).set_index("author_hash")
    comments = comments.assign(point=find_talking_points(comments))
    table = version_1.join(network_features(comments, version_1.index))
    table["ai_style_share"] = ai_style_share(comments)
    table = table.join(world.accounts)
    campaign = world.truth["campaign"].groupby(world.comments["author_hash"]).sum()
    table["campaign_comments"] = campaign.reindex(table.index).fillna(0).astype(int)
    return table.assign(level=level, seed=seed, **options).reset_index()


def build_tables(cfg, jobs, cache):
    """Run the jobs on all cores. Finished worlds are kept in `cache` and not rebuilt."""
    cache.mkdir(parents=True, exist_ok=True)

    def file_of(job):
        _, level, seed, options = job
        extra = "".join(f"_{k}{v}" for k, v in sorted(options.items()))
        return cache / f"{level}_seed{seed}{extra}.csv"

    todo = [job for job in jobs if not file_of(job).exists()]
    if todo:
        # "spawn": a forked worker can hang if the parent has already trained a model.
        with get_context("spawn").Pool() as pool:
            for job, table in zip(todo, pool.imap(world_table, todo)):
                table.to_csv(file_of(job), index=False)
                print(f"  built {file_of(job).name}: {len(table)} accounts, {int(table['target'].sum())} puppets")
    return pd.concat([pd.read_csv(file_of(job)) for job in jobs], ignore_index=True)


# ---------- scoring ----------

def new_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, random_state=0)


def recall_at_precision(y, scores, min_precision=MIN_PRECISION):
    """Best recall among cut-offs whose precision is at least `min_precision`. 0 if none."""
    precision, recall, _ = precision_recall_curve(y, scores)
    ok = precision[:-1] >= min_precision
    return float(recall[:-1][ok].max()) if ok.any() else 0.0


def score_per_world(model, test, columns):
    """Mean over test worlds of recall at 90% precision, and of PR-AUC."""
    recalls, aucs = [], []
    for _, world in test.groupby("seed"):
        scores = model.predict_proba(world[columns])[:, 1]
        recalls.append(recall_at_precision(world["target"], scores))
        aucs.append(average_precision_score(world["target"], scores))
    return float(np.mean(recalls)), float(np.mean(aucs))


def run(cfg, quick=False):
    out = cfg["reports_dir"].parent / "redteam"
    cache = cfg["data_dir"].parent / ("redteam_quick" if quick else "redteam")
    size = {"n_ordinary": 500, "n_threads": 500} if quick else {}
    train_seeds, test_seeds = ([0], [10]) if quick else ([0, 1], [10, 11, 12])
    level_keys = [level["key"] for level in LEVELS]
    sizes, shares = ([5, 20], [0.1, 0.3]) if quick else ([5, 10, 20, 40], [0.1, 0.2, 0.3, 0.5])

    def frontier_jobs(seeds):
        return [(cfg, "sparse", seed, dict(size, network_size=s, n_networks=80 // s, campaign_share=share))
                for s in sizes for share in shares for seed in seeds]

    print("Building worlds for the evasion ladder")
    jobs = [(cfg, level, seed, size) for level in level_keys for seed in train_seeds + test_seeds]
    ladder = build_tables(cfg, jobs, cache)
    print("Building worlds for the frontier (network size x campaign share)")
    f_train = build_tables(cfg, frontier_jobs(train_seeds[:1]), cache)
    f_test = build_tables(cfg, frontier_jobs(test_seeds[:2]), cache)
    train = ladder[ladder["seed"].isin(train_seeds)]
    test = ladder[ladder["seed"].isin(test_seeds)].reset_index(drop=True)
    n_worlds = len(test_seeds) * len(level_keys)

    results = {"levels": LEVELS, "min_precision": MIN_PRECISION, "quick": quick,
               "accounts_per_world": round(len(test) / n_worlds),
               "puppets_per_world": round(test["target"].sum() / n_worlds),
               "campaign_comments_per_world": [
                   round(test.loc[test["level"] == level, "campaign_comments"].sum() / len(test_seeds))
                   for level in level_keys],
               "families": [], "models": [], "controls": [], "frontier": []}

    # 1. One signal family at a time, trained and tested at the same level (its best case).
    for name, columns in FAMILIES.items():
        row = {"name": name, "new": columns[0] not in VERSION_1, "recall": [], "pr_auc": []}
        for level in level_keys:
            model = new_model().fit(train.loc[train["level"] == level, columns],
                                    train.loc[train["level"] == level, "target"])
            recall, auc = score_per_world(model, test[test["level"] == level], columns)
            row["recall"].append(recall)
            row["pr_auc"].append(auc)
        results["families"].append(row)
        print(f"  {name:34s}", " ".join(f"{r:5.2f}" for r in row["recall"]))

    # 2. Whole models that do not know the level: trained on all levels mixed.
    #    Reported two ways: best recall at 90% precision on the test worlds, and the honest
    #    operating point (cut-off chosen on train worlds only, then applied to test worlds).
    flagged_by = {}
    for name, columns in (("Version 1 signals only", VERSION_1),
                          ("Version 1 + cross-account signals", EVERYTHING)):
        model = new_model().fit(train[columns], train["target"])
        out_of_fold = cross_val_predict(new_model(), train[columns], train["target"], cv=3,
                                        method="predict_proba")[:, 1]
        threshold = pick_threshold(train["target"], out_of_fold, MIN_PRECISION)
        flagged = pd.Series(model.predict_proba(test[columns])[:, 1] >= threshold, index=test.index)
        flagged_by[name] = flagged
        row = {"name": name, "recall": [], "pr_auc": [], "applied_precision": [], "applied_recall": []}
        for level in level_keys:
            at = test["level"] == level
            recall, auc = score_per_world(model, test[at], columns)
            row["recall"].append(recall)
            row["pr_auc"].append(auc)
            row["applied_precision"].append(float(test.loc[at & flagged, "target"].mean())
                                            if (at & flagged).any() else None)
            row["applied_recall"].append(float(flagged[at & (test["target"] == 1)].mean()))
        results["models"].append(row)
        print(f"  {name:34s}", " ".join(f"{r:5.2f}" for r in row["recall"]))

    # 3. Negative controls: of each kind of real account, what share got flagged?
    #    Pooled over all levels, since the background is built the same way at each.
    for kind, label in CONTROLS.items():
        is_kind = test["kind"] == kind
        results["controls"].append({
            "kind": kind, "label": label, "accounts": int(is_kind.sum()),
            "flag_rate": {name: float(flagged[is_kind].mean()) for name, flagged in flagged_by.items()},
            "flagged": {name: int(flagged[is_kind].sum()) for name, flagged in flagged_by.items()}})
    results["disclosed_bots"] = {"per_world": 3, "scored": int((ladder["kind"] == "disclosed").sum())}

    # 4. "Sounds like AI" on its own: among the accounts it ranks highest, who are they?
    results["ai_style"] = []
    for level in level_keys:
        at = test[test["level"] == level]
        top = at.nlargest(int(at["target"].sum()), "ai_style_share")      # as many flags as there are puppets
        kinds = top["kind"].value_counts(normalize=True)
        results["ai_style"].append({"level": level, "puppets": float(kinds.get("puppet", 0)),
                                    "ai_polished": float(kinds.get("ai_polished", 0)),
                                    "translator": float(kinds.get("translator", 0)),
                                    "other": float(kinds.drop(["puppet", "ai_polished", "translator"],
                                                              errors="ignore").sum())})

    # 5. Frontier at the "sparse" level: how small and how quiet can a network be?
    model = new_model().fit(f_train[EVERYTHING], f_train["target"])
    for s in sizes:
        for share in shares:
            cell = f_test[(f_test["network_size"] == s) & (f_test["campaign_share"] == share)]
            recall, auc = score_per_world(model, cell, EVERYTHING)
            results["frontier"].append({"network_size": s, "campaign_share": share,
                                        "recall": recall, "pr_auc": auc})

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "results.md").write_text(as_markdown(results))
    draw_chart(results, out / "degradation.png")
    return out


# ---------- report ----------

def as_markdown(results):
    names = [level["name"] for level in results["levels"]]
    lines = [
        "# Red team: which signals survive smarter sock puppets?",
        "",
        "Made-up data. Each cell is recall at 90% precision: the share of puppet accounts found",
        "when at most 1 flag in 10 may land on a real person. Mean over test worlds of about",
        f"{results['accounts_per_world']} accounts, {results['puppets_per_world']} of them puppets.",
        "Each column keeps the evasions of the columns to its left.",
        "",
        "## One signal family at a time (trained at the same level: its best case)",
        "",
        "| Signal family | " + " | ".join(names) + " |",
        "| --- |" + " ---: |" * len(names),
    ]
    for row in results["families"]:
        tag = " (new)" if row["new"] else ""
        lines.append(f"| {row['name']}{tag} | " + " | ".join(f"{r:.0%}" for r in row["recall"]) + " |")
    lines += ["", "## Whole models trained on all levels mixed", "",
              "| Model | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    for row in results["models"]:
        lines.append(f"| {row['name']} | " + " | ".join(f"{r:.0%}" for r in row["recall"]) + " |")
    lines.append("| Campaign comments per world | " +
                 " | ".join(str(n) for n in results["campaign_comments_per_world"]) + " |")
    lines += ["", "## The honest operating point", "",
              "Cut-off chosen on train worlds only, then applied to unseen test worlds.",
              "Each cell: precision / recall.", "",
              "| Model | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    for row in results["models"]:
        cells = ["no flags" if p is None else f"{p:.0%} / {r:.0%}"
                 for p, r in zip(row["applied_precision"], row["applied_recall"])]
        lines.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    model_names = [row["name"] for row in results["models"]]
    lines += ["", "## Negative controls: real accounts that must not be flagged", "",
              "Share of each kind flagged at the honest operating point, pooled over all levels.", "",
              "| Kind of real account | Accounts tested | " + " | ".join(model_names) + " |",
              "| --- | ---: |" + " ---: |" * len(model_names)]
    for row in results["controls"]:
        cells = [f"{row['flag_rate'][m]:.2%} ({row['flagged'][m]})" for m in model_names]
        lines.append(f"| {row['label']} | {row['accounts']} | " + " | ".join(cells) + " |")
    bots = results["disclosed_bots"]
    lines += ["", f"Disclosed bots: {bots['per_world']} per world, set aside by rule before scoring. "
                  f"Scored: {bots['scored']}.",
              "", "## Why \"sounds like AI\" is not in the model", "",
              "Flag the accounts with the most machine-register wording, as many as there are puppets.",
              "Who gets flagged?", "",
              "| Level | Puppets | People polishing with AI | Translator users | Other real people |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row, name in zip(results["ai_style"], names):
        lines.append(f"| {name} | {row['puppets']:.0%} | {row['ai_polished']:.0%} | "
                     f"{row['translator']:.0%} | {row['other']:.0%} |")
    sizes = sorted({c["network_size"] for c in results["frontier"]})
    shares = sorted({c["campaign_share"] for c in results["frontier"]})
    cell = {(c["network_size"], c["campaign_share"]): c["recall"] for c in results["frontier"]}
    lines += ["", "## Frontier at the sparse level: network size x share of comments that are campaign", "",
              "| Accounts per network | " + " | ".join(f"{share:.0%} campaign" for share in shares) + " |",
              "| --- |" + " ---: |" * len(shares)]
    for s in sizes:
        lines.append(f"| {s} | " + " | ".join(f"{cell[(s, share)]:.0%}" for share in shares) + " |")
    return "\n".join(lines) + "\n"


def draw_chart(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [level["name"].replace("+ ", "+\n") for level in results["levels"]]
    rows = results["families"] + results["models"]
    grid = np.array([row["recall"] for row in rows])
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [2.2, 1]})

    left.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    left.set_xticks(range(len(names)), names)
    left.set_yticks(range(len(rows)), [row["name"] for row in rows])
    left.xaxis.tick_top()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            left.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center",
                      color="white" if grid[i, j] > 0.55 else "#1b2a41", fontsize=11)
    left.axhline(3.5, color="white", linewidth=4)
    left.axhline(len(results["families"]) - 1.5, color="white", linewidth=4)
    left.axhline(len(results["families"]) - 0.5, color="white", linewidth=4)
    left.set_title("Puppets found at 90% precision, by signal and evasion level\n"
                   "(top: version 1 signals, then cross-account signals, AI-style text, whole models)",
                   loc="left", fontsize=11, pad=14)
    left.tick_params(length=0)
    for side in left.spines.values():
        side.set_visible(False)

    sizes = sorted({c["network_size"] for c in results["frontier"]})
    shares = sorted({c["campaign_share"] for c in results["frontier"]})
    cell = {(c["network_size"], c["campaign_share"]): c["recall"] for c in results["frontier"]}
    frontier = np.array([[cell[(s, share)] for share in shares] for s in sizes])
    right.imshow(frontier, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    right.set_xticks(range(len(shares)), [f"{share:.0%}" for share in shares])
    right.set_yticks(range(len(sizes)), sizes)
    right.set_xlabel("Share of each puppet's comments that push the campaign")
    right.set_ylabel("Accounts per network")
    for i in range(frontier.shape[0]):
        for j in range(frontier.shape[1]):
            right.text(j, i, f"{frontier[i, j]:.0%}", ha="center", va="center",
                       color="white" if frontier[i, j] > 0.55 else "#1b2a41", fontsize=11)
    right.set_title("Sparse level: how small and quiet\ncan a network be?", loc="left", fontsize=11)
    right.tick_params(length=0)
    for side in right.spines.values():
        side.set_visible(False)
    fig.text(0.01, 0.01, "Made-up data. Shows which signals break first, not how common puppets are.",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
