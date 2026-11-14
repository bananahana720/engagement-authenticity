"""Stage 5, evaluate: write an aggregate report. No account appears in it.

Part 1 scores the "who is typing" model on held-out test predictions
(later time window, unseen accounts). Part 2 counts what the cluster
detector found. The two are separate questions and are reported separately.
"""

import json

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from ea.clusters import cluster_summary
from ea.collect import load_collected
from ea.config import paths
from ea.label import clean_labels, old_six_labels

SMALL_CELL = 10     # groups smaller than this are not reported


def evaluate(cfg):
    p = paths(cfg)
    pred = pd.read_csv(p["predictions"], dtype={"author_hash": str})
    info = json.loads(p["threshold"].read_text())
    y, flagged = pred["target"], pred["flagged"]

    true_positive = int((flagged & (y == 1)).sum())
    n_flagged, n_inauthentic = int(flagged.sum()), int(y.sum())
    precision = true_positive / n_flagged if n_flagged else float("nan")
    recall = true_positive / n_inauthentic if n_inauthentic else float("nan")

    lines = [
        f"# Evaluation: {cfg['name']}",
        "",
        "Held-out test: the later time window, accounts the model never saw. Aggregate numbers only.",
        "",
        f"- Test accounts: {len(pred)} ({n_inauthentic} labeled as software typing, {len(pred) - n_inauthentic} labeled as a person)",
        f"- Operating point: score >= {info['threshold']:.3f}, chosen on train data for precision >= {info['min_precision']}",
        f"- Precision: {precision:.2f} ({true_positive} of {n_flagged} flagged accounts were labeled as software typing)",
        f"- Recall: {recall:.2f} ({true_positive} of {n_inauthentic} software-typing accounts were flagged)",
        f"- PR-AUC: {average_precision_score(y, pred['score']):.2f}",
        f"- Brier score: {brier_score_loss(y, pred['score']):.3f} (lower is better calibrated)",
        "- Accounts below the operating point are uncertain, not cleared.",
        "",
        "## Share flagged, by \"who is typing\" answer",
        "",
        table(by_class(pred)),
        "",
        "## Share flagged, by subreddit",
        "",
        "A check that the model has not learned one community instead of a behavior.",
        "If accounts labeled as a person are flagged far more often in one subreddit, stop and look.",
        "",
        table(by_subreddit(pred, cfg)),
        "",
        "## Base-rate arithmetic",
        "",
        base_rate_note(precision, recall),
        "",
        *cluster_section(cfg),
    ]
    p["evaluation"].parent.mkdir(parents=True, exist_ok=True)
    p["evaluation"].write_text("\n".join(lines) + "\n")
    return p["evaluation"]


def by_class(pred):
    out = pred.groupby("who_types")["flagged"].agg(accounts="size", share_flagged="mean").reset_index()
    return suppress_small(out)


def by_subreddit(pred, cfg):
    """Each test account counts once, in the subreddit where it commented most."""
    _, comments = load_collected(cfg)
    comments = comments[comments["author_hash"].isin(pred["author_hash"])]
    home = (comments.groupby("author_hash")["subreddit"]
            .agg(lambda s: s.mode().iloc[0]).rename("subreddit").reset_index())
    people = pred[pred["target"] == 0].merge(home, on="author_hash")
    out = (people.groupby("subreddit")["flagged"]
           .agg(person_accounts="size", share_flagged="mean").reset_index())
    return suppress_small(out)


def cluster_section(cfg):
    """Counts from the cluster detector, in the later (test) time window."""
    p = paths(cfg)
    clusters = pd.read_csv(p["clusters"], dtype={"author_hash": str})
    clusters = clusters[clusters["window"] == "test"]
    _, comments = load_collected(cfg)
    counts = clusters["cluster"].value_counts().rename_axis("cluster").reset_index(name="accounts")

    labels = clean_labels(pd.read_csv(p["labels"], dtype={"author_hash": str}))
    labels = labels[labels["author_hash"].isin(clusters["author_hash"])]
    six = old_six_labels(labels, clusters.set_index("author_hash")["cluster"])
    six_counts = six.value_counts().sort_index().rename_axis("old_label").reset_index(name="accounts")
    six_counts["accounts"] = six_counts["accounts"].map(lambda n: n if n >= SMALL_CELL else f"fewer than {SMALL_CELL}")

    return [
        "## Coordinated clusters (set by the detector, not by labelers)",
        "",
        "Being in a cluster means acting together. It does not show intent, who is behind it, or that anything is false.",
        "",
        table(counts),
        "",
        "How far the clusters reach (breakout scale, as far as one platform can show):",
        "",
        table(cluster_summary(clusters, comments)),
        "",
        "## The old six labels, derived for charts",
        "",
        "A automated, B changed hands, C people in a cluster, D ordinary person, E declared, U uncertain.",
        "",
        table(six_counts),
    ]


def suppress_small(frame):
    """Hide the rate for any group too small to report safely."""
    count_column = frame.columns[1]
    frame = frame.copy()
    frame["share_flagged"] = frame["share_flagged"].round(2).astype(object)
    frame.loc[frame[count_column] < SMALL_CELL, "share_flagged"] = f"hidden (fewer than {SMALL_CELL})"
    return frame


def base_rate_note(precision, recall):
    """Precision measured on an oversampled label set overstates precision in the wild."""
    if recall != recall:        # nan: no inauthentic accounts in the test set
        return "Not computed: the test set has no accounts labeled as software typing."
    wild_precision = 0.02 * recall / (0.02 * recall + 0.98 * 0.01)
    return (
        "The labeled sample oversamples suspicious accounts, so real-world precision is lower "
        f"than the {precision:.2f} above. Example: if 2% of real accounts are run by software and the "
        f"model wrongly flags 1% of people, then with recall {recall:.2f} only "
        f"{wild_precision:.0%} of flagged accounts would be software. Measure the false-flag "
        "rate on a random (not oversampled) labeled sample before reporting any prevalence."
    )


def table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + rows)
