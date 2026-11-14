"""Stage 5, evaluate: write an aggregate report. No account appears in it.

Reads the held-out test predictions (later time window, unseen accounts).
"""

import json

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from ea.collect import load_collected
from ea.config import paths

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
        f"- Test accounts: {len(pred)} ({n_inauthentic} labeled inauthentic, {len(pred) - n_inauthentic} ordinary)",
        f"- Operating point: score >= {info['threshold']:.3f}, chosen on train data for precision >= {info['min_precision']}",
        f"- Precision: {precision:.2f} ({true_positive} of {n_flagged} flagged accounts were labeled inauthentic)",
        f"- Recall: {recall:.2f} ({true_positive} of {n_inauthentic} inauthentic accounts were flagged)",
        f"- PR-AUC: {average_precision_score(y, pred['score']):.2f}",
        f"- Brier score: {brier_score_loss(y, pred['score']):.3f} (lower is better calibrated)",
        "- Accounts below the operating point are uncertain, not cleared.",
        "",
        "## Share flagged, by labeled class",
        "",
        table(by_class(pred)),
        "",
        "## Share flagged, by subreddit",
        "",
        "A check that the model has not learned one community instead of a behavior.",
        "If labeled-ordinary accounts are flagged far more often in one subreddit, stop and look.",
        "",
        table(by_subreddit(pred, cfg)),
        "",
        "## Base-rate arithmetic",
        "",
        base_rate_note(precision, recall),
    ]
    p["evaluation"].parent.mkdir(parents=True, exist_ok=True)
    p["evaluation"].write_text("\n".join(lines) + "\n")
    return p["evaluation"]


def by_class(pred):
    out = pred.groupby("label")["flagged"].agg(accounts="size", share_flagged="mean").reset_index()
    return suppress_small(out)


def by_subreddit(pred, cfg):
    """Each test account counts once, in the subreddit where it commented most."""
    _, comments = load_collected(cfg)
    comments = comments[comments["author_hash"].isin(pred["author_hash"])]
    home = (comments.groupby("author_hash")["subreddit"]
            .agg(lambda s: s.mode().iloc[0]).rename("subreddit").reset_index())
    ordinary = pred[pred["target"] == 0].merge(home, on="author_hash")
    out = (ordinary.groupby("subreddit")["flagged"]
           .agg(ordinary_accounts="size", share_flagged="mean").reset_index())
    return suppress_small(out)


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
        return "Not computed: the test set has no accounts labeled inauthentic."
    wild_precision = 0.02 * recall / (0.02 * recall + 0.98 * 0.01)
    return (
        "The labeled sample oversamples suspicious accounts, so real-world precision is lower "
        f"than the {precision:.2f} above. Example: if 2% of real accounts are inauthentic and the "
        f"model wrongly flags 1% of ordinary accounts, then with recall {recall:.2f} only "
        f"{wild_precision:.0%} of flagged accounts would be inauthentic. Measure the false-flag "
        "rate on a random (not oversampled) labeled sample before reporting any prevalence."
    )


def table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + rows)
