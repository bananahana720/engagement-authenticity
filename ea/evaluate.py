"""Stage 5, evaluate: write an aggregate report. No account appears in it.

Part 1 scores the "who is typing" model on held-out test predictions
(later time window, unseen accounts). Part 2 counts what the cluster
detector found. The two are separate questions and are reported separately.
"""

import json

import matplotlib
matplotlib.use("Agg")       # draw to files, no screen needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from ea import metrics
from ea.clusters import cluster_summary
from ea.collect import load_collected
from ea.config import paths
from ea.label import clean_labels, old_six_labels

SMALL_CELL = 10     # groups smaller than this are not reported


def evaluate(cfg):
    p = paths(cfg)
    p["evaluation"].parent.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(p["predictions"], dtype={"author_hash": str})
    info = json.loads(p["threshold"].read_text())
    y, score, flagged = pred["target"], pred["score"], pred["flagged"]
    n_software, n_people = int(y.sum()), int((y == 0).sum())

    # At the operating point the model actually uses.
    caught = int((flagged & (y == 1)).sum())
    false_flags = int((flagged & (y == 0)).sum())
    catch_rate = caught / n_software if n_software else float("nan")
    false_flag_rate = false_flags / n_people if n_people else float("nan")
    precision = caught / (caught + false_flags) if caught + false_flags else float("nan")

    crossover = metrics.crossover_error_rate(y, score)
    crossover_low, crossover_high = metrics.bootstrap_interval(metrics.crossover_error_rate, y, score)
    plot_error_tradeoff(y, score, crossover, p["error_curve"])
    plot_reliability(y, score, p["reliability"])

    lines = [
        f"# Evaluation: {cfg['name']}",
        "",
        "Held-out test: the later time window, accounts the model never saw. Aggregate numbers only.",
        f"Test accounts: {len(pred)} ({n_software} labeled as software typing, {n_people} labeled as a person).",
        f"Model trained on: {info['trained_on']} data. Scored here: {info['scored_data']} data.",
        "",
        "Two kinds of error. A **false flag** is a person called software. A **miss** is software passed as a person.",
        "",
        "## Headline, for comparing models: crossover error rate",
        "",
        f"**{crossover:.1%}** (95% range {crossover_low:.1%} to {crossover_high:.1%}). "
        "The error rate at the cut-off where false flags and misses are equally common. Lower is better.",
        "",
        f"![False flags against misses]({p['error_curve'].name})",
        "",
        "## Operating KPI: software caught at a fixed false-flag rate",
        "",
        "The project runs far from the crossover, where false flags are rare.",
        "",
        table(pd.DataFrame(fixed_rate_rows(y, score, n_people))),
        "",
        "## At the operating point in use",
        "",
        f"Score >= {info['threshold']:.3f}, chosen on train data for precision >= {info['min_precision']}.",
        "",
        f"- False-flag rate: {false_flag_rate:.1%} ({false_flags} of {n_people} people flagged)",
        f"- Miss rate: {1 - catch_rate:.1%} ({n_software - caught} of {n_software} software accounts not flagged)",
        (f"- Precision on this test set: {precision:.2f}" if caught + false_flags
         else "- Nothing was flagged, so precision cannot be computed")
        + f". PR-AUC: {average_precision_score(y, score):.2f}",
        "- Accounts below the operating point are uncertain, not cleared.",
        "",
        "## Precision at realistic prevalence",
        "",
        "The labeled sample oversamples suspicious accounts, so precision above is flattering. "
        "This table redoes it for the share of software accounts one might really meet. "
        "With zero false flags observed, it uses the largest false-flag rate the test set cannot rule out.",
        "",
        table(pd.DataFrame(prevalence_rows(catch_rate, false_flags, n_people))),
        "",
        "## Calibration",
        "",
        f"Brier score: {brier_score_loss(y, score):.3f} (0 is perfect). "
        "Well calibrated means the last two columns match.",
        "",
        table(pd.DataFrame(metrics.reliability_table(y, score))),
        "",
        f"![Reliability]({p['reliability'].name})",
        "",
        "## Share flagged, by \"who is typing\" answer",
        "",
        table(by_class(pred)),
        "",
        "## False-flag rate by subreddit",
        "",
        "A check that the model has not learned one community instead of a behavior.",
        "",
        table(by_subreddit(pred, cfg)),
        "",
        *group_section(cfg),
        *cluster_section(cfg),
    ]
    p["evaluation"].write_text("\n".join(lines) + "\n")
    return p["evaluation"]


def fixed_rate_rows(y, score, n_people):
    rows = []
    for max_rate in (0.01, 0.001):
        row = {"false-flag rate at most": f"{max_rate:.1%}"}
        if metrics.enough_people_for(max_rate, n_people):
            def metric(y, score, max_rate=max_rate):
                return metrics.recall_at_false_flag_rate(y, score, max_rate)
            low, high = metrics.bootstrap_interval(metric, y, score)
            row["software caught"] = f"{metric(y, score):.1%}"
            row["95% range"] = f"{low:.1%} to {high:.1%}"
        else:
            row["software caught"] = "not measurable"
            row["95% range"] = f"needs {round(1 / max_rate):,} labeled people in the test set, has {n_people}"
        rows.append(row)
    return rows


def prevalence_rows(catch_rate, false_flags, n_people):
    # "Rule of three": with 0 false flags among n people, the true rate could still be up to 3/n.
    false_flag_rate = false_flags / n_people if false_flags else 3 / n_people
    return [{"if this share of accounts is software": f"{prevalence:.0%}",
             "share of flagged accounts that really are": f"{metrics.precision_at_prevalence(catch_rate, false_flag_rate, prevalence):.0%}"}
            for prevalence in (0.01, 0.02, 0.05, 0.10)]


def group_section(cfg):
    """Share flagged in each named group of accounts (optional file labels/groups.csv).

    negative_control groups are real-person look-alikes: every flag there is a false flag.
    red_team groups are planted software: every account not flagged there is a miss.
    """
    p = paths(cfg)
    if not p["groups"].exists():
        return ["## Control groups", "",
                "No `labels/groups.csv` yet. See docs/METRICS.md for the format.", ""]
    groups = pd.read_csv(p["groups"], dtype={"author_hash": str})
    scores = pd.read_csv(p["scores"], dtype={"author_hash": str})
    merged = groups.merge(scores[~scores["used_in_training"]], on="author_hash")   # only accounts the model never saw
    out = (merged.groupby(["kind", "group"])["flagged"]
           .agg(accounts="size", share_flagged="mean").reset_index())
    out["share_flagged"] = out["share_flagged"].map("{:.1%}".format)
    out.loc[out["accounts"] < SMALL_CELL, "share_flagged"] = f"hidden (fewer than {SMALL_CELL})"
    return [
        "## Control groups",
        "",
        "For `negative_control` groups (people who look like software) the share flagged is a false-flag rate: lower is better. "
        "For `red_team` groups (planted software) it is the share caught: higher is better.",
        "",
        table(out),
        "",
    ]


def plot_error_tradeoff(y, score, crossover, path):
    """Every cut-off the model could use: false flags on one axis, misses on the other."""
    false_flag_rate, miss_rate, _ = metrics.error_rates(y, score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(false_flag_rate, miss_rate, color="#1f4e79", linewidth=2, label="model")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="false flags = misses")
    ax.scatter([crossover], [crossover], color="#c0392b", zorder=3, label=f"crossover {crossover:.1%}")
    ax.set_xlabel("False-flag rate (people called software)")
    ax.set_ylabel("Miss rate (software passed as people)")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_title("Closer to the bottom-left corner is better")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(y, score, path):
    rows = metrics.reliability_table(y, score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="perfectly calibrated")
    ax.plot([r["average_score"] for r in rows], [r["observed_share_software"] for r in rows],
            marker="o", color="#1f4e79", label="model")
    ax.set_xlabel("Average score in the band")
    ax.set_ylabel("Share that really are software")
    ax.set_title("Does a score of 0.8 mean 80%?")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + rows)
