"""Stage 2, label: draw accounts for people to label, and check that labelers agree.

Labels (from the plan):
  A automated          B hijacked or rented    C human, incentive-driven
  D ordinary human     E disclosed automation  U uncertain

Label the account, not one comment. Every A, B or C needs one written sentence
of evidence; without it the label becomes U.
"""

import numpy as np
import pandas as pd

from ea.collect import load_collected
from ea.config import paths

LABELS = ["A", "B", "C", "D", "E", "U"]
NEEDS_EVIDENCE = {"A", "B", "C"}

# An account is "suspicious" for sampling when it is in the top 5% on any of these.
# This only decides who gets looked at first. It is not a label.
SUSPICION_SIGNALS = ["active_hours", "fast_reply_share", "near_duplicate_share",
                     "co_engage_repeat_partners", "phrase_burst_share"]


def draw_sample(cfg, n_accounts=200, examples_per_account=10, seed=0):
    """Write a labeling sheet: half suspicious-looking accounts, half everyone else.

    Oversampling the suspicious half gives the rare classes enough examples.
    The sampling weight records how many real accounts each sampled one stands
    for, so prevalence can be weighted back to the true population later.
    """
    features = pd.read_csv(paths(cfg)["features"])
    features = features[features["window"] == "train"]

    top = features[SUSPICION_SIGNALS].quantile(0.95)
    suspicious = (features[SUSPICION_SIGNALS] > top).any(axis=1)
    features = features.assign(stratum=np.where(suspicious, "suspicious", "other"))

    drawn = []
    for _, group in features.groupby("stratum"):
        take = min(len(group), n_accounts // 2)
        drawn.append(group.sample(take, random_state=seed).assign(sampling_weight=len(group) / take))
    sheet = pd.concat(drawn).sample(frac=1, random_state=seed)   # shuffle, so order gives no hint
    sheet = sheet.assign(label="", confidence="", evidence="")

    posts, comments = load_collected(cfg)
    examples = (comments[comments["author_hash"].isin(sheet["author_hash"])]
                .sort_values("time")
                .groupby("author_hash").head(examples_per_account)
                [["author_hash", "time", "subreddit", "link_id", "body"]])

    out = paths(cfg)["to_label"]
    out.parent.mkdir(parents=True, exist_ok=True)
    # "stratum" stays out of the sheet so the labeler is not primed by it.
    sheet.drop(columns=["stratum", "window"]).to_csv(out, index=False)
    sheet[["author_hash", "stratum", "sampling_weight"]].to_csv(
        out.with_name("sample_strata.csv"), index=False)
    examples.to_csv(paths(cfg)["label_examples"], index=False)
    return sheet


def clean_labels(labels):
    """Apply the evidence rule and reject unknown labels."""
    labels = labels.copy()
    labels["label"] = labels["label"].astype(str).str.strip().str.upper()
    unknown = set(labels["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}. Use one of {LABELS}.")
    no_evidence = labels["evidence"].fillna("").astype(str).str.strip() == ""
    labels.loc[labels["label"].isin(NEEDS_EVIDENCE) & no_evidence, "label"] = "U"
    return labels


def cohen_kappa(first, second):
    """Agreement between two labelers beyond what chance would give. 1 = perfect, 0 = chance."""
    first, second = np.asarray(first), np.asarray(second)
    observed = (first == second).mean()
    expected = sum((first == c).mean() * (second == c).mean() for c in set(first) | set(second))
    return 1.0 if expected == 1 else float((observed - expected) / (1 - expected))


def agreement(file_a, file_b):
    """Kappa overall and per class for two labelers' sheets of the same accounts.

    Target from the plan: 0.6 or above before labeling more.
    """
    a = clean_labels(pd.read_csv(file_a)).set_index("author_hash")["label"]
    b = clean_labels(pd.read_csv(file_b)).set_index("author_hash")["label"]
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()

    rows = [{"class": "all", "kappa": cohen_kappa(both["a"], both["b"]), "n": len(both)}]
    for label in LABELS:
        n = int(((both["a"] == label) | (both["b"] == label)).sum())
        if n:
            rows.append({"class": label, "kappa": cohen_kappa(both["a"] == label, both["b"] == label), "n": n})
    return pd.DataFrame(rows)
