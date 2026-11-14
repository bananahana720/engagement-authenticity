"""Stage 2, label: draw accounts for people to label, and check that labelers agree.

Three separate questions replace the old single six-way label (see docs/LABELS.md):

  who_types      asked of each account, answered by a person
  changed_hands  asked of each account, answered by a person
  cluster        asked of each group, answered by the detector (ea/clusters.py)

Nobody labels motive. Why an account posts cannot be seen from outside.
"""

import numpy as np
import pandas as pd

from ea.collect import load_collected
from ea.config import paths

# Question 1: who is typing?
WHO_TYPES = {
    "person": "A person writes and posts each comment.",
    "person_plus_automation": "A person is involved, with software doing part of the work "
                              "(scheduling, drafting, posting from a queue).",
    "program": "Software writes and posts, with no person involved in each comment.",
    "declared": "Says what it is: a declared bot, or an organization such as a news desk or brand.",
    "uncertain": "Evidence conflicts or is too thin. A real answer, not a failure.",
}

# Question 2: did the account change hands?
CHANGED_HANDS = {
    "no_sign": "Nothing suggests the account changed owner.",
    "likely": "Long silence, then a different persona: likely taken over or sold.",
    "uncertain": "Some signs, not enough to say.",
}

# Question 3: is the account in a coordinated cluster? Set by the detector, never by a labeler.
CLUSTER = {
    "in_cluster": "The detector found it in a group that acts together.",
    "none_found": "Tested, and no such group was found. Not proof there is none.",
    "not_tested": "Too little activity to test.",
}

QUESTIONS = {"who_types": WHO_TYPES, "changed_hands": CHANGED_HANDS}

# These answers need one written sentence of evidence. Without it they become "uncertain".
NEEDS_EVIDENCE = {"who_types": {"person_plus_automation", "program"}, "changed_hands": {"likely"}}

# An account is "suspicious" for sampling when it is in the top 5% on any of these.
# This only decides who gets looked at first. It is not a label.
SUSPICION_SIGNALS = ["active_hours", "fast_reply_share", "near_duplicate_share",
                     "co_engage_repeat_partners", "phrase_burst_share"]


def draw_sample(cfg, n_accounts=200, examples_per_account=10, seed=0):
    """Write a labeling sheet: half suspicious-looking accounts, half everyone else.

    Oversampling the suspicious half gives the rare answers enough examples.
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
    sheet = sheet.assign(who_types="", changed_hands="", confidence="", evidence="")

    _, comments = load_collected(cfg)
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
    """Reject unknown answers and apply the evidence rule to each question."""
    labels = labels.copy()
    no_evidence = labels["evidence"].fillna("").astype(str).str.strip() == ""
    for question, answers in QUESTIONS.items():
        labels[question] = labels[question].astype(str).str.strip().str.lower()
        unknown = set(labels[question]) - set(answers)
        if unknown:
            raise ValueError(f"Unknown {question} answers: {sorted(unknown)}. Use one of {list(answers)}.")
        labels.loc[labels[question].isin(NEEDS_EVIDENCE[question]) & no_evidence, question] = "uncertain"
    return labels


def old_six_labels(labels, cluster):
    """The earlier A to F style labels, derived for charts that compare with other work.

    labels: cleaned sheet with who_types and changed_hands.
    cluster: Series of cluster answers indexed by author_hash.
    The first rule that fits wins.
    """
    in_cluster = labels["author_hash"].map(cluster).eq("in_cluster")
    who, hands = labels["who_types"], labels["changed_hands"]
    rules = [
        (who == "declared", "E"),                                   # disclosed automation
        (hands == "likely", "B"),                                   # hijacked or rented
        (who.isin(["program", "person_plus_automation"]), "A"),     # automated
        ((who == "person") & in_cluster, "C"),                      # coordinated people (motive not judged)
        ((who == "person") & (hands == "no_sign"), "D"),            # ordinary person
    ]
    return pd.Series(np.select([r[0] for r in rules], [r[1] for r in rules], default="U"),
                     index=labels.index)


def cohen_kappa(first, second):
    """Agreement between two labelers beyond what chance would give. 1 = perfect, 0 = chance."""
    first, second = np.asarray(first), np.asarray(second)
    observed = (first == second).mean()
    expected = sum((first == c).mean() * (second == c).mean() for c in set(first) | set(second))
    return 1.0 if expected == 1 else float((observed - expected) / (1 - expected))


def agreement(file_a, file_b):
    """Kappa for each question, overall and per answer, for two labelers' sheets.

    Target: 0.6 or above on each question before labeling more.
    """
    a = clean_labels(pd.read_csv(file_a)).set_index("author_hash")
    b = clean_labels(pd.read_csv(file_b)).set_index("author_hash")
    shared = a.index.intersection(b.index)
    rows = []
    for question, answers in QUESTIONS.items():
        first, second = a.loc[shared, question], b.loc[shared, question]
        rows.append({"question": question, "answer": "all", "kappa": cohen_kappa(first, second), "n": len(shared)})
        for answer in answers:
            n = int(((first == answer) | (second == answer)).sum())
            if n:
                rows.append({"question": question, "answer": answer,
                             "kappa": cohen_kappa(first == answer, second == answer), "n": n})
    return pd.DataFrame(rows)
