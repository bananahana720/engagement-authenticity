"""The project's key numbers, each a small function. Explained in docs/METRICS.md.

Two kinds of error, named in plain words everywhere:
  false flag  a person called software      (the costly one for this project)
  miss        software passed as a person

y is 1 for software, 0 for a person. score is the model's 0-to-1 score.
"""

import numpy as np
from sklearn.metrics import roc_curve


def error_rates(y, score):
    """False-flag rate and miss rate at every possible score cut-off."""
    false_flag_rate, catch_rate, cutoffs = roc_curve(y, score)
    return false_flag_rate, 1 - catch_rate, cutoffs


def crossover_error_rate(y, score):
    """The error rate at the cut-off where false flags and misses are equally common.

    One number for comparing two models: lower is better. Also called the equal error rate.
    """
    false_flag_rate, miss_rate, _ = error_rates(y, score)
    # The two curves cross between two neighbouring cut-offs; take the nearest and average.
    nearest = np.argmin(np.abs(false_flag_rate - miss_rate))
    return float((false_flag_rate[nearest] + miss_rate[nearest]) / 2)


def recall_at_false_flag_rate(y, score, max_rate):
    """Share of software caught when at most `max_rate` of people may be falsely flagged."""
    false_flag_rate, miss_rate, _ = error_rates(y, score)
    allowed = false_flag_rate <= max_rate
    return float((1 - miss_rate[allowed]).max())


def bootstrap_interval(metric, y, score, rounds=1000, seed=0):
    """95% range for a metric: recompute it on many resamples of the test accounts."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        pick = rng.integers(0, len(y), len(y))
        if 0 < y[pick].sum() < len(y):          # a resample needs both kinds of account
            values.append(metric(y[pick], score[pick]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def enough_people_for(max_rate, n_people):
    """A 1% false-flag rate cannot be measured on fewer than 100 people, 0.1% on fewer than 1,000."""
    return n_people >= round(1 / max_rate)


def precision_at_prevalence(catch_rate, false_flag_rate, prevalence):
    """Of the accounts flagged, the share that really are software, if `prevalence` of all accounts are."""
    caught = prevalence * catch_rate
    falsely_flagged = (1 - prevalence) * false_flag_rate
    return caught / (caught + falsely_flagged) if caught + falsely_flagged else float("nan")


def reliability_table(y, score, bins=5):
    """Does a score of 0.8 mean 80%? Average score next to the observed share, per score band."""
    y, score = np.asarray(y), np.asarray(score)
    band = np.minimum((score * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        inside = band == b
        if inside.any():
            rows.append({"score_band": f"{b / bins:.1f} to {(b + 1) / bins:.1f}", "accounts": int(inside.sum()),
                         "average_score": round(float(score[inside].mean()), 2),
                         "observed_share_software": round(float(y[inside].mean()), 2)})
    return rows
