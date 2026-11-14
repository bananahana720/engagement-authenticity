"""Cross-account signals: compare every account with every other account, over time.

One operator running many accounts shares constraints across them even when each
account's wording and rhythm look human. Each signal below turns one such constraint
into "events" an account takes part in, then asks: does this account have partners
who keep sharing its *rare* events?

  same_thread     the same threads, at any delay (not only within minutes)
  same_points     the same talking point on the same day, whatever the wording
  same_pivots     first use of a talking point in the same 6 hours
  same_days       active on the same days
  cohort          first seen within a day of an unusual number of other accounts
  fused           for the account's strongest talking-point partners: do the other
                  leaks point at the same partners, and do those partners also
                  point at each other (a clique)? By chance anyone has a few
                  strong-looking partners among thousands of accounts, but chance
                  partners have nothing to do with one another.

A shared event counts for more the fewer accounts took part in it, so a busy news
day or a viral talking point counts for little. Each account's score is the average
over its 3 strongest partners.

Talking points must be given: a `point` column on the comments, -1 for none. On real
data that comes from text embeddings and clustering; the red team uses a stand-in.
No signal looks at which side of the topic an account is on.
"""

import numpy as np
import pandas as pd
from scipy import sparse

HOUR, DAY = 3600, 86400
TOP_PARTNERS = 3
CHANNELS = ["same_thread", "same_points", "same_pivots", "same_days", "cohort"]


def network_features(comments, accounts):
    """One row per account in `accounts`. `comments` needs author_hash, link_id, created_utc, point."""
    comments = comments[comments["author_hash"].isin(accounts)]
    row = pd.Series(np.arange(len(accounts)), index=accounts)
    who = row[comments["author_hash"]].to_numpy()
    when = comments["created_utc"].to_numpy()
    n = len(accounts)

    with_point = comments["point"].to_numpy() >= 0
    point = comments["point"].to_numpy()[with_point]
    first_use = (pd.DataFrame({"who": who[with_point], "point": point, "when": when[with_point]})
                 .groupby(["who", "point"], as_index=False)["when"].min())

    pair_scores = {
        "same_thread": shared_rare_events(who, comments["link_id"], n),
        "same_points": shared_rare_events(who[with_point], pair_id(point, when[with_point] // DAY), n),
        "same_pivots": shared_rare_events(first_use["who"].to_numpy(),
                                          pair_id(first_use["point"], first_use["when"] // (6 * HOUR)), n),
        "same_days": shared_rare_events(who, when // DAY, n),
    }
    first_seen = pd.Series(when).groupby(who).min().reindex(range(n)).to_numpy()
    in_cohort, cohort_excess = cohorts(first_seen, when.min())
    pair_scores["cohort"] = in_cohort

    out = pd.DataFrame(index=accounts)
    for name in CHANNELS[:-1]:
        out[name] = top_partner_mean(pair_scores[name])
    out["cohort_excess"] = cohort_excess

    # Fused: take the account's 5 strongest talking-point partners. Do the other leaks
    # point at the same partners, and do those partners also point at each other?
    k = min(5, n - 1)
    partners = np.argpartition(pair_scores["same_points"], -k, axis=1)[:, -k:]
    rows = np.arange(n)[:, None]
    for name in ("cohort", "same_pivots", "same_days", "same_thread"):
        out["partners_" + name] = pair_scores[name][rows, partners].mean(axis=1)
    among = pair_scores["same_points"][partners[:, :, None], partners[:, None, :]]    # n x k x k
    out["partners_clique"] = among.sum(axis=(1, 2)) / max(k * (k - 1), 1)                    # diagonal is zero

    counts = pd.Series(point).groupby(who[with_point]).value_counts()
    top3 = counts.groupby(level=0).apply(lambda c: c.nlargest(3).sum())
    total = pd.Series(who).value_counts()
    out["point_focus"] = (top3 / total).reindex(range(n)).fillna(0).to_numpy()
    return out


def pair_id(a, b):
    """One id per (a, b) combination."""
    return pd.factorize(pd.Series(np.asarray(a)).astype(str) + ":" + pd.Series(np.asarray(b)).astype(str))[0]


def shared_rare_events(who, event, n_accounts):
    """Account-by-account matrix: how much two accounts share events few others took part in.

    Scaled like a cosine (1 = identical, 0 = nothing shared), shrunk so that two
    accounts with very few events cannot score high from a single coincidence.
    """
    event = pd.factorize(np.asarray(event))[0]
    took_part = sparse.csr_matrix((np.ones(len(who), dtype=np.float32), (who, event)),
                                  shape=(n_accounts, event.max() + 1))
    took_part.sum_duplicates()
    took_part.data[:] = 1                                   # taking part twice counts once
    n_in_event = np.asarray(took_part.sum(axis=0)).ravel()
    rarity = np.where(n_in_event >= 2, np.log(n_accounts / np.maximum(n_in_event, 1)), 0)
    weighted = took_part @ sparse.diags(np.sqrt(np.clip(rarity, 0, None)).astype(np.float32))
    shared = (weighted @ weighted.T).toarray()
    own = shared.diagonal().copy()
    np.fill_diagonal(shared, 0)
    shrink = np.median(own[own > 0]) if (own > 0).any() else 1
    return shared / (np.sqrt(np.outer(own, own)) + shrink)


def cohorts(first_seen, data_start, window=DAY, baseline=14 * DAY):
    """Accounts first seen close together.

    Returns (pair matrix: 1 if two accounts were first seen within a day of each other,
    per-account excess: how many times more accounts arrived in that day than the
    surrounding four weeks would predict). The first two weeks of data are left out:
    there, everybody looks new.
    """
    known = first_seen >= data_start + baseline
    ordered = np.sort(first_seen[known])
    near = np.searchsorted(ordered, first_seen + window) - np.searchsorted(ordered, first_seen - window) - 1
    wide = np.searchsorted(ordered, first_seen + baseline) - np.searchsorted(ordered, first_seen - baseline) - 1
    expected = (wide + 1) * window / baseline
    excess = np.where(known, (near + 1) / (expected + 1), 1.0)
    together = (np.abs(first_seen[:, None] - first_seen[None, :]) < window) & known[:, None] & known[None, :]
    np.fill_diagonal(together, False)
    return together.astype(np.float32), excess


def top_partner_mean(pair_scores, k=TOP_PARTNERS):
    """Average of each account's k strongest partners."""
    k = min(k, pair_scores.shape[1] - 1)
    top = np.partition(pair_scores, -k, axis=1)[:, -k:]
    return top.mean(axis=1)
