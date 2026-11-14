"""Question 3: is the account in a coordinated cluster?

Coordination is a property of a group, so no person labels it per account.
Two accounts are linked when they
  - keep turning up in the same threads within minutes of each other, or
  - post nearly the same words within an hour, more than once.
Accounts joined by links form a cluster once the group is big enough.

"none_found" means tested and nothing found. It is not proof of innocence,
and "in_cluster" is not proof of bad intent: friends and fan groups also move together.
The thresholds are first guesses and have not been checked against labeled data.
"""

from collections import Counter

import pandas as pd

from ea.features import co_engagement_pairs, near_duplicate_pairs


def find_clusters(comments, cfg):
    """One row per account: cluster answer and a cluster number (-1 when in none)."""
    feature_settings, settings = cfg["features"], cfg["clusters"]
    counts = comments.groupby("author_hash").size()
    tested = set(counts[counts >= feature_settings["min_comments"]].index)
    comments = comments[comments["author_hash"].isin(tested)]

    links = set()
    shared_threads = co_engagement_pairs(comments, feature_settings["co_engage_minutes"])
    links.update(pair for pair, threads in shared_threads.items()
                 if len(threads) >= settings["min_shared_threads"])

    author_of = dict(zip(comments["id"], comments["author_hash"]))
    duplicate_pairs = near_duplicate_pairs(comments, feature_settings["near_duplicate_minutes"],
                                           feature_settings["near_duplicate_jaccard"])
    times_matched = Counter(tuple(sorted((author_of[a], author_of[b]))) for a, b in duplicate_pairs)
    links.update(pair for pair, n in times_matched.items() if n >= settings["min_duplicate_comments"])

    group_of = connected_groups(links)
    sizes = Counter(group_of.values())
    rows = []
    for account in counts.index:
        group = group_of.get(account)
        if account not in tested:
            rows.append((account, "not_tested", -1))
        elif group is not None and sizes[group] >= settings["min_accounts"]:
            rows.append((account, "in_cluster", group))
        else:
            rows.append((account, "none_found", -1))
    return pd.DataFrame(rows, columns=["author_hash", "cluster", "cluster_id"])


def connected_groups(links):
    """Give every linked account a group number; accounts joined by any chain of links share one."""
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        parent[root(a)] = root(b)
    numbers = {}
    return {account: numbers.setdefault(root(account), len(numbers)) for account in list(parent)}


def cluster_summary(clusters, comments):
    """Aggregate view of the clusters found: size, and how far each one reaches.

    "reach" follows Nimmo's breakout scale as far as one platform allows:
    category 1 = one community, category 2 = several communities on one platform.
    Categories 3 to 6 need other platforms and media, which this data cannot show.
    """
    found = clusters[clusters["cluster"] == "in_cluster"]
    if found.empty:
        return pd.DataFrame(columns=["breakout_category", "clusters", "median_accounts"])
    with_subs = comments.merge(found[["author_hash", "cluster_id"]], on="author_hash")
    per_cluster = pd.DataFrame({
        "accounts": found.groupby("cluster_id").size(),
        "subreddits": with_subs.groupby("cluster_id")["subreddit"].nunique(),
    })
    per_cluster["breakout_category"] = per_cluster["subreddits"].gt(1).map(
        {False: "1: one community", True: "2: several communities, one platform"})
    return (per_cluster.groupby("breakout_category")
            .agg(clusters=("accounts", "size"), median_accounts=("accounts", "median")).reset_index())
