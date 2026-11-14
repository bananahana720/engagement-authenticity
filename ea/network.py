"""Interaction network: who replies to whom, starting from accounts already known to be inauthentic.

"Seeds" are accounts with a verified label (for example the 944 accounts Reddit
disclosed in 2018). Every comment is one edge from its author to the author of
the post or comment it answers. From those edges this module measures:

  mixing        how much seed activity lands on other seeds, against a shuffle
  seed graph    shape of the seed-to-seed web: pairs, reciprocity, clustering
  selectivity   how narrow each account's choice of targets and subreddits is
  roles         accounts that only post, only comment, or do both
  relays        the same link posted by several accounts, one after another
  proximity     if we knew only half the seeds, would their contacts be the other half?

The last one is the precision check on the whole idea. Being close to a seed is
evidence for a score. It is never a label by itself.

Nothing here records which side of a topic an account is on, and the report
holds counts and rates only.
"""

import json
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from ea.collect import load_collected
from ea.config import paths

UNKNOWN = ""        # the target's author was deleted or was never collected


# ---------- edges ----------

def build_edges(posts, comments):
    """One row per comment: source account, target account, and how they are linked.

    kind is "post_reply" (a top-level comment answering a post) or
    "comment_reply" (a comment answering another comment).
    """
    post_author = pd.Series(posts["author_hash"].fillna(UNKNOWN).values, index=posts["id"])
    post_time = pd.Series(posts["created_utc"].values, index=posts["id"])
    comment_author = pd.Series(comments["author_hash"].values, index=comments["id"])
    comment_time = pd.Series(comments["created_utc"].values, index=comments["id"])

    parent = comments["parent_id"].astype(str)
    to_comment = parent.str.startswith("t1_")
    parent_key = parent.str.split("_", n=1).str[-1]
    target = np.where(to_comment, parent_key.map(comment_author), parent_key.map(post_author))
    parent_time = np.where(to_comment, parent_key.map(comment_time), parent_key.map(post_time))
    thread_owner = comments["link_id"].map(post_author)

    edges = pd.DataFrame({
        "source": comments["author_hash"].values,
        "target": pd.Series(target).fillna(UNKNOWN).values,
        "kind": np.where(to_comment, "comment_reply", "post_reply"),
        "thread_owner": thread_owner.fillna(UNKNOWN).values,
        "subreddit": comments["subreddit"].values,
        "post_id": comments["link_id"].values,
        "created_utc": comments["created_utc"].values,
        "latency": comments["created_utc"].values - pd.to_numeric(pd.Series(parent_time), errors="coerce").values,
    })
    if "disclosed_bot" in comments:
        edges = edges[~comments["disclosed_bot"].values]       # declared bots are set aside by rule
    return edges.reset_index(drop=True)


def target_class(edges, seeds):
    """self / seed / other / unknown, for each edge."""
    return np.select(
        [edges["target"] == UNKNOWN, edges["target"] == edges["source"], edges["target"].isin(seeds)],
        ["unknown", "self", "seed"], default="other")


# ---------- mixing: do seeds aim at seeds? ----------

def mixing_table(edges, seeds):
    """Where edges that start at a seed end up, by kind of reply. Counts."""
    from_seed = edges[edges["source"].isin(seeds)]
    table = pd.crosstab(from_seed["kind"], target_class(from_seed, seeds))
    return table.reindex(columns=["self", "seed", "other", "unknown"], fill_value=0)


def mixing_against_shuffle(edges, seeds, shuffles=200, seed=0):
    """Share of seed replies that hit another seed, against chance.

    Chance is a shuffle of who-was-answered among the replies in the same
    subreddit. That keeps how active each account is, how often each account
    gets answered, and where; it breaks only the choice of partner.

    Needs replies from non-seeds too. On data that holds only the seeds' own
    activity the shuffle cannot move anything, and "testable" is False.
    """
    known = edges[(edges["target"] != UNKNOWN) & (edges["target"] != edges["source"])]
    is_seed_source = known["source"].isin(seeds).values
    if is_seed_source.all() or not is_seed_source.any():
        return {"testable": False, "observed_share": share(known["target"].isin(seeds).values[is_seed_source])}
    observed = share(known["target"].isin(seeds).values[is_seed_source])
    rng = np.random.default_rng(seed)
    groups = [g.values for _, g in pd.Series(np.arange(len(known))).groupby(known["subreddit"].values)]
    target_is_seed = known["target"].isin(seeds).values
    sources, targets = known["source"].values, known["target"].values
    null = []
    for _ in range(shuffles):
        shuffled = np.arange(len(known))
        for rows in groups:
            shuffled[rows] = rng.permutation(rows)
        keep = is_seed_source & (targets[shuffled] != sources)      # a shuffle must not invent self-replies
        null.append(share(target_is_seed[shuffled][keep]))
    null = np.array(null)
    return {"testable": True, "observed_share": observed, "chance_share": float(null.mean()),
            "times_chance": float(observed / null.mean()) if null.mean() else float("inf"),
            "p_value": float((1 + (null >= observed).sum()) / (1 + shuffles))}


def share(flags):
    return float(np.mean(flags)) if len(flags) else 0.0


# ---------- the seed-to-seed web ----------

def seed_graph(edges, seeds):
    """Directed graph of replies between two different seeds. Edge weight = number of replies."""
    between = edges[edges["source"].isin(seeds) & edges["target"].isin(seeds)
                    & (edges["source"] != edges["target"])]
    graph = nx.DiGraph()
    for (source, target), n in between.groupby(["source", "target"]).size().items():
        graph.add_edge(source, target, weight=int(n))
    return graph


def graph_shape(graph):
    if graph.number_of_edges() == 0:
        return {"accounts": 0, "pairs": 0, "replies": 0, "reciprocity": 0.0, "clustering": 0.0,
                "groups": 0, "largest_group": 0}
    groups = list(nx.weakly_connected_components(graph))
    return {
        "accounts": graph.number_of_nodes(),
        "pairs": graph.number_of_edges(),                         # ordered pairs
        "replies": int(sum(w for _, _, w in graph.edges(data="weight"))),
        "reciprocity": float(nx.reciprocity(graph)),              # share of pairs answered back
        "clustering": float(nx.transitivity(graph.to_undirected())),
        "groups": len(groups),
        "largest_group": max(len(g) for g in groups),
    }


def seed_graph_against_shuffle(edges, seeds, shuffles=200, seed=0):
    """Is the seed-to-seed web tighter than its own activity levels explain?

    Shuffles who-answered-whom among seed-to-seed replies in the same subreddit,
    then compares repeat pairs, reciprocity and clustering with the real web.
    """
    between = edges[edges["source"].isin(seeds) & edges["target"].isin(seeds)
                    & (edges["source"] != edges["target"])]
    observed = graph_shape(seed_graph(between, seeds))
    if observed["replies"] < 10:
        return {"observed": observed, "chance": None}
    rng = np.random.default_rng(seed)
    measures = ("pairs", "reciprocity", "clustering")
    null = {m: [] for m in measures}
    for _ in range(shuffles):
        shuffled = between.assign(target=between.groupby("subreddit")["target"]
                                  .transform(lambda t: rng.permutation(t.values)))
        shape = graph_shape(seed_graph(shuffled, seeds))
        for m in measures:
            null[m].append(shape[m])
    return {"observed": observed,
            "chance": {m: float(np.mean(values)) for m, values in null.items()},
            # repeat pairs mean FEWER distinct pairs than chance; the other two mean MORE
            "p_value": {"pairs": tail(null["pairs"], observed["pairs"], lower=True),
                        "reciprocity": tail(null["reciprocity"], observed["reciprocity"]),
                        "clustering": tail(null["clustering"], observed["clustering"])}}


def tail(null, observed, lower=False):
    null = np.asarray(null)
    hits = (null <= observed).sum() if lower else (null >= observed).sum()
    return float((1 + hits) / (1 + len(null)))


# ---------- selectivity, range and roles ----------

def entropy_share(counts):
    """0 = everything in one place, 1 = spread evenly over the places used. Needs 2+ places."""
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def account_profiles(edges, posts):
    """One row per account: how many targets and subreddits, how concentrated, and its role."""
    known = edges[(edges["target"] != UNKNOWN) & (edges["target"] != edges["source"])]
    rows = {}
    for account, group in known.groupby("source"):
        targets = group["target"].value_counts()
        rows[account] = {"replies_to_others": len(group), "targets": len(targets),
                         "top_target_share": float(targets.iloc[0] / len(group)),
                         "target_spread": entropy_share(targets.values)}
    profiles = pd.DataFrame.from_dict(rows, orient="index")
    profiles = profiles.reindex(columns=["replies_to_others", "targets", "top_target_share", "target_spread"])

    real_posts = posts[posts["created_utc"].notna() & (posts["author_hash"].fillna(UNKNOWN) != UNKNOWN)]
    activity = pd.concat([edges[["source", "subreddit"]].rename(columns={"source": "account"}),
                          real_posts[["author_hash", "subreddit"]].rename(columns={"author_hash": "account"})])
    by_sub = activity.groupby("account")["subreddit"].value_counts()
    profiles = profiles.join(pd.DataFrame({
        "subreddits": by_sub.groupby(level=0).size(),
        "top_subreddit_share": by_sub.groupby(level=0).max() / by_sub.groupby(level=0).sum(),
        "comments": edges.groupby("source").size(),
        "posts": real_posts.groupby("author_hash").size(),
    }), how="outer")
    profiles[["comments", "posts", "replies_to_others"]] = \
        profiles[["comments", "posts", "replies_to_others"]].fillna(0).astype(int)
    profiles["self_reply_share"] = (edges[edges["target"] == edges["source"]].groupby("source").size()
                                    .reindex(profiles.index).fillna(0) / profiles["comments"].replace(0, np.nan))
    profiles["role"] = np.select([profiles["comments"] == 0, profiles["posts"] == 0],
                                 ["posts only", "comments only"], default="posts and comments")
    return profiles.rename_axis("account")


def summarize_profiles(profiles, min_replies=10):
    """Medians for one group of accounts. Selectivity only counts accounts with enough replies."""
    enough = profiles[profiles["replies_to_others"] >= min_replies]
    return {
        "accounts": len(profiles),
        "roles": {k: int(v) for k, v in profiles["role"].value_counts().items()},
        "median_subreddits": median(profiles["subreddits"]),
        "median_top_subreddit_share": median(profiles["top_subreddit_share"]),
        "accounts_with_enough_replies": len(enough),
        "median_targets": median(enough["targets"]),
        "median_top_target_share": median(enough["top_target_share"]),
        "median_target_spread": median(enough["target_spread"]),
        "share_with_narrow_targets": share((enough["top_target_share"] >= 0.5).values),
        "median_self_reply_share": median(profiles["self_reply_share"]),
    }


def median(values):
    values = pd.Series(values).dropna()
    return float(values.median()) if len(values) else None


# ---------- relays: the same link, passed along ----------

def relay_chains(posts, seeds, max_gap_hours=24):
    """Groups of 2+ different accounts posting the same link, each within max_gap_hours of the last.

    Needs a "link_key" column on posts (a hash of the link). Returns one row per chain.
    """
    if "link_key" not in posts:
        return pd.DataFrame(columns=["accounts", "posts", "seed_accounts", "subreddits", "median_gap_minutes"])
    linked = posts[posts["link_key"].fillna("").ne("") & posts["created_utc"].notna()
                   & posts["author_hash"].fillna(UNKNOWN).ne(UNKNOWN)].sort_values("created_utc")
    chains = []
    for _, group in linked.groupby("link_key"):
        if group["author_hash"].nunique() < 2:
            continue
        gaps = group["created_utc"].diff().fillna(0).values
        chain_id = (gaps > max_gap_hours * 3600).cumsum()
        for _, chain in group.groupby(chain_id):
            authors = chain["author_hash"].unique()
            if len(authors) < 2:
                continue
            chains.append({"accounts": len(authors), "posts": len(chain),
                           "seed_accounts": int(np.isin(authors, list(seeds)).sum()),
                           "subreddits": chain["subreddit"].nunique(),
                           "median_gap_minutes": float(np.median(chain["created_utc"].diff().dropna()) / 60)})
    return pd.DataFrame(chains, columns=["accounts", "posts", "seed_accounts", "subreddits", "median_gap_minutes"])


def summarize_relays(chains, posts, seeds):
    all_seed = chains[chains["seed_accounts"] == chains["accounts"]] if len(chains) else chains
    seed_posts = posts[posts["author_hash"].isin(seeds) & posts["created_utc"].notna()]
    return {
        "chains": len(chains),
        "chains_all_seed": len(all_seed),
        "longest_chain_accounts": int(chains["accounts"].max()) if len(chains) else 0,
        "median_gap_minutes": median(chains["median_gap_minutes"]) if len(chains) else None,
        "chains_across_subreddits": int((chains["subreddits"] > 1).sum()) if len(chains) else 0,
        "seed_posts": len(seed_posts),
        "seed_posts_in_chains": int(all_seed["posts"].sum()) if len(all_seed) else 0,
    }


# ---------- proximity: does the net catch the right accounts? ----------

def proximity_test(edges, seeds, trials=100, seed=0, thresholds=(1, 2, 3), min_share=0.15):
    """Hide half the seeds, cast the net from the other half, count what it catches.

    For each trial: the "known" half are the starting points. Every other account
    that replied to, or was replied to by, a known seed is a candidate.

      loose net   flag a candidate that touches at least k different known seeds
      tight net   also require that at least min_share of the candidate's own
                  replies go to known seeds (needs the candidates' own replies,
                  so it is skipped on data that holds only the seeds' activity)

    Precision is the share of flagged accounts that are hidden seeds. Recall is
    the share of hidden seeds that were flagged, counting only hidden seeds that
    show up in a reply at all.

    Precision here is a floor on real data: an unflagged "ordinary" contact may be
    an inauthentic account nobody disclosed.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted(seeds)
    known_edges = edges[(edges["target"] != UNKNOWN) & (edges["target"] != edges["source"])]
    full_data = not known_edges["source"].isin(seeds).all()
    replies_sent = known_edges.groupby("source").size()
    pairs = known_edges.groupby(["source", "target"]).size().rename("replies").reset_index()
    pairs = pairs[pairs["source"].isin(seeds) | pairs["target"].isin(seeds)]
    seen = set(pairs["source"]) | set(pairs["target"])
    nets = [(f"touches_{k}_or_more", k, 0.0) for k in thresholds]
    if full_data:
        nets += [(f"touches_{k}_or_more_and_{int(min_share * 100)}pct_of_replies", k, min_share) for k in thresholds]
    results = {name: {"precision": [], "recall": [], "flagged": []} for name, _, _ in nets}
    for _ in range(trials):
        known = set(rng.choice(seeds, size=len(seeds) // 2, replace=False))
        hidden = set(seeds) - known
        out = pairs[pairs["source"].isin(known) & ~pairs["target"].isin(known)]
        back = pairs[pairs["target"].isin(known) & ~pairs["source"].isin(known)]
        contacts = pd.concat([out.rename(columns={"source": "seed", "target": "candidate"}),
                              back.rename(columns={"target": "seed", "source": "candidate"})])
        touched = contacts.groupby("candidate")["seed"].nunique()
        sent_to_known = back.groupby("source")["replies"].sum()
        seed_share = (sent_to_known / replies_sent.reindex(sent_to_known.index)).reindex(touched.index).fillna(0)
        reachable = hidden & seen
        for name, k, needed_share in nets:
            flagged = set(touched[(touched >= k) & (seed_share >= needed_share)].index)
            hits = len(flagged & hidden)
            results[name]["flagged"].append(len(flagged))
            results[name]["precision"].append(hits / len(flagged) if flagged else np.nan)
            results[name]["recall"].append(hits / len(reachable) if reachable else np.nan)
    return {name: {"flagged": float(np.mean(r["flagged"])), "precision": nanmean(r["precision"]),
                   "recall": nanmean(r["recall"])} for name, r in results.items()}


def nanmean(values):
    values = np.asarray(values, dtype=float)
    return None if np.isnan(values).all() else float(np.nanmean(values))


# ---------- the report ----------

def load_seeds(cfg):
    """Seeds are the accounts in the seeds file, or else the labeled A/B/C accounts."""
    seeds_file = paths(cfg)["seeds"]
    if seeds_file.exists():
        return set(pd.read_csv(seeds_file, dtype=str)["author_hash"])
    labels = pd.read_csv(paths(cfg)["labels"], dtype=str)
    return set(labels[labels["label"].isin(["A", "B", "C"])]["author_hash"])


def analyze(posts, comments, seeds, shuffles=200, trials=100):
    """Every measure, as one plain dictionary of counts and rates."""
    edges = build_edges(posts, comments)
    profiles = account_profiles(edges, posts)
    is_seed = profiles.index.isin(list(seeds))
    chains = relay_chains(posts, seeds)
    from_seed = edges[edges["source"].isin(seeds)]
    fast = from_seed[from_seed["target"].isin(seeds) & (from_seed["target"] != from_seed["source"])]["latency"].dropna()
    result = {
        "counts": {"seeds": len(seeds), "seeds_active": int(is_seed.sum()),
                   "other_accounts_active": int((~is_seed).sum()),
                   "comments": len(edges), "comments_by_seeds": len(from_seed)},
        "mixing": {kind: {k: int(v) for k, v in row.items()} for kind, row in mixing_table(edges, seeds).iterrows()},
        "mixing_against_chance": mixing_against_shuffle(edges, seeds, shuffles),
        "seed_web": seed_graph_against_shuffle(edges, seeds, shuffles),
        "seed_to_seed_replies_by_subreddit": {k: int(v) for k, v in edges[
            edges["source"].isin(seeds) & edges["target"].isin(seeds) & (edges["source"] != edges["target"])
        ]["subreddit"].value_counts().head(8).items()},
        "seed_to_seed_reply_minutes": {"replies_timed": len(fast), "median": median(fast / 60),
                                       "within_10_minutes_share": share((fast <= 600).values)},
        "seed_profiles": summarize_profiles(profiles[is_seed]),
        "other_profiles": summarize_profiles(profiles[~is_seed]) if (~is_seed).any() else None,
        "relays": summarize_relays(chains, posts, seeds),
        "proximity": proximity_test(edges, seeds, trials),
    }
    return result, edges, profiles


def picture_data(edges, profiles, seeds):
    """Positions for drawing the seed-to-seed web. Accounts are numbered, never named or hashed."""
    graph = seed_graph(edges, seeds)
    if graph.number_of_nodes() == 0:
        return {"nodes": [], "links": []}
    layout = packed_layout(graph.to_undirected())
    order = {account: i for i, account in enumerate(graph.nodes)}
    group_of = {}
    for g, members in enumerate(sorted(nx.weakly_connected_components(graph), key=len, reverse=True)):
        group_of.update({m: g for m in members})
    top_sub = Counter()
    between = edges[edges["source"].isin(seeds) & edges["target"].isin(seeds) & (edges["source"] != edges["target"])]
    sub_of = {}
    for account in graph.nodes:
        used = between[(between["source"] == account) | (between["target"] == account)]["subreddit"]
        sub_of[account] = used.value_counts().index[0]
        top_sub[sub_of[account]] += 1
    nodes = [{"id": order[a], "x": round(float(layout[a][0]), 4), "y": round(float(layout[a][1]), 4),
              "sent": int(graph.out_degree(a, weight="weight")), "received": int(graph.in_degree(a, weight="weight")),
              "role": profiles.loc[a, "role"] if a in profiles.index else "", "group": group_of[a],
              "subreddit": sub_of[a]} for a in graph.nodes]
    links = [{"source": order[s], "target": order[t], "replies": int(w), "both_ways": graph.has_edge(t, s)}
             for s, t, w in graph.edges(data="weight")]
    return {"nodes": nodes, "links": links}


def packed_layout(graph, row_width=10.0, gap=0.8):
    """Lay out each connected group by itself, then set the groups side by side, largest first."""
    layout, x, y, row_height = {}, 0.0, 0.0, 0.0
    for members in sorted(nx.connected_components(graph), key=len, reverse=True):
        size = 0.9 * np.sqrt(len(members))                      # room grows with the group
        if len(members) > 2:
            local = nx.kamada_kawai_layout(graph.subgraph(members), scale=size)
        else:
            local = {m: np.array([(i - (len(members) - 1) / 2) * 0.9, 0.0]) for i, m in enumerate(sorted(members))}
        if x > 0 and x + 2 * size > row_width:
            x, y, row_height = 0.0, y + row_height + gap, 0.0
        for m, (px, py) in local.items():
            layout[m] = (x + size + px, y + size + py)
        x, row_height = x + 2 * size + gap, max(row_height, 2 * size)
    return layout


def network_report(cfg, shuffles=200, trials=100):
    """Run every measure on what was collected and write the aggregate report files."""
    posts, comments = load_collected(cfg)
    seeds = load_seeds(cfg)
    result, edges, profiles = analyze(posts, comments, seeds, shuffles, trials)
    out = paths(cfg)["network"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    paths(cfg)["network_picture"].write_text(json.dumps(picture_data(edges, profiles, seeds)))
    return out, result
