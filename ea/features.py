"""Stage 3, features: one row per account per time window.

Each block below tests one family of hypotheses from the plan:
  timing        H1 (no sleep gap), H2 (replies faster than people read)
  near-duplicates  H5 (coordinated accounts share phrasing)
  co-engagement    H6 (coordinated accounts turn up together, repeatedly)
  coded phrases    phrase pickup in sync with other accounts

No feature looks at which side of the topic an account is on.
"""

import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from ea.collect import load_collected
from ea.config import paths


def build_features(cfg):
    posts, comments = load_collected(cfg)
    comments = comments[~comments["disclosed_bot"]]     # label E: set aside by rule, not modeled
    comments = add_reply_latency(comments, posts)

    windows = {
        "train": (cfg["start"], cfg["train_end"]),
        "test": (cfg["train_end"], cfg["end"]),
    }
    tables = []
    for name, (start, end) in windows.items():
        in_window = comments[(comments["time"] >= pd.Timestamp(start, tz="UTC"))
                             & (comments["time"] < pd.Timestamp(end, tz="UTC"))]
        if len(in_window):
            table = features_for_window(in_window, cfg)
            table.insert(1, "window", name)
            tables.append(table)
    features = pd.concat(tables, ignore_index=True)

    out = paths(cfg)["features"]
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    phrase_spread(comments, cfg).to_csv(ensure_parent(paths(cfg)["phrase_spread"]), index=False)
    return features


def features_for_window(comments, cfg):
    settings = cfg["features"]
    counts = comments.groupby("author_hash").size()
    active = counts[counts >= settings["min_comments"]].index
    comments = comments[comments["author_hash"].isin(active)]

    blocks = [
        timing_features(comments, settings["fast_reply_seconds"]),
        near_duplicate_features(comments, settings["near_duplicate_minutes"],
                                settings["near_duplicate_jaccard"]),
        co_engagement_features(comments, settings["co_engage_minutes"]),
        phrase_features(comments, all_phrases(cfg), settings["phrase_burst_minutes"],
                        settings["phrase_burst_accounts"]),
    ]
    return pd.concat(blocks, axis=1).fillna(0).rename_axis("author_hash").reset_index()


# ---------- timing (H1, H2) ----------

def add_reply_latency(comments, posts):
    """Seconds between a comment and the post or comment it replies to."""
    parent_time = pd.concat([
        pd.Series(posts["created_utc"].values, index="t3_" + posts["id"]),
        pd.Series(comments["created_utc"].values, index="t1_" + comments["id"]),
    ])
    comments = comments.copy()
    comments["reply_latency"] = comments["created_utc"] - comments["parent_id"].map(parent_time)
    return comments


def timing_features(comments, fast_reply_seconds):
    comments = comments.assign(hour=comments["time"].dt.hour)
    by_account = comments.groupby("author_hash")
    hour_counts = comments.groupby(["author_hash", "hour"]).size().unstack(fill_value=0)
    hour_counts = hour_counts.reindex(columns=range(24), fill_value=0)

    latency = comments.dropna(subset=["reply_latency"])
    latency = latency[latency["reply_latency"] >= 0]
    latency_by_account = latency.groupby("author_hash")["reply_latency"]

    return pd.DataFrame({
        "n_comments": by_account.size(),
        "n_threads": by_account["link_id"].nunique(),
        "removed_share": by_account["removed"].mean(),
        # A person sleeps: several hours of the day stay empty. A scheduler does not.
        "active_hours": (hour_counts > 0).sum(axis=1),
        "hour_entropy": hour_counts.apply(entropy, axis=1),
        "longest_quiet_hours": hour_counts.apply(longest_quiet_run, axis=1),
        "median_reply_seconds": latency_by_account.median(),
        "fast_reply_share": latency_by_account.apply(lambda s: (s < fast_reply_seconds).mean()),
    })


def entropy(hour_counts):
    """0 = all activity in one hour of the day, 4.58 = spread evenly over all 24."""
    share = hour_counts[hour_counts > 0] / hour_counts.sum()
    return float(-(share * np.log2(share)).sum())


def longest_quiet_run(hour_counts):
    """Longest run of empty hours of the day, wrapping past midnight."""
    empty = list(hour_counts.values == 0)
    if all(empty):
        return 24
    longest = run = 0
    for is_empty in empty + empty:          # doubled, so a run can wrap around midnight
        run = run + 1 if is_empty else 0
        longest = max(longest, run)
    return min(longest, 24)


# ---------- near-duplicate text (H5) ----------

def normalize(text):
    text = re.sub(r"https?://\S+", " ", text.lower())
    return re.sub(r"[^a-z0-9 ]+", " ", text).split()


def shingles(words, size=4):
    """Every run of `size` consecutive words, as a set."""
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def near_duplicate_pairs(comments, minutes, min_jaccard, min_words=8):
    """Pairs of comments by DIFFERENT accounts, close in time, with mostly the same wording.

    Two comments are compared only if they share at least one 4-word run, which
    keeps this fast. Similarity is Jaccard overlap of their 4-word runs.
    """
    texts = {}
    by_shingle = defaultdict(list)
    for row in comments.itertuples():
        words = normalize(row.body)
        if len(words) < min_words:
            continue
        texts[row.id] = (shingles(words), row.created_utc, row.author_hash)
        for s in texts[row.id][0]:
            by_shingle[s].append(row.id)

    # Candidate pairs: same 4-word run, different accounts, close in time.
    candidates = set()
    for ids in by_shingle.values():
        ids.sort(key=lambda i: texts[i][1])                 # oldest first
        for position, a in enumerate(ids):
            for b in ids[position + 1:]:
                if texts[b][1] - texts[a][1] > minutes * 60:
                    break                                   # everything later is too far apart
                if texts[a][2] != texts[b][2]:
                    candidates.add((a, b))

    pairs = []
    for a, b in candidates:
        set_a, set_b = texts[a][0], texts[b][0]
        if len(set_a & set_b) / len(set_a | set_b) >= min_jaccard:
            pairs.append((a, b))
    return pairs


def near_duplicate_features(comments, minutes, min_jaccard):
    pairs = near_duplicate_pairs(comments, minutes, min_jaccard)
    matched_ids = {comment_id for pair in pairs for comment_id in pair}
    matched = comments["id"].isin(matched_ids)
    return pd.DataFrame({
        "near_duplicate_share": matched.groupby(comments["author_hash"]).mean(),
    })


# ---------- co-engagement (H6) ----------

def co_engagement_features(comments, minutes, max_group=50):
    """How often the same accounts turn up in the same thread within minutes of each other.

    One coincidence means nothing. The signal is a partner that keeps reappearing.
    """
    slot = comments["created_utc"] // (minutes * 60)
    shared_threads = defaultdict(set)       # (account, account) -> threads they co-appeared in
    for (thread, _), group in comments.groupby(["link_id", slot]):
        accounts = sorted(set(group["author_hash"]))
        if 2 <= len(accounts) <= max_group:  # a huge pile-on is a busy thread, not a ring
            for pair in combinations(accounts, 2):
                shared_threads[pair].add(thread)

    best_partner = defaultdict(int)
    repeat_partners = defaultdict(int)
    for (a, b), threads in shared_threads.items():
        for account in (a, b):
            best_partner[account] = max(best_partner[account], len(threads))
            repeat_partners[account] += len(threads) >= 3

    accounts = comments["author_hash"].unique()
    return pd.DataFrame({
        "co_engage_best_partner_threads": [best_partner[a] for a in accounts],
        "co_engage_repeat_partners": [repeat_partners[a] for a in accounts],
    }, index=accounts)


# ---------- coded phrases ----------

def all_phrases(cfg):
    """Both lists merged. From here on the code cannot tell which list a phrase came from."""
    lists = cfg["coded_phrases"]
    assert len(lists["enforcement_favoring"]) == len(lists["enforcement_opposing"]), \
        "Track the same number of phrases from each side."
    return sorted(p.lower() for p in lists["enforcement_favoring"] + lists["enforcement_opposing"])


def phrase_uses(comments, phrases):
    """One row per (comment, phrase) where the comment contains the phrase."""
    body = comments["body"].str.lower()
    uses = []
    for phrase in phrases:
        hit = body.str.contains(r"\b" + re.escape(phrase) + r"\b", regex=True)
        uses.append(comments.loc[hit, ["id", "author_hash", "created_utc", "subreddit", "time"]]
                    .assign(phrase=phrase))
    return pd.concat(uses, ignore_index=True)


def phrase_features(comments, phrases, burst_minutes, burst_accounts):
    """Share of an account's comments that use a tracked phrase, and how often that use
    lands in a burst: several different accounts using the same phrase within minutes."""
    uses = phrase_uses(comments, phrases)
    uses["slot"] = uses["created_utc"] // (burst_minutes * 60)
    accounts_in_slot = uses.groupby(["phrase", "slot"])["author_hash"].transform("nunique")
    uses["in_burst"] = accounts_in_slot >= burst_accounts

    n_comments = comments.groupby("author_hash").size()
    per_account = uses.groupby("author_hash")
    return pd.DataFrame({
        "phrase_use_share": per_account["id"].nunique() / n_comments,
        "phrase_burst_share": per_account["in_burst"].mean(),
    }).reindex(n_comments.index)


def phrase_spread(comments, cfg):
    """Aggregate table: for each phrase and week, how many uses and how many accounts."""
    uses = phrase_uses(comments, all_phrases(cfg))
    uses["week"] = uses["time"].dt.strftime("%G-W%V")
    return (uses.groupby(["phrase", "week"])
            .agg(uses=("id", "nunique"), accounts=("author_hash", "nunique"))
            .reset_index())


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
