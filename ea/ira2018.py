"""Seed data: the 944 accounts Reddit disclosed in April 2018 as suspected Internet Research Agency accounts.

Source: github.com/ALCC01/reddit-suspicious-accounts, a public-domain (CC0) copy
of those accounts' public comments and posts, taken on 11 April 2018 while
Reddit kept them visible for investigators.

What the data is, and is not:
  - It holds only what the 944 accounts wrote. Replies TO them from other
    people are not in it, and neither are those people's other comments.
  - For each comment we know who wrote the post it sits under. We know who wrote
    the comment it answers only when that was another of the 944.
  - Reddit's listing returns at most 1,000 items per account, newest first.

This module converts it to the same small records the collect stage writes:
usernames become keyed hashes (the 944 and everyone they answered alike), text
is dropped, and links and titles are kept only as hashes for relay matching.
"""

import hashlib
import json

import pandas as pd
import requests

from ea.config import paths
from ea.pseudonymize import NO_AUTHOR, is_disclosed_bot, load_key, pseudonym

FILES = ("seed.csv", "data/comments.csv", "data/submissions.csv")
NOTICE_AUTHOR = "reddit"        # the ban notice Reddit pinned on each profile, not the account's own post


def download(cfg):
    """Fetch the three source files into data/<name>/raw/. Skips files already there."""
    raw = cfg["data_dir"] / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw / name.split("/")[-1]
        if not target.exists():
            response = requests.get(f"{cfg['seeds']['base_url']}/{name}", timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
    return raw


def short_hash(text):
    return hashlib.sha1(str(text).strip().lower().encode()).hexdigest()[:12]


def convert(cfg, raw=None):
    """Write posts.jsonl, comments.jsonl and seeds.csv from the raw CSV files."""
    raw = raw or cfg["data_dir"] / "raw"
    key = load_key()

    def hashed(name):
        return "" if name in NO_AUTHOR or pd.isna(name) else pseudonym(str(name), key)

    seed_names = pd.read_csv(raw / "seed.csv", header=None, names=["name", "karma"])["name"]
    seeds = sorted({hashed(n) for n in seed_names})

    submissions = pd.read_csv(raw / "submissions.csv").drop_duplicates("id")
    submissions = submissions[submissions["author.name"].str.lower() != NOTICE_AUTHOR]
    posts = [{"id": row.id, "subreddit": subreddit_name(row.subreddit_name_prefixed),
              "author_hash": hashed(row.author), "disclosed_bot": False,
              "created_utc": int(row.created_utc), "removed": False, "title": "",
              "link_key": "" if row.is_self else short_hash(row.url)}
             for row in submissions.rename(columns={"author.name": "author"}).itertuples()]
    seed_post_ids = {p["id"] for p in posts}

    raw_comments = pd.read_csv(raw / "comments.csv").drop_duplicates("id")
    comments = [{"id": row.id, "subreddit": subreddit_name(row.sub), "author_hash": hashed(row.author),
                 "disclosed_bot": False, "created_utc": int(row.created_utc), "removed": False,
                 "link_id": row.link_id.split("_", 1)[-1], "parent_id": row.parent_id, "body": ""}
                for row in raw_comments.rename(columns={"author.name": "author",
                                                        "subreddit.display_name_prefixed": "sub"}).itertuples()]

    # Posts by other people that the 944 commented under: we know the author, not the time.
    others = raw_comments.assign(post_id=raw_comments["link_id"].str.split("_", n=1).str[-1])
    others = others[~others["post_id"].isin(seed_post_ids)].drop_duplicates("post_id")
    for row in others.rename(columns={"subreddit.display_name_prefixed": "sub"}).itertuples():
        declared = not pd.isna(row.link_author) and is_disclosed_bot(str(row.link_author))
        posts.append({"id": row.post_id, "subreddit": subreddit_name(row.sub), "author_hash": hashed(row.link_author),
                      "disclosed_bot": declared, "created_utc": None, "removed": False, "title": "", "link_key": ""})

    folder = paths(cfg)["collected"] / "ira2018"
    folder.mkdir(parents=True, exist_ok=True)
    for name, records in (("posts", posts), ("comments", comments)):
        with open(folder / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    seeds_file = paths(cfg)["seeds"]
    seeds_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"author_hash": seeds, "source": "reddit 2018 disclosure"}).to_csv(seeds_file, index=False)
    return len(posts), len(comments), len(seeds)


def subreddit_name(prefixed):
    """'r/politics' -> 'politics'. Profile posts ('u/name') all become 'own profile'."""
    prefixed = str(prefixed)
    return "own profile" if prefixed.startswith("u/") else prefixed.split("/", 1)[-1]
