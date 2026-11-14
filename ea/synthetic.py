"""Made-up data with known answers, for tests and for trying the pipeline end to end.

Nothing here is real. Three kinds of invented account:
  D ordinary     comments in waking hours, replies minutes to hours later, own wording
  A automated    comments around the clock, replies within seconds, templated text
  C coordinated  human rhythm, but a ring turns up in the same threads within
                 minutes, pushing the same talking point with small rewording
Results on this data say the code runs. They say nothing about real Reddit.
"""

import json

import numpy as np
import pandas as pd

from ea.config import paths
from ea.features import all_phrases

WORDS = ("the policy court city report people border agents officials law family local "
         "number year data claim judge state federal county work case rule plan vote news "
         "think really maybe never always because however already another against between").split()
DAY = 86400


def make_synthetic(cfg, n_ordinary=300, n_automated=30, n_rings=3, ring_size=10, n_threads=400, seed=0):
    rng = np.random.default_rng(seed)
    start = int(pd.Timestamp(cfg["start"], tz="UTC").timestamp())
    end = int(pd.Timestamp(cfg["end"], tz="UTC").timestamp())
    subs = cfg["subreddits"]["all_threads"] + cfg["subreddits"]["filtered"][:2]
    phrases = all_phrases(cfg)

    posts = [{"id": f"p{i}", "subreddit": str(rng.choice(subs)), "author_hash": f"poster{i % 40:03d}",
              "disclosed_bot": False, "created_utc": int(rng.integers(start, end - 2 * DAY)),
              "removed": False, "title": "immigration thread"} for i in range(n_threads)]
    comments, labels = [], {}

    def add(author, post, when, body):
        comments.append({"id": f"c{len(comments)}", "subreddit": post["subreddit"], "author_hash": author,
                         "disclosed_bot": False, "created_utc": int(when), "removed": False,
                         "link_id": post["id"], "parent_id": "t3_" + post["id"], "body": body})

    def own_words():
        text = " ".join(rng.choice(WORDS, size=rng.integers(10, 30)))
        return text + " " + str(rng.choice(phrases)) if rng.random() < 0.05 else text

    def waking_time(earliest, wake_hour):
        """A time after `earliest` that falls in a 15-hour waking day starting at wake_hour UTC."""
        when = earliest + rng.lognormal(np.log(2 * 3600), 1.0)
        hours_awake = ((when // 3600) - wake_hour) % 24
        return when if hours_awake < 15 else when + (24 - hours_awake) * 3600

    def ordinary_comments(author, how_many):
        wake_hour = int(rng.integers(0, 24))
        for i in rng.choice(n_threads, size=how_many, replace=False):
            add(author, posts[i], waking_time(posts[i]["created_utc"], wake_hour), own_words())

    for n in range(n_ordinary):
        labels[f"ordinary{n:04d}"] = "D"
        ordinary_comments(f"ordinary{n:04d}", int(rng.integers(12, 60)))

    templates = [" ".join(rng.choice(WORDS, size=18)) for _ in range(5)]
    for n in range(n_automated):
        labels[f"automated{n:04d}"] = "A"
        for i in rng.choice(n_threads, size=int(rng.integers(60, 150)), replace=False):
            body = str(rng.choice(templates)) + " " + str(rng.choice(WORDS))
            add(f"automated{n:04d}", posts[i], posts[i]["created_utc"] + rng.integers(5, 25), body)

    for ring in range(n_rings):
        members = [f"ring{ring}member{m:02d}" for m in range(ring_size)]
        labels.update({m: "C" for m in members})
        wake_hour = int(rng.integers(0, 24))
        for m in members:
            ordinary_comments(m, int(rng.integers(8, 20)))
        for i in rng.choice(n_threads, size=40, replace=False):
            talking_point = list(rng.choice(WORDS, size=20)) + [str(rng.choice(phrases))]
            meet = waking_time(posts[i]["created_utc"], wake_hour)
            for m in rng.choice(members, size=int(rng.integers(4, 8)), replace=False):
                reworded = talking_point.copy()
                reworded[int(rng.integers(0, 20))] = str(rng.choice(WORDS))
                add(str(m), posts[i], meet + rng.integers(0, 240), " ".join(reworded))

    folder = paths(cfg)["collected"] / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    for name, records in (("posts", posts), ("comments", comments)):
        with open(folder / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)

    label_file = paths(cfg)["labels"]
    label_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"author_hash": list(labels), "label": list(labels.values()),
                  "confidence": 3, "evidence": "synthetic: generated with this label"}
                 ).to_csv(label_file, index=False)
    return len(posts), len(comments), len(labels)


def make_synthetic_network(cfg, n_ordinary=400, ring_size=20, covert_size=20, relay_size=10,
                           n_threads=2000, seed=0):
    """Made-up data for the network stage, with three planted patterns among ordinary accounts.

      ring     posts threads, and aims most of its replies at other ring members
      covert   members never answer each other; each keeps answering the same three ordinary posters
      relay    members post the same link one after another, minutes apart, and rarely comment

    All three groups are written to the seeds file. Returns (posts, comments, seeds).
    """
    rng = np.random.default_rng(seed)
    start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
    subs = ["sub%d" % i for i in range(8)]
    ordinary = [f"ordinary{n:04d}" for n in range(n_ordinary)]
    ring = [f"ring{n:02d}" for n in range(ring_size)]
    covert = [f"covert{n:02d}" for n in range(covert_size)]
    relay = [f"relay{n:02d}" for n in range(relay_size)]
    posts, comments = [], []

    def post(author, when, sub=None, link_key=""):
        posts.append({"id": f"p{len(posts)}", "subreddit": sub or str(rng.choice(subs)), "author_hash": author,
                      "disclosed_bot": False, "created_utc": int(when), "removed": False, "title": "",
                      "link_key": link_key})
        return posts[-1]

    def reply(author, thread, parent=None):
        """Answer the post, or one earlier comment in its thread."""
        answered = parent or thread
        when = answered["created_utc"] + int(rng.lognormal(np.log(1800), 1.0)) + 1
        comments.append({"id": f"c{len(comments)}", "subreddit": thread["subreddit"], "author_hash": author,
                         "disclosed_bot": False, "created_utc": when, "removed": False, "link_id": thread["id"],
                         "parent_id": ("t1_" if parent else "t3_") + answered["id"], "body": ""})
        in_thread[thread["id"]].append(comments[-1])

    span = 120 * DAY
    for _ in range(n_threads):
        post(str(rng.choice(ordinary)), start + rng.integers(0, span), link_key=f"link{len(posts)}")
    for member in ring:
        for _ in range(4):
            post(member, start + rng.integers(0, span), link_key=f"link{len(posts)}")
    for n in range(40):                                     # relay chains of 3 to 5 accounts
        when = start + rng.integers(0, span)
        for step, member in enumerate(rng.choice(relay, size=int(rng.integers(3, 6)), replace=False)):
            post(str(member), when + step * int(rng.integers(120, 900)), link_key=f"relayed{n}")

    in_thread = {p["id"]: [] for p in posts}
    by_author = {}
    for p in posts:
        by_author.setdefault(p["author_hash"], []).append(p)
    ordinary_threads = [p for p in posts if p["author_hash"].startswith("ordinary")]
    ring_threads = [p for p in posts if p["author_hash"].startswith("ring")]

    def answer_somewhere(author, threads):
        thread = threads[int(rng.integers(len(threads)))]
        earlier = in_thread[thread["id"]]
        reply(author, thread, earlier[int(rng.integers(len(earlier)))] if earlier and rng.random() < 0.3 else None)

    for author in ordinary:
        for _ in range(int(rng.integers(10, 40))):
            answer_somewhere(author, posts)
    for author in ring:
        for _ in range(30):
            answer_somewhere(author, ring_threads if rng.random() < 0.6 else ordinary_threads)
    posters = [a for a in ordinary if a in by_author]
    for author in covert:
        favourites = [by_author[str(a)] for a in rng.choice(posters, size=3, replace=False)]
        for _ in range(30):
            mine = favourites[int(rng.integers(3))]
            if rng.random() < 0.8:
                reply(author, mine[int(rng.integers(len(mine)))])       # always the post, never a fellow member
            else:
                reply(author, ordinary_threads[int(rng.integers(len(ordinary_threads)))])
    for author in relay:
        for _ in range(3):
            answer_somewhere(author, ordinary_threads)

    folder = paths(cfg)["collected"] / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    for name, records in (("posts", posts), ("comments", comments)):
        with open(folder / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    seeds = ring + covert + relay
    seeds_file = paths(cfg)["seeds"]
    seeds_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"author_hash": seeds, "source": "synthetic"}).to_csv(seeds_file, index=False)
    return len(posts), len(comments), len(seeds)
