"""Made-up data with known answers, for tests and for trying the pipeline end to end.

Nothing here is real. Three kinds of invented account:
  ordinary       comments in waking hours, replies minutes to hours later, own wording
  automated      comments around the clock, replies within seconds, templated text
  ring members   human rhythm, but a ring turns up in the same threads within
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
        labels[f"ordinary{n:04d}"] = "person"
        ordinary_comments(f"ordinary{n:04d}", int(rng.integers(12, 60)))

    templates = [" ".join(rng.choice(WORDS, size=18)) for _ in range(5)]
    for n in range(n_automated):
        labels[f"automated{n:04d}"] = "program"
        for i in rng.choice(n_threads, size=int(rng.integers(60, 150)), replace=False):
            body = str(rng.choice(templates)) + " " + str(rng.choice(WORDS))
            add(f"automated{n:04d}", posts[i], posts[i]["created_utc"] + rng.integers(5, 25), body)

    # Harder cases, so the error rates in the demo are not all zero.
    groups = {}
    for n in range(20):          # software that hides: human-like delays, own wording, pauses 4 hours a day
        name = f"evasive{n:04d}"
        labels[name], groups[name] = "program", ("red_team", "slow, sleeping software")
        wake_hour = int(rng.integers(0, 24))
        for i in rng.choice(n_threads, size=int(rng.integers(40, 90)), replace=False):
            when = posts[i]["created_utc"] + rng.lognormal(np.log(600), 1.0)
            hours_awake = ((when // 3600) - wake_hour) % 24
            if hours_awake < 20:
                add(name, posts[i], when, own_words())
    for n in range(40):          # people who look like software: awake at all hours, quick to reply
        name = f"poweruser{n:04d}"
        labels[name], groups[name] = "person", ("negative_control", "power users")
        for i in rng.choice(n_threads, size=int(rng.integers(50, 90)), replace=False):
            add(name, posts[i], posts[i]["created_utc"] + rng.lognormal(np.log(1200), 1.0), own_words())

    for ring in range(n_rings):
        members = [f"ring{ring}member{m:02d}" for m in range(ring_size)]
        labels.update({m: "person" for m in members})     # people; the ring is for the cluster detector
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
    pd.DataFrame({"author_hash": list(labels), "who_types": list(labels.values()),
                  "changed_hands": "no_sign", "confidence": 3, "evidence": "synthetic: generated with this label"}
                 ).to_csv(label_file, index=False)
    pd.DataFrame([{"author_hash": a, "kind": kind, "group": group} for a, (kind, group) in groups.items()]
                 ).to_csv(paths(cfg)["groups"], index=False)
    return len(posts), len(comments), len(labels)
