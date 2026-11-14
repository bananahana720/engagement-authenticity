"""Fill in the threads the 2018 seed accounts touched, so the mixing test has everyone's replies.

The 2018 copy holds only what the seeds wrote. The mixing test needs the
replies of everyone else in the same threads, or it cannot tell a seed's
choice of partner from chance. This stage takes every thread a seed posted
or commented in, and fetches from Arctic Shift:

  - each thread's post (author and time), 500 ids per call,
  - every comment in the thread, about a thousand per call.

Records go through collect.minimize, so usernames are hashed with the same key
as the seeds, and comment text and titles are dropped. Output: data/ira2018/collected/threads/.
Resumable: a thread listed in threads_done.txt is not fetched again.
"""

import requests

from ea.collect import already_collected, fetch_pages, get_json, minimize, write_line
from ea.config import check_collection_allowed, paths
from ea.pseudonymize import load_key

import pandas as pd


def thread_ids(cfg):
    """Every thread a seed commented in, plus every seed post that drew comments."""
    raw = cfg["data_dir"] / "raw"
    comments = pd.read_csv(raw / "comments.csv", usecols=["link_id"])
    posts = pd.read_csv(raw / "submissions.csv", usecols=["id", "num_comments", "author.name"])
    posts = posts[(posts["author.name"].str.lower() != "reddit") & (posts["num_comments"] > 0)]
    ids = set(comments["link_id"].str.split("_", n=1).str[-1]) | set(posts["id"])
    return sorted(ids)


def collect_threads(cfg, session=None, limit=None):
    check_collection_allowed(cfg)
    session = session or requests.Session()
    key = load_key()
    base = cfg["collect"]["api_base"]
    wait = cfg["collect"]["seconds_between_requests"]
    folder = paths(cfg)["collected"] / "threads"
    folder.mkdir(parents=True, exist_ok=True)
    ids = thread_ids(cfg)[:limit]

    have_posts = already_collected(folder / "posts.jsonl")
    missing = [i for i in ids if i not in have_posts]
    with open(folder / "posts.jsonl", "a") as out:
        for start in range(0, len(missing), 500):
            batch = missing[start:start + 500]
            for raw in get_json(session, f"{base}/api/posts/ids", {"ids": ",".join(batch)}):
                small = minimize(raw, key, "post")
                if small is None:          # author deleted: keep the thread anchor, no author
                    small = {"id": raw["id"], "subreddit": raw["subreddit"], "author_hash": "",
                             "disclosed_bot": False, "created_utc": int(raw["created_utc"]),
                             "removed": True, "title": ""}
                small["title"] = ""        # titles are not needed here
                write_line(out, small)

    done_file = folder / "threads_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    todo = [i for i in ids if i not in done]
    with open(folder / "comments.jsonl", "a") as out, open(done_file, "a") as log:
        for n, thread in enumerate(todo, 1):
            for raw in fetch_pages(session, f"{base}/api/comments/search", {"link_id": thread}, wait):
                small = minimize(raw, key, "comment")
                if small is not None:
                    small["body"] = ""     # the network needs who answered whom, not what was said
                write_line(out, small)
            log.write(thread + "\n")       # written last: a listed thread is complete
            log.flush()
            if n % 250 == 0:
                print(f"{n + len(done)}/{len(ids)} threads")
    return len(ids)
