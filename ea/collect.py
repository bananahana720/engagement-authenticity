"""Stage 1, collect: fetch topic threads from Arctic Shift and keep the minimum.

For each subreddit:
  1. page through its posts for the time window,
  2. keep the posts that are on topic (by title, unless the subreddit is all on topic),
  3. fetch every comment under each kept post.

Records are cut down to the fields the features need, and usernames are
replaced by a keyed hash, BEFORE anything is written. Raw records never touch disk.

The same code can read a downloaded dump file instead of the API (see read_dump).
"""

import json
import re
import time

import pandas as pd
import requests

from ea.config import all_subreddits, paths
from ea.pseudonymize import NO_AUTHOR, is_disclosed_bot, load_key, pseudonym

USER_AGENT = "engagement-authenticity-research/0.1"


# ---------- cutting a raw record down to the minimum ----------

def minimize(record, key, kind):
    """Turn one raw Reddit post or comment into the small record we store.

    Returns None when the record has no usable author.
    """
    author = record.get("author")
    if author in NO_AUTHOR:
        return None
    meta = record.get("_meta") or {}
    removed = bool(meta.get("was_deleted_later") or meta.get("removal_type"))
    small = {
        "id": record["id"],
        "subreddit": record["subreddit"],
        "author_hash": pseudonym(author, key),
        "disclosed_bot": is_disclosed_bot(author),
        "created_utc": int(record["created_utc"]),
        "removed": removed,
    }
    if kind == "post":
        small["title"] = record.get("title", "")
    else:
        small["link_id"] = strip_prefix(record["link_id"])
        small["parent_id"] = record.get("parent_id") or record["link_id"]
        # Text that was removed or deleted on Reddit is not kept, only the fact of removal.
        small["body"] = "" if removed else record.get("body", "")
    return small


def strip_prefix(reddit_id):
    """'t3_abc' -> 'abc'."""
    return str(reddit_id).split("_", 1)[-1]


def topic_pattern(keywords):
    """One case-insensitive regex that matches any keyword on word boundaries."""
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)


# ---------- talking to the Arctic Shift API ----------

def fetch_pages(session, url, params, wait_seconds):
    """Yield every record for a query, oldest first, one page at a time.

    The API returns up to 1000 records per call, sorted by time. We ask for the
    next page starting one second before the last timestamp seen, so records that
    share that second are not lost, and skip ids we already have.
    """
    params = dict(params, sort="asc", limit="auto")
    seen_ids = set()
    while True:
        page = get_json(session, url, params)
        new = [r for r in page if r["id"] not in seen_ids]
        if not new:
            return
        yield from new
        seen_ids.update(r["id"] for r in new)
        params["after"] = int(new[-1]["created_utc"]) - 1
        time.sleep(wait_seconds)


def get_json(session, url, params, tries=5):
    """GET with patient retries. Returns the list under the response's "data" key."""
    for attempt in range(tries):
        response = session.get(url, params=params, timeout=60, headers={"User-Agent": USER_AGENT})
        if response.status_code == 200:
            return response.json().get("data") or []
        time.sleep(10 * (attempt + 1))      # rate limited or server busy: back off
    response.raise_for_status()
    raise RuntimeError(f"No answer from {url} after {tries} tries")


def measure_volumes(cfg, session=None):
    """Monthly comment and post counts per subreddit. Cheap: one call per number."""
    session = session or requests.Session()
    base = cfg["collect"]["api_base"]
    rows = []
    for sub in all_subreddits(cfg):
        for kind in ("comments", "posts"):
            params = {"key": f"r/{sub}/{kind}/count", "precision": "month",
                      "after": cfg["start"], "before": cfg["end"]}
            for point in get_json(session, f"{base}/api/time_series", params):
                rows.append({"subreddit": sub, "kind": kind, **point})
            time.sleep(cfg["collect"]["seconds_between_requests"])
    out = paths(cfg)["volumes"]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def collect(cfg, session=None, only=None):
    """Download on-topic threads for every subreddit in the config."""
    session = session or requests.Session()
    key = load_key()
    base = cfg["collect"]["api_base"]
    wait = cfg["collect"]["seconds_between_requests"]
    pattern = topic_pattern(cfg["topic"]["title_keywords"])
    window = {"after": cfg["start"], "before": cfg["end"]}

    for sub in only or all_subreddits(cfg):
        folder = paths(cfg)["collected"] / sub
        folder.mkdir(parents=True, exist_ok=True)
        keep_all = sub in cfg["subreddits"]["all_threads"]
        done = already_collected(folder / "posts.jsonl")

        with open(folder / "posts.jsonl", "a") as posts_file, \
             open(folder / "comments.jsonl", "a") as comments_file:
            post_query = dict(window, subreddit=sub)
            for raw_post in fetch_pages(session, f"{base}/api/posts/search", post_query, wait):
                if raw_post["id"] in done:
                    continue                                    # resume after a stopped run
                if not (keep_all or pattern.search(raw_post.get("title", ""))):
                    continue
                comment_query = {"link_id": raw_post["id"]}
                for raw_comment in fetch_pages(session, f"{base}/api/comments/search", comment_query, wait):
                    write_line(comments_file, minimize(raw_comment, key, "comment"))
                # The post is written last, so a post on disk means its comments are complete.
                write_line(posts_file, minimize(raw_post, key, "post") or deleted_post(raw_post))
        print(f"r/{sub}: done")


def deleted_post(raw_post):
    """A post whose author is gone still anchors its thread: keep id and time only."""
    return {"id": raw_post["id"], "subreddit": raw_post["subreddit"], "author_hash": "",
            "disclosed_bot": False, "created_utc": int(raw_post["created_utc"]),
            "removed": True, "title": ""}


def already_collected(posts_path):
    if not posts_path.exists():
        return set()
    with open(posts_path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def write_line(file, record):
    if record is not None:
        file.write(json.dumps(record) + "\n")


# ---------- reading a downloaded dump instead of the API ----------

def read_dump(path):
    """Yield raw records from a .jsonl or .zst dump file (monthly dumps are .zst)."""
    if str(path).endswith(".zst"):
        import io
        import zstandard
        with open(path, "rb") as f:
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(f)
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                yield json.loads(line)
    else:
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def collect_from_dumps(cfg, posts_path, comments_path):
    """Same output as collect(), from one posts dump and one comments dump."""
    key = load_key()
    pattern = topic_pattern(cfg["topic"]["title_keywords"])
    subs = {s.lower(): s for s in all_subreddits(cfg)}
    keep_all = {s.lower() for s in cfg["subreddits"]["all_threads"]}
    start = pd.Timestamp(cfg["start"], tz="UTC").timestamp()
    end = pd.Timestamp(cfg["end"], tz="UTC").timestamp()

    kept_posts = set()
    for raw in read_dump(posts_path):
        sub = str(raw.get("subreddit", "")).lower()
        if sub not in subs or not start <= int(raw["created_utc"]) < end:
            continue
        if sub in keep_all or pattern.search(raw.get("title", "")):
            kept_posts.add(raw["id"])
            append(cfg, subs[sub], "posts", minimize(raw, key, "post") or deleted_post(raw))

    for raw in read_dump(comments_path):
        sub = str(raw.get("subreddit", "")).lower()
        if sub in subs and strip_prefix(raw.get("link_id", "")) in kept_posts:
            append(cfg, subs[sub], "comments", minimize(raw, key, "comment"))


def append(cfg, sub, kind, record):
    folder = paths(cfg)["collected"] / sub
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{kind}.jsonl", "a") as f:
        write_line(f, record)


# ---------- loading what was collected ----------

def load_collected(cfg):
    """Return (posts, comments) DataFrames for everything collected so far."""
    root = paths(cfg)["collected"]
    posts = read_all(root.glob("*/posts.jsonl"))
    comments = read_all(root.glob("*/comments.jsonl"))
    for table in (posts, comments):
        table.drop_duplicates("id", inplace=True)
        table["time"] = pd.to_datetime(table["created_utc"], unit="s", utc=True)
    return posts, comments


def read_all(files):
    frames = [pd.read_json(f, lines=True, dtype={"id": str, "link_id": str, "author_hash": str})
              for f in files if f.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError("Nothing collected yet. Run the collect stage first.")
    return pd.concat(frames, ignore_index=True)
