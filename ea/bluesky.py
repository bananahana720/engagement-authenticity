"""Stage 1 for Bluesky: listen to the public post stream, then fetch on-topic threads.

Two steps in one run, so that nothing identifying is ever written:

  1. LISTEN. Read Jetstream (Bluesky's public JSON stream of every new post, no
     account needed) for a set number of minutes. Count how many posts match each
     topic's keywords, and remember in memory which threads the collect topic touched.
  2. FETCH. Ask the public Bluesky API for each remembered thread, cut every post
     down to the minimum, replace account and post identifiers by keyed hashes,
     and write the same files the Reddit collector writes.

A Bluesky post address contains its author's permanent identifier, so post ids are
hashed as well as author ids. Raw records and raw addresses never touch disk.

While listening, the run also watches for two things, and acts on both at the end:

  - posts their authors delete: removed from the collected files;
  - accounts that declare they do not allow their posts in bulk datasets: never
    stored, and anything already stored from them is removed. Bluesky's proposal
    for this declaration is still a draft, so today the stream carries none. The
    check is in place for the day it does.

Moderation labels on an account (spam, inauthentic and so on) are kept in a separate
file, as possible ground truth. They are never used as a feature.

See docs/SOURCES.md for the terms this is built around.
"""

import json
import re
import time
from collections import Counter

import pandas as pd
import requests

from ea.collect import USER_AGENT, topic_pattern, write_line
from ea.config import paths
from ea.pseudonymize import load_key, pseudonym

POST = "app.bsky.feed.post"
COMMUNITY = "bluesky"       # fills the "subreddit" column the later stages group by

# Accounts that say they are bots: a handle whose first part ends in -bot or _bot,
# or a "bot" label the account put on itself. Narrow on purpose, like the Reddit rule.
DISCLOSED_BOT_HANDLE = re.compile(r"(?i)[-_]bot$")

# Bluesky's own moderation service. Its identifier is stored as is. Any other labeler
# could be run by one person, so its identifier is hashed like an account's.
BLUESKY_MODERATION = "did:plc:ar7c4by46qjdydhdevvrndac"


# ---------- reading the stream ----------

def parse_event(event):
    """Return (operation, kind of record, account, record address, record), else None.

    Jetstream has two message layouts. The newer one wraps everything in "payload"
    with the commit fields at the top; the older one nests them under "commit".
    """
    payload = event.get("payload", event)
    commit = payload.get("commit") or payload
    kind, account = commit.get("collection"), payload.get("did")
    if not kind or not account:
        return None
    address = f"at://{account}/{kind}/{commit.get('rkey')}"
    return commit.get("operation"), kind, account, address, commit.get("record") or {}


def stream_address(url, kinds):
    """Add the kinds of record wanted to an address that ends in its filter name, "...?collections="."""
    name = url.rsplit("?", 1)[-1]
    return url + f"&{name}".join(kinds)


def allows_bulk_datasets(declaration):
    """False only when an account's reuse declaration says no to bulk datasets.

    Follows Bluesky's draft "User Intents for Data Reuse": a missing answer means
    the account has not stated a preference.
    """
    return (declaration.get("bulkDataset") or {}).get("allow") is not False


def stream_events(urls, seconds, kinds=(POST,), connect=None):
    """Yield parsed events of the wanted kinds from Jetstream until the time is up.

    Tries each address in turn and reconnects when a connection drops.
    """
    if connect is None:
        from websockets.sync.client import connect
    deadline = time.monotonic() + seconds
    failures = 0
    while time.monotonic() < deadline and failures < 3 * len(urls):
        url = stream_address(urls[failures % len(urls)], kinds)
        try:
            with connect(url, open_timeout=20, user_agent_header=USER_AGENT) as socket:
                while time.monotonic() < deadline:
                    parsed = parse_event(json.loads(socket.recv(timeout=30)))
                    if parsed:
                        failures = 0
                        yield parsed
        except Exception as problem:            # dropped or refused: try the next address
            failures += 1
            print(f"Stream problem on {url}: {type(problem).__name__}: {problem}")
            time.sleep(2)
    if failures >= 3 * len(urls):
        raise ConnectionError("Could not read the Bluesky stream from any configured address.")


def listen(cfg, seconds, events=None):
    """Step 1. Returns (volume table, thread addresses busiest first, deleted post ids,
    accounts that opted out of bulk datasets, accounts that withdrew an opt-out).

    Thread addresses stay in memory only. Posts and accounts are returned already hashed.
    """
    key = load_key()
    patterns = {name: topic_pattern(words) for name, words in cfg["topics"].items()}
    # Words that only count in capitals, such as "ICE" (the agency, not frozen water).
    exact = {name: re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")
             for name, words in cfg.get("topics_capitals_only", {}).items()}
    collect_topic = cfg["bluesky"]["collect_topic"]
    counts = Counter()
    threads = Counter()         # thread address -> on-topic posts seen in it
    deleted, opted_out, opted_in = set(), set(), set()
    reuse_kinds = cfg["bluesky"].get("reuse_declarations", [])
    started = time.time()
    per_minute = Counter()      # all new posts in each minute of listening, to show bursts

    live = events or stream_events(cfg["bluesky"]["stream_urls"], seconds, [POST] + reuse_kinds)
    for operation, kind, account, address, record in live:
        if kind in reuse_kinds:
            says_no = operation != "delete" and not allows_bulk_datasets(record)
            (opted_out if says_no else opted_in).add(pseudonym(account, key))
            (opted_in if says_no else opted_out).discard(pseudonym(account, key))
            continue
        if kind != POST:
            continue
        if operation == "delete":
            deleted.add(pseudonym(address, key))
            continue
        if operation != "create":
            continue
        is_reply = "reply" in record
        counts["all", is_reply] += 1
        per_minute[int(time.time() - started) // 60] += 1
        text = record.get("text", "")
        for name, pattern in patterns.items():
            if pattern.search(text) or (name in exact and exact[name].search(text)):
                counts[name, is_reply] += 1
                if name == collect_topic:
                    root = record["reply"]["root"]["uri"] if is_reply else address
                    threads[root] += 1

    minutes = max((time.time() - started) / 60, 1e-9) if events is None else seconds / 60
    busiest_first = [a for a, _ in threads.most_common()]
    volumes = volume_table(counts, list(patterns), minutes)
    if events is None and per_minute:
        tally = sorted(per_minute.values())
        print(f"All posts per minute: quietest {tally[0]}, middle {tally[len(tally) // 2]}, busiest {tally[-1]}.")
    return volumes, busiest_first, deleted, opted_out, opted_in


def volume_table(counts, topics, minutes):
    """One row per topic: posts seen, and what that rate would mean over a day."""
    total = counts["all", False] + counts["all", True]
    rows = []
    for name in ["all"] + topics:
        starts, replies = counts[name, False], counts[name, True]
        rows.append({
            "topic": name, "minutes_listened": round(minutes, 1),
            "thread_starts": starts, "replies": replies, "posts": starts + replies,
            "share_of_all_posts": round((starts + replies) / total, 5) if total else 0.0,
            "posts_per_day_at_this_rate": round((starts + replies) / minutes * 1440),
        })
    return pd.DataFrame(rows)


# ---------- fetching threads and cutting them down ----------

def fetch_thread(session, api_base, address, tries=3):
    """One thread from the public API, or None when it is gone or unavailable."""
    url = f"{api_base}/xrpc/app.bsky.feed.getPostThread"
    params = {"uri": address, "depth": 100, "parentHeight": 0}
    for attempt in range(tries):
        response = session.get(url, params=params, timeout=60, headers={"User-Agent": USER_AGENT})
        if response.status_code == 200:
            return response.json().get("thread")
        if response.status_code == 400:         # deleted, blocked or not found
            return None
        time.sleep(10 * (attempt + 1))          # rate limited or busy: back off
    return None


def walk(node):
    """Yield every visible post in a thread. Deleted and blocked posts have no "post"."""
    if not node or "post" not in node:
        return
    yield node["post"]
    for reply in node.get("replies") or []:
        yield from walk(reply)


def is_disclosed_bot(author):
    handle_start = str(author.get("handle", "")).split(".")[0]
    labels = {label.get("val") for label in author.get("labels") or []}
    return bool(DISCLOSED_BOT_HANDLE.search(handle_start)) or "bot" in labels


def minimize(post, key):
    """Turn one raw Bluesky post into ("post" or "comment", the small record we store).

    Times are when Bluesky's server first saw the post, not the time the posting
    app claimed, because an app can claim any time.
    """
    record = post.get("record") or {}
    small = {
        "id": pseudonym(post["uri"], key),
        "subreddit": COMMUNITY,
        "author_hash": pseudonym(post["author"]["did"], key),
        "disclosed_bot": is_disclosed_bot(post["author"]),
        "created_utc": int(pd.Timestamp(post.get("indexedAt") or record["createdAt"]).timestamp()),
        "removed": False,
    }
    reply = record.get("reply")
    if not reply:
        return "post", dict(small, title=record.get("text", ""))
    root, parent = reply["root"]["uri"], reply["parent"]["uri"]
    small["link_id"] = pseudonym(root, key)
    # The later stages tell thread starts from replies by these Reddit-style prefixes.
    small["parent_id"] = ("t3_" if parent == root else "t1_") + pseudonym(parent, key)
    small["body"] = record.get("text", "")
    return "comment", small


def account_labels(post, key):
    """Moderation labels on a post's author, as small records. Withdrawn labels are skipped.

    A label an account puts on itself is marked, because it is a statement, not a finding.
    """
    author = post["author"]
    return [{"author_hash": pseudonym(author["did"], key), "label": label.get("val"),
             "self_applied": label.get("src") == author["did"],
             "labeler": labeler_name(label.get("src", ""), author["did"], key)}
            for label in author.get("labels") or [] if not label.get("neg")]


def labeler_name(source, author, key):
    if source == author:
        return ""
    return "bluesky-moderation" if source == BLUESKY_MODERATION else pseudonym(source, key)


# ---------- the whole run ----------

def collect(cfg, minutes, session=None, events=None):
    """Listen, then fetch and store the busiest on-topic threads. Returns the volume table."""
    session = session or requests.Session()
    key = load_key()
    settings = cfg["bluesky"]
    volumes, threads, deleted, opted_out, opted_in = listen(cfg, minutes * 60, events)

    report = paths(cfg)["volumes"]
    report.parent.mkdir(parents=True, exist_ok=True)
    volumes.to_csv(report, index=False)

    folder = paths(cfg)["collected"] / COMMUNITY
    folder.mkdir(parents=True, exist_ok=True)
    files = {"post": folder / "posts.jsonl", "comment": folder / "comments.jsonl"}
    labels_path, opt_out_path = folder / "labels.jsonl", folder / "opted_out.jsonl"

    # Everyone who has ever opted out, minus those who have since withdrawn it.
    opted_out = (stored_values(opt_out_path, "author_hash") | opted_out) - opted_in
    opt_out_path.write_text("".join(json.dumps({"author_hash": a}) + "\n" for a in sorted(opted_out)))

    seen = {kind: stored_values(path, "id") for kind, path in files.items()}
    seen_labels = {json.dumps(row, sort_keys=True) for row in read_lines(labels_path)}
    written = Counter()
    with open(files["post"], "a") as posts_file, open(files["comment"], "a") as comments_file, \
         open(labels_path, "a") as labels_file:
        out = {"post": posts_file, "comment": comments_file}
        for address in threads[:settings["max_threads"]]:
            for raw in walk(fetch_thread(session, settings["api_base"], address)):
                kind, small = minimize(raw, key)
                if small["author_hash"] in opted_out or small["id"] in deleted:
                    continue
                if small["id"] not in seen[kind]:
                    write_line(out[kind], small)
                    seen[kind].add(small["id"])
                    written[kind] += 1
                for label in account_labels(raw, key):
                    if json.dumps(label, sort_keys=True) not in seen_labels:
                        write_line(labels_file, label)
                        seen_labels.add(json.dumps(label, sort_keys=True))
            time.sleep(settings["seconds_between_requests"])

    forgotten = sum(forget(path, "id", deleted) for path in files.values())
    withdrawn = sum(forget(path, "author_hash", opted_out) for path in [*files.values(), labels_path])
    print(f"Threads touched by the topic: {len(threads)}. Stored {written['post']} thread starts "
          f"and {written['comment']} replies. Removed {forgotten} posts their authors deleted "
          f"and {withdrawn} records from accounts that opted out of datasets.")
    return volumes


def read_lines(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stored_values(path, field):
    return {row[field] for row in read_lines(path)}


def forget(path, field, values):
    """Remove stored records whose field is in values. Returns how many went."""
    rows = read_lines(path)
    kept = [row for row in rows if row[field] not in values]
    if len(kept) < len(rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in kept))
    return len(rows) - len(kept)
