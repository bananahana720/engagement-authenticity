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
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone

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
    failures, heard_anything = 0, False
    while time.monotonic() < deadline and (heard_anything or failures < 3 * len(urls)):
        url = stream_address(urls[failures % len(urls)], kinds)
        try:
            with connect(url, open_timeout=20, user_agent_header=USER_AGENT) as socket:
                while time.monotonic() < deadline:
                    parsed = parse_event(json.loads(socket.recv(timeout=30)))
                    if parsed:
                        failures, heard_anything = 0, True
                        yield parsed
        except Exception as problem:            # dropped or refused: try the next address
            failures += 1
            say(f"Stream problem on {url}: {type(problem).__name__}: {problem}")
            # A stream that worked earlier is worth waiting for; one that never worked is not.
            time.sleep(min(2 * failures, 60))
    if not heard_anything and failures >= 3 * len(urls):
        raise ConnectionError("Could not read the Bluesky stream from any configured address.")


def listen(cfg, seconds, events=None):
    """Step 1. Returns (volume table, on-topic posts seen per thread address, deleted post ids,
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
    volumes = volume_table(counts, list(patterns), minutes)
    if events is None and per_minute:
        tally = sorted(per_minute.values())
        say(f"All posts per minute: quietest {tally[0]}, middle {tally[len(tally) // 2]}, busiest {tally[-1]}.")
    return volumes, threads, deleted, opted_out, opted_in


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

def fetch_thread(session, api_base, address, labelers=(), tries=3):
    """One thread from the public API, or None when it is gone or unavailable.

    Naming labeler services asks the API to attach their labels as well as Bluesky's own.
    """
    url = f"{api_base}/xrpc/app.bsky.feed.getPostThread"
    params = {"uri": address, "depth": 100, "parentHeight": 0}
    headers = {"User-Agent": USER_AGENT}
    if labelers:
        headers["atproto-accept-labelers"] = ", ".join(labelers)
    for attempt in range(tries):
        try:
            response = session.get(url, params=params, timeout=60, headers=headers)
        except requests.RequestException:       # network dropped: wait and try again
            time.sleep(10 * (attempt + 1))
            continue
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


def account_labels(post, key, named=None):
    """Moderation labels on a post's author, as small records. Withdrawn labels are skipped.

    A label an account puts on itself is marked, because it is a statement, not a finding.
    """
    author = post["author"]
    return [{"author_hash": pseudonym(author["did"], key), "label": label.get("val"),
             "self_applied": label.get("src") == author["did"],
             "labeler": labeler_name(label.get("src", ""), author["did"], key, named)}
            for label in author.get("labels") or [] if not label.get("neg")]


def named_labelers(cfg):
    """Labeler service identifier -> the short name it is stored under.

    Bluesky's own service and the public labeler services listed in the config are
    stored by name. Any other labeler could be one person, so it is stored hashed.
    """
    listed = cfg["bluesky"].get("labelers", {})
    return {BLUESKY_MODERATION: "bluesky-moderation", **{did: name for name, did in listed.items()}}


def labeler_name(source, author, key, named=None):
    if source == author:
        return ""
    named = named or {BLUESKY_MODERATION: "bluesky-moderation"}
    return named.get(source) or pseudonym(source, key)


# ---------- choosing and storing threads ----------

def say(text):
    """Print with the time, at once, so a log file shows progress while a long run goes on."""
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {text}", flush=True)


def pick_threads(threads, settings, rng, already=frozenset(), key=None):
    """Choose which on-topic threads to fetch. Returns [(address, how it was picked)].

    The busiest few are taken for their size. The rest are drawn at random, so that
    rates can be measured on the random ones: busy threads alone over-state everything.
    Without a `random_threads` setting, only the busiest are taken (the first samples
    were collected that way).
    """
    fresh = [a for a, _ in threads.most_common() if key is None or pseudonym(a, key) not in already]
    if "random_threads" not in settings:
        return [(a, "busiest") for a in fresh[:settings["max_threads"]]]
    n_busiest = settings.get("busiest_threads", 0)
    busiest, rest = fresh[:n_busiest], fresh[n_busiest:]
    drawn = rng.sample(rest, min(settings["random_threads"], len(rest)))
    return [(a, "busiest") for a in busiest] + [(a, "random") for a in drawn]


def store_threads(cfg, picks, folder, session, deleted=frozenset(), opted_out=frozenset(), cycle=0):
    """Fetch each picked thread, cut it down, and append it to the files in `folder`.

    Returns how many thread starts and replies were written.
    """
    key = load_key()
    settings = cfg["bluesky"]
    named = named_labelers(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    files = {"post": folder / "posts.jsonl", "comment": folder / "comments.jsonl"}
    labels_path = folder / "labels.jsonl"
    seen = {kind: stored_values(path, "id") for kind, path in files.items()}
    seen_labels = {json.dumps(row, sort_keys=True) for row in read_lines(labels_path)}
    written = Counter()
    with open(files["post"], "a") as posts_file, open(files["comment"], "a") as comments_file, \
         open(labels_path, "a") as labels_file:
        out = {"post": posts_file, "comment": comments_file}
        for address, how in picks:
            thread = fetch_thread(session, settings["api_base"], address, list(named))
            for raw in walk(thread):
                kind, small = minimize(raw, key)
                if small["author_hash"] in opted_out or small["id"] in deleted:
                    continue
                if kind == "post":
                    small.update(picked=how, cycle=cycle)       # lets rates be weighted later
                if small["id"] not in seen[kind]:
                    write_line(out[kind], small)
                    seen[kind].add(small["id"])
                    written[kind] += 1
                for label in account_labels(raw, key, named):
                    if json.dumps(label, sort_keys=True) not in seen_labels:
                        write_line(labels_file, label)
                        seen_labels.add(json.dumps(label, sort_keys=True))
            time.sleep(settings["seconds_between_requests"])
    return written["post"], written["comment"]


def remember_opt_outs(cfg, opted_out, opted_in):
    """Everyone who has ever opted out of datasets, minus those who have since withdrawn it."""
    path = paths(cfg)["collected"] / COMMUNITY / "opted_out.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    everyone = (stored_values(path, "author_hash") | opted_out) - opted_in
    path.write_text("".join(json.dumps({"author_hash": a}) + "\n" for a in sorted(everyone)))
    return everyone


def apply_removals(cfg, deleted, opted_out):
    """Drop deleted posts and opted-out accounts from every Bluesky folder. Returns (posts, records) removed."""
    forgotten = withdrawn = 0
    for folder in sorted(paths(cfg)["collected"].glob(COMMUNITY + "*")):
        for name in ("posts.jsonl", "comments.jsonl"):
            forgotten += forget(folder / name, "id", deleted)
        for name in ("posts.jsonl", "comments.jsonl", "labels.jsonl"):
            withdrawn += forget(folder / name, "author_hash", opted_out)
    return forgotten, withdrawn


# ---------- one short run ----------

def collect(cfg, minutes, session=None, events=None):
    """Listen, then fetch and store on-topic threads straight away. Returns the volume table."""
    session = session or requests.Session()
    volumes, threads, deleted, opted_out, opted_in = listen(cfg, minutes * 60, events)

    report = paths(cfg)["volumes"]
    report.parent.mkdir(parents=True, exist_ok=True)
    volumes.to_csv(report, index=False)

    opted_out = remember_opt_outs(cfg, opted_out, opted_in)
    picks = pick_threads(threads, cfg["bluesky"], random.Random(0))
    starts, replies = store_threads(cfg, picks, paths(cfg)["collected"] / COMMUNITY, session, deleted, opted_out)
    forgotten, withdrawn = apply_removals(cfg, deleted, opted_out)
    say(f"Threads touched by the topic: {len(threads)}. Stored {starts} thread starts "
        f"and {replies} replies. Removed {forgotten} posts their authors deleted "
        f"and {withdrawn} records from accounts that opted out of datasets.")
    return volumes


# ---------- a long run, in cycles ----------

def run(cfg, hours, session=None, cycles=None):
    """Listen for days. Each cycle: listen, choose threads, and fetch the threads chosen a day ago.

    Waiting a day before fetching lets a thread collect its replies first. Chosen thread
    addresses wait in memory only, so stopping the run loses the threads not yet fetched,
    and nothing identifying is ever on disk. Each day's records go in their own folder.
    Stream drops are ridden out: the run waits a minute and listens again.

    `cycles` is for tests: a list of event lists, one per cycle, in place of the live stream.
    """
    session = session or requests.Session()
    key = load_key()
    settings = cfg["bluesky"]
    cycle_seconds = settings["cycle_minutes"] * 60
    wait = 0 if cycles is not None else settings["fetch_after_hours"] * 3600
    rng = random.Random()
    waiting, chosen = [], set()         # (fetch time, address, how picked); hashed addresses ever chosen
    end = time.time() + hours * 3600
    cycle = 0
    log = paths(cfg)["volumes"].with_name("cycles.csv")
    log.parent.mkdir(parents=True, exist_ok=True)

    while (cycle < len(cycles)) if cycles is not None else (time.time() < end):
        started = int(time.time())
        try:
            events = cycles[cycle] if cycles is not None else None
            volumes, threads, deleted, opted_out, opted_in = listen(cfg, min(cycle_seconds, end - started), events)
        except ConnectionError as problem:
            say(f"{problem} Waiting a minute.")
            time.sleep(60)
            continue
        cycle += 1

        picks = pick_threads(threads, settings, rng, chosen, key)
        chosen.update(pseudonym(address, key) for address, _ in picks)
        waiting += [(started + wait, address, how) for address, how in picks]
        due = [(a, how) for when, a, how in waiting if when <= time.time()]
        waiting = [item for item in waiting if item[0] > time.time()]

        opted_out = remember_opt_outs(cfg, opted_out, opted_in)
        starts = replies = 0
        if megabytes(cfg["data_dir"]) < settings["max_disk_mb"]:
            folder = paths(cfg)["collected"] / f"{COMMUNITY}-{datetime.now(timezone.utc):%Y-%m-%d}"
            starts, replies = store_threads(cfg, due, folder, session, deleted, opted_out, started)
        elif due:
            say(f"Disk limit of {settings['max_disk_mb']} MB reached: {len(due)} threads were not fetched.")
        forgotten, withdrawn = apply_removals(cfg, deleted, opted_out)

        volumes.insert(0, "cycle", started)
        volumes.to_csv(paths(cfg)["volumes"], mode="a", index=False, header=not paths(cfg)["volumes"].exists())
        row = pd.DataFrame([{
            "cycle": started, "threads_touched": len(threads),
            "picked_busiest": sum(how == "busiest" for _, how in picks),
            "picked_random": sum(how == "random" for _, how in picks),
            "fetched_threads": len(due), "stored_thread_starts": starts, "stored_replies": replies,
            "removed_deleted": forgotten, "removed_opted_out": withdrawn, "threads_waiting": len(waiting)}])
        row.to_csv(log, mode="a", index=False, header=not log.exists())
        say(f"Cycle {cycle}: {len(threads)} threads touched, {len(picks)} chosen, {len(due)} fetched, "
            f"{starts} thread starts and {replies} replies stored, {len(waiting)} threads waiting.")

    # The run is over: fetch what is still waiting, young as it is, rather than lose it.
    if waiting and megabytes(cfg["data_dir"]) < settings["max_disk_mb"]:
        folder = paths(cfg)["collected"] / f"{COMMUNITY}-{datetime.now(timezone.utc):%Y-%m-%d}"
        starts, replies = store_threads(cfg, [(a, how) for _, a, how in waiting], folder, session,
                                        opted_out=stored_values(paths(cfg)["collected"] / COMMUNITY / "opted_out.jsonl",
                                                                "author_hash"), cycle=int(time.time()))
        say(f"End of run: fetched the {len(waiting)} threads still waiting: {starts} thread starts, {replies} replies.")
    return log


def megabytes(folder):
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6 if folder.exists() else 0.0


def read_lines(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stored_values(path, field):
    return {row[field] for row in read_lines(path)}


def forget(path, field, values):
    """Remove stored records whose field is in values. Returns how many went.

    Reads line by line, so a large file never has to fit in memory.
    """
    if not path.exists() or not values:
        return 0
    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            if line.strip() and json.loads(line)[field] in values:
                dropped += 1
            else:
                kept.append(line)
    if dropped:
        path.write_text("".join(kept))
    return dropped
