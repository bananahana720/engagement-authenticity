"""Tests on small hand-made inputs with known answers, plus one end-to-end run."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ea import clusters, collect, evaluate, features, label, metrics, model, seeds, synthetic
from ea.config import load_config, paths
from ea.pseudonymize import is_disclosed_bot, pseudonym


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("EA_PSEUDONYM_KEY", "test-key")
    cfg = load_config(Path(__file__).parent.parent / "config" / "immigration.toml")
    cfg["data_dir"], cfg["reports_dir"] = tmp_path / "data", tmp_path / "reports"
    cfg["collect"]["seconds_between_requests"] = 0
    cfg["source"] = "synthetic"         # tests use made-up data; the real config says "reddit"
    return cfg


# ---------- pseudonyms ----------

def test_pseudonym_is_stable_and_depends_on_the_key():
    assert pseudonym("SomeUser", b"k1") == pseudonym("someuser", b"k1")
    assert pseudonym("SomeUser", b"k1") != pseudonym("SomeUser", b"k2")
    assert "someuser" not in pseudonym("SomeUser", b"k1")


def test_disclosed_bot_names():
    assert is_disclosed_bot("AutoModerator") and is_disclosed_bot("RemindMeBot") and is_disclosed_bot("news_bot")
    assert not is_disclosed_bot("talbot") and not is_disclosed_bot("Abbot")


# ---------- collect ----------

RAW_COMMENT = {"id": "c1", "subreddit": "immigration", "author": "RealName", "created_utc": 1767300000,
               "link_id": "t3_p1", "parent_id": "t3_p1", "body": "hello", "author_flair_text": "x"}


def test_minimize_drops_username_and_extra_fields():
    small = collect.minimize(RAW_COMMENT, b"k", "comment")
    assert "author" not in small and "author_flair_text" not in small
    assert "RealName" not in json.dumps(small)
    assert small["link_id"] == "p1" and small["body"] == "hello"


def test_minimize_drops_deleted_authors_and_removed_text():
    assert collect.minimize(dict(RAW_COMMENT, author="[deleted]"), b"k", "comment") is None
    removed = collect.minimize(dict(RAW_COMMENT, _meta={"was_deleted_later": True}), b"k", "comment")
    assert removed["removed"] and removed["body"] == ""


def test_topic_pattern_matches_whole_words_only():
    pattern = collect.topic_pattern(["ICE", "deportation"])
    assert pattern.search("ICE agents arrive") and pattern.search("New deportation flights")
    assert not pattern.search("Police service notice")


class FakeArcticShift:
    """Stands in for the API: answers from two small in-memory lists."""

    def __init__(self, posts, comments):
        self.posts, self.comments, self.calls = posts, comments, 0

    def get(self, url, params, **_):
        self.calls += 1
        if "posts" in url:
            rows = [p for p in self.posts if p["subreddit"] == params["subreddit"]]
        else:
            rows = [c for c in self.comments
                    if ("link_id" in params and c["link_id"] == "t3_" + params["link_id"])
                    or ("link_id" not in params and c["subreddit"] == params["subreddit"])]
        after = params.get("after")
        if isinstance(after, int):
            rows = [r for r in rows if r["created_utc"] > after]
        return FakeResponse(sorted(rows, key=lambda r: r["created_utc"])[:2])   # pages of 2


class FakeResponse:
    status_code = 200

    def __init__(self, rows):
        self.rows = rows

    def json(self):
        return {"data": self.rows}


def test_collect_keeps_only_on_topic_threads_and_resumes(cfg):
    t = 1767300000
    posts = [
        {"id": "p1", "subreddit": "politics", "author": "u1", "created_utc": t, "title": "ICE raid downtown"},
        {"id": "p2", "subreddit": "politics", "author": "u2", "created_utc": t + 10, "title": "Budget vote"},
        {"id": "p3", "subreddit": "politics", "author": "u3", "created_utc": t + 20, "title": "Deportation ruling"},
    ]
    comments = [dict(RAW_COMMENT, id=f"c{i}", subreddit="politics", link_id=f"t3_{post}", parent_id=f"t3_{post}",
                     created_utc=t + 100 + i)
                for i, post in enumerate(["p1", "p1", "p1", "p2", "p3"])]
    fake = FakeArcticShift(posts, comments)
    collect.collect(cfg, session=fake, only=["politics"])

    got_posts, got_comments = collect.load_collected(cfg)
    assert set(got_posts["id"]) == {"p1", "p3"}
    assert set(got_comments["id"]) == {"c0", "c1", "c2", "c4"}       # p1 needed two pages

    collect.collect(cfg, session=fake, only=["politics"])            # second run adds nothing
    assert len(collect.load_collected(cfg)[1]) == 4


def test_collect_takes_whole_subreddit_in_bulk_and_resumes(cfg):
    t = 1767300000
    posts = [{"id": f"p{i}", "subreddit": "immigration", "author": "u", "created_utc": t + i, "title": "Visa wait"}
             for i in range(3)]
    comments = [dict(RAW_COMMENT, id=f"c{i}", link_id=f"t3_p{i % 3}", parent_id=f"t3_p{i % 3}",
                     created_utc=t + 100 + i) for i in range(5)]
    fake = FakeArcticShift(posts, comments)
    collect.collect(cfg, session=fake, only=["immigration"])
    got_posts, got_comments = collect.load_collected(cfg)
    assert len(got_posts) == 3 and len(got_comments) == 5
    first_calls = fake.calls

    collect.collect(cfg, session=fake, only=["immigration"])            # resume: nothing new, few calls
    assert len(collect.load_collected(cfg)[1]) == 5
    assert fake.calls - first_calls <= 4


def test_topic_share_counts_on_topic_threads_without_storing_titles(cfg):
    t = 1767300000
    posts = [
        {"id": "p1", "subreddit": "politics", "created_utc": t, "title": "ICE raid downtown", "num_comments": 40},
        {"id": "p2", "subreddit": "politics", "created_utc": t + 10, "title": "Budget vote", "num_comments": 7},
    ]
    cfg["subreddits"] = {"all_threads": [], "filtered": ["politics"], "city": []}
    out = collect.measure_topic_share(cfg, session=FakeArcticShift(posts, []))
    share = pd.read_csv(out)
    assert share.loc[0, "posts"] == 2 and share.loc[0, "on_topic_posts"] == 1
    assert share.loc[0, "on_topic_comments_est"] == 40
    assert "ICE" not in out.read_text()


def test_seed_wiki_table_parses():
    content = "Suspicious Accounts\r\n\r\n|Username|Karma|\r\n|-|-|\r\n|u/Some_one|-16|\r\n|u/two-2|0|\r\n|u/Big|2,591|\r\n"
    assert seeds.parse_wiki_table(content) == [("Some_one", -16), ("two-2", 0), ("Big", 2591)]


# ---------- features ----------

def comment(id, author, t, thread="p1", parent="t3_p1", body="some ordinary words here"):
    return {"id": id, "author_hash": author, "created_utc": t, "link_id": thread, "parent_id": parent,
            "body": body, "subreddit": "s", "removed": False, "time": pd.Timestamp(t, unit="s", tz="UTC")}


def test_timing_features():
    hour = 3600
    rows = [comment(f"a{h}", "bot", h * hour) for h in range(24)]           # every hour of the day
    rows += [comment(f"b{h}", "human", h * hour) for h in (9, 10, 20)]      # three daytime hours
    table = pd.DataFrame(rows)
    posts = pd.DataFrame([{"id": "p1", "created_utc": 9 * hour - 20}])
    out = features.timing_features(features.add_reply_latency(table, posts), fast_reply_seconds=30)
    assert out.loc["bot", "active_hours"] == 24 and out.loc["bot", "longest_quiet_hours"] == 0
    assert out.loc["human", "active_hours"] == 3
    assert out.loc["human", "longest_quiet_hours"] == 12                    # 21:00 to 08:00, across midnight
    assert out.loc["human", "fast_reply_share"] == pytest.approx(1 / 3)     # only the 09:00 comment, 20 s after


def test_near_duplicates_need_different_accounts_and_close_times():
    text = "they keep saying the same thing about the border policy every single day now"
    reworded = text.replace("now", "today")
    table = pd.DataFrame([
        comment("c1", "x", 0, body=text),
        comment("c2", "y", 600, body=reworded),          # other account, 10 minutes later: a pair
        comment("c3", "x", 700, body=text),              # same account as c1: not a pair with c1
        comment("c4", "z", 9000, body=text),             # 2.5 hours later: too far apart
        comment("c5", "w", 100, body="a completely unrelated comment about something else entirely here"),
    ])
    pairs = {frozenset(p) for p in features.near_duplicate_pairs(table, minutes=60, min_jaccard=0.7)}
    assert pairs == {frozenset(("c1", "c2")), frozenset(("c2", "c3"))}


def test_co_engagement_counts_repeat_partners():
    rows = []
    for n in range(3):                                   # x and y together in three threads
        rows += [comment(f"x{n}", "x", n * 10000, thread=f"p{n}"), comment(f"y{n}", "y", n * 10000 + 60, thread=f"p{n}")]
    rows.append(comment("z0", "z", 30, thread="p0"))     # z joins once
    out = features.co_engagement_features(pd.DataFrame(rows), minutes=5)
    assert out.loc["x", "co_engage_best_partner_threads"] == 3 and out.loc["x", "co_engage_repeat_partners"] == 1
    assert out.loc["z", "co_engage_best_partner_threads"] == 1 and out.loc["z", "co_engage_repeat_partners"] == 0


def test_phrases_are_merged_so_no_feature_knows_the_side(cfg):
    phrases = features.all_phrases(cfg)
    assert len(phrases) == 16
    table = pd.DataFrame(
        [comment(f"b{i}", f"acct{i}", 100 + i, body="abolish ice now") for i in range(3)]     # 3 accounts, seconds apart
        + [comment("late", "acct0", 90000, body="abolish ice again")]                         # alone, a day later
        + [comment("plain", "acct9", 50, body="no tracked phrase in this one")])
    out = features.phrase_features(table, phrases, burst_minutes=10, burst_accounts=3)
    assert out.loc["acct1", "phrase_burst_share"] == 1.0
    assert out.loc["acct0", "phrase_burst_share"] == 0.5
    assert not any("favoring" in c or "opposing" in c for c in out.columns)


# ---------- label ----------

def test_answer_without_evidence_becomes_uncertain():
    sheet = pd.DataFrame({"author_hash": ["a", "b", "c", "d"],
                          "who_types": ["program", "Program", "person", "person"],
                          "changed_hands": ["no_sign", "no_sign", "likely", "likely"],
                          "evidence": ["replies in 5 seconds, all day", "", "", "silent 3 years, new language"]})
    clean = label.clean_labels(sheet)
    assert list(clean["who_types"]) == ["program", "uncertain", "person", "person"]
    assert list(clean["changed_hands"]) == ["no_sign", "no_sign", "uncertain", "likely"]


def test_unknown_answers_are_rejected():
    sheet = pd.DataFrame({"author_hash": ["a"], "who_types": ["shill"], "changed_hands": ["no_sign"], "evidence": ["x"]})
    with pytest.raises(ValueError):
        label.clean_labels(sheet)


def test_old_six_labels_can_be_derived():
    sheet = pd.DataFrame({
        "author_hash": list("abcdefg"),
        "who_types": ["declared", "person", "program", "person_plus_automation", "person", "person", "uncertain"],
        "changed_hands": ["no_sign", "likely", "no_sign", "no_sign", "no_sign", "no_sign", "no_sign"],
    })
    cluster = pd.Series({"e": "in_cluster", "f": "none_found"})
    assert list(label.old_six_labels(sheet, cluster)) == ["E", "B", "A", "A", "C", "D", "U"]


def test_cohen_kappa():
    assert label.cohen_kappa(list("AADD"), list("AADD")) == 1.0
    assert label.cohen_kappa(list("AADD"), list("ADAD")) == 0.0


# ---------- clusters ----------

def test_clusters_need_repeated_links_and_enough_accounts(cfg):
    cfg["features"]["min_comments"] = 3
    rows = []
    for n in range(3):                                   # x, y, z together in three threads
        for k, who in enumerate("xyz"):
            rows.append(comment(f"{who}{n}", who, n * 10000 + k * 30, thread=f"p{n}"))
    for n in range(3):                                   # v and w together three times: only a pair
        rows += [comment(f"v{n}", "v", 500000 + n * 10000, thread=f"q{n}"),
                 comment(f"w{n}", "w", 500000 + n * 10000 + 30, thread=f"q{n}")]
    rows += [comment(f"s{n}", "solo", 900000 + n * 10000, thread=f"r{n}") for n in range(3)]
    rows.append(comment("once", "quiet", 990000, thread="r9"))
    found = clusters.find_clusters(pd.DataFrame(rows), cfg).set_index("author_hash")["cluster"]
    assert set(found[found == "in_cluster"].index) == {"x", "y", "z"}
    assert found["v"] == "none_found" and found["solo"] == "none_found"      # a pair is not a cluster
    assert found["quiet"] == "not_tested"                                    # too little activity


# ---------- metrics ----------

def test_crossover_error_rate():
    y = [0] * 10 + [1] * 10
    assert metrics.crossover_error_rate(y, [0.1] * 10 + [0.9] * 10) == 0.0            # perfect separation
    one_each_way = [0.1] * 9 + [0.95] + [0.05] + [0.9] * 9                            # one person high, one program low
    assert metrics.crossover_error_rate(y, one_each_way) == pytest.approx(0.1)
    assert metrics.crossover_error_rate(y, [0.5] * 20) == pytest.approx(0.5)          # no information


def test_recall_at_false_flag_rate():
    y = [0] * 100 + [1] * 10
    score = [0.1] * 98 + [0.8, 0.8] + [0.9] * 6 + [0.7] * 4       # two people score above four programs
    assert metrics.recall_at_false_flag_rate(y, score, 0.01) == pytest.approx(0.6)    # no false flags allowed below 0.8
    assert metrics.recall_at_false_flag_rate(y, score, 0.02) == pytest.approx(1.0)
    assert metrics.enough_people_for(0.01, 100) and not metrics.enough_people_for(0.001, 100)


def test_precision_at_prevalence():
    # 1% of accounts are software, all caught, 1% of people falsely flagged: half the flags are wrong.
    assert metrics.precision_at_prevalence(1.0, 0.01, 0.01) == pytest.approx(0.5, abs=0.01)


def test_bootstrap_interval_contains_the_estimate():
    y = [0] * 50 + [1] * 50
    score = [i / 100 for i in range(100)]
    score[45], score[55] = score[55], score[45]
    low, high = metrics.bootstrap_interval(metrics.crossover_error_rate, y, score, rounds=200)
    assert low <= metrics.crossover_error_rate(y, score) <= high


# ---------- the Reddit training rule ----------

def test_training_on_reddit_data_is_refused(cfg):
    synthetic.make_synthetic(cfg)
    features.build_features(cfg)
    cfg["source"] = "reddit"
    with pytest.raises(model.TrainingNotAllowed):
        model.train(cfg)
    assert not paths(cfg)["model"].exists()                      # nothing was fitted or saved


def test_reddit_data_can_be_scored_with_a_model_trained_elsewhere(cfg, tmp_path):
    synthetic.make_synthetic(cfg)
    features.build_features(cfg)
    model.train(cfg)                                             # trained on synthetic data
    trained_elsewhere = paths(cfg)["model"]

    reddit = dict(cfg, source="reddit")
    predictions = model.score(reddit, trained_elsewhere)
    assert len(predictions) and {"score", "flagged"} <= set(predictions.columns)
    assert "Model trained on: synthetic data. Scored here: reddit data." in evaluate.evaluate(reddit).read_text()


def test_a_model_trained_on_reddit_is_refused_for_scoring(cfg):
    synthetic.make_synthetic(cfg)
    features.build_features(cfg)
    cfg["source"] = "reddit"
    cfg["model"]["reddit_training_permitted"] = True             # as if permission had been granted
    model.train(cfg)
    cfg["model"]["reddit_training_permitted"] = False            # permission withdrawn
    with pytest.raises(model.TrainingNotAllowed):
        model.score(cfg, paths(cfg)["model"])


# ---------- the whole pipeline ----------

def test_end_to_end_on_synthetic_data(cfg):
    synthetic.make_synthetic(cfg)
    table = features.build_features(cfg)
    sheet = label.draw_sample(cfg, n_accounts=100)
    predictions = model.train(cfg)
    report = evaluate.evaluate(cfg).read_text()

    assert {"train", "test"} == set(table["window"])
    assert sheet["who_types"].eq("").all() and sheet["changed_hands"].eq("").all()   # arrives unlabeled
    assert "stratum" not in pd.read_csv(paths(cfg)["to_label"]).columns
    # Most planted programs are found, and very few planted people are flagged.
    # (The synthetic data includes evasive programs and look-alike people, so not perfect.)
    assert predictions.loc[predictions["target"] == 1, "flagged"].mean() > 0.6
    assert predictions.loc[predictions["target"] == 0, "flagged"].mean() < 0.05
    # Planted rings are found by the cluster detector, and ordinary accounts are not swept in.
    found = pd.read_csv(paths(cfg)["clusters"]).set_index("author_hash")
    found = found[found["window"] == "train"]["cluster"]
    ring = found[found.index.str.startswith("ring")]
    ordinary = found[found.index.str.startswith("ordinary")]
    assert (ring == "in_cluster").mean() > 0.9
    assert (ordinary == "in_cluster").mean() < 0.02
    assert "crossover error rate" in report and paths(cfg)["error_curve"].exists()
    assert "power users" in report                                      # control-group table is filled
    # The report is aggregate: no account hash appears in it.
    assert not any(h in report for h in predictions["author_hash"])


# ---------- Bluesky collect ----------

from ea import bluesky

ROOT = "at://did:plc:alice/app.bsky.feed.post/r1"
REPLY = "at://did:plc:bob/app.bsky.feed.post/r2"


def new_layout(did, rkey, record, operation="create"):
    return {"$type": "message", "payload": {"did": did, "operation": operation,
            "collection": "app.bsky.feed.post", "rkey": rkey, "record": record}}


def old_layout(did, rkey, record, operation="create"):
    return {"did": did, "kind": "commit", "commit": {"operation": operation,
            "collection": "app.bsky.feed.post", "rkey": rkey, "record": record}}


def thread_post(uri, did, handle, text, reply=None, labels=()):
    record = {"text": text, "createdAt": "1999-01-01T00:00:00Z"}
    if reply:
        record["reply"] = {"root": {"uri": reply[0]}, "parent": {"uri": reply[1]}}
    return {"uri": uri, "indexedAt": "2026-11-14T12:00:00Z", "record": record,
            "author": {"did": did, "handle": handle, "displayName": "Real Name",
                       "labels": [{"val": v} for v in labels]}}


@pytest.fixture
def bsky_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("EA_PSEUDONYM_KEY", "test-key")
    cfg = load_config(Path(__file__).parent.parent / "config" / "bluesky_immigration.toml")
    cfg["data_dir"], cfg["reports_dir"] = tmp_path / "data", tmp_path / "reports"
    cfg["bluesky"]["seconds_between_requests"] = 0
    return cfg


def test_bluesky_reads_both_stream_layouts_and_skips_other_events():
    record = {"text": "hi"}
    expected = ("create", "app.bsky.feed.post", "did:plc:alice", ROOT, record)
    assert bluesky.parse_event(new_layout("did:plc:alice", "r1", record)) == expected
    assert bluesky.parse_event(old_layout("did:plc:alice", "r1", record)) == expected
    assert bluesky.parse_event({"did": "did:plc:alice", "kind": "identity"}) is None


def test_bluesky_stream_address_asks_for_each_kind_of_record():
    assert bluesky.stream_address("wss://host/x?collections=", ["a.b", "c.d"]) == \
        "wss://host/x?collections=a.b&collections=c.d"
    assert bluesky.stream_address("wss://host/subscribe?wantedCollections=", ["a.b"]) == \
        "wss://host/subscribe?wantedCollections=a.b"


def test_bluesky_minimize_hashes_account_and_post_addresses():
    kind, small = bluesky.minimize(thread_post(REPLY, "did:plc:bob", "bob.bsky.social", "a reply",
                                               reply=(ROOT, ROOT)), b"k")
    stored = json.dumps(small)
    assert kind == "comment" and small["body"] == "a reply"
    assert "did:plc" not in stored and "bob" not in stored and "Real Name" not in stored
    assert small["link_id"] == pseudonym(ROOT, b"k") and small["parent_id"] == "t3_" + pseudonym(ROOT, b"k")
    # The server's time is kept, not the 1999 date the posting app claimed.
    assert small["created_utc"] == int(pd.Timestamp("2026-11-14T12:00:00Z").timestamp())
    kind, small = bluesky.minimize(thread_post(ROOT, "did:plc:alice", "alice.example.com", "start"), b"k")
    assert kind == "post" and small["title"] == "start" and "link_id" not in small


def test_bluesky_disclosed_bots():
    assert bluesky.is_disclosed_bot({"handle": "news-bot.bsky.social"})
    assert bluesky.is_disclosed_bot({"handle": "x.bsky.social", "labels": [{"val": "bot"}]})
    assert not bluesky.is_disclosed_bot({"handle": "talbot.bsky.social"})


def test_bluesky_listen_counts_topics_and_remembers_threads(bsky_cfg):
    events = [bluesky.parse_event(e) for e in [
        new_layout("did:plc:alice", "r1", {"text": "New deportation flights today"}),
        new_layout("did:plc:bob", "r2", {"text": "Deported where?",
                                         "reply": {"root": {"uri": ROOT}, "parent": {"uri": ROOT}}}),
        new_layout("did:plc:carol", "r3", {"text": "I like ice cream"}),
        new_layout("did:plc:dan", "r4", {"text": "Kyiv was hit overnight"}),
        new_layout("did:plc:erin", "r5", {}, operation="delete"),
        new_layout("did:plc:fay", "r6", {"text": "ICE was outside the school"}),
    ]]
    volumes, threads, deleted, opted_out, opted_in = bluesky.listen(bsky_cfg, 60, events)
    by_topic = volumes.set_index("topic")
    assert by_topic.loc["all", "posts"] == 5
    assert by_topic.loc["immigration", "posts"] == 3 and by_topic.loc["immigration", "replies"] == 1
    assert by_topic.loc["russia_ukraine", "posts"] == 1
    assert by_topic.loc["all", "posts_per_day_at_this_rate"] == 5 * 1440
    assert threads == [ROOT, "at://did:plc:fay/app.bsky.feed.post/r6"]
    assert deleted == {pseudonym("at://did:plc:erin/app.bsky.feed.post/r5", b"test-key")}


class FakeBluesky:
    """Stands in for the public API: one thread with a reply and a deleted reply."""
    def get(self, url, params, timeout, headers):
        thread = {"post": thread_post(ROOT, "did:plc:alice", "alice.bsky.social", "New deportation flights"),
                  "replies": [{"post": thread_post(REPLY, "did:plc:bob", "bob.bsky.social", "Deported where?",
                                                   reply=(ROOT, ROOT)), "replies": []},
                              {"$type": "app.bsky.feed.defs#notFoundPost", "notFound": True}]}
        return type("Response", (), {"status_code": 200, "json": lambda self: {"thread": thread}})()


def test_bluesky_collect_writes_files_the_features_stage_can_load(bsky_cfg):
    start = [bluesky.parse_event(new_layout("did:plc:alice", "r1", {"text": "New deportation flights"}))]
    bluesky.collect(bsky_cfg, 1, session=FakeBluesky(), events=start)
    bluesky.collect(bsky_cfg, 1, session=FakeBluesky(), events=start)      # a second run stores nothing twice
    posts, comments = collect.load_collected(bsky_cfg)
    assert len(posts) == 1 and len(comments) == 1
    folder = paths(bsky_cfg)["collected"] / "bluesky"
    assert "did:plc" not in (folder / "posts.jsonl").read_text() + (folder / "comments.jsonl").read_text()
    assert paths(bsky_cfg)["volumes"].exists()

    # The author of the reply deletes it; the next run that hears the delete removes it.
    gone = [bluesky.parse_event(new_layout("did:plc:bob", "r2", {}, operation="delete"))]
    bluesky.collect(bsky_cfg, 1, session=FakeBluesky(), events=gone)
    assert (folder / "comments.jsonl").read_text() == ""


def reuse_declaration(did, record, operation="create"):
    event = new_layout(did, "self", record, operation)
    event["payload"]["collection"] = "org.user-intents.declaration"
    return bluesky.parse_event(event)


def test_bluesky_reuse_declaration_has_three_answers():
    assert not bluesky.allows_bulk_datasets({"bulkDataset": {"allow": False}})
    assert bluesky.allows_bulk_datasets({"bulkDataset": {"allow": True}})
    assert bluesky.allows_bulk_datasets({"bulkDataset": {"updatedAt": "2026-01-01T00:00:00Z"}})   # no answer given
    assert bluesky.allows_bulk_datasets({"syntheticContentGeneration": {"allow": False}})         # a different question


def test_bluesky_accounts_that_opt_out_of_datasets_are_never_stored_and_are_removed(bsky_cfg):
    start = bluesky.parse_event(new_layout("did:plc:alice", "r1", {"text": "New deportation flights"}))
    folder = paths(bsky_cfg)["collected"] / "bluesky"

    bluesky.collect(bsky_cfg, 1, session=FakeBluesky(), events=[start])
    assert len((folder / "comments.jsonl").read_text().splitlines()) == 1

    # Bob says no to bulk datasets: his stored reply goes, and it is not stored again.
    says_no = reuse_declaration("did:plc:bob", {"bulkDataset": {"allow": False}})
    bluesky.collect(bsky_cfg, 1, session=FakeBluesky(), events=[says_no, start])
    assert (folder / "comments.jsonl").read_text() == ""
    assert "did:plc" not in (folder / "opted_out.jsonl").read_text()
    bluesky.collect(bsky_cfg, 1, session=FakeBluesky(), events=[start])       # the opt-out is remembered
    assert (folder / "comments.jsonl").read_text() == ""

    # Bob withdraws the declaration: his posts may be stored again.
    bluesky.collect(bsky_cfg, 1, session=FakeBluesky(), events=[reuse_declaration("did:plc:bob", {}, "delete"), start])
    assert len((folder / "comments.jsonl").read_text().splitlines()) == 1


def test_bluesky_account_labels_are_kept_apart_and_hashed():
    post = thread_post(ROOT, "did:plc:alice", "alice.bsky.social", "hi", labels=["spam"])
    post["author"]["labels"] = [{"val": "spam", "src": bluesky.BLUESKY_MODERATION},
                                {"val": "ai-account", "src": "did:plc:someones-labeler"},
                                {"val": "bot", "src": "did:plc:alice"},
                                {"val": "inauthentic", "src": "did:plc:moderation", "neg": True}]
    labels = bluesky.account_labels(post, b"k")
    assert [(l["label"], l["self_applied"], l["labeler"]) for l in labels] == \
        [("spam", False, "bluesky-moderation"), ("ai-account", False, pseudonym("did:plc:someones-labeler", b"k")),
         ("bot", True, "")]
    assert "alice" not in json.dumps(labels) and "did:plc" not in json.dumps(labels)


# ---------- network ----------

from ea import ira2018, network


def tiny_network():
    """Seeds s1, s2 answer each other; s3 only answers an ordinary poster; o1, o2 are ordinary."""
    posts = pd.DataFrame([
        {"id": "p1", "author_hash": "s1", "subreddit": "a", "created_utc": 100, "link_key": "k1"},
        {"id": "p2", "author_hash": "o1", "subreddit": "a", "created_utc": 100, "link_key": "k2"},
        {"id": "p3", "author_hash": "s2", "subreddit": "b", "created_utc": 400, "link_key": "k1"},
    ])
    rows = [("c1", "s2", "p1", "t3_p1", 160), ("c2", "s1", "p1", "t1_c1", 220), ("c3", "s1", "p1", "t3_p1", 230),
            ("c4", "s3", "p2", "t3_p2", 300), ("c5", "o2", "p2", "t1_c4", 400), ("c6", "o2", "p2", "t1_gone", 500)]
    comments = pd.DataFrame([{"id": i, "author_hash": a, "link_id": p, "parent_id": parent, "created_utc": t,
                              "subreddit": "a", "disclosed_bot": False} for i, a, p, parent, t in rows])
    return posts, comments


def test_edges_point_at_the_author_answered():
    edges = network.build_edges(*tiny_network()).set_index(pd.Index(["c1", "c2", "c3", "c4", "c5", "c6"]))
    assert edges.loc["c1", ["source", "target", "kind"]].tolist() == ["s2", "s1", "post_reply"]
    assert edges.loc["c2", ["source", "target", "kind"]].tolist() == ["s1", "s2", "comment_reply"]
    assert edges.loc["c2", "latency"] == 60
    assert edges.loc["c6", "target"] == network.UNKNOWN          # the comment it answers was never collected


def test_mixing_table_separates_self_seed_other_unknown():
    edges = network.build_edges(*tiny_network())
    table = network.mixing_table(edges, {"s1", "s2", "s3"})
    assert table.loc["post_reply"].to_dict() == {"self": 1, "seed": 1, "other": 1, "unknown": 0}
    assert table.loc["comment_reply", "seed"] == 1


def test_seed_graph_and_relay_chain():
    posts, comments = tiny_network()
    shape = network.graph_shape(network.seed_graph(network.build_edges(posts, comments), {"s1", "s2", "s3"}))
    assert shape["accounts"] == 2 and shape["pairs"] == 2 and shape["reciprocity"] == 1.0
    chains = network.relay_chains(posts, {"s1", "s2", "s3"})
    assert len(chains) == 1 and chains.iloc[0]["accounts"] == 2 and chains.iloc[0]["median_gap_minutes"] == 5


def test_entropy_share():
    assert network.entropy_share([5, 5]) == pytest.approx(1.0)
    assert network.entropy_share([10]) == 0.0
    assert network.entropy_share([99, 1]) < 0.2


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """The synthetic network, built once."""
    cfg = load_config(Path(__file__).parent.parent / "config" / "immigration.toml")
    root = tmp_path_factory.mktemp("net")
    cfg["data_dir"], cfg["reports_dir"] = root / "data", root / "reports"
    synthetic.make_synthetic_network(cfg)
    posts, comments = collect.load_collected(cfg)
    return cfg, posts, comments, network.build_edges(posts, comments), network.load_seeds(cfg)


def group(seeds, prefix):
    return {s for s in seeds if s.startswith(prefix)}


def test_planted_ring_shows_up_against_chance(planted):
    _, _, _, edges, seeds = planted
    ring = network.mixing_against_shuffle(edges, group(seeds, "ring"), shuffles=50)
    assert ring["testable"] and ring["times_chance"] > 5 and ring["p_value"] < 0.05
    covert = network.mixing_against_shuffle(edges, group(seeds, "covert"), shuffles=50)
    assert covert["observed_share"] == 0                       # covert members never answer each other


def test_covert_accounts_are_narrow_and_ordinary_accounts_are_not(planted):
    _, posts, _, edges, seeds = planted
    profiles = network.account_profiles(edges, posts)
    covert = network.summarize_profiles(profiles[profiles.index.isin(list(group(seeds, "covert")))])
    ordinary = network.summarize_profiles(profiles[~profiles.index.isin(list(seeds))])
    assert covert["median_top_target_share"] > 0.25 > ordinary["median_top_target_share"]
    assert covert["median_target_spread"] < ordinary["median_target_spread"]


def test_relay_crew_is_found(planted):
    _, posts, _, _, seeds = planted
    summary = network.summarize_relays(network.relay_chains(posts, seeds), posts, seeds)
    assert summary["chains_all_seed"] == 40 and summary["median_gap_minutes"] < 15


def test_the_net_catches_the_ring_but_not_the_covert_accounts(planted):
    _, _, _, edges, seeds = planted
    ring = network.proximity_test(edges, group(seeds, "ring"), trials=20)
    loose, tight = ring["touches_2_or_more"], ring["touches_2_or_more_and_15pct_of_replies"]
    assert loose["precision"] < 0.5 < tight["precision"]       # touching seeds is not enough; aiming at them is
    assert tight["recall"] > 0.8
    covert = network.proximity_test(edges, group(seeds, "covert"), trials=20)
    assert all((net["precision"] or 0) == 0 for net in covert.values())


def test_network_report_holds_no_account_ids(planted):
    cfg, _, _, _, seeds = planted
    out, result = network.network_report(cfg, shuffles=20, trials=10)
    text = out.read_text() + paths(cfg)["network_picture"].read_text()
    assert not any(s in text for s in seeds) and "ordinary0" not in text
    assert result["counts"]["seeds"] == 50


def test_ira2018_convert_hashes_everyone_and_drops_text(cfg):
    raw = cfg["data_dir"] / "raw"
    raw.mkdir(parents=True)
    (raw / "seed.csv").write_text("TrollOne,5\nTrollTwo,0\n")
    pd.DataFrame([
        {"id": "s1", "author.name": "TrollOne", "created_utc": 100.0, "is_self": False,
         "url": "http://example.com/a", "subreddit_name_prefixed": "r/news"},
        {"id": "s2", "author.name": "reddit", "created_utc": 900.0, "is_self": True,
         "url": "", "subreddit_name_prefixed": "u/reddit"},
    ]).to_csv(raw / "submissions.csv", index=False)
    pd.DataFrame([
        {"id": "c1", "author.name": "TrollTwo", "created_utc": 160.0, "link_id": "t3_s1", "parent_id": "t3_s1",
         "link_author": "TrollOne", "body": "secret words", "subreddit.display_name_prefixed": "r/news"},
        {"id": "c2", "author.name": "TrollTwo", "created_utc": 170.0, "link_id": "t3_x9", "parent_id": "t3_x9",
         "link_author": "OrdinaryPerson", "body": "more words", "subreddit.display_name_prefixed": "r/funny"},
    ]).to_csv(raw / "comments.csv", index=False)

    assert ira2018.convert(cfg) == (2, 2, 2)                    # the ban notice is not a post
    folder = paths(cfg)["collected"] / "ira2018"
    written = (folder / "posts.jsonl").read_text() + (folder / "comments.jsonl").read_text()
    for leak in ("TrollOne", "OrdinaryPerson", "secret words", "example.com"):
        assert leak not in written
    posts, comments = collect.load_collected(cfg)
    table = network.mixing_table(network.build_edges(posts, comments), network.load_seeds(cfg))
    assert table.loc["post_reply", "seed"] == 1 and table.loc["post_reply", "other"] == 1


# ---------- Reddit collection is paused ----------

def test_reddit_collection_is_refused_while_paused(cfg):
    """Every stage that would fetch Reddit data stops before its first request."""
    from ea import threads
    from ea.config import CollectionPaused
    cfg["source"] = "reddit"                    # what the real configs say
    fake = FakeArcticShift([], [])
    for fetch in (lambda: collect.measure_volumes(cfg, session=fake),
                  lambda: collect.measure_topic_share(cfg, session=fake),
                  lambda: collect.collect(cfg, session=fake),
                  lambda: collect.collect_from_dumps(cfg, "posts.jsonl", "comments.jsonl"),
                  lambda: seeds.fetch_reddit_2018(cfg, session=fake),
                  lambda: threads.collect_threads(cfg, session=fake)):
        with pytest.raises(CollectionPaused):
            fetch()
    assert fake.calls == 0
    assert not cfg["data_dir"].exists()


def test_shipped_configs_keep_reddit_paused_and_untrainable():
    for name in ("immigration.toml", "ira2018.toml"):
        shipped = load_config(Path(__file__).parent.parent / "config" / name)
        assert shipped["source"] == "reddit"
        assert shipped["collect"]["reddit_collection_permitted"] is False
        with pytest.raises(model.TrainingNotAllowed):
            model.check_training_allowed(shipped["source"], shipped)
