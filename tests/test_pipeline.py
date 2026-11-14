"""Tests on small hand-made inputs with known answers, plus one end-to-end run."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ea import collect, evaluate, features, label, model, seeds, synthetic
from ea.config import load_config, paths
from ea.pseudonymize import is_disclosed_bot, pseudonym


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("EA_PSEUDONYM_KEY", "test-key")
    cfg = load_config(Path(__file__).parent.parent / "config" / "immigration.toml")
    cfg["data_dir"], cfg["reports_dir"] = tmp_path / "data", tmp_path / "reports"
    cfg["collect"]["seconds_between_requests"] = 0
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

def test_label_without_evidence_becomes_uncertain():
    sheet = pd.DataFrame({"author_hash": ["a", "b", "c"], "label": ["A", "a", "D"],
                          "evidence": ["replies in 5 seconds, all day", "", ""]})
    assert list(label.clean_labels(sheet)["label"]) == ["A", "U", "D"]


def test_cohen_kappa():
    assert label.cohen_kappa(list("AADD"), list("AADD")) == 1.0
    assert label.cohen_kappa(list("AADD"), list("ADAD")) == 0.0


# ---------- the whole pipeline ----------

def test_end_to_end_on_synthetic_data(cfg):
    synthetic.make_synthetic(cfg)
    table = features.build_features(cfg)
    sheet = label.draw_sample(cfg, n_accounts=100)
    predictions = model.train(cfg)
    report = evaluate.evaluate(cfg).read_text()

    assert {"train", "test"} == set(table["window"])
    assert sheet["label"].eq("").all()                                  # the sheet arrives unlabeled
    assert "stratum" not in pd.read_csv(paths(cfg)["to_label"]).columns
    # Planted bots and rings are found, and no planted ordinary account is flagged.
    assert predictions.loc[predictions["target"] == 1, "flagged"].mean() > 0.6
    assert predictions.loc[predictions["target"] == 0, "flagged"].mean() == 0
    # The report is aggregate: no account hash appears in it.
    assert not any(h in report for h in predictions["author_hash"])
