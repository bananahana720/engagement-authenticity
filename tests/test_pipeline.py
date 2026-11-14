"""Tests on small hand-made inputs with known answers, plus one end-to-end run."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ea import clusters, collect, evaluate, features, label, metrics, model, synthetic
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
            rows = [c for c in self.comments if c["link_id"] == "t3_" + params["link_id"]]
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
