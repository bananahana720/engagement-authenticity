"""Tests for the red team: the made-up worlds and the cross-account signals."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ea import features, redteam
from ea.config import load_config
from ea.network_features import cohorts, network_features, shared_rare_events
from ea.redteam_generate import LEVELS, make_world

SMALL = {"n_ordinary": 400, "n_threads": 400}
DAY = 86400


@pytest.fixture(scope="module")
def cfg():
    return load_config(Path(__file__).parent.parent / "config" / "immigration.toml")


@pytest.fixture(scope="module")
def tables(cfg):
    """One small world per level, as feature tables."""
    return {level["key"]: redteam.world_table((cfg, level["key"], 3, SMALL)) for level in LEVELS}


# ---------- the generator ----------

def test_world_has_every_control_and_marks_only_puppets(cfg):
    world = make_world(cfg, "sparse", 1, **SMALL)
    kinds = set(world.accounts["kind"])
    assert set(redteam.CONTROLS) | {"puppet", "disclosed"} <= kinds
    assert set(world.accounts.index[world.accounts["target"] == 1]) == \
        set(world.accounts.index[world.accounts["kind"] == "puppet"])
    assert world.accounts["target"].sum() == 80
    bots = world.comments[world.comments["disclosed_bot"]]
    assert set(bots["author_hash"]) == set(world.accounts.index[world.accounts["kind"] == "disclosed"])


def test_same_seed_same_world(cfg):
    a, b = make_world(cfg, "jitter", 5, **SMALL), make_world(cfg, "jitter", 5, **SMALL)
    assert a.comments["body"].equals(b.comments["body"]) and a.comments["created_utc"].equals(b.comments["created_utc"])


def test_background_is_the_same_at_every_level(cfg):
    """Only the puppets change between levels, so levels are compared on equal ground."""
    def background(level):
        world = make_world(cfg, level, 2, **SMALL)
        ordinary = world.comments[world.comments["author_hash"].str.startswith("o")]
        return ordinary[["author_hash", "created_utc", "body"]].reset_index(drop=True)
    assert background("naive").equals(background("patient"))


def test_rewording_hides_from_wording_match_but_not_from_talking_points(cfg):
    world = make_world(cfg, "style", 1, **SMALL)
    campaign = world.comments[world.truth["campaign"].to_numpy()]
    pairs = features.near_duplicate_pairs(campaign, minutes=60, min_jaccard=0.7)
    assert len(pairs) == 0                                  # no two reworded comments match on wording

    found = redteam.find_talking_points(world.comments[~world.comments["disclosed_bot"]])
    truth = world.truth.loc[~world.comments["disclosed_bot"].to_numpy(), "point"].to_numpy()
    grouped = pd.DataFrame({"found": found, "truth": truth})[found >= 0]
    purity = grouped.groupby("found")["truth"].agg(lambda t: t.value_counts().iloc[0]).sum() / len(grouped)
    assert purity > 0.9                                     # a cluster is almost always one talking point
    assert (found[truth >= 0] >= 0).mean() > 0.4            # and it finds a good share of them


def test_text_is_invented_words_only(cfg):
    """Nothing the generator writes could pass for a real comment."""
    world = make_world(cfg, "sparse", 1, **SMALL)
    phrases = set(features.all_phrases(cfg))
    concept_words, fillers = redteam.vocabulary()
    invented = set(concept_words.ravel()) | set(fillers)
    for body in world.comments["body"].sample(300, random_state=0):
        for phrase in phrases:
            body = body.replace(phrase, "")
        assert set(body.split()) <= invented


# ---------- cross-account signals ----------

def test_shared_rare_events_beat_shared_common_events():
    # Accounts 0 and 1 share three events nobody else is in. Accounts 2..9 all share one event.
    who = [0, 0, 0, 1, 1, 1] + list(range(2, 10))
    event = ["a", "b", "c", "a", "b", "c"] + ["busy"] * 8
    scores = shared_rare_events(np.array(who), np.array(event), 10)
    assert scores[0, 1] > 1.5 * scores[2, 3]
    assert scores[0, 2] == 0 and np.allclose(scores, scores.T) and scores[0, 0] == 0


def test_cohorts_find_accounts_first_seen_together():
    start = 0
    spread = np.arange(60) * 3 * DAY + 20 * DAY              # one newcomer every three days
    batch = np.full(10, 100 * DAY) + np.arange(10) * 3600    # ten within ten hours
    early = np.array([DAY, 2 * DAY])                         # inside the first two weeks: unknown
    together, excess = cohorts(np.concatenate([spread, batch, early]).astype(float), start)
    assert excess[60:70].min() > 3 * np.median(excess[:60])
    assert together[60, 61] == 1 and together[0, 1] == 0
    assert excess[-1] == 1 and together[-1, -2] == 0         # the first two weeks say nothing


def test_network_features_one_row_per_account():
    comments = pd.DataFrame({
        "author_hash": ["a", "a", "b", "b", "c"], "link_id": ["p1", "p2", "p1", "p2", "p3"],
        "created_utc": [0, DAY, 10, DAY + 10, 5 * DAY], "point": [1, 2, 1, 2, -1]})
    table = network_features(comments, pd.Index(["a", "b", "c"]))
    assert list(table.index) == ["a", "b", "c"]
    assert table.loc["a", "same_points"] > table.loc["c", "same_points"]
    assert not table.isna().any().any()


# ---------- the ladder does what it says ----------

def puppets_and_people(table):
    return table[table["target"] == 1], table[table["target"] == 0]


def test_naive_bots_trip_the_version_1_signals(tables):
    puppets, people = puppets_and_people(tables["naive"])
    assert puppets["fast_reply_share"].median() > 0.3 > people["fast_reply_share"].quantile(0.99)
    assert puppets["near_duplicate_share"].median() > 0.5


def test_each_evasion_removes_the_signal_it_targets(tables):
    style, _ = puppets_and_people(tables["style"])
    assert style["near_duplicate_share"].max() < 0.1          # reworded: wording match is gone
    assert style["fast_reply_share"].median() > 0.3           # still replies in seconds

    jitter, people = puppets_and_people(tables["jitter"])
    assert jitter["fast_reply_share"].max() == 0              # human timing: speed is gone
    assert jitter["same_points"].median() > people["same_points"].quantile(0.99)   # the ring still shows

    sparse, _ = puppets_and_people(tables["sparse"])
    assert sparse["co_engage_repeat_partners"].max() <= 1     # one puppet per thread
    assert sparse["campaign_comments"].sum() < jitter["campaign_comments"].sum()


def test_disclosed_bots_are_never_scored(tables):
    assert not (tables["naive"]["kind"] == "disclosed").any()


def test_no_feature_records_a_side(tables):
    assert not any("favoring" in c or "opposing" in c or "side" in c for c in tables["naive"].columns)
