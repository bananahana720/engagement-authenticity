"""Tests for the model-fingerprint measurement, on made-up text (no model is run)."""

import random

from ea import fingerprint
from ea.fingerprint_generate import clean, posts, user_message


def fake_rows(mode, n_posts, tells, seed=0, length=14):
    """Each 'family' sprinkles its own tell word into otherwise shared filler."""
    rng = random.Random(seed)
    filler = "the pan is good and it works well for most people who try it at home".split()
    rows = []
    for family, tell in tells.items():
        for post_id in range(n_posts):
            words = rng.choices(filler, k=length) + ([tell] * (length // 7) if tell else [])
            rng.shuffle(words)
            rows.append({"family": family, "mode": mode, "post_id": post_id, "text": " ".join(words)})
    return rows


def test_posts_are_fixed_and_shared():
    assert posts() == posts() and len(set(posts())) == 300
    assert posts()[0] in user_message("short_plain", posts()[0])


def test_clean_drops_reasoning_and_unfinished_sentence():
    assert clean("<think>hmm</think> Fine. Half a sen", finished=False) == "Fine."
    assert clean(" All of it. And more", finished=True) == "All of it. And more"


def test_distinct_styles_are_found_and_identical_styles_sit_at_chance():
    distinct = fake_rows("m", 60, {"a": "honestly", "b": "indeed", "c": "overall"})
    result = fingerprint.cross_score(distinct, distinct, 60)
    assert result["accuracy"] > 0.95 and result["n"] == 180
    same = fake_rows("m", 60, {"a": "", "b": "", "c": ""})
    assert abs(fingerprint.cross_score(same, same, 60)["accuracy"] - 1 / 3) < 0.15


def test_evasion_that_removes_the_tell_drops_to_chance():
    tells = {"a": "honestly", "b": "indeed", "c": "overall"}
    by_mode = {"short_plain": fake_rows("short_plain", 60, tells),
               "short_human": fake_rows("short_human", 60, dict.fromkeys(tells, ""), seed=1),
               "long_plain": fake_rows("long_plain", 40, tells, seed=2, length=170)}
    out = fingerprint.run(list(tells), by_mode, 60, 40, 0)
    assert out["evasion"]["none"]["accuracy"] > 0.95
    assert out["evasion"]["human_style_unseen"]["accuracy"] < 0.55
    assert out["pooled"][3]["accuracy"] >= out["pooled"][0]["accuracy"] - 0.02  # 10 pooled replies
    assert [c["words"] for c in out["by_length"]] == fingerprint.CUTS
    assert sum(sum(row.values()) for row in out["confusion"].values()) == 180


def test_wilson_range_brackets_the_share():
    low, high = fingerprint.wilson(50, 100)
    assert low < 0.5 < high and high - low < 0.21
