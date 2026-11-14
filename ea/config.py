"""Load a topic config file and give every stage the same file paths."""

import tomllib
from pathlib import Path


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["data_dir"] = Path(cfg["data_dir"])
    cfg["reports_dir"] = Path(cfg["reports_dir"])
    return cfg


class CollectionPaused(Exception):
    pass


def check_collection_allowed(cfg):
    """Stop before any request for Reddit data. Reddit collection is paused (docs/DATA_TERMS.md).

    Every stage that would fetch Reddit data calls this first. The pause lifts only when
    the config records it: [collect] reddit_collection_permitted = true.
    """
    if cfg.get("source") == "reddit" and not cfg.get("collect", {}).get("reddit_collection_permitted", False):
        raise CollectionPaused(
            "Reddit collection is paused, so nothing was fetched. "
            "See the Status section of docs/DATA_TERMS.md.")


def all_subreddits(cfg):
    subs = cfg["subreddits"]
    return subs["all_threads"] + subs["filtered"] + subs.get("city", [])


def paths(cfg):
    """Every file the pipeline reads or writes, in one place."""
    data, reports = cfg["data_dir"], cfg["reports_dir"]
    return {
        "collected": data / "collected",             # <subreddit>/posts.jsonl, comments.jsonl
        "features": data / "features.csv",
        "clusters": data / "clusters.csv",
        "to_label": data / "labels" / "to_label.csv",
        "label_examples": data / "labels" / "to_label_examples.csv",
        "labels": data / "labels" / "labels.csv",
        "model": data / "model" / "model.joblib",
        "predictions": data / "model" / "test_predictions.csv",
        "scores": data / "model" / "test_window_scores.csv",       # every account, labeled or not
        "groups": data / "labels" / "groups.csv",                  # optional: control and red-team groups
        "error_curve": reports / "error_tradeoff.png",
        "reliability": reports / "reliability.png",
        "threshold": data / "model" / "threshold.json",
        "volumes": reports / "volumes.csv",
        "topic_share": reports / "topic_share.csv",
        "phrase_spread": reports / "phrase_spread.csv",
        "evaluation": reports / "evaluation.md",
        "seeds": data / "labels" / "seeds.csv",      # accounts with a verified label, for the network stage
        "network": reports / "network.json",
        "network_picture": reports / "network_picture.json",
    }
