"""Load a topic config file and give every stage the same file paths."""

import tomllib
from pathlib import Path


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["data_dir"] = Path(cfg["data_dir"])
    cfg["reports_dir"] = Path(cfg["reports_dir"])
    return cfg


def all_subreddits(cfg):
    subs = cfg["subreddits"]
    return subs["all_threads"] + subs["filtered"] + subs.get("city", [])


def paths(cfg):
    """Every file the pipeline reads or writes, in one place."""
    data, reports = cfg["data_dir"], cfg["reports_dir"]
    return {
        "collected": data / "collected",             # <subreddit>/posts.jsonl, comments.jsonl
        "features": data / "features.csv",
        "to_label": data / "labels" / "to_label.csv",
        "label_examples": data / "labels" / "to_label_examples.csv",
        "labels": data / "labels" / "labels.csv",
        "model": data / "model" / "model.joblib",
        "predictions": data / "model" / "test_predictions.csv",
        "threshold": data / "model" / "threshold.json",
        "volumes": reports / "volumes.csv",
        "phrase_spread": reports / "phrase_spread.csv",
        "evaluation": reports / "evaluation.md",
    }
