"""The thread filler: hashes with the seeds' key, drops text, resumes."""

import json
from pathlib import Path

import pytest

from ea import threads
from ea.config import load_config, paths
from ea.pseudonymize import pseudonym


class Fake:
    def __init__(self):
        self.calls = 0

    def get(self, url, params, **_):
        self.calls += 1
        if url.endswith("/api/posts/ids"):
            rows = [{"id": i, "subreddit": "s", "author": "Op", "created_utc": 100} for i in params["ids"].split(",")]
        elif isinstance(params.get("after"), int):
            rows = []
        else:
            rows = [{"id": "c1", "subreddit": "s", "author": "Seed_A", "created_utc": 200,
                     "link_id": "t3_" + params["link_id"], "parent_id": "t3_" + params["link_id"], "body": "words"}]
        return type("R", (), {"status_code": 200, "json": lambda self: {"data": rows}})()


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("EA_PSEUDONYM_KEY", "test-key")
    cfg = load_config(Path(__file__).parent.parent / "config" / "ira2018.toml")
    cfg["data_dir"], cfg["reports_dir"] = tmp_path / "data", tmp_path / "reports"
    cfg["collect"]["seconds_between_requests"] = 0
    cfg["source"] = "synthetic"         # the test uses a fake API; the real config says "reddit", which is paused
    raw = cfg["data_dir"] / "raw"
    raw.mkdir(parents=True)
    (raw / "comments.csv").write_text("link_id\nt3_p1\n")
    (raw / "submissions.csv").write_text("id,num_comments,author.name\np2,3,Seed_A\np3,0,Seed_A\nn1,5,reddit\n")
    return cfg


def test_threads_fetch_hashed_textless_and_resume(cfg):
    fake = Fake()
    assert threads.collect_threads(cfg, session=fake) == 2          # p1 and p2; p3 drew no comments
    folder = paths(cfg)["collected"] / "threads"
    comments = [json.loads(l) for l in (folder / "comments.jsonl").read_text().splitlines()]
    assert {c["body"] for c in comments} == {""}
    assert comments[0]["author_hash"] == pseudonym("Seed_A", b"test-key")
    calls = fake.calls
    threads.collect_threads(cfg, session=fake)
    assert fake.calls == calls                                      # everything already done
