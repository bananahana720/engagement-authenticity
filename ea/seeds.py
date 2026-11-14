"""Known-bad seed accounts from public platform disclosures.

Reddit's 2018 transparency report listed 944 accounts it suspended as part of
a Russian influence operation (wiki page /r/reddit.com/wiki/suspiciousaccounts).
reddit.com refuses plain HTTP clients, so the page is read from Arctic Shift's
wiki archive.

Two files are written under data/seeds/, never committed:
  reddit_2018_suspicious.csv         username, karma (needed to look the accounts up)
  reddit_2018_suspicious_hashed.csv  pseudonym, karma (joins to collected data)
"""

import re
from pathlib import Path

import pandas as pd
import requests

from ea.collect import get_json
from ea.pseudonymize import load_key, pseudonym

WIKI_PATH = "/r/reddit.com/wiki/suspiciousaccounts"
ROW = re.compile(r"^\|\s*u/([A-Za-z0-9_-]+)\s*\|\s*(-?[\d,]+)\s*\|", re.MULTILINE)


def parse_wiki_table(content):
    """Rows of the wiki's markdown table as (username, karma)."""
    return [(name, int(karma.replace(",", ""))) for name, karma in ROW.findall(content)]


def fetch_reddit_2018(cfg, session=None, out_dir=Path("data/seeds")):
    session = session or requests.Session()
    url = f"{cfg['collect']['api_base']}/api/subreddits/wikis"
    pages = get_json(session, url, {"paths": WIKI_PATH})
    if not pages:
        raise RuntimeError(f"Arctic Shift has no copy of {WIKI_PATH}")
    rows = parse_wiki_table(pages[0]["content"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=["username", "karma"]).drop_duplicates("username")
    table.to_csv(out_dir / "reddit_2018_suspicious.csv", index=False)
    key = load_key()
    hashed = pd.DataFrame({"author_hash": [pseudonym(u, key) for u in table["username"]],
                           "karma": table["karma"]})
    hashed.to_csv(out_dir / "reddit_2018_suspicious_hashed.csv", index=False)
    return len(table), out_dir
