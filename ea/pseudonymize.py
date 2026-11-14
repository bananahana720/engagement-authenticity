"""Replace usernames with a keyed hash before anything is written to disk.

The key lives outside git (.secrets/pseudonym.key, or the EA_PSEUDONYM_KEY
environment variable). Without the key nobody can turn a hash back into a
username or test whether a given username is in the data.
"""

import hashlib
import hmac
import os
import re
import secrets
from pathlib import Path

KEY_FILE = Path(".secrets/pseudonym.key")

# Accounts that say they are bots (label E in the plan). Checked on the raw
# username, because the hash hides it afterwards.
# Narrow on purpose: "RemindMeBot" and "news_bot" match, "talbot" does not.
DISCLOSED_BOT_NAME = re.compile(r"^(?i:automoderator)$|(?i:[_-]bot)$|[a-z0-9]Bot$")

# Reddit's placeholders for a missing author.
NO_AUTHOR = {"[deleted]", "[removed]", "", None}


def load_key():
    """Return the secret key, creating one on first use."""
    env_key = os.environ.get("EA_PSEUDONYM_KEY")
    if env_key:
        return env_key.encode()
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        KEY_FILE.write_text(secrets.token_hex(32))
        KEY_FILE.chmod(0o600)
    return KEY_FILE.read_text().strip().encode()


def pseudonym(username, key):
    """A stable 16-character stand-in for a username."""
    digest = hmac.new(key, username.lower().encode(), hashlib.sha256).hexdigest()
    return digest[:16]


def is_disclosed_bot(username):
    return bool(DISCLOSED_BOT_NAME.search(username))
