# Data sources and terms

Checked 2026-11-14. Read this before collecting.

## What was read

**Arctic Shift** (github.com/ArthurHeitmann/arctic_shift), from its README, API notes and file notes:

- A community project, not Reddit. It collects through Reddit's official API and republishes.
- Private and quarantined subreddits are not included.
- Its pages state no license and no research-use terms. The API comes with "no uptime or performance guarantees".
- It runs a removal-request form for people who want their data taken out.
- Since 2023-11, each record is fetched a second time after 36 hours. Records removed in between carry `_meta.was_deleted_later` and `_meta.removal_type`.

## What was NOT read

**Reddit's own Data API Terms, Public Content Policy and User Agreement.** The machine
this scaffold was built on could not reach reddit.com or redditinc.com. Someone has
to read them before real collection starts. Things to look for, taken from secondary
sources and so unconfirmed:

- Does Reddit require content deleted on Reddit to be deleted from copies? (Believed yes.)
- Does training a model on Reddit content need Reddit's written approval? (Believed yes for AI training. This project trains a small classifier on derived numbers such as reply times, not on text, but check the wording.)
- Do the terms reach data obtained from a third-party archive rather than the API?

## How the code is built around this

| Concern | What the code does |
| --- | --- |
| Usernames | Replaced by a keyed hash before writing. The key stays in `.secrets/`, outside git. |
| Extra fields | Dropped. Kept: id, subreddit, time, thread, parent, text, removed flag. |
| Removed or deleted text | Not kept. The row stays, with an empty body and `removed = true`. |
| Deleted authors | Their comments are not kept at all. |
| Load on a free service | One request a second, a plain user agent with no personal details, resumable so nothing is fetched twice. |
| Publication | Reports hold counts and rates only. Groups under 10 accounts are hidden. |

## Open questions for the project owner

1. A comment its author deleted keeps a row here (time and thread, no text). Stricter would be to drop the row. Dropping loses the "share removed" signal.
2. Deletions that happen after the 36-hour re-check are not visible in the dumps. Honoring them would need a periodic re-check against Arctic Shift.
