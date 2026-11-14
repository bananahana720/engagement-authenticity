# Data sources and terms

Read on 2026-11-14 from Andrew's PC, which can reach redditinc.com and Arctic Shift.
This is a plain reading by the project, not legal advice.

## Verdict

Reddit's terms do not forbid this project's collection and aggregate measurement.
One clause does reach the **model** stage: Reddit's Developer Terms bar training
"algorithmic models" on Reddit data without Reddit's permission. Until that is
settled, train only on synthetic data, or get permission first (see "Decision for
the project owner" below).

## What was read

| Document | Version | Source |
| --- | --- | --- |
| Reddit Data API Terms | Effective 2023-06-19, revised 2026-07-20 | redditinc.com/policies/data-api-terms |
| Reddit Developer Terms | Effective 2024-09-24, revised 2026-03-24 | redditinc.com/policies/developer-terms |
| Reddit User Agreement | Effective 2026-07-01, revised 2026-05-26 | redditinc.com/policies/user-agreement |
| Reddit Public Content Policy | Not read: the help-centre page returned 403, and the redditinc.com page is only a short pointer | |
| Arctic Shift README and API notes | GitHub master, 2026-11-14 | github.com/ArthurHeitmann/arctic_shift |

## What the Reddit terms say that matters here

1. **The terms follow the data, not only the API.** The Data API Terms say you may use
   "Materials you access from our Services, whether using the Data APIs or not, only
   in accordance with the Data API Terms." So the safe reading is that data from an
   archive such as Arctic Shift is still covered.
2. **Model training needs permission.** Developer Terms: you must not use Reddit data
   "to train large language, artificial intelligence, or other algorithmic models or
   related services without our permission." The Data API Terms say the same for User
   Content and add that, on termination, you must delete "any data or models that were
   derived from User Content." This project's classifier trains on derived numbers
   (reply times, overlap rates), not on text, but "other algorithmic models" is broad
   enough that it may cover it.
3. **Research beyond the free tier needs an agreement.** Both the Data API and
   Developer Terms: "research in excess of rate limits" or any use not expressly
   permitted needs a separate agreement with Reddit.
4. **Deleted content must be deleted.** Developer Terms: if User Content is deleted or
   removed on Reddit, you delete it "as soon as possible", and on request from Reddit
   or the user.
5. **Keep only what the use case needs.** Data API Terms: do not retain data "beyond
   your approved use case" and delete what is not required.
6. **No scraping Reddit itself.** The User Agreement forbids collecting data from
   Reddit's services by automated means without consent. This project never touches
   reddit.com: it reads Arctic Shift.
7. **No harassment.** Data API Terms forbid using the data to "spam, incentivize, or
   harass users." The rule against naming accounts covers this.

## What Arctic Shift says

- A community project, not Reddit. It collects through Reddit's API and republishes
  as dumps, an API and a single-subreddit download tool.
- No licence and no research-use terms are stated. The API has "no uptime or
  performance guarantees" and the docs ask nothing beyond reasonable use.
- It runs a removal-request form, and lists the forms of other archives.
- Private and quarantined subreddits are not included.
- Since 2023-11 each record is fetched again after about 36 hours. Records removed in
  between carry `_meta.was_deleted_later` and `_meta.removal_type`. Later deletions
  are not visible.
- Field selection (`fields=`) exists but cannot select `_meta`, so the collector
  downloads full records and cuts them down in memory before writing.

## How the code is built around this

| Concern | What the code does |
| --- | --- |
| Usernames | Replaced by a keyed hash before writing. The key stays in `.secrets/`, outside git. |
| Extra fields | Dropped in memory. Kept: id, subreddit, time, thread, parent, text, removed flag. |
| Removed or deleted text | Not kept. The row stays, with an empty body and `removed = true`. |
| Deleted authors | Their comments are not kept at all. |
| Titles during measurement | `python -m ea topic-share` matches titles in memory and writes only counts. |
| Seed accounts | Reddit's 2018 published list is stored under `data/seeds/` with a hashed copy for joins. Never committed, never republished. |
| Load on a free service | One request a second, a plain user agent with no personal details, resumable so nothing is fetched twice. |
| Publication | Reports hold counts and rates only. Groups under 10 accounts are hidden. |

## Decision for the project owner

1. **Model training on real Reddit data.** Options: (a) apply to Reddit's researcher
   programme (r/reddit4researchers) or write to Reddit for permission before running
   `python -m ea train` on real data; (b) keep the model stage on synthetic data and
   use transparent rule scores on real data until permission arrives. Recommended:
   (a), doing (b) meanwhile. A sign-up waits for Andrew's yes.
2. **Late deletions.** Honouring deletions after the 36-hour re-check needs a periodic
   re-check of stored ids against Arctic Shift. Recommended before any result is
   published.
3. A comment its author deleted keeps a row here (time and thread, no text). Stricter
   would be to drop the row. Dropping loses the "share removed" signal.
