# Data sources beyond Reddit: what each one's terms allow

Checked 2026-11-14 from each source's own pages unless a row says otherwise.
Reddit is paused (see [DATA_TERMS.md](DATA_TERMS.md)). This note answers: which other
sources can an independent researcher, with no university affiliation, no budget and
no new accounts, use for collection, storage, model training and analysis?

**Short answer: they are not all fair game. Bluesky is the one live source whose terms
do not stand in the way. Telegram and YouTube forbid it outright. Mastodon varies by
server and its community expects to be asked. The best-known labeled bot dataset (TwiBot-22) needs an institutional email.**

```
                    collect   store   train a   no sign-up   topic
                    posts             model     and $0       volume
 Bluesky              ✓        ✓       ~           ✓         measured below
 Mastodon             ~        ~       ~           ~         low, unmeasured
 Hacker News (API)    ✓        ✓       ~           ✓         very low
 Telegram             ✗        ✗       ✗           ✗         high, off limits
 YouTube              ✗        ✗       ~           ✗         high, off limits
 X (live)             ✗        ✗       ✗           ✗         high, off limits

 ✓ nothing in the terms forbids it    ~ silent, unclear or varies    ✗ forbidden or gated
```

## Verdict per source

"Silent" means the page was read and says nothing on the point. Silence is not
permission in writing; it means no clause forbids it.

| Source | Verdict | The clause that decides it | Access and cost | Labeled ground truth |
| --- | --- | --- | --- | --- |
| **Bluesky** | **Use. Primary source.** | Stream docs: "No authentication is required for the live tail; these instances serve the full network." Firehose docs: "No authentication is required." Terms of Service (updated 2025-08-14): "Bluesky does not control how others may use your Content." Terms, Privacy Policy and Community Guidelines are silent on scraping, research, model training and inferring political views. | Jetstream (public JSON stream) and `public.api.bsky.app` ("These direct endpoints do not support authentication"). No account, no key, $0. Live only: history needs a separate backfill. | None published. Candidates to test: Bluesky's own moderation labels and account takedown events (see "Next"). |
| **Mastodon / fediverse** | **Hold. Per-server decision.** | mastodon.social's current terms (effective 2026-08-31) have no clause on scraping, research or model training. They say "Anyone, or any computer, might access your Content" and may "index and store your Content in search engines and databases", and "neither we, nor you, have control over these uses". In June 2025 Mastodon told users that "training LLMs on the data of Mastodon users on our instances is not permitted" (quoted by The Register); that wording is not in the terms now in force. Every other server sets its own rules, so each one needs its own read. | mastodon.social serves hashtag feeds publicly (`hashtag_feeds: public`) and has switched its live feeds off. Default limit: "300 times within 5 minutes". $0. | None. |
| **Telegram** | **Do not collect.** | Content Licensing terms: "Telegram firmly prohibits the scraping, indexing, harvesting, aggregation or use of data obtained from its platform" to "train, fine-tune, validate or otherwise engage in the development" of AI. Terms of Service: the ban applies "to all users, businesses, and third-party services accessing the platform." Bot terms: "Always prohibited uses include any form of data collection aimed at creating large datasets, machine learning models". No research exception; the only exception is per-user consent. | The API needs "your own api_id", which needs an account and phone number. | Published sets only (below). |
| **YouTube comments** | **Blocked. Stays parked.** | Researcher Program: "Be a student, research-focused staff, or faculty member affiliated with an accredited, higher-education institution". The Data API needs a Google Cloud project (a sign-up). Developer Policies also ban scraping and cap storage at 30 days (read from a summary of the page, not word for word). | API key required; higher quota only through the Researcher Program. | None. |
| **Hacker News** | **Allowed through the official API only. Poor fit.** | API README: "we're making the public Hacker News data available in near real time" and "There is currently no rate limit." Site terms forbid the other route: "you will not engage in or use any data mining, robots, scraping or similar data gathering or extraction methods." | Official Firebase API, no key, $0. | None. Politics is a small share of the site (from general knowledge, not measured). |
| **X, live collection** | **Do not collect.** Already decided: no paid pull. | Developer Agreement (updated 2026-04-27) forbids using "the X API or X Content to fine-tune or train a foundation or frontier model" and using it "to derive or obtain non-public information of individual X users". Sharing is limited: "you may only distribute Post IDs, Direct Message IDs, and/or User IDs". | Paid API. | See published sets. |

## Published datasets

| Dataset | Verdict | What decides it | Labels |
| --- | --- | --- | --- |
| **TwiBot-22** | **Blocked as written.** This changes the current plan, which trains on it. | README: "Please apply for access by contacting [the author] with your **institutional email address**", plus institution, advisor and use case. Code is MIT; the data is gated. | Bot or human per account (X, collected before 2022). |
| **X information operations archive** | **Needs Andrew's yes** (it asks for an email address). | "You can download the datasets by entering your email address". "Your use of the datasets is governed by the X Developer Agreement and Policy." | "37 datasets of attributed platform manipulation campaigns originating from 17 countries", 2018 to early 2022. Small accounts are hashed. The strongest free ground truth for state-run coordination found so far. Unhashed data needs an institutional consortium. |
| **Indiana University Bot Repository** (includes Fox8-23) | **Unclear for an independent researcher.** | "The data hosted by Bot Repository is provided for academic research only." | Bot or human, several sets. The dataset list did not load. |
| **Community Archive** (X) | **Use for method tests.** | Opt-in: "tweets that members of the community have chosen to make public". Goal: "create a public domain dataset". Code license MIT. | None. 27.0M tweets from 2.4K members. |
| **Pushshift Telegram** (Zenodo 3607497) | **Usable under its own license, with a caveat.** | "Creative Commons Attribution 4.0 International". Published 2020-01-14: before the Russia-Ukraine full-scale invasion and before Telegram's current ban on building datasets. | None. 317M messages, 28K channels. |
| **TGDataset** (Telegram) | **Hold.** No license stated on the page read. | 120,979 channels, about 71 GB zipped. | Per channel: Telegram's own `scam` and `verified` flags; a labeled conspiracy list. |
| **Bluesky dumps on Hugging Face** | **No need.** The live stream gives the same data with deletions honored. | One card says "Dataset usage is subject to Bluesky's Terms of Service" while tagged Apache-2.0. | None found. |
| **Platform threat reports** (Meta, Google, OpenAI, Anthropic) | **Read as narrative, not data.** | The index pages did not expose account-level data or terms when read. Not verified further. | Named campaigns and tactics, useful for the red-team list, not for training. |

## First real samples from Bluesky

Two 15-minute listens on 2026-11-14, each followed by a fetch of every thread the
immigration topic touched. No account or key was used. Both fell between 5:30 and 6:30
in the morning US Eastern, when the US was mostly asleep, so treat the topic rates as a
floor.

| | First listen, 10:30 UTC | Second listen, about 11:20 UTC |
| --- | --- | --- |
| All posts in 15 minutes | 251,661 (**suspect**, see below) | 28,927 |
| Immigration enforcement posts | 33 | 45 |
| Russia-Ukraine posts | 343 | 237 |
| Immigration posts a day at this rate | about 3,200 | about 4,300 |
| Russia-Ukraine posts a day at this rate | about 33,000 | about 23,000 |
| Stored thread starts and replies | 32 and 143 | 45 and 1,815 |
| Accounts stored | 110 | 1,566 |

- **The topic counts agree between the two listens. The count of all posts does not.** The first listen counted about 8 times more posts overall than the second, and a direct check of the stream agreed with the second (about 1,900 new posts a minute, or about 2.8 million a day). The cause is not found: a real burst of posts at that time, or a counting fault. Quote topic counts, not shares of all posts. The 6-hour listen below settles it.
- The second listen counted "ICE" in capitals and stored far more replies, because its on-topic threads were larger.
- The later stages ran unchanged on the Bluesky files.
- Asking the stream for reuse declarations as well as posts works: the stream accepted the request and sent no declarations, as expected while the proposal is a draft.
- Keyword search of past posts works without an account on `api.bsky.app` (answered 200) and is refused on `public.api.bsky.app` (403). The docs describe both hosts as taking no authentication. Search is the likely route to past weeks of threads; not built.

## The 6-hour listen: is there enough immigration talk on Bluesky?

**Yes.** One listen from 10:47 to 16:47 UTC on 2026-11-14 (5:47 to 11:47 in the morning
US Eastern), on Andrew's computer, no account or key.

| Topic | Posts in 6 hours | Posts a day at this rate |
| --- | --- | --- |
| All of Bluesky | 1,069,559 | about 4.3 million |
| Immigration enforcement | 3,405 | about 13,600 |
| Russia-Ukraine | 7,362 | about 29,400 |

What was stored, together with the first 15-minute sample:

| Measure | Value |
| --- | --- |
| Threads the topic touched | 2,590, of which the busiest 578 were fetched (the run's limit was 600) |
| Thread starts and replies stored | 610 and 46,745 |
| Accounts that replied | 25,614 |
| Accounts with 5 or more replies (enough for a feature row) | 1,490 |
| Accounts with 20 or more replies | 40 |
| Replies per thread, middle and top tenth | 9 and 251 |
| Replies within 30 seconds of the post they answer | 1.1% |
| Replies from accounts that say they are bots | 0.14% |

- **Volume is enough to work with.** Six hours gave 784 accounts a feature row in the window the features stage looks at. A week of listening would give several times that.
- **No coordinated cluster was found**, and no account had near-duplicate replies. With the first-guess thresholds and six hours of data that is weak evidence either way: the thresholds have never been checked against labeled data.
- **The sample leans to big threads.** Fetching the busiest threads first means quiet threads are under-represented. Raise the limit or sample threads at random before quoting rates.
- **The stream dropped four times in six hours** and the run reconnected each time. Short gaps mean the daily rates are slightly low.
- **The first sample's count of all posts was a one-off.** This run and three direct checks give 1,900 to 3,000 new posts a minute depending on the hour, with no duplicates. The first sample's 16,800 a minute came from the stream at that moment, most likely a replay of a backlog, and not from the code. Its topic counts were in line with the rest.

## Why Bluesky, and what it cannot do

- **The whole network is public by design and offered without an account.** Collection is the intended use of the stream, not a workaround.
- **Deletions arrive on the same stream**, so honoring them is practical. Bluesky's Developer Guidelines say: "All services must have a method for deleting content a user has requested to be deleted."
- **Timing is trustworthy.** Every post carries the time Bluesky's server first saw it, to the second.
- **Limits:**
  - Bluesky's users are not a cross-section of the public, and political talk there leans one way. Rates measured there do not transfer to Reddit or X. Side comparisons stay gated as before.
  - The stream is live only. Past events need a backfill of account histories, which is not built.
  - Known Russia-Ukraine influence operations run mostly on Telegram and X, which are off limits. For that topic, published sets (the X archive, Pushshift Telegram) are the route, not live collection.
  - There is no labeled bot data for Bluesky.
- **A norm to respect, not a rule:** Bluesky's draft proposal "User Intents for Data Reuse" would let each account state whether it allows "generative AI", "protocol bridging", "bulk datasets" and "public archiving and preservation". It calls itself "This draft proposal", and says the signal "does carry ethical weight, but is not legally enforceable." It expects that "Companies and research teams building AI training sets are expected to respect this intent when they see it". The draft names the record `org.user-intents.declaration` as a placeholder and has not been adopted, so no account can set it today. The collector already listens for it: see the duties table.

## What the collector does about each duty

| Duty | Where it comes from | What `ea/bluesky.py` does |
| --- | --- | --- |
| Do not abuse rate limits | Community Guidelines: do not "bypass, abuse, or disrupt Bluesky's systems, security features, APIs, rate limits" | One stream connection; half a second between thread requests; at most 300 threads a run; the cached public endpoint Bluesky asks developers to use. |
| Delete what users delete | Developer Guidelines | Posts deleted before a thread is fetched are never stored. Deletes heard on the stream in any later run remove the stored post. Deletes that happen while nothing is listening are missed: see open question 1. |
| Protect user data | Developer Guidelines: "reasonable security measures" | Account identifiers and post addresses are replaced by keyed hashes before writing. Handles, display names and profiles are never stored. |
| Respect reuse preferences | Bluesky's draft "User Intents for Data Reuse" | The listener also watches for reuse declarations. An account that says no to "bulk datasets" is never stored, and anything already stored from it is removed. No answer means no stated preference, as the draft defines it. The record name is a setting, to update when the draft is adopted. |
| Authenticity, never politics | Project rule | No field records a side. Topic keywords select threads, not accounts. |

Model training on Bluesky data is not forbidden by any clause read. It is also not
granted. The model here learns from numbers about behavior (reply times, repeated
text, acting in step), not from the wording of posts.

## Possible ground truth: Bluesky's moderation labels

Bluesky's own moderation service publishes its labels, and answers queries without an
account. Its declared labels include five that bear on this project:

| Label | Bluesky's description |
| --- | --- |
| `inauthentic` | "Bot or a person pretending to be someone else." |
| `spam` | "Unwanted, repeated, or unrelated actions that bother users." |
| `engagement-farming` | "Insincere content or bulk actions aimed at gaining followers, including frequent follows, posts, and likes." |
| `impersonation` | "Pretending to be someone else without permission." |
| `scam` | "Scams, phishing & fraud." |

The collector keeps the labels on each stored account in `labels.jsonl`, apart from
the features, with the account hashed. A label an account puts on itself is marked as
such. A labeler other than Bluesky's own service is stored hashed, since one person may run it. Labels are possible ground truth only; they are never a feature, or the model
would learn to copy Bluesky's moderators.

### Measured on 2026-11-14

**Bluesky's own labels are too sparse to be ground truth for bots.**

- In the second sample, 3 of 1,566 accounts (0.19%) carried any of the five labels above, all three `spam`. 8 accounts (0.51%) carried any label from Bluesky's moderators.
- Over 3.7 days (2026-11-10 to 2026-11-14), Bluesky's moderation service put `spam` on 190 accounts, about 52 a day, and withdrew 46 of them (24%). It issued no `inauthentic`, `engagement-farming` or `scam` labels on accounts at all.
- **Account takedowns are the larger signal:** 13,484 in the same 3.7 days, about 3,600 a day, with 690 (5.1%) withdrawn. A takedown does not say why, so it is evidence of a rule broken, not of automation.
- Bluesky has switched off bulk label queries ("Support for wildcard label queries is temporarily disabled"). The label stream still replays without an account.

**Independent labeler services publish much closer labels.** 322 labeler services were found through the public account search. Those most relevant here:

| Service | Labels that matter here |
| --- | --- |
| labeler.hailey.at | `ai-agent`, `coordinated-abuse` (member of a coordinated abuse network), `new-acct-replies` (10 or more replies in an account's first hour), `reply-link-spam`, `suss-handle-change` |
| pef-moderation.org | `network` (part of a network of coordinating peers), `fake`, `repetitive`, `engagementfarmer` |
| bottags.bsky.social | `undisclosed-bot`, `bot-reply` |
| ab4n-labeler.bsky.social | `matryoshka` (pro-Kremlin disinformation burner accounts) |
| moderation.blueat.net | `bot-net`, `sus-new` |
| stechlab-labels.bsky.social | `no-gap-more-than-one-hours` and similar: posted without a long break in the last 24 hours |

How to read these:

- They are **weak labels**: each is one team's or one person's judgment, with unknown error rates. None has been checked here.
- Some are **computed from the same behavior this project measures**. The stechlab labels are a timing rule. Training on such a label and then scoring with timing features would only teach the model that rule.
- The `matryoshka` label is the one found so far that speaks to the Russia-Ukraine topic.
- Using them fits the project rules as long as the account stays hashed and no labeled account is ever named.

## Next

1. **Ground truth on Bluesky, in two parts.** (a) Listen to the label stream of Bluesky's moderators and of three or four independent labelers, hash each labeled account, and join to the collected accounts: this gives "later taken down" and "labeled by others" flags without storing any identifier. (b) Check each labeler by hand on a small sample before trusting it, and keep labels built from timing rules out of training.
2. **Listen for a week**, with threads sampled at random as well as busiest first, then set the cluster thresholds against labeled accounts.
3. **Replace TwiBot-22** as the training set, or find a route to it. Options: the X information operations archive (needs an email address), or planted synthetic behavior only.
4. **Backfill** past weeks of on-topic Bluesky threads from public account histories.

## Where this stopped (paused 2026-11-14)

Andrew paused the project until the following week. State at the pause:

**Built and tested, not yet run on real data** (this branch):
- `python -m ea bluesky --hours 168` runs for days in hourly cycles. Each cycle listens, picks the 25 busiest on-topic threads and 100 drawn at random, and fetches them a day later once their replies are in. Each stored thread start records `picked = busiest | random`, so rates can be measured on the random ones. Each day's records go in their own folder. Stream drops are ridden out. Fetching stops at 4,000 MB per topic.
- `config/bluesky_russia_ukraine.toml`: the second topic, same settings, its own phrase lists.
- The collector asks the API for the labels of five independent labeler services and stores them by short name, hashed per account. The timing-rule labeler is stored as `timing-rules` and must never be used for training.

**Not started:**
1. The week-long listen on Andrew's computer. Before starting: check free disk space, then start one run per topic with `nohup`, in the worktree `~/Projects/engagement-authenticity-bluesky` after updating it to main.
2. Joining collected accounts to Bluesky takedowns (from the moderation label stream, `!takedown` on accounts) as a second weak label.
3. Talking-point clustering with a small local embedding model, to fill the `point` column that `ea/network_features.py` expects, then running the fused cross-account signals on the 6-hour data.
4. Reporting per-label overlap, cluster results and any coordinated groups found, in aggregate only.

Nothing is running on Andrew's computer. The 6-hour data and the label sample stay in the two worktrees' git-ignored `data/` folders.

## Open questions for the project owner

1. A delete that happens while the collector is not running is missed. Stricter options: run the listener continuously, or re-check every stored post on a schedule, or drop all text older than N days.
2. Bluesky's Developer Guidelines ask developers to keep "updated contact information accessible to the public including an email address". They are written for apps and services on the network. If collection becomes continuous, publishing a contact address for removal requests would be the careful reading. That is a public posting, so it waits for a yes.
