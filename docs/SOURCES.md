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

## First real sample from Bluesky

One 15-minute listen on 2026-11-14 from 10:30 UTC (5:30 in the morning US Eastern, so the
US was mostly asleep), then a fetch of every thread the immigration topic touched.
No account or key was used. One short window at a quiet hour: treat the daily rates as
a floor, not an estimate.

| Topic | Posts in 15 minutes | Share of all posts | Posts a day at this rate |
| --- | --- | --- | --- |
| All of Bluesky | 251,661 | 100% | about 24 million |
| Immigration enforcement | 33 | 0.013% | about 3,200 |
| Russia-Ukraine | 343 | 0.136% | about 33,000 |

- Stored: 32 thread starts and 143 replies from 110 accounts. Only 3 accounts had the 5 replies needed for a feature row, so nothing can be modeled from this yet.
- The later stages ran unchanged on the Bluesky files.
- The immigration keywords for this run left out bare "ICE" to avoid "ice cream". The config now counts "ICE" when written in capitals, which should raise the count.
- Keyword search of past posts works without an account on `api.bsky.app` (answered 200) and is refused on `public.api.bsky.app` (403). The docs describe both hosts as taking no authentication. Search is the likely route to past weeks of threads; not built.

## Why Bluesky, and what it cannot do

- **The whole network is public by design and offered without an account.** Collection is the intended use of the stream, not a workaround.
- **Deletions arrive on the same stream**, so honoring them is practical. Bluesky's Developer Guidelines say: "All services must have a method for deleting content a user has requested to be deleted."
- **Timing is trustworthy.** Every post carries the time Bluesky's server first saw it, to the second.
- **Limits:**
  - Bluesky's users are not a cross-section of the public, and political talk there leans one way. Rates measured there do not transfer to Reddit or X. Side comparisons stay gated as before.
  - The stream is live only. Past events need a backfill of account histories, which is not built.
  - Known Russia-Ukraine influence operations run mostly on Telegram and X, which are off limits. For that topic, published sets (the X archive, Pushshift Telegram) are the route, not live collection.
  - There is no labeled bot data for Bluesky.
- **A norm to respect, not a rule:** Bluesky's draft proposal "User Intents for Data Reuse" would let each account state whether it allows "generative AI", "protocol bridging", "bulk datasets" and "public archiving and preservation". It calls itself "This draft proposal", and says the signal "does carry ethical weight, but is not legally enforceable." It expects that "Companies and research teams building AI training sets are expected to respect this intent when they see it". Whether accounts can set it today was not checked. The collector does not read it yet: see "Next".

## What the collector does about each duty

| Duty | Where it comes from | What `ea/bluesky.py` does |
| --- | --- | --- |
| Do not abuse rate limits | Community Guidelines: do not "bypass, abuse, or disrupt Bluesky's systems, security features, APIs, rate limits" | One stream connection; half a second between thread requests; at most 300 threads a run; the cached public endpoint Bluesky asks developers to use. |
| Delete what users delete | Developer Guidelines | Posts deleted before a thread is fetched are never stored. Deletes heard on the stream in any later run remove the stored post. Deletes that happen while nothing is listening are missed: see open question 1. |
| Protect user data | Developer Guidelines: "reasonable security measures" | Account identifiers and post addresses are replaced by keyed hashes before writing. Handles, display names and profiles are never stored. |
| Authenticity, never politics | Project rule | No field records a side. Topic keywords select threads, not accounts. |

Model training on Bluesky data is not forbidden by any clause read. It is also not
granted. The model here learns from numbers about behavior (reply times, repeated
text, acting in step), not from the wording of posts.

## Next

1. **Ground truth on Bluesky.** Test whether Bluesky's moderation labels (spam, impersonation) and account takedown events on the stream can serve as labels, the way removal flags do on Reddit. Bluesky's moderation service answers label queries without an account (checked: the query returned a list of labels). What the labels cover, and how many accounts carry one, is not yet measured.
2. **Replace TwiBot-22** as the training set, or find a route to it. Options: the X information operations archive (needs an email address), or planted synthetic behavior only.
3. **Respect reuse preferences.** Find out whether accounts can already state a "bulk datasets" preference, and if so skip accounts that say no.
4. **Backfill** past weeks of on-topic Bluesky threads from public account histories.

## Open questions for the project owner

1. A delete that happens while the collector is not running is missed. Stricter options: run the listener continuously, or re-check every stored post on a schedule, or drop all text older than N days.
2. Bluesky's Developer Guidelines ask developers to keep "updated contact information accessible to the public including an email address". They are written for apps and services on the network. If collection becomes continuous, publishing a contact address for removal requests would be the careful reading. That is a public posting, so it waits for a yes.
