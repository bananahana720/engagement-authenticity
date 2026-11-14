# Labels: three questions

Decided 2026-11-14. This page replaces section 2 of the research plan ("Account
classes and labeling guide"), which used one six-way label. The definitions the
code enforces are in `ea/label.py`.

## Why three questions

The six labels mixed three different things: who operates the account, whether the
account changed owner, and why it posts. Coordination belongs to a group, not to one
account. Motive cannot be seen from outside, so it is no longer labeled at all.

```
            per account                          per group
 ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
 │ 1 WHO IS TYPING?  │ │ 2 CHANGED HANDS?  │ │ 3 IN A CLUSTER?   │
 │   a person labels │ │   a person labels │ │   the detector    │
 └───────────────────┘ └───────────────────┘ └───────────────────┘
```

## Question 1: who is typing?

Column `who_types`. Answered by a labeler, per account.

| Answer | Meaning | Evidence that supports it |
| --- | --- | --- |
| `person` | A person writes and posts each comment | Sleep gaps, varied wording, back-and-forth conversation, replies that take reading time |
| `person_plus_automation` | A person is involved, with software doing part of the work | Human-sounding replies mixed with scheduled or queued bursts; long stretches of even spacing; pasted text alongside real conversation |
| `program` | Software writes and posts, no person per comment | Active in all 24 hours, replies within seconds, templated or near-identical text |
| `declared` | Says what it is: a declared bot, or an organization (news desk, brand, agency) | Name or profile says so; consistent single function. Organizations are the best-known false "bot" flags, so they get their own answer |
| `uncertain` | Evidence conflicts or is too thin | Fewer than about 20 comments, or strong signs both ways |

## Question 2: did the account change hands?

Column `changed_hands`. Answered by a labeler, per account.

| Answer | Meaning | Evidence that supports it |
| --- | --- | --- |
| `no_sign` | Nothing suggests a new owner | Continuous history, stable interests and rhythm |
| `likely` | Likely taken over or sold | Silence of six months or more, then a sharp change in language, topics or active hours; old history deleted |
| `uncertain` | Some signs, not enough | A gap without a clear change, or a change without a gap |

The collected data holds only topic threads, so this question usually needs a look
at the account's wider history. Until that is collected, expect many `uncertain`.

## Question 3: is the account in a coordinated cluster?

Column `cluster` in `data/<topic>/clusters.csv`. Set by `ea/clusters.py`. Labelers never answer it.

| Answer | Meaning |
| --- | --- |
| `in_cluster` | The detector found it in a group of three or more accounts that act together |
| `none_found` | Tested, nothing found. Not proof there is none |
| `not_tested` | Too little activity to test |

Two accounts are linked when they keep appearing in the same threads within minutes
of each other, or repeatedly post nearly the same words within an hour. Being in a
cluster shows acting together. It does not show intent, who is behind the group, or
that anything said is false. Friends and fan groups move together too.

## Labeling checklist

Label the account, never a single comment. One row of the sheet (`to_label.csv`) per
account; example comments are in `to_label_examples.csv`.

- [ ] Read the example comments before looking at the numbers.
- [ ] Rhythm: how many hours of the day are active? Is there a sleep gap? (`active_hours`, `longest_quiet_hours`)
- [ ] Reply speed: are replies faster than the parent could be read? (`median_reply_seconds`, `fast_reply_share`)
- [ ] Wording: varied and responsive, or templated and repeated? (`near_duplicate_share`)
- [ ] History: any long silence followed by a different persona?
- [ ] Self-description: does it say it is a bot or an organization?
- [ ] Answer `who_types`.
- [ ] Answer `changed_hands`.
- [ ] Set `confidence` from 1 (guess) to 3 (sure).
- [ ] Write one sentence in `evidence` for every `program`, `person_plus_automation` or `likely`. Without it, that answer is turned into `uncertain`.

Do not record or guess:

- which side of the topic the account is on,
- why it posts (paid, sincere, bored),
- whether its claims are true,
- which country or which AI model might be behind it,
- whether it is in a cluster. The detector answers that.

Observe only. Never reply to an account or send it a trick prompt to test it.
"Reads like AI wrote it" is weak evidence and never enough for `program`.

## Agreement

Two people label the same accounts without seeing each other's answers. Then:

```
python -m ea agreement --sheets labeler_a.csv labeler_b.csv
```

Kappa is reported for each question separately, overall and per answer. Target 0.6
or above on each question before labeling more. Where it is lower, rewrite that part
of this page with the disagreeing examples and label a fresh batch.

## The old six labels, for charts

They can still be derived (`old_six_labels` in `ea/label.py`), so results stay
comparable with earlier work. The first rule that fits wins.

| Old label | Derived from |
| --- | --- |
| E disclosed automation | `who_types` = declared |
| B hijacked or rented | `changed_hands` = likely |
| A automated | `who_types` = program or person_plus_automation |
| C (was "human, incentive-driven") | `who_types` = person and `cluster` = in_cluster. Motive is no longer part of it |
| D ordinary human | `who_types` = person, `changed_hands` = no_sign, not in a cluster |
| U uncertain | Everything else |

`person_plus_automation` has no old equivalent. It is counted under A here, which is a
judgment call.

## What the model learns

Only question 1: software typing (`program`, `person_plus_automation`) against
`person`. `declared` and `uncertain` are left out. Question 2 is not modeled yet,
and question 3 is answered by the cluster detector directly.
