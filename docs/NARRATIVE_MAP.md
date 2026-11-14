# The narrative grounding map

Design note. Written 2026-11-14. Nothing here is built yet.

The question it answers: **what are the main points of view in a topic, what does each
one rest on, does that basis hold up, and who is carrying it?**

## The idea in one picture

Every recurring point of view ("narrative") gets placed on one grid. The rows say how
well its basis holds up. The columns say who carries it.

```
                              WHO CARRIES IT
                   mostly ordinary          disproportionately
                   accounts                 coordinated or automated
                 ┌────────────────────────┬──────────────────────────┐
 holds up        │ ordinary debate        │ amplified true claim     │
                 ├────────────────────────┼──────────────────────────┤
 contested       │ live dispute           │ amplified dispute        │
                 ├────────────────────────┼──────────────────────────┤
 contradicted    │ sincere error          │ campaign candidate       │
                 ├────────────────────────┼──────────────────────────┤
 no basis given  │ rumor                  │ amplified rumor          │
                 └────────────────────────┴──────────────────────────┘
        (value statements such as "this is cruel" sit outside the rows:
         they have no basis to check, but still get a column)
```

Reading it:

- The **rows come from content only** (the claims layer). The **columns come from
  behavior only** (the detector). Neither ever looks at the other.
- "Good faith or malicious" is not a fifth measurement. It is a reading of where a
  narrative lands. The right-hand column is the only evidence of bad faith this project
  can produce, and it is evidence about **how a narrative is distributed**, never about
  a person.
- The bottom-left cell matters as much as the bottom-right. A false claim carried by
  ordinary people is a sincere error, not a campaign.
- The top-right cell matters too. Coordinated accounts pushing an accurate claim are
  still inauthentic behavior. It is not misinformation.

## What "individual record" means here

The request was for both aggregate profiles and individual records. Both exist, with
one hard limit.

```
 allowed, stored                          allowed, reported               never built
 ───────────────                          ─────────────────               ───────────
 one row per COMMENT                      one row per NARRATIVE           one row per ACCOUNT
 narrative, claim, what it cites,   ──▶   size, basis, grade,             with a narrative
 source grade                             carriage                        or a side on it
```

The individual record is the comment, not the account. A table that lists accounts
beside the narratives they push is a table of people's political views. X's terms
forbid inferring that, and the project applies the same rule to Reddit. The join from
comment to account to behavior flag happens only inside the step that computes the
narrative-level counts, and a test should fail the build if any output has one row per
account with a narrative column.

Narratives do carry a side tag, because comparing the two sides is one of the goals
(see "Comparing the two sides"). The tag lives on the narrative, is assigned by a
person, and is hidden from every step that clusters, grades or flags. It is read only
by the final comparison step. Accounts never get one.

## How a point of view ties to its basis

A point of view rests on its basis through two links, and each can break separately.
(This is Toulmin's model of an argument: grounds, warrant, claim.)

```
   SOURCE ───── link 1 ─────▶ STATED FACT ───── link 2 ─────▶ POINT OF VIEW
 "ICE detention              "61,000 people in              "detention is at a
  table, Aug 2026"            detention in August"           record high"

   does the source                is the fact                 does the fact
   say that?                      itself right?               support the point?
   (fidelity)                     (claims layer)              (support)
```

| Where it breaks | What it looks like | How it is checked |
| --- | --- | --- |
| No source at all | A factual claim with nothing cited | Automatic: no link and no named source |
| Link 1, fidelity | Source does not say it, says the opposite, or does not exist | Hand check on a sample |
| The fact | The number or event is wrong | Claims layer: fact-check rating, then records check |
| Link 2, support | The fact is real but does not carry the conclusion: cherry-picked period, wrong series, missing context | Hand check, top narratives only |
| The source itself | The source is a party to the dispute, or an advocacy paper | Source grade and source type, below |

Link 2 is where most sophisticated misleading argument lives, and it is the one the
claims layer does not cover. A true number with the wrong period passes every fact
check and still misleads.

## The pipeline

```
 comments ─▶ 1. split into ─▶ 2. cluster into ─▶ 3. extract ─▶ 4. grade ─▶ 5. measure ─▶ the grid
 (collected)    claims           narratives         the basis     the basis    carriage
                                                                  ▲            ▲
                                                     claims layer ┘            └ detector
                                                     (CLAIMS_EVIDENCE.md)        (ea/model.py)
```

### 1. Split comments into claims

One comment often holds several claims. The unit is the sentence.

- Keep sentences that assert something: contain a number, a date, a named agency, a
  link, or a reporting verb ("according to", "data shows").
- Tag each kept sentence as **factual** (could in principle be checked) or **value**
  (should, cruel, deserve, a prediction). Value sentences stay in the map and skip the
  grading. Having values is not a defect.
- Start with rules. Check them against 200 hand-tagged sentences before adding a model.

### 2. Cluster claims into narratives

A narrative is a set of sentences making the same assertion in different words.

- **Baseline first:** word-overlap vectors (TF-IDF) and the near-duplicate code already
  in `ea/features.py`. No model download.
- **Then:** sentence embeddings from a small open model run locally, density clustering
  (HDBSCAN), top terms per cluster for a name. This is the BERTopic recipe. Free, runs
  on a laptop.
- A person names each of the top clusters in one neutral sentence that states the
  assertion ("ICE detention reached a record level in 2026"), not a theme
  ("detention").
- Two sentences that assert opposite things belong in two narratives. Embeddings often
  put them together because the words match. Check for this by hand in every top
  cluster.

### 3. Extract the basis

For each sentence in a narrative, record what it leans on.

| Field | Values | How |
| --- | --- | --- |
| `basis_kind` | link, named source, personal account, none | Links by pattern; named sources from a list of agencies, courts and outlets |
| `source_type` | statistical agency, agency press statement, court, academic, fact-checker, news, think tank, advocacy group, social post or video, other | Lookup table by domain, filled by hand for the 50 most-cited domains |
| `source_grade` | A, B, C, D, or ungraded | Same grades as `CLAIMS_EVIDENCE.md`. News is ungraded: it points to a primary source or it does not |
| `funder` | Free text and where it was found | Hand-entered for think tanks and advocacy groups. A recorded fact, never a verdict |

Media-bias ratings (left/right scores for outlets) are deliberately not used. They
would bring a side label in through the back door.

The **basis profile** of a narrative is the share of its sentences in each
`source_type`. This alone is a useful output: it shows which narratives run on agency
tables, which on think-tank papers, and which on nothing.

### 4. Grade the basis

Each top narrative gets one core claim, written by a person, and that claim goes
through the claims layer unchanged. Then one summary grade:

| Grade | When |
| --- | --- |
| `holds up` | Core claim is consistent with records, and the cited facts support the point (link 2) |
| `contested` | Fact-checkers rate it mixed or missing context, or sources disagree, or the only basis is grade C material, or link 2 is partial |
| `contradicted` | Fact-checker rated false, or contradicted by records. The two are stored and reported separately, as already decided |
| `no basis given` | Factual claim, not checkable, and fewer than 10% of its sentences cite anything |
| `not checkable` | Factual, cited, but fails the four gates. Says nothing about whether it is true |
| `value statement` | Not a factual claim |

Expect `not checkable` and `no basis given` to be the largest groups. The map must
show them at full size. A map that shows only the gradeable narratives would overstate
how much is known.

### 5. Measure carriage

For each narrative, from the detector's output:

| Measure | Definition | Why |
| --- | --- | --- |
| Flagged share of accounts | Flagged accounts carrying it ÷ all accounts carrying it | The basic number |
| Flagged share of comments | Same, by comments | Flagged accounts post more, so the two differ |
| Lift | Flagged share ÷ flagged share across the whole topic, same subreddits and weeks | A narrative in a bot-heavy subreddit is not thereby special |
| Early share | Flagged share among the first 10% of sentences by time | Seeded by coordinated accounts and then picked up by ordinary people is a different pattern from the reverse |

A narrative goes in the right-hand column only when the lower end of the 95% interval
on lift is above 2 and at least 20 flagged accounts carry it. Both numbers are starter
values. Below either bar it is "mostly ordinary or unknown", which is the safe default.

Two further readings, to be added after the first experiment:

- **Persistence after correction.** Does a narrative's rate change after a fact-check
  of it is published? Compare flagged and ordinary carriers in aggregate.
- **Fidelity failures by carrier group.** Are misquoted or made-up citations
  concentrated among flagged accounts?

## Comparing the two sides

Automation is not one side's tool. Coordinated accounts can push a narrative, and
they can also be deployed against one, as a counter-voice. So the comparison has two
parts: what each side's narratives look like, and what happens where the sides meet.

```
          SIDE 1 narratives                     SIDE 2 narratives
        ┌───────────────────┐                 ┌───────────────────┐
        │ carriage          │   replies       │ carriage          │
        │ grades            │ ──────────────▶ │ grades            │
        │ basis profile     │ ◀────────────── │ basis profile     │
        │ timing            │   replies       │ timing            │
        └───────────────────┘                 └───────────────────┘
          compared like for like      counter-messaging: who replies
                                      across the line, how fast, how
                                      coordinated
```

**The side tag.** Each top narrative is tagged by a person as `favors enforcement`,
`opposes enforcement`, or `neither`. Expect a real share of `neither`: process claims,
cross-cutting complaints, plain news. Two people tag independently. Narratives they
disagree on become `neither`.

**Like-for-like measures**, each reported for both sides in the same table:

| Measure | What it compares |
| --- | --- |
| Carriage | Flagged share of accounts and of comments, lift, early share, pooled over each side's narratives |
| Concentration | Share of a side's flagged volume that sits in its top 3 narratives. A few heavily pushed lines, or a thin spread |
| Grades | Share of each side's narratives, weighted by size, in each grade, including `not checkable` |
| Basis profile | Share of sentences by `source_type` |
| Timing | Hours from a triggering event (an operation, a ruling, a data release) to a narrative's peak, and how long it lasts. Flagged and ordinary carriers separately |

**Counter-messaging.** Reddit records which comment each reply answers. A cross-side
reply is a reply containing a sentence from one side's narrative, posted under a
comment containing a sentence from the other side's.

| Measure | Definition |
| --- | --- |
| Cross-side reply rate | Cross-side replies ÷ all replies to that side's narrative comments |
| Flagged share of cross-side repliers | Compared with the flagged share of same-side repliers |
| Response time | Median seconds from parent to cross-side reply, flagged and ordinary separately |
| Swarming | Several flagged accounts replying to the same comment within minutes |

This also fixes a weakness in the grid. A reply that quotes a claim to rebut it shows
up here as counter-messaging instead of being counted as carrying the claim.

**Is the detector fair to both sides?** A side difference in carriage means nothing
if the detector simply flags one side's ordinary users more often. Three checks, all
run before any side comparison is reported:

1. **False-flag rate by side of content.** Take comments by accounts that people
   hand-labeled as ordinary. Compute the share that the detector flags, separately for
   comments in side 1 narratives and side 2 narratives. The two rates should match
   within their intervals. This is computed over comments and discarded; no account is
   assigned a side.
2. **Miss rate by side of content.** The same for accounts hand-labeled as automated
   or coordinated: is one side's coordination being caught less often?
3. **Leave-one-signal-out.** Rerun the comparison dropping each signal family in turn
   (timing, text similarity, co-engagement, coded phrases). A side difference that
   exists under only one family is a property of that signal, not of the sides.

**Why a side difference may still not mean what it seems.** Stated with every result:

- The sides live in different subreddits with different moderation. Removed comments
  are not kept, so heavier moderation hides more.
- Official data counts what agencies choose to count. One side's claims may be
  checkable more often, which moves its grades in both directions.
- The sides may coordinate in different ways. Coordination arranged off the platform
  leaves weaker timing traces than a scheduler does.
- "Two sides" is a simplification of the people involved, and it describes narratives
  only.
- The comparison says who is amplified more and on what basis. It never says which
  side is right.

## Evaluation

| What | Test | Bar |
| --- | --- | --- |
| Claim tagging (factual or value) | 200 hand-tagged sentences | Precision 0.8 or above on "factual" |
| Cluster coherence | A person reads 20 random sentences per top narrative and counts those that make the named assertion | 16 of 20, in at least 15 of the top 20 narratives |
| Cluster stability | Rerun with a different seed and a 80% subsample | Top 20 narratives reappear |
| Basis extraction | 200 sentences, hand-checked | Named-source recall reported, no bar yet |
| Grades | Two people grade the same narratives | Kappa 0.6 or above, same bar as account labels |
| Carriage | Shuffle narrative labels among comments within each subreddit and week, 1,000 times | Reported lift lies outside the shuffled range |
| Symmetry | Narratives chosen by size only; graders see the core claim without its side tag; report the `not checkable` rate for every narrative | No bar. Published with every result |
| Side tags | Two people tag independently | Kappa 0.6 or above; disagreements become `neither` |
| Detector fairness | The three checks under "Comparing the two sides" | False-flag rates for the two sides overlap within their 95% intervals, or no side comparison is reported |

## Ways this goes wrong

| Failure | What happens | Guard |
| --- | --- | --- |
| Sincere people cite bad studies | A contradicted narrative looks like a campaign | The columns exist for this. Contradicted and ordinary is "sincere error" |
| Coordinated accounts push true claims | A campaign is missed because its content checks out | The rows never clear the columns. "Amplified true claim" is its own cell |
| Circular measurement | The detector uses near-duplicate text and phrase bursts. Copies of one sentence also form one narrative. So copy-paste narratives look coordinated by construction | Recompute carriage with a detector that uses timing and co-engagement only. Report both |
| Detector error flows through | Every carriage number inherits the detector's mistakes, and the detector has not seen real data yet | Precision-first threshold makes flagged share a floor. Carry the detector's interval into the lift. No carriage claims before the detector passes its own evaluation |
| Slogans and headlines | Thousands of real people repeat one headline within an hour | Near-identical text that matches a linked headline is excluded from text-based signals |
| Opposite claims merged | "Rose" and "fell" land in one cluster | Hand check of top clusters; split on negation and direction words |
| Values graded as facts | An opinion gets marked "no basis given" | Value tag in step 1; graders can overrule |
| Missing comments | Removed text is not kept, so the most extreme material is absent, unevenly by subreddit | Report removed share per narrative-week where a narrative can still be assigned from surviving replies; otherwise state the gap |
| Quoting to rebut | A comment repeats a claim in order to argue against it | Carriage means "appears in", and the report says so. Replies across the side line are counted as counter-messaging |
| Side tag leaks onto accounts | Joining accounts to side-tagged narratives yields each account's politics | The join runs only inside the aggregation step. No output has an account column next to a side or a narrative |
| Grading drifts toward one side | Checkable claims may cluster on one side because of what official data happens to count | Selection by size, blind grading, and the `not checkable` rate shown beside every grade |
| Small cells expose people | A narrative carried by 6 accounts identifies them | No cell under 20 accounts is reported |
| Reading the grid as a verdict | "Campaign candidate" gets quoted as "campaign" | Wording rule: "carried disproportionately by coordinated accounts". Never "malicious", never about a person |

## First experiment (immigration)

Smallest version that tests whether the map is worth building. It needs real Reddit
comments, which the cloud machine still cannot reach, so it starts when collection
does.

```
 r/immigration ─▶ factual      ─▶ TF-IDF     ─▶ top 20      ─▶ hand-grade ─▶ one table,
 one month        sentences       clusters      by size        20 claims     20 rows
```

1. Collect r/immigration for one month (`python -m ea collect --only immigration`).
2. Split into sentences, keep factual ones with the rules in step 1.
3. Cluster with the baseline. Take the 20 largest narratives. Name each.
4. For each: basis profile by `source_type`, one core claim, the four gates, the
   records check where it passes.
5. Carriage: the detector has no labels yet, so use only the self-declared bot rule
   and the co-engagement feature as a provisional column, marked provisional.

6. Tag each narrative's side. Count cross-side replies and their median response time.
   The fairness checks need hand labels, so the side comparison of carriage waits.

Output: one table with 20 rows (narrative, side, size, basis profile, grade,
provisional carriage), the grid with 20 dots, and a cross-side reply count.

It succeeds if:

- at least 15 of 20 narratives pass the coherence read;
- at least 5 of 20 core claims pass the four gates. Fewer means the rows of the grid
  will be mostly `not checkable` for this topic, which is itself the finding;
- two graders reach kappa 0.6 on the grades.

If the baseline clusters fail the coherence read, repeat step 3 with embeddings before
concluding anything.

## Open choices

Defaults are in place. Each can be changed without redoing the design.

1. **Lift threshold of 2 and minimum of 20 flagged accounts.** Starter values.
2. **10% citation cut-off for `no basis given`.** A guess until the first basis
   profiles exist. Link rates in Reddit political comments are probably low.
3. **Who names the narratives and writes the core claims.** One person is a single
   point of bias. Default: one person writes, a second reviews blind.
4. **Whether a language model helps with steps 1 and 3.** It would be faster and may be
   more accurate. Default: rules and a person first, so the first experiment has no
   model in the loop to audit.

## Background

From memory, not rechecked for this note.

- Toulmin, *The Uses of Argument* (1958): claim, grounds, warrant.
- Wardle and Derakhshan, *Information Disorder* (2017): misinformation (false, no
  intent to harm), disinformation (false, intent), malinformation (true, used to harm).
  The grid's four corners are this split made measurable.
- François, "Actors, Behaviors, Content" (2019): judge influence operations by
  behavior and actors, not by content alone.
- Grootendorst, BERTopic (2022): embed, reduce, cluster, name.
- Kazemi and others, "Claim Matching Beyond English" (2021): grouping differently
  worded versions of one claim.
- Pacheco and others, "Uncovering Coordinated Networks on Social Media" (2021): the
  co-engagement approach the detector already uses.
