# Checking claims against records

Design note for the claims layer. Checked 2026-11-14. Nothing here is built yet.

## The idea in one picture

The claims layer already plans one evidence source: ratings from independent
fact-checkers. Most talking points are never rated. This note adds a second source:
official statistics, court records and event datasets that a claim can be compared
with directly.

```
                      ┌────────────────────────┐
 talking-point        │ 1. fact-checker rating │──▶ rated false / contested / unrated
 cluster ────────────▶│    (ClaimReview)       │        (unchanged)
                      └────────────────────────┘
        │
        │             ┌────────────────────────┐    contradicted by records
        └────────────▶│ 2. records check       │──▶ consistent with records
                      │    (this note)         │    sources disagree
                      └────────────────────────┘    not checkable
```

The two results are stored side by side and never overwrite each other. The detector
still judges behavior only. Neither result is ever attached to an account, and no
result records which side a claim favors.

## The rule

A claim gets a records check only if it passes all four gates. If it fails any gate
it is `not checkable`, which says nothing about whether it is true.

| Gate | The claim must | Fails, for example |
| --- | --- | --- |
| Countable | State a number, or say a specific event happened | "ICE is out of control" |
| Dated | Name a period, or have one that is unambiguous from the thread date | "deportations are at record highs" with no period |
| Scoped | Name the place, the agency and the thing counted, tightly enough to pick one data series | "millions deported" (removals? returns? self-departures? by ICE or CBP?) |
| Matched | Have a human confirm the claim and the series measure the same thing | Claim says "arrests", series counts "book-ins" |

Then the outcome is decided like this.

| Outcome | When |
| --- | --- |
| `contradicted by records` | Every grade A and B series that matches (at least one, two wherever two exist) puts the claim outside its range, by more than the tolerance |
| `consistent with records` | The claim falls inside the range of the matching grade A and B series |
| `sources disagree` | Grade A and B series conflict with each other so that the claim is inside one and outside another |
| `not checkable` | A gate failed, or only grade C or D material exists |

Five safeguards keep this from turning into ruling on truth loosely.

1. **Tolerance.** Starter value: the claim must differ by more than 25% from the nearest
   edge of the range the sources give, after including published revisions. Tune this
   on a hand-checked sample of 50 claims before trusting it.
2. **Floors only cut one way.** Several sources count only what they could verify (UN
   civilian casualties, named war dead, photographed equipment losses). They are lower
   bounds. A claim below the floor can be contradicted. A claim above it cannot.
3. **Missing is not false.** An event that appears in no dataset is `not checkable`,
   never contradicted. Event claims can be corroborated (two independent records) but
   only contradicted by a document that says otherwise, such as a court order.
4. **A party's own number is not evidence.** Figures that a side in the dispute
   announces (grade C below) are recorded and never decide an outcome. This covers
   agency press statements as much as war ministries.
5. **Frozen copies.** Each check stores the file used, its retrieval date and a hash.
   Official series get revised, so a claim is judged against the data that existed
   when it was made and again against the latest.

Claims are picked for checking by cluster size alone, so the choice cannot lean
toward one side.

### How the result feeds the buckets

Decided by the project owner on 2026-11-14 (separate second figure):

- The three fact-check buckets keep their meaning. "Rated false" still means an
  independent fact-checker rated it false.
- `contradicted by records` is reported as its own count next to them.
- Headline misinformation figure: rated false only. Second figure, always shown with
  it: rated false plus contradicted by records.
- `consistent with records` moves a claim out of nothing. It is a recorded fact about
  the claim, useful for showing that coordinated accounts also push accurate numbers.

## Source grades

| Grade | What it is | Role |
| --- | --- | --- |
| A | Official statistical series or record-level release with a published method | Can decide an outcome |
| B | Independent body's verified count or reprocessing of official records, method published | Can decide an outcome |
| C | A number announced by a party to the dispute, or a think-tank synthesis | Recorded, points to grade A and B data, never decides |
| D | Events coded by machine from news | Leads for finding events, never decides |

For immigration the government is both the record keeper and a party to the argument.
Its statistical tables and records released under public-records law are grade A. Its
press statements are grade C.

## Sources: immigration enforcement

"Read" means the source's own page was read on 2026-11-14. "Not reached" means the
page could not be loaded from the machine that wrote this note, so the entry rests on
general knowledge and must be confirmed before use.

| Source | Grade | Counts | Latest seen | Access | Known gaps and biases | Status |
| --- | --- | --- | --- | --- | --- | --- |
| DHS statistics office (OHSS) monthly enforcement tables | A | Encounters, arrests, book-ins, removals, returns in one place with shared definitions | Unknown | Spreadsheet download, no login (believed) | Press reports in 2026 describe federal immigration data becoming slower and harder to find; check the lag first | Not reached |
| CBP nationwide encounters and public data portal | A | Border Patrol apprehensions and port-of-entry inadmissibles, by border | Unknown | Dashboard; portal offers the underlying files (CSV not confirmed) | "Subject to change"; final only at fiscal year end. An encounter is an event, not a person | Page read, portal not reached |
| ICE detention management data | A | People in detention, book-ins, facilities | Unknown | Tables on the page; spreadsheet not confirmed | "Data may fluctuate until it is locked at the conclusion of the fiscal year". Independent analysts report late releases and unexplained anomalies in 2026 | Page read |
| Deportation Data Project (UC Berkeley and UCLA) | B | Record-level ICE arrests, detainers, detentions, encounters, removals; CBP, court and prosecution records | ICE actions through early March 2026 | Free download, no login mentioned; no license stated; asks for a set citation | The project itself questions the reliability of the removals data. Person identifiers do not link across releases. Arrest city is typed by hand, about 4% missing | Read |
| Vera Institute ICE Detention Trends | B | Daily population, book-ins and book-outs for 1,438 facilities | Through 2026-08-05 (updated 2026-08-28) | GitHub repository | Built from the same released records as the row above, so the two are not independent of each other. Advocacy organization; use its data, not its framing | Read |
| TRAC (Syracuse) | B | Detention, court backlog, prosecutions | Detention as of 2026-07-11 | Web lookup tools only; some tools need a paid subscription | No bulk download | Read |
| Immigration courts (EOIR) | A | New cases, completions, decisions, asylum outcomes; full case-level file | Statistics page updated 2026-10-20; case file dated September 2026 | PDF tables; one large zip, updated monthly | PDFs carry no per-table date. "Statistics provided are subject to change" | Read |
| Federal courts caseload tables | A | Criminal defendants filed by offense and district (covers illegal entry and reentry prosecutions) | Quarter ending 2026-06-30 | Spreadsheets, no login | Quarterly; offense groupings are broad | Read |
| Federal Judicial Center case database | A | Every federal civil and criminal case since 1970 | Not stated | SAS or tab-delimited files; download form needs a browser | Annual files; lags | Read |
| CourtListener bulk files (Free Law Project) | B | Court dockets and opinions, for claims about what a judge ordered | Rebuilt at the end of each quarter | Bulk CSV with no login, public domain. The live API needs an account token (5 requests a minute) | Federal docket coverage is partial. **API token is a sign-up: needs the owner's yes** | Read |
| State Department visa statistics | A | Monthly visa issuances | Not stated | Report files | "Preliminary data which are subject to change" | Read |
| USCIS immigration and citizenship data | A | Applications, asylum, DACA, parole | Unknown | Quarterly files (believed) | | Not reached |
| GAO and DHS inspector general reports | A | Findings about how agencies operate, for process claims rather than counts | Current | Report pages (PDF download not confirmed) | Slow; each covers a narrow question | Read |
| Migration Policy Institute, USAFacts | C | Syntheses of the series above | MPI fact sheet October 2026 | Web | Use only to find which primary series a number came from | Titles seen only |
| ACLED, United States | B | Protests and political violence, for "did this demonstration happen" | Weekly | **Free registration required: needs the owner's yes.** A personal email gets weekly regional totals only, not single events | Terms limit building substitutes for its data | Read |
| GDELT | D | Events and news coverage, every 15 minutes | Live | Free files, no sign-up | Machine-coded, duplicates, follows media attention | Read |

Traps specific to this topic, each of which makes a claim fail the Scoped gate until
resolved:

- "Deportation" can mean removal, return, expulsion or voluntary departure. The series differ several-fold.
- ICE and CBP count separately. Fiscal years start in October.
- Arrests, book-ins and people detained on a given day are three different numbers.
- "Criminal" can mean convicted, charged, or neither.

## Sources: Russia-Ukraine

| Source | Grade | Counts | Latest seen | Access | Known gaps and biases | Status |
| --- | --- | --- | --- | --- | --- | --- |
| UN human rights monitors (HRMMU) civilian casualty updates | B | Civilians killed and injured, individually verified | Update for June 2026, released 2026-07-14: at least 16,431 killed and 48,613 injured since February 2022. Later months likely exist | Monthly PDF | A floor. Little access to Russian-held territory | Read |
| UCDP event datasets (Uppsala) | B | Deaths from organized violence, one row per event with low, best and high estimates | Yearly version 26.1 (through 2025); monthly candidate file for September 2026 | Free CSV, no login, CC BY 4.0 | Conservative by design. Monthly candidate rows are provisional | Read |
| ACLED, Ukraine | B | Battles, strikes, shelling by place | Weekly | Registration required, as above | As above | Read |
| Mediazona and BBC named Russian dead | B | Russian soldiers confirmed dead by name; separate estimate from inheritance records | 286,000 confirmed by 2026-11-06; summary every two weeks | Web; download not described; text CC BY-SA 4.0 | A floor. Russian exile outlet | Read |
| Oryx | B | Equipment lost, each item with a photo or video | Russia 24,275 items; continuously updated | Web page; community scrapers publish tables (MIT) | A floor, stated on the page. Skews to what gets filmed | Read |
| UALosses | C | Ukrainian soldiers dead, missing, captured, by name | 234,573 entries, dates to 2026-10-25 | Web only | Operator unnamed. Its category totals do not add up to its headline. Record it, do not rely on it | Read |
| Ukrainian General Staff and Air Force daily figures | C | Claimed Russian losses; missiles and drones launched and shot down | Daily (community mirror in JSON, MIT) | GitHub; the Kaggle copy likely needs a login | A party's claims | Read |
| Russian Ministry of Defense figures | C | Claimed Ukrainian losses | Daily | Web | A party's claims. Treated exactly as the row above | Not reached |
| WHO attacks on health care | B | Attacks on clinics, staff, transport | Not seen | Dashboard with Excel export; login may be required | Certainty covers whether an attack happened, not its details. Does not name a perpetrator | Read |
| IAEA updates | A | State of nuclear sites, from inspectors on site | Not seen | Web | Describes conditions; avoids assigning blame | Read |
| World Bank damage and needs assessment | B | Cost of damage | Unknown | PDF | Produced jointly with the Ukrainian government | Not reached |
| KSE Institute damage reports | C | Cost of damage | Page updated 2026-11-09, no figure on it | Web | Run with Ukraine's presidential office and ministries, with reparations claims as a stated aim | Read |
| VIINA | D | Events coded from 16 Ukrainian and Russian news outlets; territorial control | Updated daily; 2022 to 2026 files | GitHub zips, open database license | Machine-classified; authors warn against low-accuracy fields. Sources include Russian state media | Read |
| GDELT | D | As above | Live | Free | As above | Read |
| Bellingcat civilian harm map, ISW maps, UN humanitarian data exchange | ? | | | | | Not reached |

Traps specific to this topic:

- Military casualty claims almost always quote grade C figures. The only grade B numbers are floors, so most such claims end as `not checkable`. That is the correct result.
- "Casualties" means killed plus wounded. "Losses" may add captured and missing.
- Territory claims depend on the map date to the day.

## What this changes for the fact-check source

The no-sign-up ClaimReview feed the claims layer planned to try first could not be
confirmed. Its Data Commons address now redirects to Google's Fact Check Tools page,
which shows a sign-in link and no file. If the feed is gone, fact-check ratings need a
Google account and API key, which is a sign-up that waits for the owner. The records
check needs no sign-up for its core sources, so it can be built first.

## Suggested first build

1. One immigration series end to end: federal court prosecutions (quarterly spreadsheets, stable definitions, no login) or immigration court completions.
2. A hand-made sheet of 50 checkable claims from the collected threads, run through the four gates by two people, to measure how often they agree and how many claims are checkable at all.
3. Only then automate claim extraction.

Expect a small share of claims to be checkable. That share is itself a finding to report.

## Before using any source

- Read its terms. None of the grade A and B sources above asked for a login except where marked.
- Confirm every "Not reached" row from a machine that can open the page.
- Sign-ups (ACLED, CourtListener API, Google Fact Check, Kaggle) wait for the owner's yes.
