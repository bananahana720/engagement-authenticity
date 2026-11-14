# engagement-authenticity

Research into how to tell automated, hijacked and coordinated ("astroturf") accounts
from ordinary people in political comment threads, starting with Reddit threads about
immigration enforcement and deportation.

The project measures **behavior, in aggregate**. It is not a tool for naming anyone.

## Rules the code follows

1. **Authenticity, never politics.** An account is labeled by how it behaves, never by which side it is on and never by motive. No feature records a side.
2. **Aggregate only.** Reports contain counts and rates. No account is named, and usernames are replaced by a keyed hash before anything is written to disk.
3. **Precision first.** Calling a real person a bot is the costly error. Accounts below the operating point are "uncertain", not "cleared" and not "bot".
4. **Behavior over style.** The signals are timing and cross-account coordination. "Sounds like AI" is not used.
5. **Minimum data.** Only the fields the features need are kept. Text removed on Reddit is not kept.
6. **Content is separate.** Whether a claim is false is a different question from whether an account is authentic. The detector never judges truth.

## The pipeline

```
 collect ──▶ features ──▶ label ──▶ model ──▶ evaluate
 Arctic      one row      sheet     trees +    aggregate
 Shift       per account  for       calibration report
 threads     per window   people    (time split)
```

The plan lists "label" second. In practice the labeling sheet is drawn from the
features, so `features` runs first.

| Stage | Command | Reads | Writes |
| --- | --- | --- | --- |
| Measure volumes | `python -m ea volumes` | Arctic Shift API | `reports/immigration/volumes.csv` |
| Collect | `python -m ea collect` | Arctic Shift API or dump files | `data/immigration/collected/<subreddit>/*.jsonl` |
| Features | `python -m ea features` | collected | `data/immigration/features.csv`, `data/immigration/clusters.csv`, `reports/immigration/phrase_spread.csv` |
| Label | `python -m ea sample`, then `python -m ea agreement --sheets A.csv B.csv` | features | `data/immigration/labels/to_label.csv` |
| Model | `python -m ea score --model <model.joblib>` | features, a model trained on non-Reddit data | `data/immigration/model/` |
| Evaluate | `python -m ea evaluate` | model output | `reports/immigration/evaluation.md` |

Every stage reads the previous stage's files, so any stage can be rerun alone.
`data/` and `reports/` are never committed.

## Try it in one minute

```
pip install -r requirements.txt
python -m ea demo        # the whole pipeline on made-up data
python -m pytest         # the tests
```

The demo plants invented programs and coordinated rings among invented ordinary
accounts and checks that the model finds the programs and the cluster detector finds the rings. It proves the code runs. It says
nothing about real Reddit.

## Red team: do the signals survive smarter sock puppets?

```
python -m ea redteam            # about 10 minutes on 4 cores
python -m ea redteam --quick    # one minute, to check the code
```

Builds made-up worlds where LLM-driven sock-puppet networks hide among ordinary
accounts, at five evasion levels (copy-paste bots, reworded text, human timing,
sparse and distributed, patient operator), and measures which signals still find
them at 90% precision. The background includes nine kinds of real account that
must not be flagged (people who polish posts with AI, organic brigades, power
users and more), and the report gives the false-flag rate on each.

Results and how to read them: [docs/redteam/RESULTS.md](docs/redteam/RESULTS.md).

Model-fingerprint test (which model family wrote a short reply, and what hides it):
[docs/redteam/fingerprint/RESULTS.md](docs/redteam/fingerprint/RESULTS.md).
All text is built from invented words, so nothing it writes could be posted anywhere.

## Run it on real data

Read [docs/DATA_TERMS.md](docs/DATA_TERMS.md) first.

1. `python -m ea volumes` counts comments per subreddit per month. Look at the numbers before downloading anything: they decide which subreddits are practical.
2. `python -m ea collect --only immigration` downloads one subreddit. It is slow on purpose (one request a second to a free community service) and can be stopped and restarted.
3. `python -m ea features`, then `python -m ea sample` to get a labeling sheet.
4. Two people label the same sheet following [docs/LABELS.md](docs/LABELS.md) (columns `who_types`, `changed_hands`, `confidence`, `evidence`), then `python -m ea agreement`. Target kappa 0.6 or above on each question before labeling more.
5. Save the agreed labels as `data/immigration/labels/labels.csv`, then `python -m ea score --model data/demo/model/model.joblib` and `python -m ea evaluate`.

**No model is trained on Reddit data** until Reddit gives permission. `python -m ea train` refuses Reddit data; models are trained on non-Reddit data and only score Reddit accounts. See [docs/DATA_TERMS.md](docs/DATA_TERMS.md).

Have dump files instead? `python -m ea collect --posts-dump RS_2026-01.zst --comments-dump RC_2026-01.zst`.

## What each file does

| File | What it does |
| --- | --- |
| `config/immigration.toml` | The topic: subreddits, dates, title keywords, coded phrases, thresholds |
| `ea/pseudonymize.py` | Username to keyed hash; spots self-declared bots |
| `ea/collect.py` | Download, cut down to the minimum, load |
| `ea/features.py` | Timing, near-duplicate text, co-engagement, coded-phrase bursts |
| `ea/label.py` | The label questions and their allowed answers; labeling sheet with sampling weights; Cohen's kappa |
| `ea/clusters.py` | Finds groups of accounts that act together |
| `ea/model.py` | Learns "is software typing?"; gradient-boosted trees, calibrated; split by time and by account |
| `ea/metrics.py` | Crossover error rate, software caught at a fixed false-flag rate, confidence ranges, calibration |
| `ea/evaluate.py` | The aggregate report: those metrics, checks by subreddit and control group, cluster counts and reach |
| `ea/synthetic.py` | Made-up data with known answers |
| `ea/network_features.py` | Cross-account signals: shared talking points, pivots, active days, arrival cohorts |
| `ea/redteam_generate.py` | Made-up sock-puppet networks at five evasion levels, plus negative controls |
| `ea/redteam.py` | The red-team experiment: which signals survive each level |
| `ea/fingerprint_generate.py` | Replies from small open models run locally, for the fingerprint test |
| `ea/fingerprint.py` | Can a classifier name the model family behind a text? By length and evasion |

## How it is judged

A **false flag** is a person called software. A **miss** is software passed as a person. The headline for comparing models is the crossover error rate; the operating KPI is the share of software caught when at most 1% of people are falsely flagged. Details: [docs/METRICS.md](docs/METRICS.md).

## Labels: three questions

| Question | Answers | Who answers |
| --- | --- | --- |
| Who is typing? | person, person plus automation, program, declared bot or organization, uncertain | A labeler, per account |
| Did the account change hands? | no sign, likely, uncertain | A labeler, per account |
| In a coordinated cluster? | in a cluster, none found, not tested | The detector, per group |

Motive is never labeled. Full guide and checklist: [docs/LABELS.md](docs/LABELS.md).

## Signals (version 1)

| Feature | Tests | Idea |
| --- | --- | --- |
| `active_hours`, `hour_entropy`, `longest_quiet_hours` | H1 | People sleep; schedulers do not |
| `median_reply_seconds`, `fast_reply_share` | H2 | Replies faster than anyone can read |
| `near_duplicate_share` | H5 | Different accounts, nearly the same words, within an hour |
| `co_engage_best_partner_threads`, `co_engage_repeat_partners` | H6 | The same accounts turning up together again and again |
| `phrase_use_share`, `phrase_burst_share` | coded phrases | Several accounts picking up the same rallying phrase within minutes |
| `removed_share` | support | Share of an account's comments later removed |

Coded phrases are tracked from both sides with equal-sized lists and merged before
any feature is computed, so no feature can tell which list a phrase came from.

## Not built yet

- Signals and a model for "changed hands" (H4: long dormancy, then a new persona). Needs account history beyond the topic threads.
- Checking the cluster thresholds against labeled data. They are first guesses.
- Karma-farming history (H8) and account age (H3). The comment records carry no account creation date.
- Holding out whole coordinated clusters in the test set.
- Prevalence estimates weighted back to the real population, with confidence intervals. The sampling weights are already recorded.
- The claims layer: cluster recurring talking points and match them to independent fact-check ratings (rated false / contested / unrated). Separate from the detector by design.
- X data (Community Archive, TwiBot-22) for method tests.
