# Metrics

How this project measures whether the detector is any good. The code is in
`ea/metrics.py`; the report is written by `python -m ea evaluate`.

## Two kinds of error

| Name here | What happened | Also called | Cost |
| --- | --- | --- | --- |
| **False flag** | A person was called software | False positive, false accept | High. A real person is wrongly accused |
| **Miss** | Software passed as a person | False negative, false reject | Lower. The estimate of how much is automated comes out too small |

Every cut-off on the model's score trades one for the other:

```
 cut-off low  ◀───────────────────────────────▶  cut-off high
 many false flags, few misses      few false flags, many misses
                        ▲                    ▲
                    crossover          where we operate
```

## The numbers, and what each is for

| Number | Question it answers | Use it for |
| --- | --- | --- |
| **Crossover error rate** | At the cut-off where false flags and misses are equally common, how common are they? | The headline for comparing two models or two feature sets. Lower is better |
| **Software caught at a 1% false-flag rate** (and 0.1%) | If at most 1 person in 100 may be falsely flagged, what share of software do we catch? | The operating KPI. This is how the detector is actually run |
| **Precision at realistic prevalence** | Of flagged accounts, how many really are software, if software is 1%, 2%, 5% or 10% of all accounts? | Deciding whether a flag means anything in the wild |
| **Calibration** (Brier score, reliability table) | Does a score of 0.8 mean about 80%? | Turning scores into "an estimated X% of comments" |
| **False-flag rate per control group** | Which kinds of real people does the detector wrongly flag? | Finding who gets hurt, before publishing |
| **Share caught per red-team group** | Which evasion tactics get past the detector? | Finding what an adversary would do next |

Every rate comes with a 95% range, found by recomputing it on 1,000 resamples of the
test accounts (bootstrap).

## Crossover error rate, and its limit

The crossover error rate (equal error rate) comes from biometrics and access control,
where it is the standard single number for comparing systems. It ignores the cut-off,
so two models can be compared fairly.

It treats both errors as equally bad. This project does not: a false flag costs far
more than a miss. So the crossover is the comparison number, never the operating
point. Two models with the same crossover can differ a lot in the low-false-flag
corner, so always read it next to "software caught at a 1% false-flag rate".

## How many labeled people are needed

A rate cannot be measured more finely than the test set allows.

| To measure a false-flag rate of | Labeled people needed in the test set |
| --- | --- |
| 1% | at least 100 |
| 0.1% | at least 1,000 |

The report says "not measurable" rather than print a number the sample cannot
support. With zero false flags among n people, the true rate could still be as
high as 3/n, and the prevalence table uses that worst case.

## Why precision is recomputed for prevalence

The labeled sample deliberately oversamples suspicious accounts, so precision
measured on it is flattering. Worked example: software is 1% of accounts, all of it
is caught, and 1% of people are falsely flagged. Out of 10,000 accounts that is 100
software accounts flagged and 99 people flagged. Half the flags are wrong.

## Control groups and red-team groups

Optional file `data/<topic>/labels/groups.csv`:

```
author_hash,kind,group
3f9a...,negative_control,power users
77c1...,negative_control,people who polish posts with AI
b20e...,red_team,level 3 sock puppets
```

- `negative_control`: real people, or stand-ins for them, who look like software. Every flag is a false flag. Lower is better.
- `red_team`: planted software. The share flagged is the share caught. Higher is better.

The report gives the share flagged per group, hides groups under 10 accounts, and
leaves out accounts the model trained on.

## A security view of the same numbers

- **The adversary adapts.** A score on yesterday's bots says little about tomorrow's. Red-team groups by evasion level show how fast detection decays as the adversary tries harder.
- **Test on the future.** Train on the earlier window, test on the later one, so the test resembles deployment.
- **Report the cost to the attacker, not only the catch rate.** A signal that is cheap to evade (reply speed: add a delay) matters less than one that is expensive (acting together at scale).
- **Assume the detector is known.** Nothing here relies on the method staying secret.

## Not built yet

- Curves on the normal-deviate scale usual for DET plots. The current plot uses plain axes.
- Prevalence estimates for the real population, using the recorded sampling weights.
- Co-engagement counts are not adjusted for how active an account is, so very active real people look more coordinated than they are. Seen on synthetic data while building this.
