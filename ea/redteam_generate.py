"""Red team, part 1: made-up sock-puppet networks hidden among made-up ordinary accounts.

Nothing here is real and nothing here is fit to post anywhere: comment text is built
from invented words ("bavo ketu ..."), so it carries meaning only for this experiment.

How text works. There are 300 invented *concepts*, each with 4 invented *words* that
mean the same thing. A *talking point* is a set of 8 concepts. Two comments push the
same talking point when they draw on the same concepts, even if they share no word.
That is the stand-in for "an LLM reworded it": same meaning, different surface.

Text also has a *register*: 20 of the 80 filler words are "machine register", the
smooth connective words that machine-written text overuses. That is the stand-in for
"sounds like AI". People use them rarely; LLM-written text uses them often, and less
often when told to imitate a person.

The ordinary background is built to cause false alarms on purpose. All of these are
real people (or disclosed bots) that must NOT be flagged. Each is a negative control:
  ordinary     own rhythm, mostly own words, sometimes a popular talking point
  junkie       post a lot and pick up new talking points within minutes
  community    real groups who share 3 favourite talking points (a fan community)
  ai_polished  write their own comments, then have an AI polish the wording
  translator   non-native speakers who post through a machine translator
  regular      power users and moderators: fixed daily hours, or replies within minutes
  crossposter  paste their own comment into several threads
  brigade      real users who arrive together from one linked post, 4 times
  newcomer     joined during a breaking news event, with many others
  disclosed    bots that say they are bots (set aside by rule before any scoring)
plus, across everyone: news bursts (dozens of people reacting to the same story
within 15 minutes), viral copy-pasted slogans, and shift workers with no sleep gap.

The puppet networks climb an evasion ladder. Each level keeps the evasions below it:
  naive     copy-paste text, replies in seconds, around the clock, whole ring per thread
  style     every comment reworded (shares no wording with its siblings)
  jitter    each persona keeps human hours and human reply delays
  sparse    one puppet per thread, 30% campaign comments hidden in ordinary cover
  patient   the operator also gives up speed: accounts made over months, no shared
            push days, new talking points picked up no faster than real people do

Through "sparse" one operator still leaks: accounts start in a batch, pushes happen on
the operator's days, and the ring pivots to a new talking point within hours of news.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ea.features import all_phrases

HOUR, DAY = 3600, 86400
N_CONCEPTS, WORDS_PER_CONCEPT, N_FILLER, N_MACHINE = 300, 4, 80, 20
# Share of an account's filler words that are machine register: (low, high), drawn per account.
HUMAN_REGISTER = 0.03
POLISHED_REGISTER = (0.1, 0.8)          # people differ in how much they let the AI rewrite
LLM_AS_HUMAN_REGISTER = (0.1, 0.5)      # an LLM told to write like a person still leaks some
N_EVERGREEN, N_EVENTS, CONCEPTS_PER_POINT = 20, 40, 8

LEVELS = [
    {"key": "naive", "name": "Naive bots",
     "text": "copy", "timing": "robotic", "spread": "swarm", "operator": "hasty"},
    {"key": "style", "name": "+ human-style text",
     "text": "reword", "timing": "robotic", "spread": "swarm", "operator": "hasty"},
    {"key": "jitter", "name": "+ human timing",
     "text": "reword", "timing": "human", "spread": "swarm", "operator": "hasty"},
    {"key": "sparse", "name": "+ sparse and distributed",
     "text": "reword", "timing": "human", "spread": "sparse", "operator": "hasty"},
    {"key": "patient", "name": "+ patient operator",
     "text": "reword", "timing": "human", "spread": "sparse", "operator": "patient"},
]
LEVEL_BY_KEY = {level["key"]: level for level in LEVELS}


def vocabulary():
    """The same invented words in every world: (concept words, filler words).
    The first N_MACHINE filler words are the machine register."""
    rng = np.random.default_rng(12345)
    consonants, vowels = list("bdfgklmnprstvz"), list("aeiou")
    words = set()
    while len(words) < N_CONCEPTS * WORDS_PER_CONCEPT + N_FILLER:
        syllables = rng.integers(2, 4)
        words.add("".join(rng.choice(consonants) + rng.choice(vowels) for _ in range(syllables)))
    words = sorted(words)
    rng.shuffle(words)
    concept_words = np.array(words[:N_CONCEPTS * WORDS_PER_CONCEPT]).reshape(N_CONCEPTS, WORDS_PER_CONCEPT)
    return concept_words, np.array(words[N_CONCEPTS * WORDS_PER_CONCEPT:])


@dataclass
class World:
    posts: pd.DataFrame
    comments: pd.DataFrame
    accounts: pd.DataFrame      # index author_hash; columns kind, target (1 = puppet)
    truth: pd.DataFrame         # per comment: true talking point (-1 = none), campaign or not.
                                # For checking the generator and the detector. Never a feature.


def make_world(cfg, level, seed, n_ordinary=2000, n_networks=4, network_size=20,
               campaign_share=0.3, n_threads=1500):
    """One world: ordinary background plus puppet networks at one evasion level."""
    level = LEVEL_BY_KEY[level]
    rng = np.random.default_rng(seed)
    start = int(pd.Timestamp(cfg["start"], tz="UTC").timestamp())
    end = int(pd.Timestamp(cfg["end"], tz="UTC").timestamp())
    subs = cfg["subreddits"]["all_threads"] + cfg["subreddits"]["filtered"]
    phrases = all_phrases(cfg)
    concept_words, fillers = vocabulary()

    # ----- talking points: some always around, some born at a news event -----
    n_points = N_EVERGREEN + N_EVENTS
    point_concepts = np.array([rng.choice(N_CONCEPTS, CONCEPTS_PER_POINT, replace=False)
                               for _ in range(n_points)])
    event_times = np.sort(rng.integers(start + 5 * DAY, end - 10 * DAY, N_EVENTS))
    born = np.concatenate([np.full(N_EVERGREEN, start - 60 * DAY), event_times])
    weight = rng.permutation(1 / np.arange(1, n_points + 1) ** 0.8)
    slogans = [" ".join(list(concept_words[point_concepts[p], 0]) + list(rng.choice(fillers, 4)))
               for p in range(n_points)]

    def popularity(times):
        """How much each talking point is in the air at each time: rows times, columns points."""
        age = (np.asarray(times)[:, None] - born[None, :]) / DAY
        fresh = np.where(np.arange(n_points) >= N_EVERGREEN, 6 * np.exp(-np.clip(age, 0, None) / 4), 0)
        return np.where(age >= 0, weight * (1 + fresh), 0)

    # ----- threads: a steady flow plus bursts after news -----
    after_news = rng.random(n_threads) < 0.3
    thread_event = np.where(after_news, rng.integers(0, N_EVENTS, n_threads), -1)
    thread_time = np.where(after_news, event_times[thread_event] + rng.exponential(DAY, n_threads),
                           rng.integers(start, end - 2 * DAY, n_threads))
    thread_time = np.clip(thread_time, start, end - 2 * DAY).astype(np.int64)
    order = np.argsort(thread_time)
    thread_time, thread_event = thread_time[order], thread_event[order]
    thread_weight = rng.lognormal(0, 1, n_threads)
    thread_sub = rng.choice(subs, n_threads)
    posts = pd.DataFrame({
        "id": [f"p{i}" for i in range(n_threads)], "subreddit": thread_sub,
        "author_hash": [f"poster{i % 97:03d}" for i in range(n_threads)],
        "disclosed_bot": False, "created_utc": thread_time, "removed": False,
        "title": "immigration thread"})

    rows, kinds = [], {}

    def add(author, thread, when, body, point=-1, campaign=False):
        rows.append((author, thread, int(min(when, end - 1)), body, point, campaign))

    # ----- text -----
    def reworded(point, machine=HUMAN_REGISTER):
        """The talking point in fresh words: same concepts, different surface, extra chatter."""
        concepts = rng.choice(point_concepts[point], rng.integers(5, 8), replace=False)
        words = list(concept_words[concepts, rng.integers(0, WORDS_PER_CONCEPT, len(concepts))])
        words += list(chatter(rng.integers(4, 10), machine))
        rng.shuffle(words)
        return " ".join(words)

    def chatter(n, machine):
        concepts = rng.integers(0, N_CONCEPTS, n)
        own = concept_words[concepts, rng.integers(0, WORDS_PER_CONCEPT, n)]
        filler = np.where(rng.random(n) < machine, rng.choice(fillers[:N_MACHINE], n),
                          rng.choice(fillers[N_MACHINE:], n))
        return np.where(rng.random(n) < 0.4, filler, own)

    def own_words(machine=HUMAN_REGISTER):
        return " ".join(chatter(rng.integers(8, 20), machine))

    def with_phrase(text, chance):
        return text + " " + str(rng.choice(phrases)) if rng.random() < chance else text

    # ----- human rhythm -----
    def keep_waking_hours(times, wake_hour):
        """Move times that fall in a persona's 9 sleeping hours to after they wake."""
        times = np.asarray(times, dtype=float)
        hours_awake = ((times // HOUR) - wake_hour) % 24
        return np.where(hours_awake < 15, times, times + (24 - hours_awake) * HOUR)

    def human_delay(n, median=2 * HOUR, spread=1.2):
        return rng.lognormal(np.log(median), spread, n)

    def phrase_habit():
        """How often an account uses a coded phrase: most rarely, a few in every third comment."""
        return float(rng.beta(0.5, 6))

    def ordinary_comments(author, how_many, joined, wake_hour=None, taste=None, point_share=0.45,
                          delay_median=2 * HOUR, arrive_now=False, habit=None, machine=HUMAN_REGISTER,
                          fixed_hours=None, crosspost=False):
        """Comments the way a real person makes them. Also used as cover for puppets."""
        if habit is None:
            habit = phrase_habit()
        if wake_hour is None:
            wake_hour = int(rng.integers(0, 24))
        first = np.searchsorted(thread_time, joined)
        first = min(first, n_threads - 20)
        p = thread_weight[first:] / thread_weight[first:].sum()
        threads = first + rng.choice(n_threads - first, size=min(how_many, n_threads - first), replace=False, p=p)
        if arrive_now:                                  # came for the news: first comment right away
            threads[0] = min(first + int(rng.integers(0, 5)), n_threads - 1)
        times = thread_time[threads] + human_delay(len(threads), delay_median)
        if fixed_hours is not None:                     # only ever online in the same two hours of the day
            slot = rng.choice(fixed_hours, len(times))
            times = (times // DAY + 1) * DAY + slot * HOUR + rng.uniform(0, HOUR, len(times))
        elif wake_hour >= 0:
            times = keep_waking_hours(times, wake_hour)
        in_air = popularity(times) * (taste if taste is not None else 1)
        hot = thread_event[threads]
        in_air[hot >= 0, N_EVERGREEN + hot[hot >= 0]] *= 4      # the thread's own news story
        for thread, when, air in zip(threads, times, in_air):
            point = -1
            if rng.random() < point_share and air.sum() > 0:
                point = int(rng.choice(n_points, p=air / air.sum()))
                text = slogans[point] if rng.random() < 0.03 else reworded(point, machine)
            else:
                text = own_words(machine)
            text = with_phrase(text, habit)
            add(author, thread, when, text, point)
            if crosspost and rng.random() < 0.5:        # the same text again in nearby threads
                for other in rng.integers(max(thread - 5, 0), min(thread + 6, n_threads), 2):
                    add(author, int(other), max(when, thread_time[other]) + rng.uniform(60, 600), text, point)

    def join_time():
        if rng.random() < 0.5:
            return start, False
        if rng.random() < 0.25:
            return int(rng.choice(event_times) + rng.exponential(DAY)), True
        return int(rng.integers(start, end - 20 * DAY)), False

    def how_many_comments(low=6):
        return int(np.clip(rng.lognormal(np.log(18), 0.7), low, 150))

    # ----- ordinary background: every kind here is a real person and must not be flagged -----
    community_taste = []
    for _ in range(6):
        taste = np.ones(n_points)
        taste[rng.choice(n_points, 3, replace=False)] = 25
        community_taste.append(taste)
    share = n_ordinary / 2000                           # control group sizes for 2000 accounts
    sizes = [("community", 72), ("junkie", 100), ("ai_polished", 160), ("translator", 60),
             ("regular", 40), ("crossposter", 40), ("brigade", 75)]
    plan = sum(([kind] * max(6, round(size * share)) for kind, size in sizes), [])
    plan += ["ordinary"] * (n_ordinary - len(plan))
    joined_at = {}
    for n, kind in enumerate(plan):
        author = f"o{n}"
        joined, arrive_now = join_time()
        wake_hour = -1 if rng.random() < 0.05 else None         # 5% keep no regular sleep gap
        options = {"arrive_now": arrive_now}
        how_many = how_many_comments()
        if kind == "community":
            how_many = how_many_comments(10)
            options.update(taste=community_taste[n % 6], point_share=0.6)
        elif kind == "junkie":
            how_many *= 2
            options.update(point_share=0.5, delay_median=20 * 60)
        elif kind in ("ai_polished", "translator"):
            options.update(machine=rng.uniform(*POLISHED_REGISTER))
        elif kind == "regular" and n % 2:                       # scheduled: the same two hours every day
            how_many = how_many_comments(40)
            options.update(fixed_hours=rng.choice(24, 2, replace=False))
        elif kind == "regular":                                 # moderator: there within minutes, all day
            how_many = how_many_comments(40)
            options.update(delay_median=3 * 60)
        elif kind == "crossposter":
            options.update(crosspost=True)
        elif kind == "brigade":
            joined, options = start, {}
        elif arrive_now:
            kind = "newcomer"
        kinds[author] = kind
        joined_at[author] = joined
        ordinary_comments(author, how_many, joined, wake_hour, **options)

    people = np.array(list(kinds))
    joined_of = np.array([joined_at[a] for a in people])

    def crowd(thread, size):
        """Real accounts that could turn up in this thread: already joined when it was posted."""
        here = people[joined_of <= thread_time[thread]]
        return rng.choice(here, min(size, len(here)), replace=False)

    # Brigades: real users who follow one link to one thread and all say their piece.
    brigade_members = people[[kinds[a] == "brigade" for a in people]]
    for members in np.array_split(brigade_members, 3):
        for thread in rng.choice(n_threads, 4, replace=False):
            point = int(rng.integers(0, N_EVERGREEN))
            arrive = thread_time[thread] + rng.uniform(20 * 60, 6 * HOUR)
            for m in members[rng.random(len(members)) < 0.6]:
                add(m, int(thread), arrive + rng.uniform(0, 30 * 60), reworded(point), point)

    # News bursts: a story breaks and dozens of people react to it within 15 minutes.
    for event in range(N_EVENTS):
        thread = min(np.searchsorted(thread_time, event_times[event]), n_threads - 1)
        for a in crowd(thread, max(5, round(30 * share))):
            add(a, int(thread), thread_time[thread] + rng.uniform(60, 900),
                reworded(N_EVERGREEN + event), N_EVERGREEN + event)

    # Viral copypasta: the same slogan, word for word, from unrelated people within an hour.
    for _ in range(10):
        thread, point = int(rng.integers(0, n_threads)), int(rng.integers(0, N_EVERGREEN))
        for a in crowd(thread, max(5, round(15 * share))):
            add(a, thread, thread_time[thread] + rng.uniform(0, HOUR), slogans[point], point)

    # Disclosed bots: one fixed notice under every thread, two seconds after it is posted.
    disclosed = [f"bot{n}" for n in range(3)]
    for n, bot in enumerate(disclosed):
        kinds[bot] = "disclosed"
        notice = own_words()
        for thread in range(n % 3, n_threads, 3):
            add(bot, thread, thread_time[thread] + 2, notice)

    # ----- puppet networks -----
    for net in range(n_networks):
        members = [f"n{net}m{m}" for m in range(network_size)]
        kinds.update({m: "puppet" for m in members})
        wake = {m: int(rng.integers(0, 24)) for m in members}
        batch = int(rng.integers(start + 20 * DAY, start + 90 * DAY))
        late_events = np.flatnonzero(event_times > batch + 2 * DAY)
        campaign = list(rng.choice(N_EVERGREEN, 2, replace=False)) + \
            list(N_EVERGREEN + rng.choice(late_events, min(5, len(late_events)), replace=False))
        phrase_of = {p: str(rng.choice(phrases)) for p in campaign}
        template = {p: reworded(p) + " " + reworded(p) for p in campaign}   # written once, by the operator
        register = {m: HUMAN_REGISTER if level["text"] == "copy" else rng.uniform(*LLM_AS_HUMAN_REGISTER)
                    for m in members}
        # A swarm is told to use the slogan. A sparse network blends in: each persona
        # uses coded phrases as often as a real person might.
        habit = {m: 0.5 if level["spread"] == "swarm" else phrase_habit() for m in members}

        def campaign_text(point, m):
            if level["text"] == "copy":
                words = template[point].split()
                if rng.random() < 0.5:                              # half are word for word
                    words[int(rng.integers(0, len(words)))] = str(rng.choice(fillers))
                text = " ".join(words)
            else:
                text = reworded(point, register[m])
            return text + " " + phrase_of[point] if rng.random() < habit[m] else text

        # The operator's pushes: right after the news for a new point, random days for old ones.
        pushes = []
        for point in campaign:
            if point >= N_EVERGREEN:
                pushes.append((point, born[point] + rng.uniform(1, 3) * HOUR, 36 * HOUR))
            else:
                pushes += [(point, t, 24 * HOUR) for t in rng.integers(batch + DAY, end - 4 * DAY, 5)]

        def threads_in(begin, finish):
            lo, hi = np.searchsorted(thread_time, [begin, finish])
            found = np.arange(lo, hi)
            return found[np.argsort(-thread_weight[found])]         # busiest first: targeted

        if level["spread"] == "swarm":
            for point, push_start, length in pushes:
                for thread in threads_in(push_start, push_start + length)[:6]:
                    for m in rng.choice(members, min(network_size, int(rng.integers(4, 9))), replace=False):
                        if level["timing"] == "robotic":
                            when = thread_time[thread] + rng.uniform(5, 60)
                        else:
                            when = keep_waking_hours(thread_time[thread] + human_delay(1), wake[m])[0]
                        add(m, thread, when, campaign_text(point, m), point, True)
            continue

        for m in members:
            total = how_many_comments(15)
            n_campaign = max(2, round(campaign_share * total))
            if level["operator"] == "hasty":
                joined = batch + rng.uniform(0, 2 * DAY)            # the batch: all start within 2 days
                ordinary_comments(m, total - n_campaign, joined, wake[m], arrive_now=True, habit=habit[m],
                                  machine=register[m])
                used = set()
                for _ in range(n_campaign):
                    point, push_start, length = pushes[int(rng.integers(0, len(pushes)))]
                    options = [t for t in threads_in(push_start - 12 * HOUR, push_start + length)
                               if (point, t) not in used][:8]       # one puppet per thread per push
                    if not options:
                        continue
                    thread = int(rng.choice(options))
                    used.add((point, thread))
                    when = max(thread_time[thread], push_start) + human_delay(1)[0]
                    add(m, thread, keep_waking_hours(when, wake[m]), campaign_text(point, m), point, True)
            else:
                # Patient: join like anyone, push each point only as fast as real people pick it up.
                joined, arrive_now = join_time()
                ordinary_comments(m, total - n_campaign, joined, wake[m], arrive_now=arrive_now, habit=habit[m],
                                  machine=register[m])
                first = min(np.searchsorted(thread_time, joined), n_threads - 20)
                for _ in range(n_campaign):
                    point = int(rng.choice(campaign))
                    p = thread_weight[first:] * popularity(thread_time[first:])[:, point]
                    if p.sum() == 0:
                        continue
                    thread = first + int(rng.choice(n_threads - first, p=p / p.sum()))
                    when = thread_time[thread] + human_delay(1)[0]
                    add(m, thread, keep_waking_hours(when, wake[m]), campaign_text(point, m), point, True)

    # ----- tables in the same shape the collect stage produces -----
    author, thread, when, body, true_point, is_campaign = zip(*rows)
    thread = np.array(thread)
    comments = pd.DataFrame({
        "id": [f"c{i}" for i in range(len(rows))], "subreddit": thread_sub[thread],
        "author_hash": author, "disclosed_bot": [a.startswith("bot") for a in author],
        "created_utc": when, "removed": False,
        "link_id": [f"p{t}" for t in thread], "parent_id": [f"t3_p{t}" for t in thread], "body": body})
    for table in (posts, comments):
        table["time"] = pd.to_datetime(table["created_utc"], unit="s", utc=True)
    accounts = pd.DataFrame({"kind": pd.Series(kinds)})
    accounts["target"] = (accounts["kind"] == "puppet").astype(int)
    truth = pd.DataFrame({"id": comments["id"], "point": true_point, "campaign": is_campaign})
    return World(posts, comments, accounts, truth)
