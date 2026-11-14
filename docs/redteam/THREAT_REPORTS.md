# Real operations: what provider and platform threat reports say

Read 2026-11-14. Purpose: check the red team's assumptions against documented covert
influence operations that used LLMs. Only what the reports state is recorded; "not
stated" means the report does not say. Reports name operations, never ordinary users,
and this file follows them.

**How far to trust this file.** It is a first pass, not a full catalogue.
- OpenAI's October 2025, February 2026 and June 2025 reports could not be read in
  full (only introductions loaded). They hold many more cases, including one OpenAI
  calls "Nine-emdash Line".
- The Meta report was read from a copy on a third-party site, not from Meta. Its
  cover says "Second Half 2026, August 2026" while its summary says it covers the
  first half of 2026. Treat Meta rows as unverified until checked against Meta's site.
- Rows marked (summary) come from an automated summary of a long page, not from
  quoted text.

## Cases

| Source | Operation | Origin as attributed | Where | What the LLM did | Models named | What gave it away | Reach as assessed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI, June 2026 Threat Report | "Data Center Bandwagon" | Likely China; likely a social media operations team at a private company with provincial government clients | X, Facebook, YouTube outreach; US audiences and overseas Chinese | Short English comments, images, persona messages, code to automate logins and interactions across platforms, polishing internal reports on avoiding platform detection | ChatGPT (reached through VPN) | Linked to an X network by "open-source behavioral indicators" | Category One on the Breakout Scale: "little or no observable engagement" |
| OpenAI, June 2026 Threat Report | "Tech and Tariffs" | Likely China; affiliation not established | X; English, Italian, Japanese, Traditional Chinese | Short comments and cartoons in bulk, colloquial tone; harassment comments | ChatGPT (reached through VPN) | Accounts created in late 2025 with few followers; quote-posts of the same account "within a few hours of each other"; overlap with another cluster | Category One: "little or no observable engagement" |
| Anthropic, "Detecting and countering malicious uses of Claude: March 2025" (published 2025-04-23) | Unnamed "influence-as-a-service" | Commercial service; state link not confirmed | X and Facebook; several countries and languages | Wrote replies, and decided when each of 100+ persona accounts should comment, like or re-share; kept personas' political alignments consistent | Claude; unnamed image tools | Provider-side review of the operator's own conversations with the model | "No content achieved viral status"; engaged with tens of thousands of real accounts |
| Meta, Adversarial Threat Report (unverified copy) (summary) | Israel-origin influence-for-hire network | Israel | Facebook, Instagram, TikTok, X; eight countries | "Mass-produce contextually relevant comments" through an AI platform's API; automated accounts commented on the network's own posts | Not named | Meta investigation | 215 Facebook and 1,044 Instagram accounts; about 248,000 Instagram followers |
| Meta (unverified copy) (summary) | Ukraine and Bulgaria network | A Ukrainian expatriate in Bulgaria | Facebook, Instagram, TikTok, X, Telegram | LLM-written articles under invented bylines | Not named | One cluster posted "identical narratives across accounts in quick succession" | 72 Facebook and 16 Instagram accounts; about 500 followers |
| Meta (unverified copy) (summary) | Russia-origin network, fabricated think tank | Russia | Facebook, Instagram, Telegram, TikTok, YouTube, X | AI profile photos, bios, articles, videos | Not named | Found after information from OpenAI | About 400 followers |
| Meta (unverified copy) (summary) | Russia-origin network targeting Hungary | Russia | Facebook | AI-generated content for fake news sites | Not named | Dormant pages made months ahead and renamed before launch; proxy servers | About 15,200 followers |
| Meta (unverified copy) (summary) | Iran-origin network targeting Africa | Iran | Facebook, Instagram, X | Polished graphics; accounts commented on own posts "to simulate popularity" | Not named | Meta investigation; proxy addresses, clean devices | About 17,000 followers |
| Google, "Adversarial Misuse of Generative AI" (2025-01-29) (summary) | DRAGONBRIDGE | China | YouTube | Research, article drafts; AI-generated news presenters | Gemini; video tool not named | Channels terminated on discovery | AI video "has not resulted in significantly higher engagement from real viewers" |
| Google, same report (summary) | CopyCop | Pro-Russia | Fake news sites aimed at the US and Europe | Rewrote real articles with a slant | Not named | Article metadata showed an LLM had been prompted | Not assessed |
| Google, "GTIG AI Threat Tracker" (2026-02-12) | Unnamed actors from China, Iran, Russia, Saudi Arabia | By country only | Not stated | Articles, satire, posters | Gemini | Not described | "We did not see evidence of successful automation" |

## What this says about our assumptions

**Detection in these reports is behavioral, or comes from the provider seeing the
operator's prompts.** No report read mentions a text watermark or "this text reads
as AI" as how an operation was found. Meta: "behavioral detection, coordination
analysis, and network-level enforcement continue to identify and disrupt threats."
That matches the project rule to weight timing and coordination over text style.

**The leaks the red team assumes do show up.** Accounts created together in late
2025; the same account quote-posted within hours; identical narratives across
accounts in quick succession; dormant assets made months ahead and renamed. These
are the batch, shared-push and lockstep signals at the "hasty operator" levels.

**Operators already work on the leak, not on the wording.** OpenAI's June 2026
report describes an operator playbook of "cross-account interactions to amplify
narratives while preserving the appearance of organic engagement" and separating
operations "to avoid the platform detecting coordination". Anthropic describes an
LLM deciding *when* each persona acts. That is the "sparse" and "patient" end of the
ladder, in the wild.

**Careful operations mostly got little reach.** OpenAI rates both June 2026 cases
Category One (one platform, no breakout). Anthropic: nothing went viral. This fits
the red team's cost-of-evasion finding, with one caution: these are the operations
that were caught.

**Models: providers see mixing, and closed models are used.** Meta: "threat actors
typically use a variety of models and platforms." OpenAI notes its June 2026 cases
used ChatGPT "rather than Chinese models", and separately that China's "cyber
special operations" strategy "emphasizes the use of locally deployed Chinese
open-weights models". So "closed-model providers will see little bot activity" is
not supported: they report plenty, and they find it through account-level review of
prompts, not a watermark. Activity on locally run open-weight models is the part no
provider can see.

## Still to do

- Read the OpenAI reports that failed to load (October 2025, February 2026, June 2025).
- Check the Meta rows against Meta's own site and finish its remaining cases.
- Add Microsoft Threat Analysis Center and Graphika reports.

## Sources

- OpenAI, June 2026 Threat Report: https://cdn.openai.com/pdf/96b559fa-c165-4575-805d-e636909e2f78/June-2026-Threat-Report.pdf
- Anthropic, March 2025 report: https://www.anthropic.com/news/detecting-and-countering-malicious-uses-of-claude-march-2025
- Google GTIG, Adversarial Misuse of Generative AI: https://cloud.google.com/blog/topics/threat-intelligence/adversarial-misuse-generative-ai?hl=en
- Google GTIG, AI Threat Tracker: https://cloud.google.com/blog/topics/threat-intelligence/distillation-experimentation-integration-ai-adversarial-use?hl=en
- Meta report, third-party copy (unverified): https://combatantisemitism.org/wp-content/uploads/2026/09/Meta-H2-2026-Adversarial-Threat-Report.pdf
