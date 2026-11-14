"""Run one pipeline stage:  python -m ea <stage> [--config config/immigration.toml]

  volumes    count comments and posts per subreddit per month (Reddit: paused, refuses)
  topic-share  count on-topic threads and their comments per subreddit per month (Reddit: paused, refuses)
  seed-list  fetch Reddit's 2018 list of suspended influence-operation accounts (Reddit: paused, refuses)
  collect    download on-topic threads (Reddit: paused, refuses)
  bluesky    listen to Bluesky's public stream for --minutes, then store on-topic threads:
             --config config/bluesky_immigration.toml (needs internet)
  features   compute one row of signals per account
  sample     write a labeling sheet
  agreement  compare two labelers' sheets
  train      fit the model on labeled accounts (refused for Reddit data, see docs/DATA_TERMS.md)
  score      score accounts with a model trained elsewhere: --model path/to/model.joblib
  evaluate   write the aggregate report
  demo       run features -> sample -> train -> evaluate on made-up data
  seeds      download and convert the 2018 disclosed accounts (use --config config/ira2018.toml)
  network    who-replies-to-whom measures, starting from the seed accounts
  network-demo  the network measures on made-up data with planted patterns
  redteam    made-up sock-puppet networks at rising evasion levels: which signals survive?
  fingerprint-generate  write replies with small open models run locally (hours on CPU; needs
             torch, transformers, llama-cpp-python and access to huggingface.co)
  fingerprint  can a classifier name the model family that wrote a text? by length and evasion
"""

import argparse
from pathlib import Path

import json

from ea import (bluesky, collect, evaluate, features, ira2018, label, model, network, redteam, seeds,
                synthetic, threads)
from ea.config import CollectionPaused, load_config, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["volumes", "topic-share", "seed-list", "collect", "bluesky", "features", "sample",
                                          "agreement", "train", "score", "evaluate", "demo",
                                          "seeds", "threads", "network", "network-demo", "redteam",
                                          "fingerprint-generate", "fingerprint"])
    parser.add_argument("--config", default="config/immigration.toml")
    parser.add_argument("--only", nargs="*", help="collect: just these subreddits")
    parser.add_argument("--posts-dump", help="collect: read posts from this dump file instead of the API")
    parser.add_argument("--comments-dump", help="collect: read comments from this dump file")
    parser.add_argument("--minutes", type=float, default=20, help="bluesky: how long to listen")
    parser.add_argument("--sheets", nargs=2, metavar=("A.csv", "B.csv"), help="agreement: the two label sheets")
    parser.add_argument("--model", help="score: a model.joblib trained on non-Reddit data")
    parser.add_argument("--accounts", type=int, default=200, help="sample: how many accounts to draw")
    parser.add_argument("--quick", action="store_true", help="redteam: small run to check the code")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.stage == "volumes":
        print("Wrote", collect.measure_volumes(cfg))
    elif args.stage == "topic-share":
        print("Wrote", collect.measure_topic_share(cfg))
    elif args.stage == "seed-list":
        print("Wrote %d accounts to %s" % seeds.fetch_reddit_2018(cfg))
    elif args.stage == "collect":
        if args.posts_dump and args.comments_dump:
            collect.collect_from_dumps(cfg, args.posts_dump, args.comments_dump)
        else:
            collect.collect(cfg, only=args.only)
    elif args.stage == "bluesky":
        if cfg["source"] != "bluesky":
            raise SystemExit("Use a Bluesky config: --config config/bluesky_immigration.toml")
        print(bluesky.collect(cfg, args.minutes).to_string(index=False))
    elif args.stage == "features":
        print(f"Wrote {len(features.build_features(cfg))} rows to {paths(cfg)['features']}")
    elif args.stage == "sample":
        print(f"Wrote {len(label.draw_sample(cfg, args.accounts))} accounts to {paths(cfg)['to_label']}")
    elif args.stage == "agreement":
        print(label.agreement(*args.sheets).round(2).to_string(index=False))
    elif args.stage == "train":
        try:
            print(f"Scored {len(model.train(cfg))} held-out test accounts")
        except model.TrainingNotAllowed as refusal:
            raise SystemExit(f"Not trained. {refusal}")
    elif args.stage == "score":
        print(f"Scored {len(model.score(cfg, args.model))} accounts with {args.model}")
    elif args.stage == "evaluate":
        print("Wrote", evaluate.evaluate(cfg))
    elif args.stage == "demo":
        run_demo(cfg)
    elif args.stage == "seeds":
        ira2018.download(cfg)
        print("Wrote %d posts, %d comments, %d seed accounts" % ira2018.convert(cfg))
    elif args.stage == "threads":
        print("Fetched %d threads" % threads.collect_threads(cfg))
    elif args.stage == "network":
        out, result = network.network_report(cfg)
        print(json.dumps(result, indent=2))
        print("Wrote", out)
    elif args.stage == "network-demo":
        cfg = dict(cfg, name="network demo (synthetic data)",
                   data_dir=Path("data/network_demo"), reports_dir=Path("reports/network_demo"))
        print("Made %d posts, %d comments, %d seed accounts" % synthetic.make_synthetic_network(cfg))
        out, result = network.network_report(cfg)
        print(json.dumps(result, indent=2))
        print("Wrote", out)
    elif args.stage == "redteam":
        out = redteam.run(cfg, quick=args.quick)
        print((out / "results.md").read_text())
    elif args.stage == "fingerprint-generate":
        from ea import fingerprint_generate
        fingerprint_generate.main()
    elif args.stage == "fingerprint":
        from ea import fingerprint
        main_result = fingerprint.main()["main"]
        print(f"Short replies: {main_result['short']['accuracy']:.0%}, long replies: "
              f"{main_result['long']['accuracy']:.0%}, chance {main_result['chance']:.0%}. "
              f"Wrote {fingerprint.REPORT_DIR}")


def run_demo(cfg):
    """The whole pipeline on synthetic data, in its own folders."""
    cfg = dict(cfg, name="demo (synthetic data, not real Reddit)", source="synthetic",
               data_dir=Path("data/demo"), reports_dir=Path("reports/demo"))
    print("Made %d posts, %d comments, %d labeled accounts" % synthetic.make_synthetic(cfg))
    features.build_features(cfg)
    label.draw_sample(cfg)
    model.train(cfg)
    print(evaluate.evaluate(cfg).read_text())


if __name__ == "__main__":
    try:
        main()
    except CollectionPaused as refusal:
        raise SystemExit(str(refusal))
