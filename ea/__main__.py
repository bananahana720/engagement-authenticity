"""Run one pipeline stage:  python -m ea <stage> [--config config/immigration.toml]

  volumes    count comments and posts per subreddit per month (needs internet)
  collect    download on-topic threads (needs internet)
  features   compute one row of signals per account
  sample     write a labeling sheet
  agreement  compare two labelers' sheets
  train      fit the model on labeled accounts
  evaluate   write the aggregate report
  demo       run features -> sample -> train -> evaluate on made-up data
  seeds      download and convert the 2018 disclosed accounts (use --config config/ira2018.toml)
  network    who-replies-to-whom measures, starting from the seed accounts
  network-demo  the network measures on made-up data with planted patterns
"""

import argparse
from pathlib import Path

import json

from ea import collect, evaluate, features, ira2018, label, model, network, synthetic, threads
from ea.config import load_config, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["volumes", "collect", "features", "sample",
                                          "agreement", "train", "evaluate", "demo",
                                          "seeds", "threads", "network", "network-demo"])
    parser.add_argument("--config", default="config/immigration.toml")
    parser.add_argument("--only", nargs="*", help="collect: just these subreddits")
    parser.add_argument("--posts-dump", help="collect: read posts from this dump file instead of the API")
    parser.add_argument("--comments-dump", help="collect: read comments from this dump file")
    parser.add_argument("--sheets", nargs=2, metavar=("A.csv", "B.csv"), help="agreement: the two label sheets")
    parser.add_argument("--accounts", type=int, default=200, help="sample: how many accounts to draw")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.stage == "volumes":
        print("Wrote", collect.measure_volumes(cfg))
    elif args.stage == "collect":
        if args.posts_dump and args.comments_dump:
            collect.collect_from_dumps(cfg, args.posts_dump, args.comments_dump)
        else:
            collect.collect(cfg, only=args.only)
    elif args.stage == "features":
        print(f"Wrote {len(features.build_features(cfg))} rows to {paths(cfg)['features']}")
    elif args.stage == "sample":
        print(f"Wrote {len(label.draw_sample(cfg, args.accounts))} accounts to {paths(cfg)['to_label']}")
    elif args.stage == "agreement":
        print(label.agreement(*args.sheets).to_string(index=False))
    elif args.stage == "train":
        print(f"Scored {len(model.train(cfg))} held-out test accounts")
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


def run_demo(cfg):
    """The whole pipeline on synthetic data, in its own folders."""
    cfg = dict(cfg, name="demo (synthetic data, not real Reddit)",
               data_dir=Path("data/demo"), reports_dir=Path("reports/demo"))
    print("Made %d posts, %d comments, %d labeled accounts" % synthetic.make_synthetic(cfg))
    features.build_features(cfg)
    label.draw_sample(cfg)
    model.train(cfg)
    print(evaluate.evaluate(cfg).read_text())


if __name__ == "__main__":
    main()
