"""Run one pipeline stage:  python -m ea <stage> [--config config/immigration.toml]

  volumes    count comments and posts per subreddit per month (needs internet)
  collect    download on-topic threads (needs internet)
  features   compute one row of signals per account
  sample     write a labeling sheet
  agreement  compare two labelers' sheets
  train      fit the model on labeled accounts (refused for Reddit data, see docs/DATA_TERMS.md)
  score      score accounts with a model trained elsewhere: --model path/to/model.joblib
  evaluate   write the aggregate report
  demo       run features -> sample -> train -> evaluate on made-up data
  redteam    made-up sock-puppet networks at rising evasion levels: which signals survive?
  fingerprint-generate  write replies with small open models run locally (hours on CPU; needs
             torch, transformers, llama-cpp-python and access to huggingface.co)
  fingerprint  can a classifier name the model family that wrote a text? by length and evasion
"""

import argparse
from pathlib import Path

from ea import collect, evaluate, features, label, model, redteam, synthetic
from ea.config import load_config, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["volumes", "collect", "features", "sample",
                                          "agreement", "train", "score", "evaluate", "demo", "redteam",
                                          "fingerprint-generate", "fingerprint"])
    parser.add_argument("--config", default="config/immigration.toml")
    parser.add_argument("--only", nargs="*", help="collect: just these subreddits")
    parser.add_argument("--posts-dump", help="collect: read posts from this dump file instead of the API")
    parser.add_argument("--comments-dump", help="collect: read comments from this dump file")
    parser.add_argument("--sheets", nargs=2, metavar=("A.csv", "B.csv"), help="agreement: the two label sheets")
    parser.add_argument("--model", help="score: a model.joblib trained on non-Reddit data")
    parser.add_argument("--accounts", type=int, default=200, help="sample: how many accounts to draw")
    parser.add_argument("--quick", action="store_true", help="redteam: small run to check the code")
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
    main()
