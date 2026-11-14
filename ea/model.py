"""Stage 4, model: learn "is software doing the typing?" from labeled accounts.

Target, from the who_types question:
  1  program, or person plus automation
  0  person
Declared accounts and uncertain answers are left out of training.

Two splits at once, so the test is honest:
  by time     train on the earlier window, test on the later one
  by account  no account is in both train and test

TRAINING RULE: no model is fitted on Reddit data until Reddit gives permission
(see docs/DATA_TERMS.md). train() refuses a Reddit-sourced config. Reddit data
is scored with a model trained elsewhere: see score().
"""

import hashlib
import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ea.config import paths
from ea.label import clean_labels

AUTOMATED = {"program", "person_plus_automation"}
NOT_FEATURES = ["author_hash", "window", "who_types", "target", "in_test"]


class TrainingNotAllowed(Exception):
    pass


def check_training_allowed(source, cfg):
    """Stop before any model sees Reddit rows, unless the config records Reddit's permission."""
    if source == "reddit" and not cfg.get("model", {}).get("reddit_training_permitted", False):
        raise TrainingNotAllowed(
            "This data comes from Reddit, and Reddit has not given permission to train models on it. "
            "Train on non-Reddit data instead and score this data with: "
            "python -m ea score --model <path to model.joblib>. See docs/DATA_TERMS.md.")


def in_test_set(author_hash, share):
    """Assign an account to train or test from its hash: stable across reruns."""
    bucket = int(hashlib.sha256(author_hash.encode()).hexdigest(), 16) % 100
    return bucket < share * 100


def labeled_features(cfg):
    features = pd.read_csv(paths(cfg)["features"], dtype={"author_hash": str})
    labels = clean_labels(pd.read_csv(paths(cfg)["labels"], dtype={"author_hash": str}))
    table = features.merge(labels[["author_hash", "who_types"]], on="author_hash")
    table = table[table["who_types"].isin(AUTOMATED | {"person"})]
    table["target"] = table["who_types"].isin(AUTOMATED).astype(int)
    share = cfg["model"]["test_account_share"]
    table["in_test"] = table["author_hash"].map(lambda h: in_test_set(h, share))
    return table


def new_model():
    """Gradient-boosted trees, then calibration so a score of 0.8 means about 80%."""
    # min_samples_leaf is small because the first labeled sets will be a few hundred accounts.
    trees = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, min_samples_leaf=5, random_state=0)
    return CalibratedClassifierCV(trees, method="sigmoid", cv=folds())


def folds():
    """Three shuffled folds. Shuffling matters: the rows arrive sorted, so unshuffled folds
    would each hold out a whole kind of account the model then never trained on."""
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def pick_threshold(y_true, scores, min_precision):
    """Lowest score cut-off that still reaches the precision we require."""
    precision, _, thresholds = precision_recall_curve(y_true, scores)
    good = [t for p, t in zip(precision[:-1], thresholds) if p >= min_precision]
    return float(min(good)) if good else 1.0      # 1.0 = flag nobody


def train(cfg):
    """Fit on this config's labeled accounts, then score its held-out test accounts."""
    check_training_allowed(cfg["source"], cfg)
    table = labeled_features(cfg)
    train_rows = table[(table["window"] == "train") & ~table["in_test"]]
    test_rows = table[(table["window"] == "test") & table["in_test"]]
    columns = [c for c in table.columns if c not in NOT_FEATURES]

    x, y = train_rows[columns], train_rows["target"]
    # The threshold is chosen on out-of-fold train scores, never on the test set.
    out_of_fold = cross_val_predict(new_model(), x, y, cv=folds(), method="predict_proba")[:, 1]
    bundle = {
        "model": new_model().fit(x, y),
        "columns": columns,
        "threshold": pick_threshold(y, out_of_fold, cfg["model"]["min_precision"]),
        "min_precision": cfg["model"]["min_precision"],
        "trained_on": cfg["source"],
        "n_train_accounts": int(len(train_rows)),
    }
    paths(cfg)["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, paths(cfg)["model"])
    return write_scores(cfg, bundle, test_rows, trained_accounts=set(train_rows["author_hash"]))


def score(cfg, model_path):
    """Score this config's accounts with a model trained somewhere else. Nothing is fitted."""
    bundle = joblib.load(model_path)
    check_training_allowed(bundle["trained_on"], cfg)      # a model fitted on Reddit is refused too
    table = labeled_features(cfg) if paths(cfg)["labels"].exists() else None
    # No account here was trained on, so every labeled account in the later window is a fair test.
    test_rows = table[table["window"] == "test"] if table is not None else None
    return write_scores(cfg, bundle, test_rows, trained_accounts=set())


def write_scores(cfg, bundle, labeled_test_rows, trained_accounts):
    """Write scores for the labeled test accounts and for every account in the later window."""
    model, columns, threshold = bundle["model"], bundle["columns"], bundle["threshold"]
    p = paths(cfg)
    p["predictions"].parent.mkdir(parents=True, exist_ok=True)

    everyone = pd.read_csv(p["features"], dtype={"author_hash": str})
    everyone = everyone[everyone["window"] == "test"]
    scores = everyone[["author_hash"]].copy()
    scores["score"] = model.predict_proba(everyone[columns])[:, 1]
    scores["flagged"] = scores["score"] >= threshold
    scores["used_in_training"] = scores["author_hash"].isin(trained_accounts)
    scores.to_csv(p["scores"], index=False)

    p["threshold"].write_text(json.dumps({
        "threshold": threshold, "min_precision": bundle["min_precision"],
        "trained_on": bundle["trained_on"], "scored_data": cfg["source"],
        "n_train_accounts": bundle["n_train_accounts"],
    }, indent=2))

    if labeled_test_rows is None:
        return scores
    predictions = labeled_test_rows[["author_hash", "who_types", "target"]].copy()
    predictions["score"] = model.predict_proba(labeled_test_rows[columns])[:, 1]
    predictions["flagged"] = predictions["score"] >= threshold
    predictions.to_csv(p["predictions"], index=False)
    return predictions
