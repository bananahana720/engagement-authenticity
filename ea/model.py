"""Stage 4, model: learn "inauthentic or not" from the labeled accounts.

Target: 1 for labels A, B, C (automated, hijacked, incentive-driven), 0 for D.
E (disclosed bots) and U (uncertain) are left out of training.

Two splits at once, so the test is honest:
  by time     train on the earlier window, test on the later one
  by account  no account is in both train and test
"""

import hashlib
import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

from ea.config import paths
from ea.label import clean_labels

INAUTHENTIC = {"A", "B", "C"}
NOT_FEATURES = ["author_hash", "window", "label", "target", "in_test"]


def in_test_set(author_hash, share):
    """Assign an account to train or test from its hash: stable across reruns."""
    bucket = int(hashlib.sha256(author_hash.encode()).hexdigest(), 16) % 100
    return bucket < share * 100


def labeled_features(cfg):
    features = pd.read_csv(paths(cfg)["features"], dtype={"author_hash": str})
    labels = clean_labels(pd.read_csv(paths(cfg)["labels"], dtype={"author_hash": str}))
    table = features.merge(labels[["author_hash", "label"]], on="author_hash")
    table = table[table["label"].isin(INAUTHENTIC | {"D"})]
    table["target"] = table["label"].isin(INAUTHENTIC).astype(int)
    share = cfg["model"]["test_account_share"]
    table["in_test"] = table["author_hash"].map(lambda h: in_test_set(h, share))
    return table


def new_model():
    """Gradient-boosted trees, then calibration so a score of 0.8 means about 80%."""
    trees = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, random_state=0)
    return CalibratedClassifierCV(trees, method="sigmoid", cv=3)


def pick_threshold(y_true, scores, min_precision):
    """Lowest score cut-off that still reaches the precision we require."""
    precision, _, thresholds = precision_recall_curve(y_true, scores)
    good = [t for p, t in zip(precision[:-1], thresholds) if p >= min_precision]
    return float(min(good)) if good else 1.0      # 1.0 = flag nobody


def train(cfg):
    table = labeled_features(cfg)
    train_rows = table[(table["window"] == "train") & ~table["in_test"]]
    test_rows = table[(table["window"] == "test") & table["in_test"]]
    columns = [c for c in table.columns if c not in NOT_FEATURES]

    x, y = train_rows[columns], train_rows["target"]
    # The threshold is chosen on out-of-fold train scores, never on the test set.
    out_of_fold = cross_val_predict(new_model(), x, y, cv=3, method="predict_proba")[:, 1]
    threshold = pick_threshold(y, out_of_fold, cfg["model"]["min_precision"])

    model = new_model().fit(x, y)
    predictions = test_rows[["author_hash", "label", "target"]].copy()
    predictions["score"] = model.predict_proba(test_rows[columns])[:, 1]
    predictions["flagged"] = predictions["score"] >= threshold

    p = paths(cfg)
    p["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": columns}, p["model"])
    predictions.to_csv(p["predictions"], index=False)
    p["threshold"].write_text(json.dumps({
        "threshold": threshold,
        "min_precision": cfg["model"]["min_precision"],
        "n_train_accounts": int(len(train_rows)),
        "n_test_accounts": int(len(test_rows)),
    }, indent=2))
    return predictions
