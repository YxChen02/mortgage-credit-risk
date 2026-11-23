import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def score_psi(reference, current):
    cuts = np.unique(np.quantile(reference, np.arange(1, 10) / 10))
    edges = np.r_[-np.inf, cuts, np.inf]
    shares = []
    for scores in (reference, current):
        counts = np.histogram(scores, bins=edges)[0] + 0.5
        shares.append(counts / counts.sum())
    expected, actual = shares
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def run(csv_path, cutoff, horizon, output, booster):
    raw = pd.read_csv(csv_path)
    loan_key = ["id", "orig_time"]
    period_key = [*loan_key, "time"]
    if raw[[*period_key, "status_time"]].isna().any().any():
        raise ValueError("Missing borrower, origination, period or status")
    data = raw.drop_duplicates()
    duplicate_rows = len(raw) - len(data)
    conflicts = data.loc[data.duplicated(period_key, keep=False), loan_key]
    ambiguous_loans = conflicts.drop_duplicates()
    keep = ~pd.MultiIndex.from_frame(data[loan_key]).isin(
        pd.MultiIndex.from_frame(ambiguous_loans)
    )
    data = data.loc[keep].copy()
    if not data.status_time.isin([0, 1, 2]).all():
        raise ValueError("Unknown loan status")
    with sqlite3.connect(":memory:") as conn:
        data.to_sql("mortgage", conn, index=False)
        conn.execute("CREATE UNIQUE INDEX borrower_period ON mortgage(id, orig_time, time)")
        sql = Path(__file__).with_name("features.sql").read_text()
        sample = pd.read_sql_query(sql, conn, params={
            "horizon": horizon, "data_end": int(data.time.max()),
        })
    observed = sample.dropna(subset=["target"])
    train = observed[observed.time + horizon <= cutoff]
    test = observed[observed.time > cutoff]
    if any(frame.target.nunique() != 2 for frame in (train, test)):
        raise ValueError("Select a cutoff with both classes in train and OOT periods")

    features = sample.columns.difference(["id", "time", "orig_time", "target"])
    models = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression()
        ),
    }
    if booster == "lightgbm":
        from lightgbm import LGBMClassifier

        models[booster] = LGBMClassifier(
            n_estimators=150, num_leaves=15, random_state=42, verbosity=-1
        )
    else:
        models[booster] = HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, random_state=42
        )
    metrics = {}
    monitoring = []
    oot = sample[sample.time > cutoff]
    for name, model in models.items():
        model.fit(train[features], train.target)
        reference = model.predict_proba(train[features])[:, 1]
        scores = model.predict_proba(test[features])[:, 1]
        fpr, tpr, _ = roc_curve(test.target, scores)
        metrics[name] = {
            "auc": roc_auc_score(test.target, scores),
            "ks": float((tpr - fpr).max()),
            "pr_auc_ap": average_precision_score(test.target, scores),
        }
        for period, batch in oot.groupby("time"):
            batch_scores = model.predict_proba(batch[features])[:, 1]
            monitoring.append({
                "model": name, "period": period, "count": len(batch),
                "unknown_labels": int(batch.target.isna().sum()),
                "score_psi": score_psi(reference, batch_scores),
            })

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(monitoring).to_csv(output / "monitoring.csv", index=False)
    summary = {
        "cutoff": cutoff, "horizon": horizon,
        "input_rows": len(raw), "exact_duplicates": duplicate_rows,
        "ambiguous_loans_excluded": len(ambiguous_loans),
        "train_rows": len(train), "oot_labeled_rows": len(test),
        "oot_unknown_labels": int(oot.target.isna().sum()),
        "train_bad_rate": train.target.mean(), "oot_bad_rate": test.target.mean(),
        "borrowers_in_both": len(set(train.id) & set(test.id)),
        "metrics": metrics,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mortgage early warning backtest")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--cutoff", required=True, type=int)
    parser.add_argument("--horizon", default=3, type=int)
    parser.add_argument("--output", default=Path("results"), type=Path)
    parser.add_argument("--booster", choices=["histgb", "lightgbm"], default="histgb")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    run(args.csv, args.cutoff, args.horizon, args.output, args.booster)
