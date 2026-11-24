import numpy as np
import pandas as pd


def fit_woe(frame, target):
    """Fit quantile bins and log(bad share / good share) on training rows."""
    rules = {}
    details = []
    for feature, values in frame.items():
        observed = values.dropna()
        cuts = np.unique(observed.quantile(np.arange(1, 10) / 10)) if len(observed) else []
        edges = np.r_[-np.inf, cuts, np.inf]
        codes = np.searchsorted(edges[1:-1], values, side="right")
        codes[values.isna()] = len(edges) - 1
        counts = pd.crosstab(codes, target.to_numpy()).reindex(
            index=range(len(edges)), columns=[0, 1], fill_value=0
        )
        shares = (counts + 0.5) / (counts + 0.5).sum()
        woe = np.log(shares[1] / shares[0])
        iv = (shares[1] - shares[0]) * woe
        rules[feature] = (edges, woe.to_numpy())
        for i in range(len(edges)):
            details.append({
                "feature": feature, "bin": i,
                "lower_inclusive": edges[i] if i < len(edges) - 1 else None,
                "upper_exclusive": edges[i + 1] if i < len(edges) - 1 else None,
                "missing_bin": i == len(edges) - 1,
                "good": counts.loc[i, 0], "bad": counts.loc[i, 1],
                "woe": woe[i], "iv_component": iv[i], "feature_iv": iv.sum(),
            })
    return rules, pd.DataFrame(details)


def transform_woe(frame, rules):
    result = {}
    for feature, (edges, woe) in rules.items():
        values = frame[feature]
        codes = np.searchsorted(edges[1:-1], values, side="right")
        codes[values.isna()] = len(edges) - 1
        result[feature] = woe[codes]
    return pd.DataFrame(result, index=frame.index)
