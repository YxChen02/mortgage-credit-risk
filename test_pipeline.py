import sqlite3
from pathlib import Path

import pandas as pd

from woe import fit_woe, transform_woe


def test_labels():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE mortgage (
        id, time, orig_time, status_time, balance_time, LTV_time,
        interest_rate_time, FICO_orig_time, LTV_orig_time)""")
    histories = {
        1: [(1, 0), (2, 0), (3, 1)],
        2: [(1, 0), (2, 2)],
        3: [(1, 0), (3, 1)],
        4: [(1, 0), (2, 0)],
        5: [(1, 0), (2, 0), (3, 0)],
    }
    conn.executemany("INSERT INTO mortgage VALUES (?, ?, 0, ?, ?, 80, 4, 700, 85)", [
        (loan, period, status, 100 * period)
        for loan, history in histories.items() for period, status in history
    ])
    sql = Path(__file__).with_name("features.sql").read_text()
    rows = conn.execute(sql, dict(horizon=2, data_end=3)).fetchall()
    assert {row[0]: (row[-4], row[-1]) for row in rows} == {
        1: (100, 1), 2: (100, 0), 3: (100, None), 4: (100, None), 5: (100, 0),
    }


def test_woe():
    train = pd.DataFrame({"ltv": [10, 20, 30, None]})
    rules, report = fit_woe(train, pd.Series([0, 0, 1, 1]))
    result = transform_woe(pd.DataFrame({"ltv": [-5, 900, None]}), rules)
    assert result.notna().all().all()
    assert report.good.sum() == report.bad.sum() == 2
    assert report.feature_iv.iloc[0] > 0


if __name__ == "__main__":
    test_labels()
    test_woe()
    print("SQL lifecycle and WOE checks passed")
