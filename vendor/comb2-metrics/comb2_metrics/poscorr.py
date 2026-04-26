from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def pos_corr(
    pos1_path: str | Path,
    pos2_path: str | Path,
    start: int,
    end: int,
    *,
    min_valid: int = 1000,
) -> pd.Series:
    pos1 = pd.read_parquet(pos1_path)
    pos2 = pd.read_parquet(pos2_path)

    pos1 = pos1.loc[(pos1.index >= start) & (pos1.index < end)]
    pos2 = pos2.loc[(pos2.index >= start) & (pos2.index < end)]

    common_index = pos1.index.intersection(pos2.index)
    pos1 = pos1.loc[common_index]
    pos2 = pos2.loc[common_index]

    corr = {}
    for date, row1 in pos1.iterrows():
        row2 = pos2.loc[date]
        valid = row1.notna() & row2.notna() & (row1 != 0) & (row2 != 0)
        if int(valid.sum()) < min_valid:
            corr[date] = np.nan
        else:
            corr[date] = float(np.corrcoef(row1[valid], row2[valid])[0, 1])

    return pd.Series(corr, name="pos_corr")
