"""Shared data loading and preprocessing for the MathModel project.

Provides the extract_id / classify_sample utilities, oxide name constants,
and a single load_and_merge() function used by all problem scripts.
"""
import re

import pandas as pd

OXIDES = [
    "二氧化硅(SiO2)", "氧化钠(Na2O)", "氧化钾(K2O)", "氧化钙(CaO)",
    "氧化镁(MgO)", "氧化铝(Al2O3)", "氧化铁(Fe2O3)", "氧化铜(CuO)",
    "氧化铅(PbO)", "氧化钡(BaO)", "五氧化二磷(P2O5)", "氧化锶(SrO)",
    "氧化锡(SnO2)", "二氧化硫(SO2)",
]
OXIDE_SHORT = ["SiO2","Na2O","K2O","CaO","MgO","Al2O3","Fe2O3","CuO",
               "PbO","BaO","P2O5","SrO","SnO2","SO2"]


def extract_id(name):
    """Parse zero-padded artifact ID from a sampling-point name string."""
    m = re.match(r"(\d+)", str(name))
    return f"{int(m.group(1)):02d}" if m else None


def classify_sample(name):
    """Classify a sampling point as 严重风化点, 未风化点, or 普通."""
    s = str(name)
    if "严重风化点" in s:
        return "严重风化点"
    if "未风化点" in s:
        return "未风化点"
    return "普通"


def load_and_merge(verbose=True):
    """Load 表单1 + 表单2, merge, validate, return (valid, oxide_cols).

    Returns
    -------
    valid : pd.DataFrame
        Valid rows (oxide sum 85%-105%), indexed by artifact ID,
        with oxide columns filled (NaN → 0), plus 类型, 表面风化, 采样类型.
    oxide_cols : list of str
        Column names for the 14 oxide columns.
    """
    df_meta = pd.read_csv("data/表单1_文物信息.csv", skiprows=1)
    df_comp = pd.read_csv("data/表单2_化学成分.csv", skiprows=1)

    df_comp["文物编号"] = df_comp.iloc[:, 0].apply(extract_id)
    df_comp["采样类型"] = df_comp.iloc[:, 0].apply(classify_sample)
    df_comp = df_comp.set_index("文物编号")

    df_meta["文物编号"] = df_meta.iloc[:, 0].astype(str).str.zfill(2)
    df_meta = df_meta.set_index("文物编号")

    merged = df_comp.merge(df_meta[["类型", "表面风化"]], left_index=True, right_index=True, how="left")

    if verbose:
        print("Step 1 — Merge result:")
        print(f"Total rows: {len(merged)}")
        print(f"类型 missing: {merged['类型'].isna().sum()}")
        print(f"采样类型 distribution:\n{merged['采样类型'].value_counts()}\n")

    assert len(merged) == 69, f"Expected 69 rows, got {len(merged)}"
    assert merged["类型"].isna().sum() == 0, "Some artifacts couldn't be matched to metadata"

    oxide_cols = [merged.columns[i] for i in range(1, 15)]
    merged["sum_pct"] = merged[oxide_cols].fillna(0).sum(axis=1)

    if verbose:
        print("Step 2 — Validity:")
        print(f"Valid (85-105%): {merged['sum_pct'].between(85, 105).sum()}/{len(merged)}")
        print(f"Invalid rows:\n{merged[~merged['sum_pct'].between(85, 105)][['sum_pct']].to_string()}\n")

    assert merged["sum_pct"].between(85, 105).sum() / len(merged) > 0.8, "Too many invalid rows"

    valid = merged[merged["sum_pct"].between(85, 105)].copy()
    valid[oxide_cols] = valid[oxide_cols].fillna(0)
    return valid, oxide_cols
