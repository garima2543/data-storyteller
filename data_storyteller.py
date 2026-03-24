"""Data Storyteller - core analysis functions"""
import pandas as pd
import numpy as np


def validate_df(df):
    msgs = []
    if df.empty:
        msgs.append("Uploaded file is empty.")
    if df.shape[0] < 2:
        msgs.append("Dataset has fewer than 2 rows; limited analysis possible.")
    if df.shape[1] < 1:
        msgs.append("Dataset has no columns.")
    df.columns = [str(c) for c in df.columns]
    return (len(msgs) == 0, msgs)


def summary_stats(df):
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])
    return {
        'numeric_describe': numeric.describe().T,
        'categorical_describe': {
            col: df[col].value_counts().head(10).to_dict()
            for col in categorical.columns
        }
    }


def missing_summary(df):
    ms = df.isna().sum()
    pct = (ms / len(df) * 100).round(2)
    return pd.DataFrame({'missing_count': ms, 'missing_pct': pct}).sort_values(
        'missing_pct', ascending=False
    )


def top_correlations(df, n=10):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    corr = numeric.corr().abs().unstack().reset_index()
    corr.columns = ['feature_1', 'feature_2', 'corr']
    corr = corr[corr['feature_1'] != corr['feature_2']]
    corr = corr.sort_values('corr', ascending=False).drop_duplicates(subset=['corr'])
    return corr.head(n)


def generate_insights(df, max_insights=6):
    insights = []

    ms = missing_summary(df)
    high_missing = ms[ms['missing_pct'] > 30]
    if not high_missing.empty:
        cols = ', '.join(high_missing.index.tolist()[:5])
        insights.append(("warn", f"Columns with >30% missing values: {cols}."))
    else:
        insights.append(("success", "No column has more than 30% missing values — dataset is largely complete."))

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 0:
        means = numeric.mean().sort_values(ascending=False)
        top3 = ', '.join(means.index.astype(str)[:3])
        insights.append(("info", f"Top 3 numeric columns by mean: {top3}."))

        most_var = numeric.var().sort_values(ascending=False)
        top3v = ', '.join(most_var.index.astype(str)[:3])
        insights.append(("info", f"Top 3 columns by variance: {top3v}."))

    cat = df.select_dtypes(exclude=[np.number])
    if cat.shape[1] > 0:
        sample = cat.iloc[:, 0]
        top = sample.value_counts().head(3)
        top_str = ', '.join([f"{i} ({c})" for i, c in top.items()])
        insights.append(("info", f"'{sample.name}' top values: {top_str}."))

    corr_df = top_correlations(df, n=5)
    if not corr_df.empty:
        r = corr_df.iloc[0]
        strength = "very strong" if r.corr > 0.8 else "moderate" if r.corr > 0.5 else "weak"
        tag = "alert" if r.corr > 0.7 else "info"
        insights.append((tag, f"Strongest correlation: '{r.feature_1}' ↔ '{r.feature_2}' (|r|={r.corr:.2f}, {strength})."))

    insights.append(("info", f"Dataset: {len(df):,} rows × {df.shape[1]} columns — "
                             f"{numeric.shape[1]} numeric, {cat.shape[1]} categorical."))

    return insights[:max_insights]
