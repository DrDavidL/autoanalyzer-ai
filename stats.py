# Statistical analysis functions for AutoAnalyzer

import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm

def run_ttest(df, numeric_col, group_col):
    groups = df[group_col].unique()
    if len(groups) != 2:
        return None, "Group column must have exactly 2 unique values."
    group1 = df[df[group_col] == groups[0]][numeric_col]
    group2 = df[df[group_col] == groups[1]][numeric_col]
    t_stat, p_val = stats.ttest_ind(group1, group2)
    return t_stat, p_val

def run_anova(df, numeric_col, group_col):
    groups = [df[df[group_col] == g][numeric_col] for g in df[group_col].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    return f_stat, p_val

def run_mannwhitney(df, numeric_col, group_col):
    groups = df[group_col].unique()
    if len(groups) != 2:
        return None, "Group column must have exactly 2 unique values."
    group1 = df[df[group_col] == groups[0]][numeric_col]
    group2 = df[df[group_col] == groups[1]][numeric_col]
    u_stat, p_val = stats.mannwhitneyu(group1, group2)
    return u_stat, p_val

def run_kruskal(df, numeric_col, group_col):
    groups = [df[df[group_col] == g][numeric_col] for g in df[group_col].unique()]
    h_stat, p_val = stats.kruskal(*groups)
    return h_stat, p_val

def run_chi2(df, col1, col2):
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p, dof, expected

def run_simple_linear_regression(df, x_col, y_col):
    X = df[[x_col]]
    y = df[y_col]
    model = linear_model.LinearRegression()
    model.fit(X, y)
    coef = model.coef_[0]
    intercept = model.intercept_
    return coef, intercept

def run_multiple_linear_regression(df, x_cols, y_col):
    X = df[x_cols]
    y = df[y_col]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    intercept = regr.intercept_
    coef = regr.coef_
    # statsmodels summary
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    summary = model.summary2()
    try:
        summary_table = summary.tables[1]
    except Exception:
        summary_table = None
    return regr, intercept, coef, summary, summary_table

def calculate_rr_arr_nnt(tn, fp, fn, tp):
    """
    Calculate Relative Risk (RR), Absolute Risk Reduction (ARR), and Number Needed to Treat (NNT)
    from a 2x2 confusion matrix: tn, fp, fn, tp.
    Returns a dict with RR, ARR, NNT, EER, CER.
    """
    # Experimental Event Rate (EER): tp / (tp + fp)
    # Control Event Rate (CER): fn / (fn + tn)
    try:
        eer = tp / (tp + fp) if (tp + fp) != 0 else np.nan
        cer = fn / (fn + tn) if (fn + tn) != 0 else np.nan
        rr = eer / cer if cer not in [0, np.nan] else np.nan
        arr = cer - eer if not np.isnan(cer) and not np.isnan(eer) else np.nan
        nnt = 1 / arr if arr not in [0, np.nan] else np.nan
    except Exception:
        eer = cer = rr = arr = nnt = np.nan
    return {'RR': rr, 'ARR': arr, 'NNT': nnt, 'EER': eer, 'CER': cer}


