# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as S
import statsmodels.api as sm
import pingouin as pg

import re
import json
import os

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score, f1_score, PrecisionRecallDisplay
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pyper
import datetime

from typing import Tuple

import statsmodels.api as sm
# -

# Mocks for compatibility reasons
np.int = int
np.float = float
np.bool = bool


class ALTERN(int):
    OTHER = 0b00
    WILDTYPE = 0b01
    MIXED = 0b10
    PERMUTATION = 0b100
    INSERT = 0b1000
    DELETION = 0b10000
    REPETITION = 0b100000
    
    def __repr__(self):
        texts = []
        if self == self.OTHER:
            return "OTHER"
        if self & self.WILDTYPE:
            texts.append("WILDTYPE")
        if self & self.MIXED:
            texts.append("MIXED")
        if self & self.PERMUTATION:
            texts.append("PERMUTATION")
        if self & self.INSERT:
            texts.append("INSERT")
        if self & self.DELETION:
            texts.append("DELETION")
        if self & self.REPETITION:
            texts.append("REPETITION")
        return f'ALTERN({" + ".join(texts)})'


def exon_filter(x, exon: Tuple[str, list[int]], altern: ALTERN):
    if not isinstance(x, list):
        return False
    name, exons = exon
    for item in x:
        if not (item["altern"] & altern): continue
        available_exons = item['exons'].get(name, [])
        for e in available_exons:
            if e in exons: return True
    return False


def d842_permutation_filter(x):
    if isinstance(x, str):
        x = json.loads(x)
    if not isinstance(x, list):
        return False
    for item in x:
        available_exons = item['exons'].get('pdgfra', [])
        if 18 in available_exons:
            for codon in item.get('codons_pairs', []):
                available_codons = codon['codons']
                if 842 in available_codons and codon['altern'] & ALTERN.PERMUTATION:
                    return True
    return False

def pr_auc_score(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


# # Load data

df2 = pd.read_csv("data.csv.gz", index_col=0)
df_rad = pd.read_csv("radiomics.csv.gz", index_col=0)

# # Baseline characteristics

# +
_location_sort = {"胃": 0, "十二指肠": 1, "小肠": 2, "直肠": 3, "系膜": 4, "后腹膜": 5, "定位困难": 6, "其它": 7}
_location_rename = {
    "胃": "Stomach",
    "十二指肠": "Duodenum",
    "小肠": "Small intestine",
    "直肠": "Rectum",
    "系膜": "Mesenterium",
    "网膜": "Periterium",
    "后腹膜": "Retroperiterium",
    "定位困难": "Difficult to locate",
    "其它": "Others",
}

genes_of_interest = [
    ("kit", [9]),
    ("kit", [11]),
    ("kit", [13]),
    ("kit", [17]),
    ("pdgfra", [12]),
    #("pdgfra", [14]),
    ("pdgfra", [18]),
]
gene_cols = [f"custom_{i[0]}{i[1][0]}" for i in genes_of_interest]


# +
def t_test(value_col, table, filter_vals=[True, False], filter_col="custom_train"):
    args = [table[table[filter_col] == f][value_col] for f in filter_vals]
    return S.ttest_ind(*args)[1]

def contingency(table):
    if np.min(table) > 5:
        return S.chi2_contingency(table)
    elif table.shape == (2, 2):
        return S.fisher_exact(table)
    else:
        return [np.nan, np.nan]

def _format_p(p: float):
    return "%.3f" % p if p >= 0.001 else "< 0.001"

def baseline(
    data=df_rad,
    group_col="custom_train",
    groups=[True, False],
    group_names=["Training cohort", "Validation cohort"],
    gene_mutation=True,
):
    results = pd.DataFrame()
    _size = data.groupby(group_col).size()
    _name_map = dict(zip(groups, group_names))

    # Age
    _age_mean = data.groupby(group_col)["custom_age"].mean()
    _age_std = data.groupby(group_col)["custom_age"].std()
    _age_p_value = t_test("custom_age", data, filter_col=group_col, filter_vals=groups)
    _age_p = _format_p(_age_p_value)
    _age = pd.concat([_age_mean, _age_std], axis=1).apply(lambda r: "%.2f ± %.2f" % (r.iloc[0], r.iloc[1]), axis=1)
    _age.loc["p"] = _age_p
    _age = _age.rename("Age (years)")
    # Gender
    _gender_data = data.groupby(["custom_gender", group_col]).size().reset_index()
    _gender_pivot = _gender_data.pivot_table(0, index=[group_col], columns="custom_gender").fillna(0)
    _gender_pivot_pct = (_gender_pivot.T / _size).T
    _gender_text = pd.DataFrame(np.apply_along_axis(lambda r: "%3d (%.2f%%)" % (r[0], r[1] * 100), 0, np.stack([_gender_pivot, _gender_pivot_pct])), index=_gender_pivot.index, columns=_gender_pivot.columns)
    _gender_text = _gender_text.rename({
        0: "Female",
        1: "Male",
    }, axis=1)
    _gender_p_value = contingency(_gender_pivot)[1]
    _gender_p = pd.Series({"p": _format_p(_gender_p_value)}, name="Sex")

    # Size
    _diameter_mean = data.groupby([group_col])["original_shape2D_MaximumDiameter"].mean()
    _diameter_std = data.groupby([group_col])["original_shape2D_MaximumDiameter"].std()
    _diameter_p_value = t_test("original_shape2D_MaximumDiameter", data, filter_col=group_col, filter_vals=groups)
    _diameter_p = _format_p(_diameter_p_value)
    _diameter = pd.concat([_diameter_mean, _diameter_std], axis=1).apply(lambda r: "%.2f ± %.2f" % (r.iloc[0], r.iloc[1]), axis=1)
    _diameter.loc["p"] = _diameter_p
    _diameter = _diameter.rename("Diameter (mm)")

    # Location
    _location_data = data.groupby(["custom_location", group_col]).size().reset_index()
    _location_pivot = _location_data.pivot_table(0, index=[group_col], columns="custom_location").fillna(0)
    _location_pivot = _location_pivot.sort_index(axis=1, key=lambda x: x.map(_location_sort))
    _location_pivot_pct = (_location_pivot.T / _size).T
    _location_text = pd.DataFrame(np.apply_along_axis(lambda r: "%3d (%.2f%%)" % (r[0], r[1] * 100), 0, np.stack([_location_pivot, _location_pivot_pct])), index=_location_pivot.index, columns=_location_pivot.columns)
    _location_text = _location_text.rename(lambda x: _location_rename.get(x, x), axis=1)
    _location_p_value = contingency(_location_pivot)[1]
    _location_p = pd.Series({"p": _format_p(_location_p_value)}, name="Location")

    # Gene mutation
    if gene_mutation:
        _gene_p = pd.Series({}, name="Gene mutation")
        _gene_values = []
        for gene, gene_col in zip(genes_of_interest, gene_cols):
            _gene_label = f"{gene[0].upper()} {gene[1][0]}"
            _gene_data = data.groupby([gene_col, group_col]).size().reset_index()
            _gene_pivot = _gene_data.pivot_table(0, index=[group_col], columns=gene_col).fillna(0)
            _gene_p_value = contingency(_gene_pivot)[1]
            _gene_data = _gene_pivot.loc[:, True]
            _gene_data_pct = (_gene_data.T / _size).T
            _gene_text = pd.Series(
                np.apply_along_axis(lambda r: "%d (%.2f%%)" % (r[0], r[1] * 100), 0, np.stack([_gene_data, _gene_data_pct])),
                name=_gene_label
            )
            _gene_text["p"] = _format_p(_gene_p_value)
            _gene_values.append(_gene_text)

        gene_col = "custom_genewt"
        _gene_data = data.groupby([gene_col, group_col]).size().reset_index()
        _gene_pivot = _gene_data.pivot_table(0, index=[group_col], columns=gene_col).fillna(0)
        _gene_p_value = contingency(_gene_pivot)[1]
        _gene_data = _gene_pivot.loc[:, True]
        _gene_data_pct = (_gene_data.T / _size).T
        _gene_text = pd.Series(
            np.apply_along_axis(lambda r: "%d (%.2f%%)" % (r[0], r[1] * 100), 0, np.stack([_gene_data, _gene_data_pct])),
            name="Wildtype"
        )
        _gene_text["p"] = _format_p(_gene_p_value)
        _gene_values.append(_gene_text)
        _gene_values = pd.concat(_gene_values, axis=1)

    results = pd.concat([results, _age, _gender_p, _gender_text, _location_p, _location_text], axis=1)
    if gene_mutation:
        results = pd.concat([results, _gene_p, _gene_values], axis=1)

    _sort = {True: 0, False: 1, "p": 2}
    results = results.sort_index(axis=0, key=lambda x: x.map(_sort))
    results.rename(index=lambda x: "%s (n = %d)" % (_name_map.get(x, x), _size[x]) if x != "p" else x, inplace=True)
    return results.T.fillna('')

baseline()
# -

_stomach_data = df_rad[["custom_train", "custom_location"]].copy()
_stomach_data["custom_location"] = _stomach_data["custom_location"].map(lambda x: x == "其它")
_stomach_data = _stomach_data.groupby(["custom_train", "custom_location"]).size().reset_index()
_stomach_pivot = _stomach_data.pivot_table(0, index=["custom_location"], columns="custom_train").fillna(0)
contingency(_stomach_pivot)


baseline(
    data=df_rad[df_rad["custom_train"]],
    group_col="custom_d842v",
    groups=[True, False],
    group_names=["D842V-mutant", "D842V-wildtype"],
    gene_mutation=False
)

_stomach_data = df_rad[df_rad["custom_train"]][["custom_d842v", "custom_location"]].copy()
_stomach_data["custom_location"] = _stomach_data["custom_location"].map(lambda x: x == "胃")
_stomach_data = _stomach_data.groupby(["custom_d842v", "custom_location"]).size().reset_index()
_stomach_pivot = _stomach_data.pivot_table(0, index=["custom_location"], columns="custom_d842v").fillna(0)
contingency(_stomach_pivot)

baseline(
    data=df_rad[~df_rad["custom_train"]],
    group_col="custom_d842v",
    groups=[True, False],
    group_names=["D842V-mutant", "D842V-wildtype"],
    gene_mutation=False
)

_stomach_data = df_rad[~df_rad["custom_train"]][["custom_d842v", "custom_location"]].copy()
_stomach_data["custom_location"] = _stomach_data["custom_location"].map(lambda x: x == "胃")
_stomach_data = _stomach_data.groupby(["custom_d842v", "custom_location"]).size().reset_index()
_stomach_pivot = _stomach_data.pivot_table(0, index=["custom_location"], columns="custom_d842v").fillna(0)
contingency(_stomach_pivot)

# +
_a =  baseline(
    data=df_rad[df_rad["custom_train"]],
    group_col="custom_d842v",
    groups=[True, False],
    group_names=["D842V-mutant", "D842V-wildtype"],
    gene_mutation=False
)
_b = baseline(
    data=df_rad[~df_rad["custom_train"]],
    group_col="custom_d842v",
    groups=[True, False],
    group_names=["D842V-mutant", "D842V-wildtype"],
    gene_mutation=False
)

pd.concat([_a, _b], axis=1)
# -

baseline(
    data=df_rad[~df_rad["custom_train"]],
    group_col="custom_genewt",
    groups=[True, False],
    group_names=["KIT- and PDGFRA-wildtype", "KIT- or D842V-mutant"],
    gene_mutation=False
)

# # ICC

#columns = [i for i in df_rad.columns if i.startswith("original") and "shape" not in i]
columns = [i for i in df_rad.columns if not i.startswith("custom") and not i.startswith("diagnostics") and not i.startswith("lbp-3D") and "shape" not in i]
#columns = [i for i in mdf_train_case.columns if i.startswith("original") or i.startswith("sq") or i.startswith("logarithm") or i.startswith("exponential") or i.startswith("gradient") and "shape" not in i]

# Remove all-values-were same columns
columns = np.array(columns)[np.where(df_rad[df_rad.custom_train][columns].std() != 0)]

len(columns)

# +
# Use ICC to select columns
cache_intra_file = f"cache/intra.txt"

if os.path.exists(cache_intra_file):
    with open(cache_intra_file, "r") as f:
        available_intra = f.read().split()
else:
    origin = pd.read_csv("results-ct4phase-renji-anyty-resampled-2024-04-29.csv", index_col="custom_studyId")
    intra = pd.read_csv("results-ct4phase-renji-anyty-resampled-intraobserver-2024-05-02.csv", index_col="custom_studyId")
    origin = origin.loc[origin.custom_phase == "v"]
    intra = intra.loc[intra.custom_phase == "v"]
    _intra = intra.copy()
    _intra["rater"] = 0
    _intra["target"] = _intra.index.values
    _origin = origin.loc[_intra.index.values, columns]
    _origin["rater"] = 1
    _origin["target"] = _origin.index.values
    _intra_all = pd.concat([_intra, _origin], ignore_index=True, axis=0)
    
    available_intra = []
    for c in columns:
        # Two-way random effects, absolute agreement, single rater/measurement
        icc = pg.intraclass_corr(_intra_all, targets="target", raters="rater", ratings=c)
        icc = icc.set_index("Type").loc["ICC2", "ICC"]
        if icc > 0.75:
            available_intra.append(c)

    with open(cache_intra_file, "w") as f:
        f.write("\n".join(available_intra))

print(f"Available columns (intraobserver): {len(available_intra)} / {len(columns)}")

# +
# Use ICC to select columns
cache_inter_file = f"cache/inter.txt"

if os.path.exists(cache_inter_file):
    with open(cache_inter_file, "r") as f:
        available_inter = f.read().split()
else:
    origin = pd.read_csv("results-ct4phase-renji-anyty-resampled-2024-04-29.csv", index_col="custom_studyId")
    inter = pd.read_csv("results-ct4phase-renji-anyty-resampled-interobserver-smooth-2024-05-26.csv", index_col="custom_studyId")
    origin = origin.loc[origin.custom_phase == "v"]
    inter = inter.loc[intra.custom_phase == "v"]
    _inter = inter.copy()
    _inter["rater"] = 0
    _inter["target"] = _inter.index.values
    _origin = origin.loc[_inter.index.values, columns]
    _origin["rater"] = 1
    _origin["target"] = _origin.index.values
    _inter_all = pd.concat([_inter, _origin], ignore_index=True, axis=0)
    
    available_inter = []
    for c in columns:
        # Two-way random effects, absolute agreement, single rater/measurement
        icc = pg.intraclass_corr(_inter_all, targets="target", raters="rater", ratings=c)
        icc = icc.set_index("Type").loc["ICC2", "ICC"]
        if icc > 0.75:
            available_inter.append(c)

    with open(cache_inter_file, "w") as f:
        f.write("\n".join(available_inter))

print(f"Available columns (interobserver): {len(available_inter)} / {len(columns)}")

# +
# Filter columns with ICC
num_all_columns = len(columns)
columns = sorted(list(set(available_intra).intersection(set(available_inter))))

print(f"Available columns (all): {len(columns)} / {num_all_columns}")
# -

# # Feature selection

RANDOM_STATE=11


# +
class CustomOpt:
    def __init__(self, model):
        self.best_estimator_ = model

clin_columns = ["custom_age", "custom_gender", "original_shape2D_MaximumDiameter", "custom_is-gastric"]
columns_to_use = list(columns) + clin_columns[:-1]

X_train = df_rad[df_rad.custom_train]
y_train = X_train["custom_d842v"]
X_test = df_rad[~df_rad.custom_train]
y_test = X_test["custom_d842v"]
X_train = pd.concat([X_train[columns_to_use], pd.Series(X_train["custom_location"] == "胃", name="custom_is-gastric")], axis=1)
X_test = pd.concat([X_test[columns_to_use], pd.Series(X_test["custom_location"] == "胃", name="custom_is-gastric")], axis=1)

# +
max_nodes = 20
num_trees = 10

sm = RandomUnderSampler(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = sm.fit_resample(X_train[columns], y_train)
rf = RandomForestClassifier(num_trees, random_state=RANDOM_STATE, max_depth=2)
rf.fit(X_train_sm, y_train_sm)
impRF = rf.feature_importances_ / rf.feature_importances_.max()

r = pyper.R(use_pandas='True')
r.assign("x_train", X_train_sm.values)
r.assign("y_train", y_train_sm.values)
r.assign("seed", RANDOM_STATE)
r.assign("ntree", num_trees)
r.assign("max_nodes", max_nodes)
r.assign("coefReg", 0.5)
r.assign("imp", impRF)
r("""
library(RRF);set.seed(seed)
gamma <- 1
coefReg <- (1-gamma)+gamma*imp
grf <- RRF(x_train,as.factor(y_train), ntree=ntree, max_nodes=max_nodes, flagReg = 1, max_nodes=max_nodes, coefReg=coefReg)
""")
best_features = np.array(columns)[r.get("grf$feaSet") - 1].tolist()
if isinstance(best_features, str):
    best_features = [best_features]

best_features


# +
class CustomOpt:
    def __init__(self, model):
        self.best_estimator_ = model

columns_to_use = best_features + clin_columns

model = RandomForestClassifier(50, random_state=RANDOM_STATE, class_weight="balanced", max_depth=2)
model.fit(X_train[columns_to_use], y_train)

opt = CustomOpt(model)

# +
y_pred_train = opt.best_estimator_.predict_proba(X_train[columns_to_use])[:, 1]
y_pred_test = opt.best_estimator_.predict_proba(X_test[columns_to_use])[:, 1]

train_results = {
    "pred_radioclinical": y_pred_train,
    "label": y_train,
}
test_results = {
    "pred_radioclinical": y_pred_test,
    "label": y_test,
}
models = {
    "radioclinical": model,
}

_results = {
    "AUC (train)": roc_auc_score(y_train, y_pred_train),
    "AUC (test)": roc_auc_score(y_test, y_pred_test),
    "AP (train)": average_precision_score(y_train, y_pred_train),
    "AP (test)": average_precision_score(y_test, y_pred_test),
}

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train)

_results.update({
    "Confusion Matrix (train, 0.5)": confusion_matrix(y_train, y_pred_train > 0.5),
    "Confusion Matrix (test, 0.5)": confusion_matrix(y_test, y_pred_test > 0.5),
    "G-mean (train, 0.5)": geometric_mean_score(y_train, y_pred_train > 0.5),
    "G-mean (test, 0.5)": geometric_mean_score(y_test, y_pred_test > 0.5),
    "F1-score (train, 0.5)": f1_score(y_train, y_pred_train > 0.5),
    "F1-score (test, 0.5)": f1_score(y_test, y_pred_test > 0.5),
})

_results

# +
results_train = permutation_importance(opt.best_estimator_, X_train[columns_to_use], y_train, scoring=lambda est, x, y: roc_auc_score(y, est.predict_proba(x)[:, 1]), random_state=RANDOM_STATE, n_jobs=-1)
results_test = permutation_importance(opt.best_estimator_, X_test[columns_to_use], y_test, scoring=lambda est, x, y: roc_auc_score(y, est.predict_proba(x)[:, 1]), random_state=RANDOM_STATE, n_jobs=-1)

_fi = pd.DataFrame(
    np.array([columns_to_use, results_train["importances_mean"], results_test["importances_mean"]]).T,
    columns=["Column", "Permutation Importance (train)", "Permutation Importance (test)"]
).sort_values("Permutation Importance (train)", ascending=False).set_index("Column")

# Plot train
sorted_importances_idx = results_train.importances_mean.argsort()
importances_train = pd.DataFrame(
    results_train.importances[sorted_importances_idx].T,
    columns=np.array(columns_to_use)[sorted_importances_idx],
)
ax = importances_train.plot(kind="box", vert=False, whis=10, figsize=(8, 6))
ax.set_title(f"Permutation Importances")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
# -
# # Alternative models

# +
X_train_clin = X_train[clin_columns]
X_test_clin = X_test[clin_columns]

model_clinical = RandomForestClassifier(50, random_state=RANDOM_STATE, class_weight='balanced', max_depth=2)
model_clinical.fit(X_train_clin, y_train)
models["clinical"] = model_clinical
opt = CustomOpt(model_clinical)

# +
y_pred_train = opt.best_estimator_.predict_proba(X_train_clin)[:, 1]
y_pred_test = opt.best_estimator_.predict_proba(X_test_clin)[:, 1]

train_results["pred_clinical"] = y_pred_train
test_results["pred_clinical"] = y_pred_test

_results = {
    "AUC (train)": roc_auc_score(y_train, y_pred_train),
    "AUC (test)": roc_auc_score(y_test, y_pred_test),
    "AP (train)": average_precision_score(y_train, y_pred_train),
    "AP (test)": average_precision_score(y_test, y_pred_test),
}

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train)

_results.update({
    "Confusion Matrix (train, 0.5)": confusion_matrix(y_train, y_pred_train > 0.5),
    "Confusion Matrix (test, 0.5)": confusion_matrix(y_test, y_pred_test > 0.5),
    "G-mean (train, 0.5)": geometric_mean_score(y_train, y_pred_train > 0.5),
    "G-mean (test, 0.5)": geometric_mean_score(y_test, y_pred_test > 0.5),
    "F1-score (train, 0.5)": f1_score(y_train, y_pred_train > 0.5),
    "F1-score (test, 0.5)": f1_score(y_test, y_pred_test > 0.5),
})

_results

# +
X_train_radiol = X_train[best_features]
X_test_radiol = X_test[best_features]

model_radiol = RandomForestClassifier(50, random_state=RANDOM_STATE, class_weight='balanced', max_depth=2)
models["radiol"] = model_radiol
model_radiol.fit(X_train_radiol, y_train)
opt = CustomOpt(model_radiol)

# +
y_pred_train = opt.best_estimator_.predict_proba(X_train_radiol)[:, 1]
y_pred_test = opt.best_estimator_.predict_proba(X_test_radiol)[:, 1]

train_results["pred_radiol"] = y_pred_train
test_results["pred_radiol"] = y_pred_test

_results = {
    "AUC (train)": roc_auc_score(y_train, y_pred_train),
    "AUC (test)": roc_auc_score(y_test, y_pred_test),
    "AP (train)": average_precision_score(y_train, y_pred_train),
    "AP (test)": average_precision_score(y_test, y_pred_test),
}

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train)

_results.update({
    "Confusion Matrix (train, 0.5)": confusion_matrix(y_train, y_pred_train > 0.5),
    "Confusion Matrix (test, 0.5)": confusion_matrix(y_test, y_pred_test > 0.5),
    "G-mean (train, 0.5)": geometric_mean_score(y_train, y_pred_train > 0.5),
    "G-mean (test, 0.5)": geometric_mean_score(y_test, y_pred_test > 0.5),
    "F1-score (train, 0.5)": f1_score(y_train, y_pred_train > 0.5),
    "F1-score (test, 0.5)": f1_score(y_test, y_pred_test > 0.5),
})

_results


# -

# ## Plots

# +
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# color palettes
C = AttributeDict({
    "blue": "#81A4CD",
    "yellow": "#D7B377",
    "red": "#E27893",
    "pink": "#E07B94",
    "purple": "#A69BBF",
})
C.update({
    "darkblue": "#5F8CBF",
    "lightblue": "#C5D5E8",
    "darkpink": "#DC6A87",
})
# -

# Demostration of undersampling

# +
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.graph_objects as go
import plotly.express as px

def _gene_mutation(x):
    for gene_col in gene_cols:
        if x[gene_col]:
            return gene_col
    return "custom_genewt"

lda = LinearDiscriminantAnalysis(n_components=2)

_df = df_rad[df_rad.custom_train]
_gene = _df.apply(_gene_mutation, axis=1)

_df = pd.DataFrame(lda.fit_transform(_df[columns], _gene), columns=["axis-0", "axis-1"], index=X_train.index)
_df.loc[:, "label"] = y_train
_df.loc[:, "selected"] = 0
_df.loc[X_train_sm.index, "selected"] = 1
_df.loc[:, "label_selected"] = _df.loc[:, "label"].map(lambda x: x << 1) + _df.loc[:, "selected"]
_df.loc[:, "color"] = _df.loc[:, "label_selected"].map({
    0b00: C.lightblue,
    0b01: C.darkblue,
    0b10: C.red,
    0b11: C.darkpink,
})
_df.loc[:, "axis-1"] += 2

fig = px.scatter(_df, x="axis-0", y="axis-1", color="color", color_discrete_map="identity")
fig.update_layout(
    template="plotly_white",
    xaxis_title='',
    yaxis_title='',
    xaxis=dict(range=(-5, 5), showticklabels=False),
    yaxis=dict(range=(-5.5, 5.5), scaleanchor="x", showticklabels=False, tickvals=(-5, 0, 5)),
    width=300, height=300,
    margin=dict(t=5, b=5, l=5, r=5),
)
#fig.write_image("figures/figure-2_3_1.svg")
fig.show()
# -

# Demostration of feature selection

_pi = permutation_importance(
    rf, X_train_sm, y_train_sm, random_state=RANDOM_STATE
)
_index = np.argsort(_pi.importances_mean)
_df = pd.DataFrame({
    "imp": _pi.importances_mean[_index[-10:]],
    "err": _pi.importances_std[_index[-10:]],
})
fig = px.bar(_df, x="imp", error_x="err")
fig.update_layout(
    template="plotly_white",
    xaxis_title='',
    yaxis_title='',
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False),
    width=150, height=300,
    margin=dict(t=5, b=5, l=5, r=5),
)
#fig.write_image("figures/figure-2_3_2.svg")
fig.show()

# ROC Plot

# +
from sklearn.metrics import auc
import plotly.graph_objects as go
import plotly.express as px

fig = go.Figure()
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

# The histogram of scores compared to true labels
fpr, tpr, thresholds = roc_curve(train_results["label"], train_results["pred_clinical"])
_auc = auc(fpr, tpr)
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line_color=C.blue, line_dash="dot",
    name=f"Training cohort (clinical, AUC={'%.2f' % _auc})",
))

fpr, tpr, thresholds = roc_curve(test_results["label"], test_results["pred_clinical"])
_auc = auc(fpr, tpr)
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line_color=C.blue,
    name=f"Validation cohort (clinical, AUC={'%.2f' % _auc})",
))

fpr, tpr, thresholds = roc_curve(train_results["label"], train_results["pred_radioclinical"])
_auc = auc(fpr, tpr)
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line_color=C.red, line_dash="dot",
    name=f"Training cohort (clinical+radiomics, AUC={'%.2f' % _auc})",
))

fpr, tpr, thresholds = roc_curve(test_results["label"], test_results["pred_radioclinical"])
_auc = auc(fpr, tpr)
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line_color=C.red,
    name=f"Validation cohort (clinical+radiomics, AUC={'%.2f' % _auc})",
))

fig.update_layout(
    template="plotly_white",
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=600, height=600,
    legend=dict(y=0.05, x=0.42, xref="paper", traceorder='reversed', font_size=12),
    margin=dict(t=5, b=5, l=5, r=5),
)
#fig.write_image("figures/figure-3.png", width=600, height=600, scale=5)
fig.show()
# -

fig.update_layout(
    showlegend=False,
    xaxis_title='',
    yaxis_title='',
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False),
    width=300, height=300,
)
#fig.write_image("figures/figure-2_4_1.svg")
fig.show()

# PRC Plot

# +
# from sklearn.metrics import auc
import plotly.graph_objects as go
import plotly.express as px

chance_level = 17 / 385
fig = go.Figure()
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=chance_level, y1=chance_level,
)

# The histogram of scores compared to true labels
precision, recall, _ = precision_recall_curve(train_results["label"], train_results["pred_clinical"])
_ap = average_precision_score(train_results["label"], train_results["pred_clinical"])
fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line_color=C.blue, line_dash="dot", line_shape="hv",
    name=f"Training cohort (clinical, AP={_ap:.2f})",
))

precision, recall, _ = precision_recall_curve(test_results["label"], test_results["pred_clinical"])
_ap = average_precision_score(test_results["label"], test_results["pred_clinical"])
fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line_color=C.blue, line_shape="hv",
    name=f"Validation cohort (clinical, AP={_ap:.2f})",
))

precision, recall, _ = precision_recall_curve(train_results["label"], train_results["pred_radioclinical"])
_ap = average_precision_score(train_results["label"], train_results["pred_radioclinical"])
fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line_color=C.red, line_dash="dot", line_shape="hv",
    name=f"Training cohort (clinical+radiomics, AP={_ap:.2f})",
))

precision, recall, _ = precision_recall_curve(test_results["label"], test_results["pred_radioclinical"])
_ap = average_precision_score(test_results["label"], test_results["pred_radioclinical"])
fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line_color=C.red, line_shape="hv",
    name=f"Validation cohort (clinical+radiomics, AP={_ap:.2f})",
))

fig.update_layout(
    template="plotly_white",
    xaxis_title='Recall',
    yaxis_title='Precision',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=600, height=600,
    legend=dict(y=0.95, x=0.42, xref="paper", traceorder='reversed', font_size=12),
    margin=dict(t=5, b=5, l=5, r=5),
)
#fig.write_image("figures/figure-4.png", scale=5)
fig.show()
# -

fig.update_layout(
    showlegend=False,
    xaxis_title='',
    yaxis_title='',
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False),
    width=300, height=300,
)
#fig.write_image("figures/figure-2_4_2.svg")
fig.show()

# Scatter

# +
import plotly.express as px

_df = pd.concat([X_train, y_train.rename("D842V mutation")], axis=1)
#_df = _df[np.logical_and(X_train["custom_is-gastric"], X_train["custom_gender"] == 1)]
_df = _df[X_train["custom_is-gastric"]]
#_data = _df[best_features].copy()
#_data[clin_columns] = np.nan
_data = _df[best_features + clin_columns]
_df["prediction"] = models["radioclinical"].predict_proba(_data)[:, 1]

fig = px.scatter(_df, *best_features[:2], color="prediction", color_continuous_scale=px.colors.sequential.RdBu_r,
                 symbol="D842V mutation", symbol_sequence=["circle-open", "circle"],
                 log_x=True, log_y=True)

fig.update_layout(
    template="plotly_white",
    width=600, height=600,
    coloraxis_colorbar_y = 0.4,
    coloraxis_colorbar_len = 0.8,
)
#fig.write_image("figures/figure-5a.png", scale=5)
fig

# +
import plotly.express as px

_df = pd.concat([X_test, y_test.rename("D842V mutation")], axis=1)
#_df = _df[np.logical_and(X_test["custom_is-gastric"], X_test["custom_gender"] == 1)]
_df = _df[X_test["custom_is-gastric"]]
#_data = _df[best_features].copy()
#_data[clin_columns] = np.nan
_data = _df[best_features + clin_columns]
_df["prediction"] = models["radioclinical"].predict_proba(_data)[:, 1]

fig = px.scatter(_df, *best_features[:2], color="prediction", color_continuous_scale=px.colors.sequential.RdBu_r,
                 symbol="D842V mutation", symbol_sequence=["circle-open", "circle"],
                 log_x=True, log_y=True)

fig.update_layout(
    template="plotly_white",
    width=600, height=600,
    coloraxis_colorbar_y = 0.4,
    coloraxis_colorbar_len = 0.8,
)
#fig.write_image("figures/figure-5b.png", scale=5)
fig
# -

# Visualization of texture features

# +
from scipy import ndimage

def show_image(im, seg=None, edge=False, pos=100, win=400, alpha=0.2, seg_cmap="flag_r"):
    low = pos - win / 2
    high = pos + win/2 / 2

    fig = plt.figure(figsize=(8, 8))
    plt.imshow((np.clip(im, low, high) - low) / (high - low) , cmap="gray")

    if seg is not None:
        if edge:
            seg = ndimage.binary_dilation(seg, iterations=edge) - seg
        plt.imshow(seg, alpha=alpha, cmap=seg_cmap)


# -

sub001 = np.load("sample_data/sub001.npz")

show_image(sub001["im"][50:300, 150:400], sub001["seg"][50:300, 150:400], edge=2)
plt.axis("off")
plt.tight_layout()
#plt.savefig("figures/figure-2_2_1.png")

# +
from skimage.feature import graycomatrix, graycoprops

_window = 4
_im = sub001["im"][50:300, 150:400]
_im = (np.clip(_im, -50, 269) / 5 + 10).astype(int) # Normalized
_seg = sub001["seg"][50:300, 150:400]

_dissimilarity = np.zeros_like(_im)
_contrast = np.zeros_like(_im)

# tumor location 125:215 230:320
for x in range(75, 165):
    for y in range(80, 170):
        _patch = _im[x - _window:x + _window, y - _window:y + _window]
        glcm = graycomatrix(_patch, distances=[5], angles=[0], levels=64, normed=True)
        _dissimilarity[x, y] = graycoprops(glcm, 'dissimilarity').item()
        _contrast[x, y] = graycoprops(glcm, 'contrast').item()

# +
show_image(_im, np.where(_seg, _dissimilarity, 0), alpha=0.3, seg_cmap="gist_ncar")

plt.axis("off")
plt.tight_layout()
#plt.savefig("figures/figure-2_2_2.png")
# -
# ## Compare results of different models

# +
all_results = pd.DataFrame([], columns=["C", "C-CI", "CR", "CR-CI", "p"])

def _format_ci(ci):
    return f"[{ci.confidence_interval.low:.3f}, {ci.confidence_interval.high:.3f}]"


# -

b1 = S.bootstrap(
    (test_results["label"].values, test_results["pred_radioclinical"]),
    lambda x, y: average_precision_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
b2 = S.bootstrap(
    (test_results["label"].values, test_results["pred_clinical"]),
    lambda x, y: average_precision_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
all_results.loc["AP", "CR"] = '%.3f' % average_precision_score(test_results["label"].values, test_results["pred_radioclinical"])
all_results.loc["AP", "C"] = '%.3f' % average_precision_score(test_results["label"].values, test_results["pred_clinical"])
all_results.loc["AP", "CR-CI"] = _format_ci(b1)
all_results.loc["AP", "C-CI"] = _format_ci(b2)
b1.confidence_interval, b2.confidence_interval

_auc_diffs = average_precision_score(test_results["label"].values, test_results["pred_clinical"]) - average_precision_score(test_results["label"].values, test_results["pred_radioclinical"])
_diffs = np.std(b1.bootstrap_distribution - b2.bootstrap_distribution)
_p = 2 * S.norm.cdf(-np.abs(_auc_diffs / _diffs))
all_results.loc["AP", "p"] = "%.3f" % _p
_p

S.norm.cdf(_auc_diffs / _diffs)

b1 = S.bootstrap(
    (test_results["label"].values, test_results["pred_radioclinical"]),
    lambda x, y: pr_auc_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
b2 = S.bootstrap(
    (test_results["label"].values, test_results["pred_clinical"]),
    lambda x, y: pr_auc_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
all_results.loc["PR-AUC", "CR"] = '%.3f' % pr_auc_score(test_results["label"].values, test_results["pred_radioclinical"])
all_results.loc["PR-AUC", "C"] = '%.3f' % pr_auc_score(test_results["label"].values, test_results["pred_clinical"])
all_results.loc["PR-AUC", "CR-CI"] = _format_ci(b1)
all_results.loc["PR-AUC", "C-CI"] = _format_ci(b2)
b1.confidence_interval, b2.confidence_interval

_auc_diffs = pr_auc_score(test_results["label"].values, test_results["pred_clinical"]) - pr_auc_score(test_results["label"].values, test_results["pred_radioclinical"])
_diffs = np.std(b1.bootstrap_distribution - b2.bootstrap_distribution)
_p = 2 * S.norm.cdf(-np.abs(_auc_diffs / _diffs))
all_results.loc["PR-AUC", "p"] = "%.3f" % _p
_p

S.norm.cdf(_auc_diffs / _diffs)

b1 = S.bootstrap(
    (test_results["label"].values, test_results["pred_radioclinical"] > 0.5),
    lambda x, y: geometric_mean_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
b2 = S.bootstrap(
    (test_results["label"].values, test_results["pred_clinical"] > 0.5),
    lambda x, y: geometric_mean_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
all_results.loc["G-Mean", "CR"] = '%.3f' % geometric_mean_score(test_results["label"].values, test_results["pred_radioclinical"] > 0.5)
all_results.loc["G-Mean", "C"] = '%.3f' % geometric_mean_score(test_results["label"].values, test_results["pred_clinical"] > 0.5)
all_results.loc["G-Mean", "CR-CI"] = _format_ci(b1)
all_results.loc["G-Mean", "C-CI"] = _format_ci(b2)
b1.confidence_interval, b2.confidence_interval

_auc_diffs = geometric_mean_score(test_results["label"].values, test_results["pred_clinical"] > 0.5) - geometric_mean_score(test_results["label"].values, test_results["pred_radioclinical"] > 0.5)
_diffs = np.std(b1.bootstrap_distribution - b2.bootstrap_distribution)
_p = 2 * S.norm.cdf(-np.abs(_auc_diffs / _diffs))
all_results.loc["G-Mean", "p"] = "%.3f" % _p
_p

b1 = S.bootstrap(
    (test_results["label"].values, test_results["pred_radioclinical"] > 0.5),
    lambda x, y: f1_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
b2 = S.bootstrap(
    (test_results["label"].values, test_results["pred_clinical"] > 0.5),
    lambda x, y: f1_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
all_results.loc["F1", "CR"] = '%.3f' % f1_score(test_results["label"].values, test_results["pred_radioclinical"] > 0.5)
all_results.loc["F1", "C"] = '%.3f' % f1_score(test_results["label"].values, test_results["pred_clinical"] > 0.5)
all_results.loc["F1", "CR-CI"] = _format_ci(b1)
all_results.loc["F1", "C-CI"] = _format_ci(b2)
b1.confidence_interval, b2.confidence_interval

_auc_diffs = f1_score(test_results["label"].values, test_results["pred_clinical"] > 0.5) - f1_score(test_results["label"].values, test_results["pred_radioclinical"] > 0.5)
_diffs = np.std(b1.bootstrap_distribution - b2.bootstrap_distribution)
_p = 2 * S.norm.cdf(-np.abs(_auc_diffs / _diffs))
all_results.loc["F1", "p"] = "%.3f" % _p
_p

b1 = S.bootstrap(
    (test_results["label"].values, test_results["pred_radioclinical"]),
    lambda x, y: roc_auc_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
b2 = S.bootstrap(
    (test_results["label"].values, test_results["pred_clinical"]),
    lambda x, y: roc_auc_score(x, y),
    n_resamples=1000,
    paired=True,
    random_state=RANDOM_STATE,
)
all_results.loc["ROC-AUC", "CR"] = '%.3f' % roc_auc_score(test_results["label"].values, test_results["pred_radioclinical"])
all_results.loc["ROC-AUC", "C"] = '%.3f' % roc_auc_score(test_results["label"].values, test_results["pred_clinical"])
all_results.loc["ROC-AUC", "CR-CI"] = _format_ci(b1)
all_results.loc["ROC-AUC", "C-CI"] = _format_ci(b2)
b1.confidence_interval, b2.confidence_interval

_auc_diffs = roc_auc_score(test_results["label"].values, test_results["pred_clinical"]) - roc_auc_score(test_results["label"].values, test_results["pred_radioclinical"])
_diffs = np.std(b1.bootstrap_distribution - b2.bootstrap_distribution)
_p = 2 * S.norm.cdf(-np.abs(_auc_diffs / _diffs))
all_results.loc["ROC-AUC", "p"] = "%.3f" % _p
_p

# +
from sklearn.metrics import accuracy_score

tn, fp, fn, tp = confusion_matrix(test_results["label"].values, test_results["pred_radioclinical"] > 0.5).ravel() 
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = accuracy_score(test_results["label"].values, test_results["pred_radioclinical"] > 0.5)
all_results.loc["Sensitivity", "CR"] = '%.3f' % sensitivity
all_results.loc["Specificity", "CR"] = '%.3f' % specificity
all_results.loc["Accuracy", "CR"] = '%.3f' % accuracy

tn, fp, fn, tp = confusion_matrix(test_results["label"].values, test_results["pred_clinical"] > 0.5).ravel() 
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = accuracy_score(test_results["label"].values, test_results["pred_clinical"] > 0.5)
all_results.loc["Sensitivity", "C"] = '%.3f' % sensitivity
all_results.loc["Specificity", "C"] = '%.3f' % specificity
all_results.loc["Accuracy", "C"] = '%.3f' % accuracy
# -

all_results.rename(columns={
    "C": "clinical",
    "C-CI": "clinical (95\\% CI)",
    "CR": "clinical+radiomics",
    "CR-CI": "clinical+radiomics (95\\% CI)",
    "p": "$p$",
}, inplace=True)
all_results
