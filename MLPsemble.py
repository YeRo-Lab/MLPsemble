import sklearn
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
import scanpy as sc
import scipy.sparse as sparse


class MetaMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim1=256, hidden_dim2=128, dropout=0.2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def preprocess(adata, labels):
    X = adata.X
    n_counts = np.asarray(X.sum(axis=1)).ravel()
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    min_counts = np.percentile(n_counts, 1)
    max_counts = np.percentile(n_counts, 99)
    min_genes = np.percentile(n_genes, 1)
    qc_mask = (n_counts >= min_counts) & (n_counts <= max_counts) & (n_genes >= min_genes)
    adata = adata[qc_mask].copy()
    labels = labels.loc[qc_mask]
    print(f"Cells after QC: {adata.n_obs}")
    data = pd.DataFrame(
        adata.X.toarray() if sparse.issparse(adata.X) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )
    print("Data loaded...")
    assert len(data) == len(labels)

    return data, labels


def fit_hvg(adata, labels):
    min_cells = int(0.05 * adata.n_obs)
    gene_mask = np.asarray((adata.X > 0).sum(axis=0)).ravel() >= min_cells
    adata = adata[:, gene_mask].copy()

    genes_after_filter = adata.var_names.copy()

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, 1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=int(0.25 * adata.n_vars),
        flavor="seurat_v3",
        layer="counts"
    )

    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask]

    adata = adata[:, hvg_mask]

    df = pd.DataFrame(
        adata.X.toarray() if sparse.issparse(adata.X) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    return df, labels, genes_after_filter, hvg_genes


def transform_hvg(adata, labels, genes_after_filter, hvg_genes):
    adata = adata[:, adata.var_names.isin(genes_after_filter)]
    adata = adata[:, genes_after_filter].copy()

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, 1e4)
    sc.pp.log1p(adata)
    adata = adata[:, hvg_genes].copy()

    df = pd.DataFrame(
        adata.X.toarray() if sparse.issparse(adata.X) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    return df, labels


def train_base(x_train, y_train, num_classes):

    base_models = [
        ("log", sklearn.linear_model.LogisticRegression(max_iter=5000)),
        ("knn", sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)),
        ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(x_train), len(base_models) * num_classes))

    for i, (name, model) in enumerate(base_models):
        print("Fitting ", name)
        for tr, va in kf.split(x_train, y_train):
            model.fit(x_train.iloc[tr], y_train.iloc[tr])
            proba = model.predict_proba(x_train.iloc[va])
            oof[va, i * num_classes:(i + 1) * num_classes] = proba
            print("Fold finished")

    for _, model in base_models:
        model.fit(x_train, y_train)

    return oof, base_models


def meta(X, base_models, num_classes):
    out = np.zeros((len(X), len(base_models) * num_classes))
    for i, (_, model) in enumerate(base_models):
        out[:, i * num_classes:(i + 1) * num_classes] = model.predict_proba(X)
    return out


# ================================
# Evaluation
# ================================
def test_transemble(model, X_meta_test, y_meta_test):

    with torch.no_grad():
        logits = model(X_meta_test)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        acc = accuracy_score(y_meta_test, preds)
        bal_acc = balanced_accuracy_score(y_meta_test, preds)
        f1 = f1_score(y_meta_test, preds, average="macro")
        precision = precision_score(y_meta_test, preds, average="macro")
        recall = recall_score(y_meta_test, preds, average="macro")
        loss = nn.CrossEntropyLoss()(logits, y_meta_test).item()

    return acc, bal_acc, f1, precision, recall, loss
