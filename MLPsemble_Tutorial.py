import os
from MLPsemble import *
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import optuna


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def main():
    print("\n--- Loading dataset ---")
    adata = sc.read_h5ad("Data/RuiChen/RuiChen.h5ad")
    labels = adata.obs["label"].astype("category")


    print(adata.shape)
    data, labels = preprocess(adata, labels)
    adata = adata[data.index].copy()

    label_map = {k: i for i, k in enumerate(sorted(labels.unique()))}
    labels = labels.map(label_map)

    X = data
    y = labels.values.astype(np.int64)
    num_classes = len(label_map)

    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []

    print("\n===== 5-FOLD NESTED CV =====")

    for fold, (train_val_idx, test_idx) in enumerate(outer_skf.split(X, y), 1):

        print(f"\n--- Fold {fold} ---")

        X_train_val = X.iloc[train_val_idx]
        y_train_val = y[train_val_idx]

        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=fold)
        train_idx_inner, val_idx_inner = next(inner_skf.split(X_train_val, y_train_val))

        adata_train = adata[train_val_idx[train_idx_inner]].copy()
        adata_val = adata[train_val_idx[val_idx_inner]].copy()
        adata_test = adata[test_idx].copy()

        labels_train = labels.iloc[train_val_idx[train_idx_inner]]
        labels_val = labels.iloc[train_val_idx[val_idx_inner]]
        labels_test = labels.iloc[test_idx]

        X_train, y_train, genes_after_filter, hvg_genes = fit_hvg(
            adata_train, labels_train
        )

        X_val, y_val = transform_hvg(
            adata_val, labels_val, genes_after_filter, hvg_genes
        )

        X_test, y_test = transform_hvg(
            adata_test, labels_test, genes_after_filter, hvg_genes
        )

        y_train = pd.Series(y_train.values)
        y_val = torch.tensor(y_val.values.astype(np.int64), dtype=torch.long)
        y_test = torch.tensor(y_test.values.astype(np.int64), dtype=torch.long)

        scaler_base = StandardScaler()

        X_train = pd.DataFrame(
            scaler_base.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )

        X_val = pd.DataFrame(
            scaler_base.transform(X_val),
            index=X_val.index,
            columns=X_val.columns
        )

        X_test = pd.DataFrame(
            scaler_base.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )

        # Base models
        X_meta_train, base_models = train_base(X_train, y_train, num_classes)

        X_meta_val = meta(X_val, base_models, num_classes)
        X_meta_test = meta(X_test, base_models, num_classes)

        scaler = StandardScaler()
        X_meta_train = torch.tensor(scaler.fit_transform(X_meta_train), dtype=torch.float32)
        X_meta_val = torch.tensor(scaler.transform(X_meta_val), dtype=torch.float32)
        X_meta_test = torch.tensor(scaler.transform(X_meta_test), dtype=torch.float32)

        y_meta_train = torch.tensor(y_train.values, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_meta_train, y_meta_train), batch_size=128, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_meta_val, y_val), batch_size=256)

        # Optimization
        def objective(trial):

            hidden_dim1 = trial.suggest_int("hidden_dim1", 64, 512, step=64)
            hidden_dim2 = trial.suggest_int("hidden_dim2", 64, 512, step=32)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            model = MetaMLP(X_meta_train.shape[1], num_classes, hidden_dim1, hidden_dim2, dropout)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.CrossEntropyLoss()

            best_val = float("inf")
            patience, wait = 10, 0

            for epoch in range(100):
                model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_loss = sum(
                        loss_fn(model(xb), yb).item()
                        for xb, yb in val_loader
                    ) / len(val_loader)

                if val_loss < best_val:
                    best_val = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            return best_val

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=25, show_progress_bar=False)

        print("Best params:", study.best_trial.params)

        p = study.best_trial.params

        # Create model and optimizer using best Params
        model = MetaMLP(X_meta_train.shape[1], num_classes,
                        p["hidden_dim1"], p["hidden_dim2"], p["dropout"])

        optimizer = optim.Adam(model.parameters(),
                               lr=p["lr"],
                               weight_decay=p["weight_decay"])

        loss_fn = nn.CrossEntropyLoss()

        best, wait = float("inf"), 0
        patience = 20

        # Train
        for epoch in range(200):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    loss_fn(model(xb), yb).item()
                    for xb, yb in val_loader
                ) / len(val_loader)

            if val_loss < best:
                best = val_loss
                wait = 0
                best_state = model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        acc, bal_acc, f1, precision, recall, loss = \
            test_transemble(model, X_meta_test, y_test)
        print(f"Fold {fold} Accuracy: {acc*100:.2f}%")

        fold_metrics.append([acc, bal_acc, f1, precision, recall, loss])

    fold_metrics = np.array(fold_metrics)

    print("\n===== FINAL RESULTS =====")
    print(f"Accuracy : {fold_metrics[:,0].mean()*100:.2f}% ± {fold_metrics[:,0].std()*100:.2f}")
    print(f"F1-score : {fold_metrics[:,2].mean():.4f} ± {fold_metrics[:,2].std():.4f}")
    print(f"Precision: {fold_metrics[:,3].mean():.4f} ± {fold_metrics[:,3].std():.4f}")
    print(f"Recall   : {fold_metrics[:,4].mean():.4f} ± {fold_metrics[:,4].std():.4f}")
    print(f"Loss     : {fold_metrics[:,5].mean():.4f} ± {fold_metrics[:,5].std():.4f}")


if __name__ == "__main__":
    main()