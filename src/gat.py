# Script that receives a graph in torch geometric format and trains a GAT model on it
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

EPOCHS      = 200
LR          = 1e-3
WEIGHT_DECAY= 5e-4
BATCH_SIZE  = 32
N_FOLDS     = 5
HIDDEN_DIM  = 16
HEADS       = 4
DROPOUT     = 0.3
NUM_CLASSES = 2
SEED        = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

logger = logging.getLogger(__name__)


def setup_logging():
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()       # stdout
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler("log.txt", mode="w")  # overwrite instead of append
    fh.setFormatter(fmt)
    logger.addHandler(fh)


class GAT(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, heads: int,
                 out_channels: int, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1,
                               concat=False, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)
        return self.mlp(x)


def load_dataset() -> tuple[list[Data], np.ndarray]:
    df         = pd.read_csv("rcpd_annotation_fix.csv")
    labels     = df["csam"].values.astype(int)
    graph_list = torch.load("graph_data.pt", weights_only=False)

    empty_graphs = 0
    valid_graphs = []
    valid_labels = []
    for graph, label in zip(graph_list, labels):
        graph.x          = torch.tensor(graph.x, dtype=torch.float32)
        graph.edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        graph.y          = torch.tensor(label, dtype=torch.long)

        if graph.x.shape[0] == 0 or graph.edge_index.shape[1] == 0:
            empty_graphs += 1
            continue  # skip empty graphs

        valid_graphs.append(graph)
        valid_labels.append(label)

    logger.info(f"Skipped {empty_graphs} empty graphs")
    return valid_graphs, np.array(valid_labels)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch  = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss   = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        all_labels.extend(batch.y.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:      # only one class present in fold
        auc = float("nan")
    cm  = confusion_matrix(all_labels, all_preds)
    return acc, f1, auc, cm


def run_kfold(dataset: list[Data], labels: np.ndarray):
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    in_channels  = dataset[0].x.shape[1]

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels), start=1):
        logger.info('─' * 50)
        logger.info(f"Fold {fold}/{N_FOLDS}  |  train={len(train_idx)}  val={len(val_idx)}")
        logger.info('─' * 50)

        train_loader = DataLoader([dataset[i] for i in train_idx],
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader([dataset[i] for i in val_idx],
                                  batch_size=BATCH_SIZE, shuffle=False)

        model     = GAT(in_channels, HIDDEN_DIM, HEADS, NUM_CLASSES, DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                     weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            if epoch % 20 == 0 or epoch == 1:
                acc, f1, auc, _ = evaluate(model, val_loader)
                logger.info(f"Epoch {epoch:>3d}  loss={loss:.4f}  "
                            f"val_acc={acc:.4f}  val_f1={f1:.4f}  val_auc={auc:.4f}")

        acc, f1, auc, cm = evaluate(model, val_loader)
        logger.info(f"► Final   acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
        logger.info(f"► Confusion matrix:\n{cm}")
        fold_results.append({"fold": fold, "acc": acc, "f1": f1, "auc": auc})

    return fold_results


def print_summary(fold_results: list[dict]):
    accs = [r["acc"] for r in fold_results]
    f1s  = [r["f1"]  for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    logger.info('═' * 50)
    logger.info(f"{N_FOLDS}-Fold Cross-Validation Summary")
    logger.info('═' * 50)
    logger.info(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info(f"F1 Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"ROC-AUC  : {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")
    logger.info('═' * 50)


if __name__ == "__main__":
    setup_logging()
    logger.info(f"Using device: {device}")
    dataset, labels = load_dataset()
    logger.info(f"Loaded {len(dataset)} graphs  |  "
                f"positives={labels.sum()}  negatives={(labels == 0).sum()}")

    fold_results = run_kfold(dataset, labels)
    print_summary(fold_results)