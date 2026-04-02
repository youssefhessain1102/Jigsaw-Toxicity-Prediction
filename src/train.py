import os
import json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataset import ToxicityDataset
from model import DistilBERTClassifier
from trainer import Trainer
from preprocessor import load_and_preprocess, save_cleaned
from utils.logger import get_logger
from utils.constants import LABELS

from sklearn.metrics import f1_score, roc_auc_score
from skmultilearn.model_selection import iterative_train_test_split
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import yaml
import random
import tempfile


def load_config() -> dict:
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


def seed_setting(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_and_save(df: pd.DataFrame, config: dict, log) -> tuple[str, str]:
    val_split = config['training']['val_split']

    X = np.arange(len(df)).reshape(-1, 1)
    Y = df[LABELS].values

    X_train, Y_train, X_val, Y_val = iterative_train_test_split(X, Y, test_size=val_split)

    train = df.iloc[X_train.flatten()].reset_index(drop=True)
    val   = df.iloc[X_val.flatten()].reset_index(drop=True)

    log.info(f"Split sizes → Train: {len(train)} | Val: {len(val)}")

    log.info(f"\n{'Label':<15} {'Full':>8} {'Train':>8} {'Val':>8}")
    log.info("-" * 42)
    for col in LABELS:
        log.info(
            f"{col:<15} {df[col].mean():>8.4f} {train[col].mean():>8.4f} {val[col].mean():>8.4f}"
        )

    def save_temp(data: pd.DataFrame) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        data.to_csv(tmp.name, index=False)
        return tmp.name

    return save_temp(train), save_temp(val)


def tune_thresholds(model, val_loader, device, log) -> dict:
    """
    Sweep threshold candidates per label on the val set and pick the one
    that maximises per-label F1.  Returns a dict {label: threshold}.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs  = np.vstack(all_probs)   # (N, 6)
    all_labels = np.vstack(all_labels)  # (N, 6)

    candidates = np.round(np.arange(0.10, 0.91, 0.05), 4) # 0.10, 0.15, 0.20, 0.25, ..., 0.90
    thresholds = {}

    log.info("\n── Per-Label Threshold Tuning (val set) ──")
    log.info(f"{'Label':<15} {'Threshold':>10} {'F1':>8}")
    log.info("-" * 36)

    for i, label in enumerate(LABELS):
        best_t, best_f1 = 0.5, 0.0
        for t in candidates:
            preds = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[label] = best_t
        log.info(f"{label:<15} {best_t:>10.2f} {best_f1:>8.4f}")

    macro_f1 = f1_score(
        all_labels,
        np.vstack([(all_probs[:, i] >= thresholds[l]).astype(int) for i, l in enumerate(LABELS)]).T,
        average="macro", zero_division=0
    )
    log.info("-" * 36)
    log.info(f"{'Macro F1 (tuned)':<15} {'':>10} {macro_f1:>8.4f}")
    log.info("──────────────────────────────────────────")

    return thresholds


def evaluate_test(model, test_loader, device, thresholds: dict, log):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds  = np.vstack(all_preds)

    # Apply a different threshold per column
    thresh_arr   = np.array([thresholds[l] for l in LABELS])   # shape (6,)
    binary_preds = (all_preds >= thresh_arr).astype(int)

    f1_macro  = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
    auc_macro = roc_auc_score(all_labels, all_preds, average='macro')

    log.info("\n── Test Set Results ──────────────────────")
    log.info(f"  F1  (macro): {f1_macro:.4f}")
    log.info(f"  AUC (macro): {auc_macro:.4f}")
    log.info("──────────────────────────────────────────")

    # Per-label breakdown
    log.info(f"\n{'Label':<15} {'Threshold':>10} {'F1':>8} {'AUC':>8}")
    log.info("-" * 44)
    for i, label in enumerate(LABELS):
        lf1 = f1_score(all_labels[:, i], binary_preds[:, i], zero_division=0)
        try:
            lauc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            lauc = float('nan')
        log.info(f"{label:<15} {thresholds[label]:>10.2f} {lf1:>8.4f} {lauc:>8.4f}")


def main():
    config = load_config()
    seed_setting(config['training']['seed'])

    log = get_logger('train', config['paths']['logs_dir'])
    log.info("Started training pipeline")

    train_df, test_df = load_and_preprocess(config)
    cleaned_df = save_cleaned(train_df, config)
    df = pd.read_csv(cleaned_df)

    train_path, val_path = split_and_save(df, config, log)

    def save_temp(data: pd.DataFrame) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        data.to_csv(tmp.name, index=False)
        return tmp.name

    test_path = save_temp(test_df)

    tokenizer = DistilBertTokenizer.from_pretrained(config['model']['name'])
    model = DistilBERTClassifier(
        config['model']['name'],
        dropout=config['model']['dropout'],
        num_labels=config['model']['num_labels']
    )

    max_length = config['model']['max_length']
    train_dataset = ToxicityDataset(max_length=max_length, df=train_path, tokenizer=tokenizer)
    test_dataset  = ToxicityDataset(max_length=max_length, df=test_path,  tokenizer=tokenizer)
    val_dataset   = ToxicityDataset(max_length=max_length, df=val_path,   tokenizer=tokenizer)

    loader_kwargs = {
        "batch_size": config['training']['batch_size'],
        "num_workers": config['training']['dataloader_num_workers'],
        "pin_memory": config['training']['pin_memory'],
    }

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    trainer.train()

    # ── Load best checkpoint ───────────────────────────────────────────────
    best_model_path = os.path.join(
        config['paths']['model_save_dir'],
        config['paths']['best_model_name']
    )
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ── Tune thresholds on val set, save to disk ───────────────────────────
    thresholds = tune_thresholds(model, val_loader=val_loader, device=device, log=log)

    thresholds_path = os.path.join(
        config['paths']['model_save_dir'],
        config['paths']['thresholds_name']
    )
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"Thresholds saved → {thresholds_path}")

    # ── Final test evaluation with tuned thresholds ────────────────────────
    evaluate_test(model, test_loader=test_loader, device=device, thresholds=thresholds, log=log)

    for path in [train_path, val_path, test_path]:
        os.remove(path)
    print("\nTemp files cleaned up successfully")


if __name__ == "__main__":
    print("started")
    main()